"""
Video Highlight Pipeline: Long Video → Short Douyin-style Highlight with AI Narration

Analyzes videos (up to 60 minutes) using Qwen2.5-VL, selects key moments,
generates Chinese narration with emotion (IndexTTS2), and assembles a
2-5 minute highlight video with effects suitable for Douyin/TikTok.

Pipeline:
    Step 1: Extract audio (ffmpeg)
    Step 2: Speech transcription (WhisperX)
    Step 3: Video analysis + event localization (Qwen2.5-VL)
    Step 4: Script generation + effect choreography (LLM API)
    Step 5: Narration synthesis + emotion injection (IndexTTS2)
    Step 6: Video assembly + effects rendering (ffmpeg + ASS)

Usage:
    # Single video
    python highlight_pipeline.py video.mp4 --ref-audio voice.wav -o highlight.mp4

    # Batch mode
    python highlight_pipeline.py --batch /path/to/videos --ref-audio voice.wav -o /output

    # Resume after interruption (auto-detected)
    python highlight_pipeline.py video.mp4 --ref-audio voice.wav

Requires: whisperx, transformers, qwen-vl-utils, openai, ffmpeg
"""

import argparse
import gc
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_DURATION_MIN = 120       # minimum output duration (seconds)
TARGET_DURATION_MAX = 300       # maximum output duration (seconds)
MIN_IMPORTANCE = 4              # minimum VL importance score to consider
CROSSFADE_DURATION = 0.8        # transition duration (seconds)
NARRATION_VOLUME = 1.0          # narration audio volume
ORIGINAL_VOLUME = 0.25          # original audio volume in mix
TITLE_CARD_DURATION = 2.5       # title card duration (seconds)
PORTRAIT_WIDTH = 1080            # Douyin standard width
PORTRAIT_HEIGHT = 1920           # Douyin standard height
CHECKPOINT_FILE = "checkpoint.json"
TTS_SAMPLE_RATE = 24000
X264_ARGS = ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
AAC_ARGS = ["-c:a", "aac", "-b:a", "128k"]

# Qwen2.5-VL defaults (tuned for L4 24GB VRAM with 4-bit quantization)
VL_FPS = 1.0
VL_TOTAL_PIXELS = 8192 * 28 * 28
VL_MIN_PIXELS = 128 * 28 * 28
VL_MAX_PIXELS = 360 * 420
VL_MAX_FRAMES = 256

# ffprobe result cache (avoids repeated subprocess spawns for the same file)
_video_info_cache = {}

# ---------------------------------------------------------------------------
# ffmpeg version detection (for xfade transition compatibility)
# ---------------------------------------------------------------------------

def _get_ffmpeg_version():
    """返回 ffmpeg 主版本号，如 (5, 1)。检测失败返回 (4, 0)。"""
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        m = re.search(r"ffmpeg version (\d+)\.(\d+)", r.stdout)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    except Exception:
        pass
    return (4, 0)

_FFMPEG_VER = _get_ffmpeg_version()

# Available effects for LLM to choose from
_TRANSITIONS_BASE = [
    "fade", "wipeleft", "wiperight", "slideright", "slideleft",
    "circleopen", "circleclose", "radial", "pixelize", "dissolve",
    "fadeblack", "fadewhite",
]
_TRANSITIONS_5X = [
    "zoomin", "smoothleft", "smoothright", "smoothup", "smoothdown",
    "squeezeh", "squeezev", "hlwind", "hrwind", "vuwind", "vdwind",
    "coverleft", "coverright", "revealleft", "revealright",
]
TRANSITIONS = _TRANSITIONS_BASE + (_TRANSITIONS_5X if _FFMPEG_VER >= (5, 0) else [])
COLOR_GRADES = ["cinematic_warm", "cinematic_cool", "vintage", "high_contrast", "none"]
SUBTITLE_STYLES = ["word_by_word_highlight", "fade_in", "pop_up", "typewriter"]
EMOTION_DIMS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_ffmpeg(*args):
    """Run an ffmpeg command, raising with stderr on failure."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", *args],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode()}") from e


def _run_ffprobe(*args):
    """Run an ffprobe command and return stdout."""
    result = subprocess.run(
        ["ffprobe", *args],
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def _get_video_info(video_path):
    """Get video duration, width, height, and fps (cached)."""
    if video_path in _video_info_cache:
        return _video_info_cache[video_path]
    info = {}
    raw = _run_ffprobe(
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    )
    data = json.loads(raw)
    stream = data.get("streams", [{}])[0]
    info["width"] = int(stream.get("width", 1920))
    info["height"] = int(stream.get("height", 1080))
    fps_str = stream.get("r_frame_rate", "30/1")
    num, den = fps_str.split("/")
    info["fps"] = float(num) / float(den) if float(den) != 0 else 30.0
    # Duration: prefer format duration, fall back to stream
    fmt_dur = data.get("format", {}).get("duration")
    stream_dur = stream.get("duration")
    info["duration"] = float(fmt_dur or stream_dur or 0)
    _video_info_cache[video_path] = info
    return info


def _free_vram():
    """Run garbage collection and free GPU memory. Caller must `del` models first."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _detect_device():
    """Detect available compute device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_audio_duration(path):
    """Return duration of an audio file in seconds."""
    info = sf.info(path)
    return info.duration


# ---------------------------------------------------------------------------
# Checkpoint — save/resume pipeline progress
# ---------------------------------------------------------------------------

def _save_checkpoint(work_dir, step, **data):
    """Save pipeline progress after each step."""
    data.pop("step", None)
    data["step"] = step
    path = os.path.join(work_dir, CHECKPOINT_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Checkpoint saved: step {step}")


def _load_checkpoint(work_dir):
    """Load checkpoint if exists."""
    path = os.path.join(work_dir, CHECKPOINT_FILE)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _clear_checkpoint(work_dir):
    """Remove checkpoint file after successful completion."""
    path = os.path.join(work_dir, CHECKPOINT_FILE)
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# LLM Client (same pattern as dub_pipeline)
# ---------------------------------------------------------------------------

class LLMClient:
    """Wrapper for OpenAI-compatible chat APIs."""

    def __init__(self, api_key, api_base="https://api.openai.com/v1", model="gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model

    def chat(self, prompt, temperature=0.3):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Step 1: Extract audio
# ---------------------------------------------------------------------------

def extract_audio(video_path, work_dir):
    """Extract audio track from video as WAV."""
    audio_path = os.path.join(work_dir, "full_audio.wav")
    if os.path.exists(audio_path):
        print("  Audio already extracted, skipping.")
        return audio_path
    _run_ffmpeg(
        "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path,
    )
    print(f"  Extracted audio: {audio_path}")
    return audio_path


# ---------------------------------------------------------------------------
# Step 2: Speech transcription (WhisperX)
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path, work_dir):
    """Transcribe audio using WhisperX with timestamps."""
    import whisperx

    transcript_path = os.path.join(work_dir, "transcript.json")
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        print(f"  Loaded cached transcript ({len(transcript)} segments)")
        return transcript

    device = _detect_device()
    compute_type = "float16" if device == "cuda" else "int8"

    print("  Loading Whisper model...")
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    print("  Transcribing...")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)

    # Word-level alignment
    print("  Aligning words...")
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device,
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device,
    )

    # Flatten to simple segment list
    transcript = []
    for seg in result["segments"]:
        transcript.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        })

    # Save
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    # Unload Whisper
    del model, align_model
    _free_vram()
    print(f"  Transcribed {len(transcript)} segments, Whisper unloaded.")
    return transcript


# ---------------------------------------------------------------------------
# Step 3: Qwen2.5-VL video analysis + event localization
# ---------------------------------------------------------------------------

def _build_vl_prompt(transcript_text, video_duration_str):
    """Build the analysis prompt for Qwen2.5-VL."""
    return f"""你是一位专业的短视频内容策划师。请分析这个视频（总时长约{video_duration_str}），完成以下任务：

1. 判断视频类型（教育/新闻/娱乐/纪录片/演讲等）
2. 概括整体内容
3. **定位所有关键事件**，给出精确的起止时间（秒数），并评估每个事件作为短视频素材的价值

以下是该视频的语音转录文本，辅助你理解内容：
{transcript_text}

请返回严格的JSON格式（不要添加markdown围栏）：
{{
  "video_type": "视频类型",
  "overall_summary": "整体内容概要（2-3句）",
  "events": [
    {{
      "start_time": "起始秒数",
      "end_time": "结束秒数",
      "description": "事件描述",
      "visual_highlights": "视觉亮点",
      "emotional_tone": "情感基调描述",
      "emotion_tags": ["从以下选择: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm"],
      "importance": 8,
      "reason": "为什么这个片段适合/不适合作为精华"
    }}
  ]
}}

注意：
- importance 评分 1-10，越高越适合做短视频素材
- emotion_tags 必须从这8个中选择: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm
- 时间戳必须是秒数（如 "123.5"），与视频实际时间对齐
- 请尽量发现所有有价值的片段，宁多勿少"""


def analyze_video(video_path, transcript, work_dir, vl_model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    """Analyze video using Qwen2.5-VL with native video understanding."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    analysis_path = os.path.join(work_dir, "video_analysis.json")
    if os.path.exists(analysis_path):
        with open(analysis_path, "r", encoding="utf-8") as f:
            analysis = json.load(f)
        print(f"  Loaded cached analysis ({len(analysis.get('events', []))} events)")
        return analysis

    video_info = _get_video_info(video_path)
    duration_min = video_info["duration"] / 60
    video_duration_str = f"{duration_min:.1f}分钟"

    # Build transcript text (truncate if too long)
    transcript_lines = []
    for seg in transcript:
        t = f"[{seg['start']:.1f}s] {seg['text']}"
        transcript_lines.append(t)
    transcript_text = "\n".join(transcript_lines)
    # Limit transcript to ~4000 chars to leave room for video tokens
    if len(transcript_text) > 4000:
        transcript_text = transcript_text[:4000] + "\n... (转录文本已截断)"

    prompt = _build_vl_prompt(transcript_text, video_duration_str)

    # Load model (4-bit quantized to fit L4 24GB with video frames)
    print(f"  Loading {vl_model_name}...")
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"
    try:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        print("  Using 4-bit quantization (saves ~10GB VRAM)")
    except ImportError:
        quantization_config = None
        print("  bitsandbytes not available, loading in bfloat16")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vl_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        quantization_config=quantization_config,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(vl_model_name)

    # Extract frames as JPEG images with ffmpeg, then pass as frame list to
    # Qwen2.5-VL. This bypasses all video decoding libraries (torchvision,
    # torchcodec, decord) and their associated crashes on Colab.
    # Temporal encoding is preserved via the fps parameter → mRoPE position IDs.
    video_info = _get_video_info(video_path)
    video_dur = video_info["duration"]
    nframes = min(int(video_dur * VL_FPS), VL_MAX_FRAMES)
    extract_fps = nframes / video_dur if video_dur > 0 else VL_FPS
    frames_dir = os.path.join(work_dir, "vl_frames")
    os.makedirs(frames_dir, exist_ok=True)
    existing_frames = sorted(
        f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".jpg")
    )
    if len(existing_frames) < nframes:
        print(f"  Extracting {nframes} frames from {video_dur:.0f}s video (ffmpeg)...")
        _run_ffmpeg(
            "-i", video_path,
            "-vf", f"fps={extract_fps:.6f},scale=480:-2",
            "-q:v", "2",
            os.path.join(frames_dir, "frame_%04d.jpg"),
        )
        existing_frames = sorted(
            f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".jpg")
        )
    print(f"  Using {len(existing_frames)} extracted frames")

    # Build frame paths and compute fps for temporal alignment
    frame_paths = [
        f"file://{os.path.abspath(os.path.join(frames_dir, f))}"
        for f in existing_frames
    ]
    frame_fps = len(existing_frames) / video_dur if video_dur > 0 else VL_FPS

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frame_paths,
                    "fps": frame_fps,
                    "total_pixels": VL_TOTAL_PIXELS,
                    "min_pixels": VL_MIN_PIXELS,
                    "max_pixels": VL_MAX_PIXELS,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Process and generate
    print("  Analyzing video (this may take a while for long videos)...")
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        **video_kwargs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=8192)
    # Trim input tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]

    # Unload VL model
    del model, processor
    _free_vram()
    print("  Qwen2.5-VL unloaded.")

    # Parse JSON response
    analysis = _parse_json_response(response)
    if analysis is None:
        raise RuntimeError(f"Failed to parse VL response:\n{response[:500]}")

    # Save
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"  Found {len(analysis.get('events', []))} events.")
    return analysis


def _parse_json_response(text):
    """Parse JSON from LLM/VL response, stripping markdown fences.

    Handles truncated JSON by attempting to repair incomplete output
    (e.g. closing unclosed arrays/objects).
    """
    # Strip markdown code fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON object/array in text
    for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue
    # Try to repair truncated JSON (output cut off by max_new_tokens)
    return _repair_truncated_json(text)


def _repair_truncated_json(text):
    """Attempt to repair truncated JSON by closing unclosed structures."""
    # Find the start of the JSON object
    start = text.find("{")
    if start < 0:
        return None
    text = text[start:]
    # Truncate at the last complete value boundary (after , or after })
    # Remove trailing incomplete key-value pair
    text = re.sub(r',\s*"[^"]*"?\s*:?\s*("(?:[^"\\]|\\.)*)?$', "", text)
    text = re.sub(r',\s*\{[^}]*$', "", text)  # remove trailing incomplete object in array
    # Count unclosed brackets and close them
    opens = 0
    open_sq = 0
    for ch in text:
        if ch == "{":
            opens += 1
        elif ch == "}":
            opens -= 1
        elif ch == "[":
            open_sq += 1
        elif ch == "]":
            open_sq -= 1
    text += "]" * open_sq + "}" * opens
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Step 4: LLM script generation + effect choreography
# ---------------------------------------------------------------------------

def _build_script_prompt(analysis, target_duration):
    """Build the script generation prompt for LLM."""
    events = analysis.get("events", [])
    # Filter by importance
    events = [e for e in events if e.get("importance", 0) >= MIN_IMPORTANCE]

    return f"""你是一位抖音爆款视频的脚本编剧 + 视频特效导演。

以下是一个{analysis.get('video_type', '未知')}类型视频的关键事件分析：

整体概要：{analysis.get('overall_summary', '')}

事件列表：
{json.dumps(events, ensure_ascii=False, indent=2)}

请从中选取最精华的事件，生成一个约{target_duration}秒的短视频解说脚本。

## 你需要输出：

1. 为每个选中的片段编写中文旁白（口语化、有感染力、适合配音朗读）
2. 为每条旁白设计**情感向量**（8维浮点数组，每个0-1）
3. 为每个片段选择**视觉特效**

## 情感向量说明
格式：[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
- 紧张悬疑 → 提高 afraid/surprised
- 感人段落 → 提高 sad/melancholic
- 激昂振奋 → 提高 happy
- 平静叙述 → 提高 calm
- 向量总和建议不超过0.8

## 可用特效菜单

转场(transition): {', '.join(TRANSITIONS)}
调色(color_grade): {', '.join(COLOR_GRADES)}
字幕风格(subtitle_style): {', '.join(SUBTITLE_STYLES)}
可选特效(effects):
  - slow_motion: 指定时间范围和倍率（如0.5=半速）
  - ken_burns: 缩放平移（适合静态画面）
标题卡片(title_card): 段落间2-3秒的文字过渡卡（可选，不是每段都需要）

## 要求
1. 总旁白时长控制在{target_duration}秒左右
2. 开头必须有hook（吸引观众3秒内不划走）
3. 结尾有总结或金句
4. 按叙事逻辑排列，确保连贯性
5. 特效要配合内容情绪，不要为了炫技而炫技

请返回严格的JSON数组（不要添加markdown围栏）：
[
  {{
    "segment_id": 0,
    "start_time": 123.5,
    "end_time": 245.8,
    "narration": "你绝对想不到，接下来发生的事情...",
    "emo_vector": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.2],
    "transition": "circleopen",
    "effects": {{
      "color_grade": "cinematic_warm",
      "subtitle_style": "word_by_word_highlight",
      "slow_motion": {{"time_range": [130.0, 133.0], "factor": 0.5}}
    }},
    "title_card": "开头标题文字（可选，设为null表示无）"
  }}
]"""


def generate_script(analysis, llm_client, target_duration=180):
    """Generate narration script with emotion vectors and effect choreography."""
    prompt = _build_script_prompt(analysis, target_duration)

    print("  Generating script...")
    response = llm_client.chat(prompt, temperature=0.7)

    script = _parse_json_response(response)
    if script is None:
        raise RuntimeError(f"Failed to parse LLM script response:\n{response[:500]}")

    if isinstance(script, dict):
        script = script.get("segments", script.get("script", []))

    # Validate and clean up each item
    cleaned = []
    for item in script:
        cleaned_item = {
            "segment_id": item.get("segment_id", len(cleaned)),
            "start_time": float(item.get("start_time", 0)),
            "end_time": float(item.get("end_time", 0)),
            "narration": item.get("narration", ""),
            "emo_vector": item.get("emo_vector", [0, 0, 0, 0, 0, 0, 0, 0.5]),
            "transition": item.get("transition", "fade"),
            "effects": item.get("effects", {}),
            "title_card": item.get("title_card"),
        }
        # Validate transition
        if cleaned_item["transition"] not in TRANSITIONS:
            cleaned_item["transition"] = "fade"
        # Validate emo_vector length
        vec = cleaned_item["emo_vector"]
        if len(vec) != 8:
            cleaned_item["emo_vector"] = [0, 0, 0, 0, 0, 0, 0, 0.5]
        # Clamp emo_vector values to [0, 1]
        cleaned_item["emo_vector"] = [max(0.0, min(1.0, v)) for v in cleaned_item["emo_vector"]]
        cleaned.append(cleaned_item)

    print(f"  Script generated: {len(cleaned)} segments")
    for i, item in enumerate(cleaned):
        dur = item["end_time"] - item["start_time"]
        print(f"    [{i}] {item['start_time']:.1f}-{item['end_time']:.1f}s "
              f"({dur:.1f}s) transition={item['transition']} | "
              f"{item['narration'][:40]}...")

    return cleaned


# ---------------------------------------------------------------------------
# Step 5: TTS narration generation with emotion injection
# ---------------------------------------------------------------------------

def _init_tts(model_dir, use_fp16):
    """Initialize IndexTTS2 model."""
    from indextts.infer_v2 import IndexTTS2
    return IndexTTS2(
        cfg_path=os.path.join(model_dir, "config.yaml"),
        model_dir=model_dir,
        use_fp16=use_fp16,
    )


def generate_narration(script, ref_audio, work_dir, model_dir="checkpoints", use_fp16=True):
    """Generate narration audio with emotion injection for each script segment."""
    tts_dir = os.path.join(work_dir, "tts_output")
    os.makedirs(tts_dir, exist_ok=True)

    # Check if all already generated
    all_exist = all(
        os.path.exists(os.path.join(tts_dir, f"narration_{i:04d}.wav"))
        for i in range(len(script))
    )
    if all_exist:
        print("  All narration files already exist, loading durations...")
        for i, item in enumerate(script):
            wav_path = os.path.join(tts_dir, f"narration_{i:04d}.wav")
            item["wav_path"] = wav_path
            item["wav_duration"] = get_audio_duration(wav_path)
        return script

    print("  Loading IndexTTS2...")
    tts = _init_tts(model_dir, use_fp16)

    for i, item in enumerate(script):
        output_path = os.path.join(tts_dir, f"narration_{i:04d}.wav")

        if os.path.exists(output_path):
            item["wav_path"] = output_path
            item["wav_duration"] = get_audio_duration(output_path)
            print(f"  [{i}] Cached: {item['wav_duration']:.1f}s")
            continue

        narration = item["narration"]
        emo_vector = item.get("emo_vector")

        # Normalize emotion vector
        if emo_vector and any(v > 0 for v in emo_vector):
            emo_vector = tts.normalize_emo_vec(emo_vector)
        else:
            emo_vector = None

        try:
            tts.infer(
                spk_audio_prompt=ref_audio,
                text=narration,
                output_path=output_path,
                emo_vector=emo_vector,
                verbose=False,
            )
        except Exception as e:
            # Fallback: generate silence placeholder
            print(f"  [{i}] TTS failed ({e}), using silence placeholder")
            silence_dur = max(2.0, len(narration) * 0.15)
            silence = np.zeros(int(silence_dur * TTS_SAMPLE_RATE), dtype=np.float32)
            sf.write(output_path, silence, TTS_SAMPLE_RATE)

        item["wav_path"] = output_path
        item["wav_duration"] = get_audio_duration(output_path)
        print(f"  [{i}] Generated: {item['wav_duration']:.1f}s | "
              f"emo={[f'{v:.2f}' for v in (emo_vector or [])]} | "
              f"{narration[:30]}...")

    # Unload TTS
    del tts
    _free_vram()
    print("  IndexTTS2 unloaded.")
    return script


# ---------------------------------------------------------------------------
# Step 6: Video assembly + effects rendering
# ---------------------------------------------------------------------------

# 6a. Portrait mode adaptation

def prepare_portrait(video_path, work_dir):
    """Convert video to 9:16 portrait if needed, with blurred background fill."""
    info = _get_video_info(video_path)
    w, h = info["width"], info["height"]

    portrait_path = os.path.join(work_dir, "portrait_source.mp4")
    if os.path.exists(portrait_path):
        return portrait_path

    aspect = w / h
    if aspect > 1.0:
        # Landscape → portrait with blurred background fill
        _run_ffmpeg(
            "-i", video_path,
            "-filter_complex",
            f"[0:v]scale={PORTRAIT_WIDTH}:{PORTRAIT_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={PORTRAIT_WIDTH}:{PORTRAIT_HEIGHT},gblur=sigma=30[bg];"
            f"[0:v]scale={PORTRAIT_WIDTH}:{PORTRAIT_HEIGHT}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2",
            "-c:a", "copy",
            portrait_path,
        )
    else:
        # Already portrait or square — scale and pad to 9:16
        _run_ffmpeg(
            "-i", video_path,
            "-vf", f"scale={PORTRAIT_WIDTH}:{PORTRAIT_HEIGHT}:force_original_aspect_ratio=decrease,"
                   f"pad={PORTRAIT_WIDTH}:{PORTRAIT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black",
            "-c:a", "copy",
            portrait_path,
        )

    print(f"  Portrait video prepared: {w}x{h} → {PORTRAIT_WIDTH}x{PORTRAIT_HEIGHT}")
    return portrait_path


# 6b. Cut segments and apply per-segment effects

def _get_color_grade_filter(grade):
    """Return ffmpeg filter string for color grading."""
    grades = {
        "cinematic_warm": "eq=saturation=1.2:brightness=0.02,colorbalance=rs=0.1:gm=0.05:bh=-0.05",
        "cinematic_cool": "eq=saturation=1.1:brightness=-0.02,colorbalance=rs=-0.08:gm=0.02:bh=0.12",
        "vintage": "curves=preset=vintage,eq=saturation=0.85",
        "high_contrast": "eq=contrast=1.3:saturation=1.2",
        "none": "null",
    }
    return grades.get(grade, "null")


def cut_and_apply_effects(portrait_path, script, work_dir):
    """Cut segments from portrait video and apply per-segment effects."""
    segments_dir = os.path.join(work_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    for i, item in enumerate(script):
        seg_path = os.path.join(segments_dir, f"seg_{i:04d}.mp4")
        if os.path.exists(seg_path):
            item["seg_path"] = seg_path
            continue

        start = item["start_time"]
        end = item["end_time"]
        effects = item.get("effects", {})

        # Build video filter chain
        vfilters = []

        # Color grading
        color_grade = effects.get("color_grade", "none")
        grade_filter = _get_color_grade_filter(color_grade)
        if grade_filter != "null":
            vfilters.append(grade_filter)

        # Ken Burns zoom+pan (for static/chart-heavy segments)
        if effects.get("ken_burns"):
            vfilters.append(
                f"zoompan=z='min(zoom+0.001,1.3)':d=125:"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                f":s={PORTRAIT_WIDTH}x{PORTRAIT_HEIGHT}:fps=30"
            )

        # Slow motion (applied to entire segment for simplicity)
        slow = effects.get("slow_motion")
        if slow:
            factor = slow.get("factor", 0.5)
            vfilters.append(f"setpts={1/factor}*PTS")

        vfilter_str = ",".join(vfilters) if vfilters else "null"

        # Audio filter for slow motion
        afilter = "anull"
        if slow:
            factor = slow.get("factor", 0.5)
            # atempo only supports 0.5-2.0
            atempo_val = max(0.5, min(2.0, factor))
            afilter = f"atempo={atempo_val}"

        # -ss before -i for fast input-level seeking; -t is relative to seek point
        duration = end - start
        _run_ffmpeg(
            "-ss", str(start),
            "-i", portrait_path,
            "-t", str(duration),
            "-vf", vfilter_str,
            "-af", afilter,
            *X264_ARGS, *AAC_ARGS,
            seg_path,
        )
        item["seg_path"] = seg_path
        item["seg_duration"] = _get_video_info(seg_path)["duration"]
        print(f"  [{i}] Cut {start:.1f}-{end:.1f}s ({duration:.1f}s) grade={color_grade}")

    return script


# 6c. Title cards

def generate_title_cards(script, work_dir):
    """Generate title card video clips for segments that have them."""
    cards_dir = os.path.join(work_dir, "title_cards")
    os.makedirs(cards_dir, exist_ok=True)

    for i, item in enumerate(script):
        title = item.get("title_card")
        if not title:
            item["card_path"] = None
            continue

        card_path = os.path.join(cards_dir, f"card_{i:04d}.mp4")
        if os.path.exists(card_path):
            item["card_path"] = card_path
            continue

        # Create a black background with animated text
        # Use drawtext with fade-in effect
        _run_ffmpeg(
            "-f", "lavfi",
            "-i", f"color=c=black:s={PORTRAIT_WIDTH}x{PORTRAIT_HEIGHT}:d={TITLE_CARD_DURATION}:r=30",
            "-f", "lavfi",
            "-i", f"anullsrc=r=44100:cl=stereo:d={TITLE_CARD_DURATION}",
            "-vf",
            f"drawtext=text='{_escape_ffmpeg_text(title)}':"
            f"fontsize=60:fontcolor=white:"
            f"x=(w-text_w)/2:y=(h-text_h)/2:"
            f"alpha='if(lt(t,0.5),t/0.5,if(gt(t,{TITLE_CARD_DURATION - 0.5}),"
            f"({TITLE_CARD_DURATION}-t)/0.5,1))'",
            *X264_ARGS, "-c:a", "aac",
            "-shortest",
            card_path,
        )
        item["card_path"] = card_path
        print(f"  [{i}] Title card: {title[:30]}...")

    return script


def _escape_ffmpeg_text(text):
    """Escape special characters for ffmpeg drawtext filter."""
    # Escape characters that have special meaning in drawtext
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "'\\''")
    text = text.replace(":", "\\:")
    text = text.replace("%", "%%")
    return text


# 6d. Concatenation with transitions

def concat_with_transitions(script, work_dir):
    """Concatenate segments with xfade transitions."""
    # Build list of clips (interleave title cards and segments)
    clips = []
    for item in script:
        if item.get("card_path"):
            clips.append({"path": item["card_path"], "transition": "fadeblack"})
        clips.append({"path": item["seg_path"], "transition": item.get("transition", "fade")})

    if not clips:
        raise RuntimeError("No clips to concatenate")

    if len(clips) == 1:
        return clips[0]["path"]

    concat_path = os.path.join(work_dir, "concatenated.mp4")
    if os.path.exists(concat_path):
        return concat_path

    # Get duration of each clip
    for clip in clips:
        info = _get_video_info(clip["path"])
        clip["duration"] = info["duration"]

    # Determine effective crossfade (must be shorter than any clip)
    min_clip_dur = min(c["duration"] for c in clips)
    xfade_dur = min(CROSSFADE_DURATION, min_clip_dur * 0.4)  # cap at 40% of shortest clip
    xfade_dur = max(0.1, xfade_dur)  # floor at 0.1s

    # Build xfade filter chain (N clips → N-1 xfade operations)
    filter_parts = []

    offset = clips[0]["duration"] - xfade_dur
    prev_label = "[0:v]"
    prev_alabel = "[0:a]"

    for i in range(1, len(clips)):
        transition = clips[i].get("transition", "fade")
        out_label = f"[v{i}]" if i < len(clips) - 1 else "[vout]"
        aout_label = f"[a{i}]" if i < len(clips) - 1 else "[aout]"

        filter_parts.append(
            f"{prev_label}[{i}:v]xfade=transition={transition}:"
            f"duration={xfade_dur}:offset={offset:.3f}{out_label}"
        )
        filter_parts.append(
            f"{prev_alabel}[{i}:a]acrossfade=d={xfade_dur}{aout_label}"
        )

        # Update offset for next transition
        offset += clips[i]["duration"] - xfade_dur
        prev_label = out_label
        prev_alabel = aout_label

    filter_complex = ";".join(filter_parts)

    # Build ffmpeg command
    cmd = []
    for clip in clips:
        cmd.extend(["-i", clip["path"]])
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        *X264_ARGS, *AAC_ARGS,
        concat_path,
    ])

    _run_ffmpeg(*cmd)
    print(f"  Concatenated {len(clips)} clips with transitions.")
    return concat_path


# 6e. Timeline + ASS subtitle generation


def _compute_timeline(script):
    """Compute start offset for each script segment in the concatenated video.

    Returns list of floats (one per script item) — the start time of each
    narration in the final concatenated timeline.
    """
    offsets = []
    current_time = 0.0
    for i, item in enumerate(script):
        if item.get("card_path"):
            crossfade = CROSSFADE_DURATION if i > 0 else 0
            current_time += TITLE_CARD_DURATION - crossfade
        offsets.append(current_time)
        seg_dur = item.get("seg_duration") or _get_video_info(item["seg_path"])["duration"]
        crossfade = CROSSFADE_DURATION if i < len(script) - 1 else 0
        current_time += seg_dur - crossfade
    return offsets


def _format_ass_time(seconds):
    """Format seconds as ASS timestamp: H:MM:SS.CC"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _generate_ass_header():
    """Generate ASS file header with styles."""
    return """[Script Info]
Title: Highlight Narration
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Sans SC,52,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,30,30,120,1
Style: Highlight,Noto Sans SC,52,&H0000BFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,30,30,120,1
Style: TitleCard,Noto Sans SC,64,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,5,30,30,400,1
Style: ProgressBar,Arial,10,&H0000BFFF,&H000000FF,&H0000BFFF,&H0000BFFF,0,0,0,0,100,100,0,0,3,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def generate_ass_subtitles(script, total_duration, work_dir):
    """Generate ASS subtitle file with animated effects."""
    ass_path = os.path.join(work_dir, "subtitles.ass")
    lines = [_generate_ass_header()]
    offsets = _compute_timeline(script)

    for i, item in enumerate(script):
        narration = item["narration"]
        wav_dur = item.get("wav_duration", 5.0)
        subtitle_style = item.get("effects", {}).get("subtitle_style", "fade_in")
        start = offsets[i]
        end = start + wav_dur

        if subtitle_style == "word_by_word_highlight":
            chars = list(narration)
            if chars:
                char_dur = wav_dur / len(chars)
                for j in range(len(chars)):
                    before = narration[:j]
                    current = narration[j]
                    after = narration[j + 1:]
                    text = f"{before}{{\\c&H00BFFF&}}{current}{{\\c&HFFFFFF&}}{after}"
                    c_start = start + j * char_dur
                    c_end = start + (j + 1) * char_dur
                    lines.append(
                        f"Dialogue: 0,{_format_ass_time(c_start)},{_format_ass_time(c_end)},"
                        f"Default,,0,0,0,,{text}\n"
                    )
        elif subtitle_style == "pop_up":
            text = f"{{\\fscx80\\fscy80\\t(0,300,\\fscx100\\fscy100)\\fad(200,200)}}{narration}"
            lines.append(
                f"Dialogue: 0,{_format_ass_time(start)},{_format_ass_time(end)},"
                f"Default,,0,0,0,,{text}\n"
            )
        elif subtitle_style == "typewriter":
            chars = list(narration)
            if chars:
                char_dur = wav_dur / len(chars)
                for j in range(len(chars)):
                    c_start = start + j * char_dur
                    c_end = start + (j + 1) * char_dur if j < len(chars) - 1 else end
                    visible_text = narration[:j + 1]
                    lines.append(
                        f"Dialogue: 0,{_format_ass_time(c_start)},{_format_ass_time(c_end)},"
                        f"Default,,0,0,0,,{visible_text}\n"
                    )
        else:
            text = f"{{\\fad(300,300)}}{narration}"
            lines.append(
                f"Dialogue: 0,{_format_ass_time(start)},{_format_ass_time(end)},"
                f"Default,,0,0,0,,{text}\n"
            )

    # Progress bar
    lines.append(
        f"Dialogue: 1,{_format_ass_time(0)},{_format_ass_time(total_duration)},"
        f"ProgressBar,,0,0,0,,{{\\p1\\clip(0,1910,0,1920)"
        f"\\t(0,{int(total_duration * 1000)},\\clip(0,1910,1080,1920))}}"
        f"m 0 0 l 1080 0 l 1080 10 l 0 10{{\\p0}}\n"
    )

    with open(ass_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"  Generated ASS subtitles: {ass_path}")
    return ass_path


# 6f. Final mix

def generate_srt(script, work_dir):
    """Generate SRT subtitle file."""
    srt_path = os.path.join(work_dir, "subtitles.srt")
    offsets = _compute_timeline(script)

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(script):
            start = offsets[i]
            end = start + item.get("wav_duration", 5.0)
            f.write(f"{i + 1}\n")
            f.write(f"{_format_srt_time(start)} --> {_format_srt_time(end)}\n")
            f.write(f"{item['narration']}\n\n")

    print(f"  Generated SRT: {srt_path}")
    return srt_path


def _format_srt_time(seconds):
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    ms = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{int(s):02d},{ms:03d}"


def final_mix(concat_path, script, audio_path, work_dir, output_path):
    """Mix narration + original audio, burn subtitles, output final video."""
    # Build narration audio track
    narration_track = os.path.join(work_dir, "narration_full.wav")
    if not os.path.exists(narration_track):
        _build_narration_track(script, concat_path, narration_track)

    # Get total duration
    total_duration = _get_video_info(concat_path)["duration"]

    # Generate ASS subtitles
    ass_path = generate_ass_subtitles(script, total_duration, work_dir)

    # Generate SRT (standalone)
    srt_output = str(Path(output_path).with_suffix(".srt"))
    generate_srt(script, work_dir)
    # Copy SRT to output location
    import shutil
    shutil.copy2(os.path.join(work_dir, "subtitles.srt"), srt_output)

    # Final mix: video + narration (full vol) + original audio (low vol) + subtitles
    _run_ffmpeg(
        "-i", concat_path,
        "-i", narration_track,
        "-filter_complex",
        f"[0:a]volume={ORIGINAL_VOLUME}[bg];"
        f"[1:a]volume={NARRATION_VOLUME}[speech];"
        f"[speech][bg]amix=inputs=2:duration=longest:normalize=0[aout];"
        f"[0:v]ass='{ass_path.replace(':', '\\:').replace('\\', '/')}'[vout]",
        "-map", "[vout]", "-map", "[aout]",
        *X264_ARGS, "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    )
    print(f"  Final output: {output_path}")
    return output_path


def _build_narration_track(script, concat_path, output_path):
    """Build a full-length narration audio track aligned to the concatenated video."""
    total_duration = _get_video_info(concat_path)["duration"]
    total_samples = int(total_duration * TTS_SAMPLE_RATE)
    track = np.zeros(total_samples, dtype=np.float32)
    offsets = _compute_timeline(script)

    for i, item in enumerate(script):
        wav_path = item.get("wav_path")
        if not wav_path or not os.path.exists(wav_path):
            continue
        audio, audio_sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio_sr != TTS_SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=audio_sr, target_sr=TTS_SAMPLE_RATE)

        start_sample = int(offsets[i] * TTS_SAMPLE_RATE)
        end_sample = min(start_sample + len(audio), total_samples)
        actual_len = end_sample - start_sample
        if actual_len > 0:
            track[start_sample:end_sample] = audio[:actual_len]

    sf.write(output_path, track, TTS_SAMPLE_RATE)
    print(f"  Built narration track: {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def create_highlight(
    video_path,
    ref_audio,
    output_path=None,
    work_dir="highlight_workspace",
    llm_api_key=None,
    llm_api_base="https://api.openai.com/v1",
    llm_model="gpt-4o-mini",
    vl_model="Qwen/Qwen2.5-VL-7B-Instruct",
    model_dir="checkpoints",
    target_duration=180,
    use_fp16=True,
    cleanup=False,
):
    """
    Main pipeline: create a highlight video from a long video.

    Args:
        video_path: Input video file path.
        ref_audio: Reference audio for TTS voice cloning.
        output_path: Output video path. Defaults to {input}_highlight.mp4.
        work_dir: Directory for intermediate files.
        llm_api_key: API key for OpenAI-compatible LLM.
        llm_api_base: Base URL for the LLM API.
        llm_model: Model name for script generation.
        vl_model: Qwen2.5-VL model name/path.
        model_dir: IndexTTS2 checkpoint directory.
        target_duration: Target output duration in seconds.
        use_fp16: Use FP16 for IndexTTS2 inference.
        cleanup: Delete intermediate files after completion.
    """
    video_path = str(video_path)
    ref_audio = str(ref_audio)
    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(video_path).parent / f"{stem}_highlight.mp4")

    video_work_dir = os.path.join(work_dir, Path(video_path).stem)
    os.makedirs(video_work_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Highlight Pipeline")
    print(f"Input:   {video_path}")
    print(f"Ref:     {ref_audio}")
    print(f"Output:  {output_path}")
    print(f"Work:    {video_work_dir}")
    print(f"Target:  {target_duration}s ({target_duration/60:.1f}min)")
    print(f"ffmpeg:  {_FFMPEG_VER[0]}.{_FFMPEG_VER[1]} ({len(TRANSITIONS)} transitions available)")
    print(f"{'=' * 60}")

    # Load checkpoint
    ckpt = _load_checkpoint(video_work_dir)
    done = ckpt.get("step", 0) if ckpt else 0
    state = ckpt or {}

    # --- Step 1: Extract audio ---
    if done < 1:
        print("\n[Step 1/6] Extracting audio...")
        audio_path = extract_audio(video_path, video_work_dir)
        state["audio_path"] = audio_path
        _save_checkpoint(video_work_dir, 1, **state)
    else:
        audio_path = state["audio_path"]
        print("[Step 1/6] Skipped (cached)")

    # --- Step 2: Transcription ---
    if done < 2:
        print("\n[Step 2/6] Transcribing audio...")
        transcript = transcribe_audio(audio_path, video_work_dir)
        state["transcript"] = transcript
        _save_checkpoint(video_work_dir, 2, **state)
    else:
        transcript = state.get("transcript", [])
        print(f"[Step 2/6] Skipped (cached, {len(transcript)} segments)")

    # --- Step 3: Video analysis ---
    if done < 3:
        print("\n[Step 3/6] Analyzing video with Qwen2.5-VL...")
        analysis = analyze_video(video_path, transcript, video_work_dir, vl_model)
        state["analysis"] = analysis
        _save_checkpoint(video_work_dir, 3, **state)
    else:
        analysis = state.get("analysis", {})
        print(f"[Step 3/6] Skipped (cached, {len(analysis.get('events', []))} events)")

    # --- Step 4: Script generation ---
    if done < 4:
        print("\n[Step 4/6] Generating script and effects...")
        llm_client = LLMClient(api_key=llm_api_key, api_base=llm_api_base, model=llm_model)
        script = generate_script(analysis, llm_client, target_duration)
        # Save script to file for inspection
        script_path = os.path.join(video_work_dir, "script.json")
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(script, f, ensure_ascii=False, indent=2)
        state["script"] = script
        _save_checkpoint(video_work_dir, 4, **state)
    else:
        script = state.get("script", [])
        print(f"[Step 4/6] Skipped (cached, {len(script)} segments)")

    # --- Step 5: TTS narration ---
    if done < 5:
        print("\n[Step 5/6] Generating narration with emotion...")
        script = generate_narration(script, ref_audio, video_work_dir, model_dir, use_fp16)
        state["script"] = script
        _save_checkpoint(video_work_dir, 5, **state)
    else:
        script = state.get("script", [])
        print(f"[Step 5/6] Skipped (cached)")

    # --- Step 6: Video assembly ---
    print("\n[Step 6/6] Assembling highlight video...")

    # 6a. Portrait mode
    print("  6a. Preparing portrait format...")
    portrait_path = prepare_portrait(video_path, video_work_dir)

    # 6b. Cut segments + effects
    print("  6b. Cutting segments and applying effects...")
    script = cut_and_apply_effects(portrait_path, script, video_work_dir)

    # 6c. Title cards
    print("  6c. Generating title cards...")
    script = generate_title_cards(script, video_work_dir)

    # 6d. Concatenate with transitions
    print("  6d. Concatenating with transitions...")
    concat_path = concat_with_transitions(script, video_work_dir)

    # 6f. Final mix + subtitles
    print("  6e. Final mix + subtitles...")
    final_mix(concat_path, script, audio_path, video_work_dir, output_path)

    _clear_checkpoint(video_work_dir)

    # Summary
    output_info = _get_video_info(output_path)
    print(f"\n{'=' * 60}")
    print(f"Done! Output: {output_path}")
    print(f"Duration: {output_info['duration']:.1f}s ({output_info['duration']/60:.1f}min)")
    print(f"Resolution: {output_info['width']}x{output_info['height']}")
    print(f"{'=' * 60}")

    if cleanup:
        import shutil
        shutil.rmtree(video_work_dir, ignore_errors=True)
        print(f"Cleaned up: {video_work_dir}")

    return output_path


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def highlight_batch(input_dir, output_dir, **kwargs):
    """Process all videos in a directory."""
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = str(input_dir) + "_highlights"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}
    videos = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in video_exts])

    if not videos:
        print(f"No videos found in {input_dir}")
        return

    print(f"Found {len(videos)} videos to process.")
    results = {"success": [], "failed": []}

    for i, video in enumerate(videos):
        print(f"\n[{i + 1}/{len(videos)}] Processing: {video.name}")
        output_path = str(output_dir / f"{video.stem}_highlight.mp4")
        try:
            create_highlight(str(video), output_path=output_path, **kwargs)
            results["success"].append(video.name)
        except Exception as e:
            print(f"  ERROR: {e}")
            results["failed"].append({"file": video.name, "error": str(e)})

    print(f"\n{'=' * 60}")
    print(f"Batch complete: {len(results['success'])} success, {len(results['failed'])} failed")
    for name in results["success"]:
        print(f"  OK: {name}")
    for item in results["failed"]:
        print(f"  FAIL: {item['file']} — {item['error']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create highlight videos from long videos using AI analysis and narration"
    )
    parser.add_argument("input", help="Input video file or directory (with --batch)")
    parser.add_argument("--ref-audio", required=True, help="Reference audio for TTS voice cloning")
    parser.add_argument("-o", "--output", default=None, help="Output file or directory")
    parser.add_argument("--batch", action="store_true", help="Batch mode: process all videos in directory")
    parser.add_argument("--work-dir", default="highlight_workspace", help="Working directory")
    parser.add_argument("--model-dir", default="checkpoints", help="IndexTTS2 model directory")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 for TTS (default: True)")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16")
    parser.add_argument("--target-duration", type=int, default=180, help="Target output duration in seconds (default: 180)")
    parser.add_argument("--llm-api-key", default=None, help="LLM API key (or set LLM_API_KEY env var)")
    parser.add_argument("--llm-api-base", default="https://api.openai.com/v1", help="LLM API base URL")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--vl-model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Qwen2.5-VL model name")
    parser.add_argument("--cleanup", action="store_true", help="Delete intermediate files after completion")

    args = parser.parse_args()

    llm_api_key = args.llm_api_key or os.environ.get("LLM_API_KEY")
    use_fp16 = args.fp16 and not args.no_fp16

    if not llm_api_key:
        print("Error: LLM API key required. Use --llm-api-key or set LLM_API_KEY env var.")
        sys.exit(1)

    common_kwargs = dict(
        ref_audio=args.ref_audio,
        work_dir=args.work_dir,
        llm_api_key=llm_api_key,
        llm_api_base=args.llm_api_base,
        llm_model=args.llm_model,
        vl_model=args.vl_model,
        model_dir=args.model_dir,
        target_duration=args.target_duration,
        use_fp16=use_fp16,
        cleanup=args.cleanup,
    )

    if args.batch:
        highlight_batch(args.input, args.output, **common_kwargs)
    else:
        create_highlight(args.input, output_path=args.output, **common_kwargs)


if __name__ == "__main__":
    main()
