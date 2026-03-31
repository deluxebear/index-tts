"""
Video Dubbing Pipeline: English → Chinese using IndexTTS2

Dubs English TED-style speeches into Chinese with:
- Vocal/background separation (demucs)
- ASR + speaker diarization (whisperx)
- Context-aware LLM translation (OpenAI-compatible API)
- Voice cloning + emotion preservation (IndexTTS2)
- Duration alignment with gap absorption (pyrubberband)
- Crossfade between segments
- Final video assembly (ffmpeg)
- SRT subtitle output (CN + EN)
- Checkpoint/resume on interruption

Usage:
    # Single video
    python dub_pipeline.py video.mp4 -o output_cn.mp4

    # Batch mode
    python dub_pipeline.py --batch /path/to/input_dir -o /path/to/output_dir

    # Resume after interruption (auto-detected)
    python dub_pipeline.py video.mp4  # picks up from last checkpoint

Requires: demucs, whisperx, pyrubberband, soundfile, openai, ffmpeg
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

CHARS_PER_SECOND = 4.5
AUDIO_COMFORT_LIMIT = 1.2
VIDEO_SLOWDOWN_MAX = 2.0
MIN_REF_DURATION = 3.0
GAP_ABSORB_MAX = 2.0
BG_VOLUME = 0.3
FADE_MS = 10
CHECKPOINT_FILE = "checkpoint.json"


def _run_ffmpeg(*args):
    """Run an ffmpeg command, raising with stderr on failure."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", *args],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode()}") from e


# ---------------------------------------------------------------------------
# Checkpoint — save/resume pipeline progress
# ---------------------------------------------------------------------------

def _save_checkpoint(work_dir, step, segments=None, **extra):
    """Save pipeline progress after each step."""
    data = {"step": step, "segments": segments}
    data.update(extra)
    path = os.path.join(work_dir, CHECKPOINT_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  Checkpoint saved: step {step}")


def _load_checkpoint(work_dir):
    """Load checkpoint if exists."""
    path = os.path.join(work_dir, CHECKPOINT_FILE)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    print(f"  Resuming from checkpoint: step {data['step']}")
    return data


def _clear_checkpoint(work_dir):
    """Remove checkpoint after successful completion."""
    path = os.path.join(work_dir, CHECKPOINT_FILE)
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# LLM Client — works with any OpenAI-compatible API
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
# Step 1: Preprocessing — extract audio and silent video
# ---------------------------------------------------------------------------

def extract_tracks(video_path, work_dir):
    """Extract audio (native sample rate for demucs quality) and silent video."""
    audio_path = os.path.join(work_dir, "full_audio.wav")
    video_only_path = os.path.join(work_dir, "video_only.mp4")

    _run_ffmpeg("-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                audio_path)

    _run_ffmpeg("-i", video_path, "-an", "-c:v", "copy", video_only_path)

    print(f"  Extracted audio: {audio_path}")
    print(f"  Extracted video: {video_only_path}")
    return audio_path, video_only_path


# ---------------------------------------------------------------------------
# Step 2: Source separation — vocals vs background
# ---------------------------------------------------------------------------

def separate_vocals(audio_path, work_dir):
    """Separate vocals from background audio using demucs."""
    import demucs.separate

    demucs.separate.main([
        "--two-stems", "vocals",
        "-n", "htdemucs",
        "-o", work_dir,
        audio_path,
    ])

    stem_name = Path(audio_path).stem
    vocals_path = os.path.join(work_dir, "htdemucs", stem_name, "vocals.wav")
    bg_path = os.path.join(work_dir, "htdemucs", stem_name, "no_vocals.wav")

    if not os.path.exists(vocals_path):
        raise FileNotFoundError(f"Demucs output not found: {vocals_path}")

    print(f"  Vocals: {vocals_path}")
    print(f"  Background: {bg_path}")
    return vocals_path, bg_path


# ---------------------------------------------------------------------------
# Step 3: ASR + speaker diarization
# ---------------------------------------------------------------------------

def _detect_device():
    """Detect available compute device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _init_tts(model_dir, use_fp16):
    """Initialize IndexTTS2 model."""
    from indextts.infer_v2 import IndexTTS2
    return IndexTTS2(
        cfg_path=os.path.join(model_dir, "config.yaml"),
        model_dir=model_dir,
        use_fp16=use_fp16,
    )


def transcribe_and_diarize(vocals_path, hf_token, num_speakers=None):
    """Transcribe with word-level timestamps and speaker labels."""
    import whisperx

    device = _detect_device()
    compute_type = "float16" if device == "cuda" else "int8"

    print("  Loading Whisper model...")
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    result = model.transcribe(vocals_path, batch_size=16)
    del model

    print("  Aligning words...")
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, vocals_path, device
    )
    del align_model

    print("  Running speaker diarization...")
    from whisperx.diarize import DiarizationPipeline
    diarize_model = DiarizationPipeline(
        token=hf_token, device=device
    )
    diarize_kwargs = {}
    if num_speakers is not None:
        diarize_kwargs["min_speakers"] = max(1, num_speakers - 1)
        diarize_kwargs["max_speakers"] = num_speakers + 1
    diarize_segments = diarize_model(vocals_path, **diarize_kwargs)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    del diarize_model

    segments = result["segments"]
    speakers = set(seg.get("speaker", "UNKNOWN") for seg in segments)
    print(f"  Transcribed {len(segments)} segments, {len(speakers)} speakers: {speakers}")
    return segments


# ---------------------------------------------------------------------------
# Step 3.5: Gap absorption (Pyvideotrans technique)
# ---------------------------------------------------------------------------

def absorb_gaps(segments):
    """Extend each segment's available time by absorbing inter-segment silence.

    Uses 'end_padded' to preserve original 'end' for subtitle accuracy,
    while giving duration alignment more room to work with.
    """
    for i in range(len(segments) - 1):
        gap = segments[i + 1]["start"] - segments[i]["end"]
        if 0 < gap <= GAP_ABSORB_MAX:
            segments[i]["end_padded"] = segments[i + 1]["start"]
        else:
            segments[i]["end_padded"] = segments[i]["end"]
    if segments:
        segments[-1]["end_padded"] = segments[-1]["end"]

    absorbed = sum(1 for s in segments if s.get("end_padded", s["end"]) > s["end"])
    print(f"  Absorbed gaps for {absorbed}/{len(segments)} segments")
    return segments


# ---------------------------------------------------------------------------
# Step 4: Speaker reference audio extraction
# ---------------------------------------------------------------------------

def extract_audio_segment(audio_path, start, end, output_path):
    """Cut a segment from an audio file using soundfile (no ffmpeg spawn)."""
    info = sf.info(audio_path)
    start_frame = int(start * info.samplerate)
    num_frames = int((end - start) * info.samplerate)
    audio, sr = sf.read(audio_path, start=start_frame, frames=num_frames)
    sf.write(output_path, audio, sr)


def extract_speaker_refs(segments, vocals_path, work_dir, fallback_refs=None):
    """Extract the best (longest) reference audio per speaker."""
    speakers = {}
    for seg in segments:
        spk = seg.get("speaker", "UNKNOWN")
        if spk not in speakers:
            speakers[spk] = []
        speakers[spk].append(seg)

    speaker_refs = {}
    refs_dir = os.path.join(work_dir, "speaker_refs")
    os.makedirs(refs_dir, exist_ok=True)

    for spk, segs in speakers.items():
        best_seg = max(segs, key=lambda s: s["end"] - s["start"])
        ref_path = os.path.join(refs_dir, f"ref_{spk}.wav")
        extract_audio_segment(vocals_path, best_seg["start"], best_seg["end"], ref_path)

        if fallback_refs and spk in fallback_refs:
            fallback = fallback_refs[spk]
        else:
            fallback = ref_path

        speaker_refs[spk] = {"fallback": fallback, "best_auto": ref_path}
        dur = best_seg["end"] - best_seg["start"]
        print(f"  {spk}: best ref {dur:.1f}s, fallback={'user-provided' if fallback_refs and spk in fallback_refs else 'auto'}")

    return speaker_refs


def get_ref_for_segment(seg, speaker_refs, vocals_path, seg_ref_dir):
    """Select reference audio: use segment itself if >= MIN_REF_DURATION, else fallback."""
    duration = seg["end"] - seg["start"]
    spk = seg.get("speaker", "UNKNOWN")

    if duration >= MIN_REF_DURATION:
        seg_ref = os.path.join(seg_ref_dir, f"seg_{seg['start']:.2f}_{seg['end']:.2f}.wav")
        if not os.path.exists(seg_ref):
            extract_audio_segment(vocals_path, seg["start"], seg["end"], seg_ref)
        return seg_ref
    else:
        return speaker_refs[spk]["fallback"]


# ---------------------------------------------------------------------------
# Step 5: Two-phase context-aware translation
# ---------------------------------------------------------------------------

MAX_CONTEXT_TOKENS = 3000


def _estimate_tokens(text):
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4


def _chunk_segments_for_context(segments, max_tokens=MAX_CONTEXT_TOKENS):
    """Split segments into chunks that each fit within the LLM token budget."""
    chunks = []
    current_chunk = []
    current_tokens = 0
    for seg in segments:
        line = f"[{seg.get('speaker', '?')}] ({seg['start']:.1f}s-{seg['end']:.1f}s) {seg['text']}"
        tokens = _estimate_tokens(line)
        if current_tokens + tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        current_chunk.append(seg)
        current_tokens += tokens
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def _merge_contexts(contexts):
    """Merge multiple context analysis results into one."""
    merged = dict(contexts[-1])  # base: last chunk for topic/domain/tone
    merged["key_terms"] = {}
    merged["keep_original"] = []
    merged["speakers"] = {}
    seen_keep = set()
    for ctx in contexts:
        merged["key_terms"].update(ctx.get("key_terms", {}))
        merged["speakers"].update(ctx.get("speakers", {}))
        for term in ctx.get("keep_original", []):
            if term not in seen_keep:
                seen_keep.add(term)
                merged["keep_original"].append(term)
    return merged


def _analyze_context_single(segments, llm_client):
    """Analyze a single chunk of segments for context."""
    transcript = "\n".join(
        f"[{seg.get('speaker', '?')}] ({seg['start']:.1f}s-{seg['end']:.1f}s) {seg['text']}"
        for seg in segments
    )

    prompt = f"""你是一位专业的视频翻译顾问。以下是一段英文演讲/对话的字幕。
请分析并返回 JSON：

{{
  "topic": "演讲/对话的核心主题",
  "domain": "所属领域（科技/商业/教育/医学等）",
  "tone": "整体语气风格（正式/幽默/激情/学术等）",
  "key_terms": {{"english_term": "推荐的中文翻译"}},
  "keep_original": ["ChatGPT", "iPhone", "...品牌名、产品名、公司名、技术专有名词等应保留英文原文的词汇"],
  "speakers": {{"SPEAKER_00": "角色描述"}},
  "translation_notes": "翻译时需要特别注意的事项"
}}

注意 keep_original 字段：列出所有不应翻译、应保持英文原文的词汇，包括但不限于：
- 品牌名（ChatGPT, Google, Tesla, TikTok 等）
- 产品名（iPhone, Model S, GPT-4 等）
- 公司/组织名（OpenAI, Meta, NASA 等）
- 广泛使用的技术术语（API, GPU, transformer, fine-tuning 等）
- 人名的英文形式

字幕内容：
{transcript}

只返回 JSON，不要解释。"""

    response = llm_client.chat(prompt)
    response = re.sub(r"^```(?:json)?\s*", "", response.strip())
    response = re.sub(r"\s*```$", "", response.strip())
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print(f"  Warning: Failed to parse context analysis JSON, using defaults")
        return {
            "topic": "unknown", "domain": "general", "tone": "neutral",
            "key_terms": {}, "keep_original": [], "speakers": {},
            "translation_notes": "",
        }


def analyze_context(segments, llm_client):
    """Phase A: Analyze overall speech context before translating.

    For long transcripts, splits into chunks, analyzes each independently,
    and merges results (union of key_terms/keep_original/speakers).
    """
    chunks = _chunk_segments_for_context(segments)

    if len(chunks) == 1:
        return _analyze_context_single(chunks[0], llm_client)

    from concurrent.futures import ThreadPoolExecutor

    print(f"  Splitting {len(segments)} segments into {len(chunks)} chunks for context analysis")

    def _analyze_chunk(args):
        i, chunk = args
        print(f"  Analyzing chunk {i + 1}/{len(chunks)} ({len(chunk)} segments)...")
        return _analyze_context_single(chunk, llm_client)

    with ThreadPoolExecutor(max_workers=min(3, len(chunks))) as pool:
        contexts = list(pool.map(_analyze_chunk, enumerate(chunks)))

    merged = _merge_contexts(contexts)
    print(f"  Merged: {len(merged.get('key_terms', {}))} terms, "
          f"{len(merged.get('keep_original', []))} keep-original, "
          f"{len(merged.get('speakers', {}))} speakers")
    return merged


def translate_with_context(segments, context, llm_client, batch_size=12,
                           skip_translated=False):
    """Phase B: Translate in context-aware batches with sliding window."""
    translated = []
    keep_original = context.get("keep_original", [])
    keep_original_str = "、".join(keep_original) if keep_original else "无"

    for i in range(0, len(segments), batch_size):
        batch = segments[i : i + batch_size]

        # Single pass: collect already-translated and filter needing translation
        need_translation = []
        for j, seg in enumerate(batch):
            idx = i + j
            if seg.get("zh_text"):
                translated.append({"id": idx, "zh_text": seg["zh_text"]})
                if skip_translated:
                    continue
            need_translation.append((idx, seg))

        if not need_translation:
            continue

        prev_lines = ""
        if translated:
            prev_items = translated[-3:]
            prev_lines = "【前文已翻译】\n" + "\n".join(
                f"#{item['id']}: {item['zh_text']}" for item in prev_items
            ) + "\n\n"

        batch_lines = []
        for idx, seg in need_translation:
            duration = seg["end"] - seg["start"]
            target_chars = max(4, int(duration * CHARS_PER_SECOND))
            batch_lines.append(
                f"#{idx} [{seg.get('speaker', '?')}] ({duration:.1f}s, ~{target_chars}字) {seg['text']}"
            )

        prompt = f"""你是一位专业的中文配音翻译。

【背景分析】
主题：{context.get('topic', 'unknown')}
领域：{context.get('domain', 'general')}
风格：{context.get('tone', 'neutral')}
术语表：{json.dumps(context.get('key_terms', {}), ensure_ascii=False)}
保持原文不翻译：{keep_original_str}
注意事项：{context.get('translation_notes', '')}

{prev_lines}【待翻译段落】
{chr(10).join(batch_lines)}

翻译要求：
1. 口语化，适合朗读配音，不要书面语
2. 每句括号中标注了目标时长和建议字数，请严格控制字数
3. 联系上下文，保持前后连贯和术语一致
4. 保持原文的语气和情感色彩
5. 人名/专有名词保持一致
6. 品牌名、产品名、公司名、技术专有名词保留英文原文，不要翻译（如 ChatGPT 不要译为"聊天GPT"）

返回格式（每行一句，#号对应原句编号）：
#{need_translation[0][0]} 翻译结果
#{need_translation[-1][0]} 翻译结果
..."""

        response = llm_client.chat(prompt)
        parsed = _parse_translation_response(response)

        for idx, seg in need_translation:
            zh_text = parsed.get(idx)
            if zh_text is None:
                print(f"  Warning: segment #{idx} translation missing, using original text")
                zh_text = seg["text"]
            seg["zh_text"] = zh_text
            translated.append({"id": idx, "zh_text": zh_text})

        translated_ids = [idx for idx, _ in need_translation]
        print(f"  Translated segments {translated_ids[0]}-{translated_ids[-1]}")

    return segments


def _parse_translation_response(response):
    """Parse numbered translation lines from LLM response."""
    result = {}
    for line in response.strip().split("\n"):
        line = line.strip()
        match = re.match(r"#(\d+)\s+(.+)", line)
        if match:
            idx = int(match.group(1))
            text = match.group(2).strip()
            result[idx] = text
    return result


def translate_segments(segments, llm_client, batch_size=12, skip_translated=False):
    """Full translation pipeline: analyze context, then batch translate."""
    print("  Phase A: Analyzing speech context...")
    context = analyze_context(segments, llm_client)
    print(f"  Topic: {context.get('topic', '?')}")
    print(f"  Domain: {context.get('domain', '?')}")
    print(f"  Tone: {context.get('tone', '?')}")
    if context.get("key_terms"):
        print(f"  Key terms: {json.dumps(context['key_terms'], ensure_ascii=False)}")

    print("  Phase B: Translating with context...")
    segments = translate_with_context(segments, context, llm_client, batch_size,
                                      skip_translated=skip_translated)
    return segments, context


# ---------------------------------------------------------------------------
# Step 4.5: External subtitle loading (skip LLM when subs exist)
# ---------------------------------------------------------------------------

# Non-dialogue patterns to strip from external subtitles
_CREDIT_RE = re.compile(
    r"^(翻译|译者|审核|校对|校订|字幕|时间轴|压制|后期|特效)"
    r"(人员|者)?[：:]\s*\S+",
    re.MULTILINE,
)
_CREDIT_EN_RE = re.compile(
    r"^(Translated|Reviewed|Subtitl|Timing|Encoded)\s+by\b",
    re.IGNORECASE | re.MULTILINE,
)
# Speaker label: "比拉瓦尔·西杜（BS）:" or "ES：" at start of cue.
# Full-name form always matches; bare abbreviation form collected dynamically.
_SPEAKER_FULL_RE = re.compile(
    r"^(?:[\w\s·\-]+)[（(]([A-Za-z]{1,5})[)）][：:]\s*"
)
_SPEAKER_BARE_RE_TMPL = r"^(?:{})[：:]\s*"
_SOUND_BRACKET_RE = re.compile(r"[\[【（\(]([^)\]】）]*)[\]】）\)]")
_MUSIC_RE = re.compile(r"[♪♫🎵🎶].*?[♪♫🎵🎶]|^[♪♫🎵🎶]+$", re.MULTILINE)
_SOUND_KEYWORDS = frozenset(
    "笑 笑声 掌声 音乐 欢呼 鼓掌 叹气 哭泣 尖叫 咳嗽 叹息 嘘声 欢笑 喝彩 "
    "响起 播放 停顿 沉默 哄笑 嘻笑 抽泣 啜泣 喘息 呻吟 鼓掌声 欢呼声".split()
)
_SOUND_KEYWORDS_EN = frozenset(
    "applause laughter laughing music cheering clapping sighing crying "
    "screaming coughing silence pause chuckling sobbing gasping".split()
)

# Common Traditional Chinese characters that differ from Simplified
_TRAD_CHARS = frozenset(
    "國學點這說對開時過還從們來會個經機關東與給當應進種頭體動問裡間發實無義區單導質線環節條"
    "讓設圖產書總門辦連課認記調選華戰網雲視電話費師題險難際試將場構園價觀記論結紀確萬帶態"
    "邊響歲夢優齊壓執歡滅濟燈營鑰議護邏較載輸輪軍達運過選鄰鏡開閃閱關際隊隨險電預領養駕"
)


def _parse_srt_timestamp(s):
    """Parse SRT timestamp 'HH:MM:SS,mmm' to float seconds."""
    h, m, rest = s.strip().split(":")
    sec, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000


def _read_subtitle_file(filepath, mode="read"):
    """Read a subtitle file, trying multiple encodings."""
    for encoding in ("utf-8-sig", "gb18030", "big5"):
        try:
            with open(filepath, "r", encoding=encoding) as f:
                return f.readlines() if mode == "readlines" else f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(f"Cannot decode subtitle file: {filepath}")


def parse_srt(filepath):
    """Parse an SRT file into a list of cues: [{start, end, text}, ...]."""
    content = _read_subtitle_file(filepath)

    cues = []
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        # Find the timestamp line (may not be the first line)
        ts_match = None
        ts_idx = 0
        for li, line in enumerate(lines):
            ts_match = re.match(
                r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{3})",
                line.strip(),
            )
            if ts_match:
                ts_idx = li
                break
        if not ts_match:
            continue
        start = _parse_srt_timestamp(ts_match.group(1).replace(".", ","))
        end = _parse_srt_timestamp(ts_match.group(2).replace(".", ","))
        text = " ".join(l.strip() for l in lines[ts_idx + 1:] if l.strip())
        if text:
            cues.append({"start": start, "end": end, "text": text})
    return cues


def _parse_ass_timestamp(ts):
    """Parse ASS timestamp 'H:MM:SS.cc' to float seconds."""
    h, m, rest = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def parse_ass(filepath):
    """Parse an ASS/SSA file into the same cue format as parse_srt."""
    lines = _read_subtitle_file(filepath, mode="readlines")

    in_events = False
    fmt_cols = None
    cues = []
    for line in lines:
        line = line.strip()
        if line.lower() == "[events]":
            in_events = True
            continue
        if line.startswith("[") and in_events:
            break
        if not in_events:
            continue
        if line.lower().startswith("format:"):
            fmt_cols = [c.strip().lower() for c in line.split(":", 1)[1].split(",")]
            continue
        if not line.startswith("Dialogue:") or fmt_cols is None:
            continue

        parts = line.split(":", 1)[1].split(",", len(fmt_cols) - 1)
        if len(parts) < len(fmt_cols):
            continue
        row = dict(zip(fmt_cols, [p.strip() for p in parts]))

        start = _parse_ass_timestamp(row.get("start", "0:00:00.00"))
        end = _parse_ass_timestamp(row.get("end", "0:00:00.00"))
        text = row.get("text", "")
        text = re.sub(r"\{[^}]*\}", "", text)
        text = text.replace("\\N", " ").replace("\\n", " ").strip()
        if text:
            cues.append({"start": start, "end": end, "text": text})
    return cues


def _is_sound_description(match):
    """Check if a bracketed expression is a sound/stage direction."""
    content = match.group(1).strip().lower()
    if any(kw in content for kw in _SOUND_KEYWORDS):
        return True
    if any(kw in content for kw in _SOUND_KEYWORDS_EN):
        return True
    raw = match.group(1).strip()
    if re.match(r"^[A-Z\s]+$", raw) and len(raw) > 1:
        return True
    return False


def clean_subtitle_cues(cues):
    """Remove non-dialogue entries (credits, sound descriptions, music) from cues."""
    # Pass 1: collect known speaker abbreviations from full-name labels
    # e.g. "比拉瓦尔·西杜（BS）:" → "BS"
    speaker_abbrevs = set()
    for cue in cues:
        m = _SPEAKER_FULL_RE.match(cue["text"])
        if m:
            speaker_abbrevs.add(m.group(1))

    # Build bare-label regex only for known speaker abbreviations
    if speaker_abbrevs:
        bare_pattern = _SPEAKER_BARE_RE_TMPL.format(
            "|".join(re.escape(a) for a in speaker_abbrevs)
        )
        speaker_re = re.compile(
            _SPEAKER_FULL_RE.pattern + r"|" + bare_pattern
        )
    else:
        speaker_re = _SPEAKER_FULL_RE

    cleaned = []
    for cue in cues:
        text = cue["text"]

        # Strip speaker labels (keep dialogue after the label)
        text = speaker_re.sub("", text)

        # Then check if remaining text is a credit line
        if _CREDIT_RE.search(text) or _CREDIT_EN_RE.search(text):
            continue

        # Remove sound descriptions in brackets
        text = _SOUND_BRACKET_RE.sub(
            lambda m: "" if _is_sound_description(m) else m.group(0), text
        )

        # Remove music markers
        text = _MUSIC_RE.sub("", text)

        text = text.strip()
        if not text:
            continue

        cue = dict(cue)
        cue["text"] = text
        cleaned.append(cue)

    removed = len(cues) - len(cleaned)
    if removed:
        print(f"  Cleaned subtitles: removed {removed} non-dialogue entries")
    return cleaned


def detect_subtitle_language(text):
    """Detect whether text is Simplified Chinese, Traditional Chinese, or English."""
    # Count CJK characters
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return "en"
    cjk = [c for c in chars if "\u4e00" <= c <= "\u9fff"]
    if len(cjk) < len(chars) * 0.1:
        return "en"
    # Check for Traditional Chinese characters
    trad_count = sum(1 for c in cjk if c in _TRAD_CHARS)
    if trad_count > len(cjk) * 0.05 or trad_count > 10:
        return "zht"
    return "zh"


def convert_traditional_to_simplified(cues):
    """Convert Traditional Chinese cues to Simplified Chinese (mainland conventions)."""
    from opencc import OpenCC
    cc = OpenCC("t2s")
    for cue in cues:
        cue["text"] = cc.convert(cue["text"])
    return cues


def discover_subtitle(video_path):
    """Find the best matching subtitle file for a video.

    Globs for {stem}*.srt and {stem}*.ass, extracts language code from filename
    (e.g. '.TED.zh-CN.srt' → 'zh'), and returns the best match by priority.

    Returns (path, language_hint) where language_hint is 'zh', 'zht', 'en', or 'unknown'.
    Returns (None, None) if no subtitle found.
    """
    video = Path(video_path)
    stem = video.stem
    parent = video.parent

    # Language code → internal hint
    _lang_map = {
        "zh-cn": "zh", "zh-hans": "zh", "chs": "zh", "zh": "zh",
        "zh-tw": "zht", "zh-hant": "zht", "cht": "zht", "zht": "zht",
        "en": "en", "eng": "en",
    }
    _priority = {"zh": 0, "zht": 1, "unknown": 2, "en": 3}

    # Glob for subtitle files that start with the video stem
    sub_files = []
    for pattern in (f"{stem}*.srt", f"{stem}*.ass"):
        sub_files.extend(parent.glob(pattern))

    candidates = []
    for path in sub_files:
        # Skip the video file itself
        if path.suffix in (".mp4", ".mkv", ".mov", ".avi"):
            continue
        # Extract language tag from the part after the video stem
        suffix_part = path.stem[len(stem):]  # e.g. ".TED.zh-CN" or ".zh" or ""
        parts = [p.lower() for p in suffix_part.split(".") if p]

        lang_hint = "unknown"
        for part in reversed(parts):  # check right-to-left: "zh-cn" before "ted"
            if part in _lang_map:
                lang_hint = _lang_map[part]
                break

        candidates.append((path, lang_hint, _priority.get(lang_hint, 2)))

    if not candidates:
        return None, None

    # Best priority first, then shorter filename (more specific match)
    candidates.sort(key=lambda x: (x[2], len(x[0].name)))
    best_path, lang_hint, _ = candidates[0]

    # For unknown language, detect from content
    if lang_hint == "unknown":
        parser = parse_ass if best_path.suffix == ".ass" else parse_srt
        cues = parser(str(best_path))
        if not cues:
            return None, None
        sample_text = " ".join(c["text"] for c in cues[:50])
        lang_hint = detect_subtitle_language(sample_text)

    print(f"  Found subtitle: {best_path.name} (detected: {lang_hint})")
    return str(best_path), lang_hint


def match_srt_to_segments(cues, segments):
    """Match external subtitle cues to ASR segments by timing overlap.

    Uses a two-pointer sweep over sorted cues for O(n+m) performance.
    Sets seg['zh_text'] for matched segments.
    Returns (segments, matched_count).
    """
    if not cues:
        return segments, 0

    sorted_cues = sorted(cues, key=lambda c: c["start"])
    sorted_segs = sorted(enumerate(segments), key=lambda x: x[1]["start"])

    matched = 0
    cue_ptr = 0

    for seg_idx, seg in sorted_segs:
        seg_start, seg_end = seg["start"], seg["end"]
        seg_dur = seg_end - seg_start
        if seg_dur <= 0:
            continue

        # Advance cue_ptr past cues that end before this segment starts
        while cue_ptr > 0 and sorted_cues[cue_ptr - 1]["end"] > seg_start:
            cue_ptr -= 1  # back up if needed for overlapping cues
        while cue_ptr < len(sorted_cues) and sorted_cues[cue_ptr]["end"] <= seg_start:
            cue_ptr += 1

        best_overlap = 0
        best_cue = None
        collected = []  # (cue_start, overlap, text)

        # Scan forward from cue_ptr
        for j in range(cue_ptr, len(sorted_cues)):
            cue = sorted_cues[j]
            if cue["start"] >= seg_end:
                break
            overlap = min(seg_end, cue["end"]) - max(seg_start, cue["start"])
            if overlap <= 0:
                continue
            ratio = overlap / seg_dur
            if ratio > 0.3:
                collected.append((cue["start"], cue["text"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_cue = cue

        if collected:
            if len(collected) > 1:
                collected.sort(key=lambda x: x[0])  # sort by time
                seen = set()
                texts = []
                for _, t in collected:
                    if t not in seen:
                        seen.add(t)
                        texts.append(t)
                seg["zh_text"] = "".join(texts)
            else:
                seg["zh_text"] = collected[0][1]
            matched += 1
        elif best_cue and best_overlap / seg_dur > 0.15:
            seg["zh_text"] = best_cue["text"]
            matched += 1

    return segments, matched


def load_external_subtitles(video_path, segments):
    """Load and match external subtitles to ASR segments.

    Returns (segments, subtitle_source) where subtitle_source is a description
    string like 'zh:video.zh.srt' or None if no usable subs found.
    """
    sub_path, lang_hint = discover_subtitle(video_path)
    if sub_path is None:
        return segments, None

    # English subs don't help — fall through to LLM
    if lang_hint == "en":
        print(f"  Found English-only subtitle, will use LLM translation instead")
        return segments, None

    # Parse
    parser = parse_ass if sub_path.endswith(".ass") else parse_srt
    cues = parser(sub_path)
    if not cues:
        print(f"  Warning: subtitle file is empty: {sub_path}")
        return segments, None

    # Clean non-dialogue entries
    cues = clean_subtitle_cues(cues)
    if not cues:
        print(f"  Warning: all subtitle cues were non-dialogue, skipping")
        return segments, None

    # Convert Traditional → Simplified if needed
    source_desc = lang_hint
    if lang_hint == "zht":
        print(f"  Converting Traditional Chinese → Simplified Chinese...")
        cues = convert_traditional_to_simplified(cues)
        source_desc = "zht→zh"

    # Match to ASR segments
    segments, matched = match_srt_to_segments(cues, segments)
    total = len([s for s in segments if s.get("text", "").strip()])
    print(f"  Matched {matched}/{total} segments from external subtitles")

    if matched == 0:
        return segments, None

    return segments, f"{source_desc}:{Path(sub_path).name}"


# ---------------------------------------------------------------------------
# Step 6: TTS generation with IndexTTS2
# ---------------------------------------------------------------------------

def get_audio_duration(path):
    """Get audio duration in seconds."""
    info = sf.info(path)
    return info.duration


def generate_speech(segments, speaker_refs, vocals_path, work_dir, tts):
    """Generate Chinese speech for each segment using IndexTTS2."""
    tts_dir = os.path.join(work_dir, "tts_output")
    seg_ref_dir = os.path.join(work_dir, "seg_refs")
    os.makedirs(tts_dir, exist_ok=True)
    os.makedirs(seg_ref_dir, exist_ok=True)

    for i, seg in enumerate(segments):
        zh_text = seg.get("zh_text", "")
        if not zh_text.strip():
            seg["wav_path"] = None
            seg["actual_duration"] = 0
            continue

        output_path = os.path.join(tts_dir, f"tts_{i:04d}.wav")

        # Skip if already generated (for resume)
        if os.path.exists(output_path):
            seg["wav_path"] = output_path
            seg["actual_duration"] = get_audio_duration(output_path)
        else:
            ref_audio = get_ref_for_segment(seg, speaker_refs, vocals_path, seg_ref_dir)
            tts.infer(
                spk_audio_prompt=ref_audio,
                text=zh_text,
                output_path=output_path,
                emo_audio_prompt=ref_audio,
                verbose=False,
            )
            seg["wav_path"] = output_path
            seg["actual_duration"] = get_audio_duration(output_path)

        target_dur = seg["end"] - seg["start"]
        ratio = seg["actual_duration"] / target_dur if target_dur > 0 else 1.0
        print(f"  [{i:3d}] {seg.get('speaker', '?')} | "
              f"target={target_dur:.1f}s actual={seg['actual_duration']:.1f}s "
              f"ratio={ratio:.2f} | {zh_text[:30]}...")

    return segments


# ---------------------------------------------------------------------------
# Step 7: Duration alignment (with gap absorption)
# ---------------------------------------------------------------------------

def align_durations(segments, work_dir):
    """Align generated audio with 50/50 burden split between audio and video.

    - ratio <= AUDIO_COMFORT_LIMIT (1.2): audio-only speedup
    - ratio > AUDIO_COMFORT_LIMIT: audio caps at 1.2x, video slows down the rest
    - Stores seg["video_slowdown"] for later video processing
    """
    import pyrubberband as pyrb

    aligned_dir = os.path.join(work_dir, "aligned")
    os.makedirs(aligned_dir, exist_ok=True)

    for i, seg in enumerate(segments):
        if seg.get("wav_path") is None:
            seg["aligned_path"] = None
            seg["video_slowdown"] = 1.0
            continue

        target_dur = seg.get("end_padded", seg["end"]) - seg["start"]
        actual_dur = seg["actual_duration"]

        if target_dur <= 0 or actual_dur <= 0:
            seg["aligned_path"] = seg["wav_path"]
            seg["video_slowdown"] = 1.0
            continue

        ratio = actual_dur / target_dur

        if 0.99 <= ratio <= 1.01:
            seg["aligned_path"] = seg["wav_path"]
            seg["aligned_duration"] = actual_dur
            seg["video_slowdown"] = 1.0
            continue

        aligned_path = os.path.join(aligned_dir, f"aligned_{i:04d}.wav")
        audio, sr = sf.read(seg["wav_path"])

        if ratio > 1.0:
            if ratio <= AUDIO_COMFORT_LIMIT:
                audio_rate = ratio
                seg["video_slowdown"] = 1.0
            else:
                # 50/50 split: audio caps at comfort limit, video absorbs the rest
                video_slowdown = min(ratio / AUDIO_COMFORT_LIMIT, VIDEO_SLOWDOWN_MAX)
                audio_rate = ratio / video_slowdown
                seg["video_slowdown"] = video_slowdown
            stretched = pyrb.time_stretch(audio, sr, rate=audio_rate)
            sf.write(aligned_path, stretched, sr)
        else:
            # Too short: keep original speed, pad with silence at the end.
            # Do NOT slow down — slowed speech sounds unnatural.
            seg["video_slowdown"] = 1.0
            target_samples = int(target_dur * sr)
            if len(audio) < target_samples:
                padded = np.zeros(target_samples, dtype=audio.dtype)
                padded[: len(audio)] = audio
                stretched = padded
            else:
                stretched = audio
            sf.write(aligned_path, stretched, sr)

        seg["aligned_path"] = aligned_path
        seg["aligned_duration"] = len(stretched) / sr

        if seg["video_slowdown"] > 1.01:
            print(f"  [{i:3d}] video_slowdown={seg['video_slowdown']:.2f}x, audio_rate={ratio/seg['video_slowdown']:.2f}x")

    slowdown_count = sum(1 for s in segments if s.get("video_slowdown", 1.0) > 1.01)
    if slowdown_count:
        print(f"  {slowdown_count} segments need video slowdown")
    return segments


# ---------------------------------------------------------------------------
# Step 7.5: Timeline computation for video slowdown
# ---------------------------------------------------------------------------

def _get_video_duration(video_path):
    """Get video duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def compute_shifted_timeline(segments, video_duration):
    """Build timeline mapping from original to new timestamps after video slowdowns.

    Returns (timeline, new_total_duration).
    timeline: list of (orig_start, orig_end, new_start, new_end, slowdown)
    Also sets seg["new_start"] on each segment.
    """
    sorted_segs = sorted(
        [s for s in segments if s.get("aligned_path") is not None],
        key=lambda s: s["start"],
    )

    timeline = []
    cumulative_shift = 0.0
    prev_orig_end = 0.0

    for seg in sorted_segs:
        orig_start = seg["start"]
        orig_end = seg.get("end_padded", seg["end"])
        slowdown = seg.get("video_slowdown", 1.0)

        # Gap before this segment inherits the cumulative shift
        new_start = orig_start + cumulative_shift
        orig_dur = orig_end - orig_start
        new_dur = orig_dur * slowdown
        added_time = new_dur - orig_dur
        new_end = new_start + new_dur
        cumulative_shift += added_time

        timeline.append((orig_start, orig_end, new_start, new_end, slowdown))
        seg["new_start"] = new_start
        prev_orig_end = orig_end

    # Set new_start for segments without aligned audio (empty text, etc.)
    for seg in segments:
        if "new_start" not in seg:
            shift = sum(
                (oe - os) * (sd - 1.0)
                for (os, oe, _, _, sd) in timeline
                if os < seg["start"]
            )
            seg["new_start"] = seg["start"] + shift

    new_total = video_duration + cumulative_shift
    return timeline, new_total


def _build_video_with_slowdowns(video_only_path, timeline, video_duration, output_path):
    """Rebuild video with per-segment slowdowns using ffmpeg filter_complex."""
    filter_parts = []
    concat_inputs = []
    idx = 0
    prev_end = 0.0

    for (orig_start, orig_end, _, _, slowdown) in timeline:
        # Passthrough gap before this segment
        if orig_start > prev_end + 0.01:
            label = f"v{idx}"
            filter_parts.append(
                f"[0:v]trim={prev_end:.3f}:{orig_start:.3f},setpts=PTS-STARTPTS[{label}]"
            )
            concat_inputs.append(f"[{label}]")
            idx += 1

        label = f"v{idx}"
        if slowdown > 1.01:
            filter_parts.append(
                f"[0:v]trim={orig_start:.3f}:{orig_end:.3f},setpts={slowdown:.4f}*(PTS-STARTPTS)[{label}]"
            )
        else:
            filter_parts.append(
                f"[0:v]trim={orig_start:.3f}:{orig_end:.3f},setpts=PTS-STARTPTS[{label}]"
            )
        concat_inputs.append(f"[{label}]")
        idx += 1
        prev_end = orig_end

    # Tail after last segment
    if prev_end < video_duration - 0.01:
        label = f"v{idx}"
        filter_parts.append(
            f"[0:v]trim={prev_end:.3f},setpts=PTS-STARTPTS[{label}]"
        )
        concat_inputs.append(f"[{label}]")
        idx += 1

    concat_str = "".join(concat_inputs)
    filter_parts.append(f"{concat_str}concat=n={idx}:v=1:a=0[vout]")
    filter_complex = ";".join(filter_parts)

    _run_ffmpeg(
        "-i", video_only_path,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an",
        output_path,
    )
    print(f"  Built slowdown video: {output_path}")


def _stretch_background_audio(bg_audio_path, timeline, new_total_duration, output_path):
    """Stretch background audio to match the shifted video timeline."""
    import pyrubberband as pyrb

    bg_audio, sr = sf.read(bg_audio_path, dtype="float32")
    if bg_audio.ndim == 2:
        bg_audio = np.mean(bg_audio, axis=1)
    new_total_samples = int(new_total_duration * sr)
    new_bg = np.zeros(new_total_samples, dtype=np.float32)

    prev_orig_end = 0.0
    prev_new_end = 0.0

    for (orig_start, orig_end, new_start, new_end, slowdown) in timeline:
        # Copy passthrough region before this segment
        if orig_start > prev_orig_end + 0.001:
            src_s = int(prev_orig_end * sr)
            src_e = int(orig_start * sr)
            dst_s = int(prev_new_end * sr)
            chunk = bg_audio[src_s:src_e]
            end_idx = min(len(chunk), new_total_samples - dst_s)
            if end_idx > 0:
                new_bg[dst_s:dst_s + end_idx] = chunk[:end_idx]

        # Stretch slowed region
        src_s = int(orig_start * sr)
        src_e = int(orig_end * sr)
        dst_s = int(new_start * sr)
        chunk = bg_audio[src_s:src_e]
        if len(chunk) > 0 and slowdown > 1.01:
            stretched = pyrb.time_stretch(chunk, sr, rate=1.0 / slowdown)
            end_idx = min(len(stretched), new_total_samples - dst_s)
            if end_idx > 0:
                new_bg[dst_s:dst_s + end_idx] = stretched[:end_idx]
        elif len(chunk) > 0:
            end_idx = min(len(chunk), new_total_samples - dst_s)
            if end_idx > 0:
                new_bg[dst_s:dst_s + end_idx] = chunk[:end_idx]

        prev_orig_end = orig_end
        prev_new_end = new_end

    # Copy tail
    if prev_orig_end < len(bg_audio) / sr:
        src_s = int(prev_orig_end * sr)
        dst_s = int(prev_new_end * sr)
        chunk = bg_audio[src_s:]
        end_idx = min(len(chunk), new_total_samples - dst_s)
        if end_idx > 0:
            new_bg[dst_s:dst_s + end_idx] = chunk[:end_idx]

    sf.write(output_path, new_bg, sr)
    print(f"  Stretched background audio: {output_path}")


# ---------------------------------------------------------------------------
# Step 8-9: Audio assembly and final video mix
# ---------------------------------------------------------------------------

def _apply_fade(audio, sr):
    """Apply fade-in and fade-out to avoid click noise at segment boundaries."""
    fade_len = int(FADE_MS / 1000 * sr)
    fade_len = min(fade_len, len(audio) // 4)
    if fade_len > 0:
        audio[:fade_len] *= np.linspace(0, 1, fade_len, dtype=audio.dtype)
        audio[-fade_len:] *= np.linspace(1, 0, fade_len, dtype=audio.dtype)
    return audio


def _build_speech_track(segments, sr, total_samples):
    """Place aligned audio segments onto a timeline-aware speech track."""
    import librosa

    speech_track = np.zeros(total_samples, dtype=np.float32)
    for seg in segments:
        aligned_path = seg.get("aligned_path")
        if aligned_path is None or not os.path.exists(aligned_path):
            continue

        audio, audio_sr = sf.read(aligned_path, dtype="float32")
        if audio_sr != sr:
            audio = librosa.resample(audio, orig_sr=audio_sr, target_sr=sr)

        audio = _apply_fade(audio, sr)

        # Use new_start (shifted timeline) if available, else original start
        start_time = seg.get("new_start", seg["start"])
        start_sample = int(start_time * sr)
        end_sample = start_sample + len(audio)

        if start_sample >= total_samples:
            continue
        if end_sample > total_samples:
            audio = audio[: total_samples - start_sample]
            end_sample = total_samples

        speech_track[start_sample:end_sample] += audio[: end_sample - start_sample]

    return speech_track


def _merge_video_audio(video_path, speech_path, bg_path, output_path):
    """Merge video with speech and background audio tracks."""
    _run_ffmpeg(
        "-i", video_path,
        "-i", speech_path,
        "-i", bg_path,
        "-filter_complex",
        f"[1:a]volume=1.0[speech];[2:a]volume={BG_VOLUME}[bg];"
        "[speech][bg]amix=inputs=2:duration=longest:normalize=0[aout]",
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy", "-shortest",
        output_path,
    )


def assemble_final(segments, bg_audio_path, video_only_path, output_path, work_dir,
                   timeline=None, new_total_duration=None):
    """Assemble final video. Two paths:
    - Fast: no video slowdown needed → -c:v copy
    - Slow: video slowdown needed → ffmpeg trim+setpts+concat + bg stretch
    """
    if timeline is not None:
        print("  Using video slowdown path (re-encoding)...")
        video_duration = _get_video_duration(video_only_path)

        slowed_video = os.path.join(work_dir, "video_slowed.mp4")
        _build_video_with_slowdowns(video_only_path, timeline, video_duration, slowed_video)

        stretched_bg = os.path.join(work_dir, "bg_stretched.wav")
        _stretch_background_audio(bg_audio_path, timeline, new_total_duration, stretched_bg)

        bg_info = sf.info(stretched_bg)
        sr = bg_info.samplerate
        total_samples = int(new_total_duration * sr)
        speech_track = _build_speech_track(segments, sr, total_samples)
        speech_path = os.path.join(work_dir, "chinese_speech.wav")
        sf.write(speech_path, speech_track, sr)
        del speech_track

        _merge_video_audio(slowed_video, speech_path, stretched_bg, output_path)
    else:
        print("  Using fast path (no video slowdown)...")
        bg_info = sf.info(bg_audio_path)
        sr = bg_info.samplerate
        total_samples = bg_info.frames
        speech_track = _build_speech_track(segments, sr, total_samples)
        speech_path = os.path.join(work_dir, "chinese_speech.wav")
        sf.write(speech_path, speech_track, sr)
        del speech_track

        _merge_video_audio(video_only_path, speech_path, bg_audio_path, output_path)

    print(f"  Final video: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Step 10: SRT subtitle output
# ---------------------------------------------------------------------------

def _format_srt_time(seconds):
    """Convert seconds to SRT time format: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_srt(segments, output_path, lang="zh"):
    """Generate SRT subtitle file aligned to actual dubbed audio timing.

    Uses new_start (shifted timeline) and aligned_duration (post-stretch).
    """
    text_key = "zh_text" if lang == "zh" else "text"
    with open(output_path, "w", encoding="utf-8") as f:
        idx = 0
        for seg in segments:
            text = seg.get(text_key, "")
            if not text.strip():
                continue
            idx += 1
            srt_start = seg.get("new_start", seg["start"])
            aligned_dur = seg.get("aligned_duration")
            if aligned_dur is not None:
                srt_end = srt_start + aligned_dur
            else:
                srt_end = srt_start + (seg["end"] - seg["start"])
            f.write(f"{idx}\n{_format_srt_time(srt_start)} --> {_format_srt_time(srt_end)}\n{text}\n\n")
    print(f"  Saved SRT: {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def dub_video(
    video_path,
    output_path=None,
    work_dir="dub_workspace",
    hf_token=None,
    llm_api_key=None,
    llm_api_base="https://api.openai.com/v1",
    llm_model="gpt-4o-mini",
    model_dir="checkpoints",
    fallback_refs=None,
    num_speakers=None,
    use_fp16=True,
    tts=None,
    cleanup=False,
    external_subs=None,
    no_external_subs=False,
):
    """
    Main pipeline: dub an English video into Chinese.

    Args:
        video_path: Input video file path.
        output_path: Output video path. Defaults to {input_name}_cn.mp4.
        work_dir: Directory for intermediate files.
        hf_token: HuggingFace token for pyannote speaker diarization.
        llm_api_key: API key for OpenAI-compatible translation LLM.
        llm_api_base: Base URL for the LLM API.
        llm_model: Model name for translation.
        model_dir: IndexTTS2 checkpoint directory.
        fallback_refs: Dict mapping speaker IDs to fallback reference audio paths.
        num_speakers: Hint for number of speakers (2-4).
        use_fp16: Use FP16 for IndexTTS2 inference.
        tts: Pre-initialized IndexTTS2 instance (for batch mode).
        cleanup: Delete intermediate files after successful completion.
        external_subs: Explicit path to external subtitle file (.srt/.ass).
        no_external_subs: Disable auto-discovery of external subtitles.
    """
    video_path = str(video_path)
    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(video_path).parent / f"{stem}_cn.mp4")

    video_work_dir = os.path.join(work_dir, Path(video_path).stem)
    os.makedirs(video_work_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Dubbing: {video_path}")
    print(f"Output:  {output_path}")
    print(f"Work:    {video_work_dir}")
    print(f"{'=' * 60}")

    # Load checkpoint if resuming
    ckpt = _load_checkpoint(video_work_dir)
    done = ckpt["step"] if ckpt else 0
    segments = ckpt.get("segments") if ckpt else None
    paths = ckpt.get("paths", {}) if ckpt else {}

    # --- Step 1: Extract tracks ---
    if done < 1:
        print("\n[Step 1/11] Extracting audio and video tracks...")
        audio_path, video_only_path = extract_tracks(video_path, video_work_dir)
        paths.update(audio_path=audio_path, video_only_path=video_only_path)
        _save_checkpoint(video_work_dir, 1, paths=paths)
    else:
        audio_path = paths["audio_path"]
        video_only_path = paths["video_only_path"]
        print(f"\n[Step 1/11] Skipped (cached)")

    # --- Step 2: Source separation ---
    if done < 2:
        print("\n[Step 2/11] Separating vocals from background...")
        vocals_path, bg_path = separate_vocals(audio_path, video_work_dir)
        paths.update(vocals_path=vocals_path, bg_path=bg_path)
        _save_checkpoint(video_work_dir, 2, paths=paths)
    else:
        vocals_path = paths["vocals_path"]
        bg_path = paths["bg_path"]
        print(f"\n[Step 2/11] Skipped (cached)")

    # --- Step 3: ASR + diarization ---
    if done < 3:
        print("\n[Step 3/11] Transcribing and diarizing...")
        segments = transcribe_and_diarize(vocals_path, hf_token, num_speakers)
        transcript_path = os.path.join(video_work_dir, "transcript.json")
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        print(f"  Saved transcript: {transcript_path}")
        _save_checkpoint(video_work_dir, 3, segments=segments, paths=paths)
    else:
        print(f"\n[Step 3/11] Skipped (cached, {len(segments)} segments)")

    # --- Step 3.5: Gap absorption ---
    if done < 4:
        print("\n[Step 3.5/11] Absorbing inter-segment gaps...")
        segments = absorb_gaps(segments)
        _save_checkpoint(video_work_dir, 4, segments=segments, paths=paths)
    else:
        print(f"\n[Step 3.5/11] Skipped (cached)")

    # --- Step 4: Speaker references ---
    if done < 5:
        print("\n[Step 4/11] Extracting speaker references...")
        speaker_refs = extract_speaker_refs(segments, vocals_path, video_work_dir, fallback_refs)
        paths["speaker_refs"] = speaker_refs
        _save_checkpoint(video_work_dir, 5, segments=segments, paths=paths)
    else:
        speaker_refs = paths.get("speaker_refs", {})
        print(f"\n[Step 4/11] Skipped (cached)")

    # --- Step 5: Translation (with external subtitle support) ---
    if done < 6:
        print("\n[Step 5/11] Translating to Chinese...")

        # Try external subtitles first
        if no_external_subs:
            sub_source = None
        else:
            ext_sub = external_subs or video_path
            segments, sub_source = load_external_subtitles(ext_sub, segments)

        untranslated = sum(1 for s in segments if not s.get("zh_text") and s.get("text", "").strip())

        if sub_source and untranslated == 0:
            print(f"  All segments matched from external subtitles ({sub_source})")
        else:
            if sub_source:
                print(f"  {len(segments) - untranslated} from external subs, {untranslated} need LLM")
            llm_client = LLMClient(api_key=llm_api_key, api_base=llm_api_base, model=llm_model)
            segments, _ = translate_segments(segments, llm_client,
                                             skip_translated=bool(sub_source))
        translations_path = os.path.join(video_work_dir, "translations.json")
        with open(translations_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"id": i, "speaker": s.get("speaker"), "start": s["start"], "end": s["end"],
                  "en": s["text"], "zh": s.get("zh_text", "")}
                 for i, s in enumerate(segments)],
                f, ensure_ascii=False, indent=2,
            )
        print(f"  Saved translations: {translations_path}")
        _save_checkpoint(video_work_dir, 6, segments=segments, paths=paths)
    else:
        print(f"\n[Step 5/11] Skipped (cached)")

    # --- Step 6: TTS generation ---
    if done < 7:
        print("\n[Step 6/11] Generating Chinese speech with IndexTTS2...")
        if tts is None:
            tts = _init_tts(model_dir, use_fp16)
        segments = generate_speech(segments, speaker_refs, vocals_path, video_work_dir, tts)
        _save_checkpoint(video_work_dir, 7, segments=segments, paths=paths)
    else:
        print(f"\n[Step 6/11] Skipped (cached)")

    # --- Step 7: Duration alignment (with 50/50 video slowdown) ---
    if done < 8:
        print("\n[Step 7/11] Aligning durations (audio + video)...")
        segments = align_durations(segments, video_work_dir)
        _save_checkpoint(video_work_dir, 8, segments=segments, paths=paths)
    else:
        print(f"\n[Step 7/11] Skipped (cached)")

    # --- Step 7.5: Compute shifted timeline ---
    needs_slowdown = any(seg.get("video_slowdown", 1.0) > 1.01 for seg in segments)
    timeline = None
    new_total = None
    if needs_slowdown:
        if done < 9:
            print("\n[Step 7.5/11] Computing shifted timeline for video slowdown...")
            video_duration = _get_video_duration(video_only_path)
            timeline, new_total = compute_shifted_timeline(segments, video_duration)
            print(f"  Original duration: {video_duration:.1f}s → New duration: {new_total:.1f}s (+{new_total - video_duration:.1f}s)")
            _save_checkpoint(video_work_dir, 9, segments=segments, paths=paths,
                             timeline=timeline, new_total=new_total)
        else:
            timeline = ckpt.get("timeline")
            new_total = ckpt.get("new_total")
            # Restore new_start from checkpoint segments
            print(f"\n[Step 7.5/11] Skipped (cached)")
    else:
        for seg in segments:
            seg["new_start"] = seg["start"]

    # --- Step 8-9: Assembly ---
    if done < 10:
        print("\n[Step 8-9/11] Assembling final video...")
        assemble_final(segments, bg_path, video_only_path, output_path, video_work_dir,
                       timeline=timeline, new_total_duration=new_total)
        _save_checkpoint(video_work_dir, 10, segments=segments, paths=paths)
    else:
        print(f"\n[Step 8-9/11] Skipped (cached)")

    # --- Step 10-11: SRT subtitles ---
    print("\n[Step 10-11/11] Generating subtitles...")
    srt_base = output_path.rsplit(".", 1)[0]
    generate_srt(segments, f"{srt_base}.srt", lang="zh")
    generate_srt(segments, f"{srt_base}_en.srt", lang="en")

    # Mark done
    _clear_checkpoint(video_work_dir)

    # Cleanup if requested
    if cleanup:
        import shutil
        shutil.rmtree(video_work_dir)
        print(f"  Cleaned up: {video_work_dir}")

    print(f"\nDone! Output: {output_path}")
    return output_path


def dub_batch(
    input_dir,
    output_dir=None,
    video_extensions=(".mp4", ".mkv", ".mov", ".avi"),
    **kwargs,
):
    """
    Batch process all videos in a directory.

    Args:
        input_dir: Input directory containing video files.
        output_dir: Output directory. Defaults to {input_dir}_cn/.
        video_extensions: Tuple of video file extensions to process.
        **kwargs: All other arguments passed to dub_video().
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir.parent / f"{input_dir.name}_cn"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        f for f in input_dir.iterdir() if f.suffix.lower() in video_extensions
    )

    if not videos:
        print(f"No video files found in {input_dir}")
        return

    print(f"Found {len(videos)} videos to process")
    print(f"Output directory: {output_dir}")

    tts = _init_tts(kwargs.get("model_dir", "checkpoints"), kwargs.get("use_fp16", True))

    results = []
    for i, video in enumerate(videos):
        print(f"\n{'#' * 60}")
        print(f"[{i + 1}/{len(videos)}] {video.name}")
        print(f"{'#' * 60}")

        out_path = str(output_dir / f"{video.stem}_cn.mp4")
        try:
            dub_video(str(video), out_path, tts=tts, **kwargs)
            results.append((video.name, "OK", out_path))
        except Exception as e:
            print(f"FAILED: {video.name} — {e}")
            results.append((video.name, "FAILED", str(e)))
            continue

    print(f"\n{'=' * 60}")
    print("BATCH SUMMARY")
    print(f"{'=' * 60}")
    for name, status, info in results:
        print(f"  {'OK' if status == 'OK' else 'FAIL'} | {name} | {info}")
    ok_count = sum(1 for _, s, _ in results if s == "OK")
    print(f"\n  {ok_count}/{len(results)} succeeded")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dub English videos into Chinese using IndexTTS2"
    )
    parser.add_argument(
        "input", help="Input video file or directory (with --batch)"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output file or directory. Default: {input}_cn.mp4 or {input_dir}_cn/",
    )
    parser.add_argument("--batch", action="store_true", help="Batch mode: process all videos in input directory")
    parser.add_argument("--work-dir", default="dub_workspace", help="Working directory for intermediate files")
    parser.add_argument("--model-dir", default="checkpoints", help="IndexTTS2 model directory")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 inference (default: True)")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 inference")
    parser.add_argument("--num-speakers", type=int, default=None, help="Hint for number of speakers")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--llm-api-key", default=None, help="LLM API key (or set LLM_API_KEY env var)")
    parser.add_argument("--llm-api-base", default="https://api.openai.com/v1", help="LLM API base URL")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--cleanup", action="store_true", help="Delete intermediate files after completion")
    parser.add_argument("--external-subs", default=None, help="Path to external subtitle file (.srt or .ass)")
    parser.add_argument("--no-external-subs", action="store_true", help="Disable auto-discovery of external subtitles")

    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    llm_api_key = args.llm_api_key or os.environ.get("LLM_API_KEY")
    use_fp16 = args.fp16 and not args.no_fp16

    if not hf_token:
        print("Error: HuggingFace token required. Use --hf-token or set HF_TOKEN env var.")
        sys.exit(1)
    if not llm_api_key:
        print("Error: LLM API key required. Use --llm-api-key or set LLM_API_KEY env var.")
        sys.exit(1)

    common_kwargs = dict(
        work_dir=args.work_dir,
        hf_token=hf_token,
        llm_api_key=llm_api_key,
        llm_api_base=args.llm_api_base,
        llm_model=args.llm_model,
        model_dir=args.model_dir,
        num_speakers=args.num_speakers,
        use_fp16=use_fp16,
        cleanup=args.cleanup,
        external_subs=args.external_subs,
        no_external_subs=args.no_external_subs,
    )

    if args.batch:
        dub_batch(args.input, args.output, **common_kwargs)
    else:
        dub_video(args.input, args.output, **common_kwargs)


if __name__ == "__main__":
    main()
