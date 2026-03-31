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
STRETCH_MIN = 0.85
STRETCH_MAX = 1.15
STRETCH_HARD_LIMIT = 1.5
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
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Resuming from checkpoint: step {data['step']}")
    return data


def _clear_checkpoint(work_dir):
    """Remove checkpoint after successful completion."""
    path = os.path.join(work_dir, CHECKPOINT_FILE)
    if os.path.exists(path):
        os.remove(path)


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

def analyze_context(segments, llm_client):
    """Phase A: Analyze overall speech context before translating."""
    full_transcript = "\n".join(
        f"[{seg.get('speaker', '?')}] ({seg['start']:.1f}s-{seg['end']:.1f}s) {seg['text']}"
        for seg in segments
    )

    prompt = f"""你是一位专业的视频翻译顾问。以下是一段英文演讲/对话的完整字幕。
请分析并返回 JSON：

{{
  "topic": "演讲/对话的核心主题",
  "domain": "所属领域（科技/商业/教育/医学等）",
  "tone": "整体语气风格（正式/幽默/激情/学术等）",
  "key_terms": {{"english_term": "推荐的中文翻译"}},
  "speakers": {{"SPEAKER_00": "角色描述"}},
  "translation_notes": "翻译时需要特别注意的事项"
}}

完整字幕：
{full_transcript}

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
            "key_terms": {}, "speakers": {}, "translation_notes": "",
        }


def translate_with_context(segments, context, llm_client, batch_size=12):
    """Phase B: Translate in context-aware batches with sliding window."""
    translated = []

    for i in range(0, len(segments), batch_size):
        batch = segments[i : i + batch_size]

        prev_lines = ""
        if translated:
            prev_items = translated[-3:]
            prev_lines = "【前文已翻译】\n" + "\n".join(
                f"#{item['id']}: {item['zh_text']}" for item in prev_items
            ) + "\n\n"

        batch_lines = []
        for j, seg in enumerate(batch):
            idx = i + j
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
注意事项：{context.get('translation_notes', '')}

{prev_lines}【待翻译段落】
{chr(10).join(batch_lines)}

翻译要求：
1. 口语化，适合朗读配音，不要书面语
2. 每句括号中标注了目标时长和建议字数，请严格控制字数
3. 联系上下文，保持前后连贯和术语一致
4. 保持原文的语气和情感色彩
5. 人名/专有名词保持一致

返回格式（每行一句，#号对应原句编号）：
#{i} 翻译结果
#{i+1} 翻译结果
..."""

        response = llm_client.chat(prompt)
        parsed = _parse_translation_response(response)

        for j, seg in enumerate(batch):
            idx = i + j
            zh_text = parsed.get(idx)
            if zh_text is None:
                print(f"  Warning: segment #{idx} translation missing, using original text")
                zh_text = seg["text"]
            seg["zh_text"] = zh_text
            translated.append({"id": idx, "zh_text": zh_text})

        print(f"  Translated segments {i}-{i + len(batch) - 1}")

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


def translate_segments(segments, llm_client, batch_size=12):
    """Full translation pipeline: analyze context, then batch translate."""
    print("  Phase A: Analyzing speech context...")
    context = analyze_context(segments, llm_client)
    print(f"  Topic: {context.get('topic', '?')}")
    print(f"  Domain: {context.get('domain', '?')}")
    print(f"  Tone: {context.get('tone', '?')}")
    if context.get("key_terms"):
        print(f"  Key terms: {json.dumps(context['key_terms'], ensure_ascii=False)}")

    print("  Phase B: Translating with context...")
    segments = translate_with_context(segments, context, llm_client, batch_size)
    return segments, context


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
    """Align generated audio to target duration using time-stretching.

    Uses 'end_padded' (from absorb_gaps) for more available time.
    No hard truncation — prefers aggressive stretching over cutting audio.
    """
    import pyrubberband as pyrb

    aligned_dir = os.path.join(work_dir, "aligned")
    os.makedirs(aligned_dir, exist_ok=True)

    for i, seg in enumerate(segments):
        if seg.get("wav_path") is None:
            seg["aligned_path"] = None
            continue

        # Use padded end (with absorbed gap) for more room
        target_dur = seg.get("end_padded", seg["end"]) - seg["start"]
        actual_dur = seg["actual_duration"]

        if target_dur <= 0 or actual_dur <= 0:
            seg["aligned_path"] = seg["wav_path"]
            continue

        ratio = actual_dur / target_dur
        aligned_path = os.path.join(aligned_dir, f"aligned_{i:04d}.wav")

        if 0.99 <= ratio <= 1.01:
            seg["aligned_path"] = seg["wav_path"]
            continue

        audio, sr = sf.read(seg["wav_path"])

        if ratio > 1.0:
            # Too long: speed up (no truncation)
            stretch_rate = min(ratio, STRETCH_HARD_LIMIT)
            stretched = pyrb.time_stretch(audio, sr, rate=stretch_rate)
            sf.write(aligned_path, stretched, sr)
        else:
            # Too short: slow down, pad remainder with silence
            stretched = pyrb.time_stretch(audio, sr, rate=max(ratio, 0.5))
            target_samples = int(target_dur * sr)
            if len(stretched) < target_samples:
                padded = np.zeros(target_samples, dtype=stretched.dtype)
                padded[: len(stretched)] = stretched
                stretched = padded
            sf.write(aligned_path, stretched, sr)

        seg["aligned_path"] = aligned_path

    return segments


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


def assemble_final(segments, bg_audio_path, video_only_path, output_path, work_dir):
    """Assemble Chinese speech + background + video into final output."""
    bg_info = sf.info(bg_audio_path)
    sr = bg_info.samplerate
    total_samples = bg_info.frames

    speech_track = np.zeros(total_samples, dtype=np.float32)
    for seg in segments:
        aligned_path = seg.get("aligned_path")
        if aligned_path is None or not os.path.exists(aligned_path):
            continue

        audio, audio_sr = sf.read(aligned_path, dtype="float32")
        if audio_sr != sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=audio_sr, target_sr=sr)

        # Apply fade to avoid click at segment boundaries
        audio = _apply_fade(audio, sr)

        start_sample = int(seg["start"] * sr)
        end_sample = start_sample + len(audio)

        if start_sample >= total_samples:
            continue
        if end_sample > total_samples:
            audio = audio[: total_samples - start_sample]
            end_sample = total_samples

        # Additive mix (not overwrite) — handles overlapping segments gracefully
        speech_track[start_sample:end_sample] += audio[: end_sample - start_sample]

    speech_path = os.path.join(work_dir, "chinese_speech.wav")
    sf.write(speech_path, speech_track, sr)
    del speech_track

    # Mix speech + background (with volume control) and merge with video
    _run_ffmpeg(
        "-i", video_only_path,
        "-i", speech_path,
        "-i", bg_audio_path,
        "-filter_complex",
        f"[1:a]volume=1.0[speech];[2:a]volume={BG_VOLUME}[bg];"
        "[speech][bg]amix=inputs=2:duration=longest:normalize=0[aout]",
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy", "-shortest",
        output_path,
    )

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
    """Generate SRT subtitle file from segments."""
    text_key = "zh_text" if lang == "zh" else "text"
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            text = seg.get(text_key, "")
            if not text.strip():
                continue
            start = _format_srt_time(seg["start"])
            end = _format_srt_time(seg["end"])  # Original end, not end_padded
            f.write(f"{i + 1}\n{start} --> {end}\n{text}\n\n")
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
        print("\n[Step 1/10] Extracting audio and video tracks...")
        audio_path, video_only_path = extract_tracks(video_path, video_work_dir)
        paths.update(audio_path=audio_path, video_only_path=video_only_path)
        _save_checkpoint(video_work_dir, 1, paths=paths)
    else:
        audio_path = paths["audio_path"]
        video_only_path = paths["video_only_path"]
        print(f"\n[Step 1/10] Skipped (cached)")

    # --- Step 2: Source separation ---
    if done < 2:
        print("\n[Step 2/10] Separating vocals from background...")
        vocals_path, bg_path = separate_vocals(audio_path, video_work_dir)
        paths.update(vocals_path=vocals_path, bg_path=bg_path)
        _save_checkpoint(video_work_dir, 2, paths=paths)
    else:
        vocals_path = paths["vocals_path"]
        bg_path = paths["bg_path"]
        print(f"\n[Step 2/10] Skipped (cached)")

    # --- Step 3: ASR + diarization ---
    if done < 3:
        print("\n[Step 3/10] Transcribing and diarizing...")
        segments = transcribe_and_diarize(vocals_path, hf_token, num_speakers)
        transcript_path = os.path.join(video_work_dir, "transcript.json")
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        print(f"  Saved transcript: {transcript_path}")
        _save_checkpoint(video_work_dir, 3, segments=segments, paths=paths)
    else:
        print(f"\n[Step 3/10] Skipped (cached, {len(segments)} segments)")

    # --- Step 3.5: Gap absorption ---
    if done < 4:
        print("\n[Step 3.5/10] Absorbing inter-segment gaps...")
        segments = absorb_gaps(segments)
        _save_checkpoint(video_work_dir, 4, segments=segments, paths=paths)
    else:
        print(f"\n[Step 3.5/10] Skipped (cached)")

    # --- Step 4: Speaker references ---
    if done < 5:
        print("\n[Step 4/10] Extracting speaker references...")
        speaker_refs = extract_speaker_refs(segments, vocals_path, video_work_dir, fallback_refs)
        paths["speaker_refs"] = speaker_refs
        _save_checkpoint(video_work_dir, 5, segments=segments, paths=paths)
    else:
        speaker_refs = paths.get("speaker_refs", {})
        print(f"\n[Step 4/10] Skipped (cached)")

    # --- Step 5: Translation ---
    if done < 6:
        print("\n[Step 5/10] Translating to Chinese...")
        llm_client = LLMClient(api_key=llm_api_key, api_base=llm_api_base, model=llm_model)
        segments, context = translate_segments(segments, llm_client)
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
        print(f"\n[Step 5/10] Skipped (cached)")

    # --- Step 6: TTS generation ---
    if done < 7:
        print("\n[Step 6/10] Generating Chinese speech with IndexTTS2...")
        if tts is None:
            tts = _init_tts(model_dir, use_fp16)
        segments = generate_speech(segments, speaker_refs, vocals_path, video_work_dir, tts)
        _save_checkpoint(video_work_dir, 7, segments=segments, paths=paths)
    else:
        print(f"\n[Step 6/10] Skipped (cached)")

    # --- Step 7: Duration alignment ---
    if done < 8:
        print("\n[Step 7/10] Aligning durations...")
        segments = align_durations(segments, video_work_dir)
        _save_checkpoint(video_work_dir, 8, segments=segments, paths=paths)
    else:
        print(f"\n[Step 7/10] Skipped (cached)")

    # --- Step 8-9: Assembly ---
    if done < 9:
        print("\n[Step 8-9/10] Assembling final video...")
        assemble_final(segments, bg_path, video_only_path, output_path, video_work_dir)
        _save_checkpoint(video_work_dir, 9, segments=segments, paths=paths)
    else:
        print(f"\n[Step 8-9/10] Skipped (cached)")

    # --- Step 10: SRT subtitles ---
    print("\n[Step 10/10] Generating subtitles...")
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
    )

    if args.batch:
        dub_batch(args.input, args.output, **common_kwargs)
    else:
        dub_video(args.input, args.output, **common_kwargs)


if __name__ == "__main__":
    main()
