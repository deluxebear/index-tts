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

CHARS_PER_SECOND = 3.8
AUDIO_COMFORT_LIMIT = 1.2
VIDEO_SLOWDOWN_MAX = 2.0
MIN_REF_DURATION = 3.0
GAP_ABSORB_MAX = 2.0
BG_VOLUME = 0.3
FADE_MS = 10
SLOWDOWN_COMFORT_LIMIT = 0.75  # min rate for natural-sounding slowdown
CHECKPOINT_FILE = "checkpoint.json"
TTS_CHECKPOINT_EVERY = 10
DURATION_FACTOR_MIN = 0.5
DURATION_FACTOR_MAX = 2.0
SKIP_ASR_REF_SECONDS = 8.0

_EN_MONTHS = {
    "january": 1, "jan": 1, "jan.": 1,
    "february": 2, "feb": 2, "feb.": 2,
    "march": 3, "mar": 3, "mar.": 3,
    "april": 4, "apr": 4, "apr.": 4,
    "may": 5,
    "june": 6, "jun": 6, "jun.": 6,
    "july": 7, "jul": 7, "jul.": 7,
    "august": 8, "aug": 8, "aug.": 8,
    "september": 9, "sep": 9, "sept": 9, "sep.": 9, "sept.": 9,
    "october": 10, "oct": 10, "oct.": 10,
    "november": 11, "nov": 11, "nov.": 11,
    "december": 12, "dec": 12, "dec.": 12,
}
_MONTH_ALT = "|".join(
    re.escape(k) for k in sorted(_EN_MONTHS, key=len, reverse=True)
)
_ORDINAL = r"(?:st|nd|rd|th)?"


def _zh_date(year=None, month=None, day=None):
    parts = []
    if year:
        parts.append(f"{int(year)}年")
    if month:
        parts.append(f"{int(month)}月")
    if day:
        parts.append(f"{int(day)}日")
    return "".join(parts)


def normalize_spoken_dates(text):
    """Rewrite leftover English / numeric dates into spoken Chinese (年/月/日).

    Idempotent on already-normalized strings such as ``2024年1月15日``.
    """
    if not text:
        return text

    def month_num(name):
        return _EN_MONTHS[name.lower().rstrip(".")]

    def repl_mdy(m):
        return _zh_date(m.group(3), month_num(m.group(1)), m.group(2))

    def repl_dmy(m):
        return _zh_date(m.group(3), month_num(m.group(2)), m.group(1))

    def repl_the_of(m):
        return _zh_date(month=month_num(m.group(2)), day=m.group(1))

    def repl_md(m):
        return _zh_date(month=month_num(m.group(1)), day=m.group(2))

    def repl_my(m):
        return _zh_date(m.group(2), month_num(m.group(1)))

    def repl_iso(m):
        return _zh_date(m.group(1), m.group(2), m.group(3))

    def repl_num(m):
        a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if a > 12:
            return _zh_date(y, b, a)
        return _zh_date(y, a, b)

    text = re.sub(
        rf"(?i)\b({_MONTH_ALT})\s+(\d{{1,2}}){_ORDINAL},?\s+(\d{{4}})\b",
        repl_mdy, text,
    )
    text = re.sub(
        rf"(?i)\b(\d{{1,2}}){_ORDINAL}\s+(?:of\s+)?({_MONTH_ALT}),?\s+(\d{{4}})\b",
        repl_dmy, text,
    )
    text = re.sub(
        rf"(?i)\bthe\s+(\d{{1,2}}){_ORDINAL}\s+of\s+({_MONTH_ALT})\b",
        repl_the_of, text,
    )
    text = re.sub(
        rf"(?i)\b({_MONTH_ALT})\s+(\d{{4}})\b",
        repl_my, text,
    )
    text = re.sub(
        rf"(?i)\b({_MONTH_ALT})\s+(\d{{1,2}}){_ORDINAL}\b",
        repl_md, text,
    )
    text = re.sub(r"\b(20\d{2}|19\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b", repl_iso, text)
    text = re.sub(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](20\d{2}|19\d{2})\b", repl_num, text)
    return text


_EN_HOUR_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_HOUR_WORD_ALT = "|".join(sorted(_EN_HOUR_WORDS, key=len, reverse=True))
_MERIDIEM = r"(?:a\.?m\.?|p\.?m\.?)"


def _parse_hour_token(token):
    token = token.lower()
    if token.isdigit():
        return int(token)
    return _EN_HOUR_WORDS[token]


def _zh_clock(hour, minute=0, meridiem=None):
    hour = int(hour)
    minute = int(minute or 0)
    if meridiem:
        mer = re.sub(r"[.\s]", "", meridiem.lower())
        if mer == "am":
            if hour == 12:
                hour = 0
        elif mer == "pm" and hour != 12:
            hour += 12
    hour = hour % 24
    if meridiem is None and 1 <= hour <= 12:
        if minute == 0:
            return f"{hour}点"
        if minute == 30:
            return f"{hour}点半"
        return f"{hour}点{minute}分"
    if hour == 0:
        period, h12 = "凌晨", 12
    elif hour < 6:
        period, h12 = "凌晨", hour
    elif hour < 12:
        period, h12 = "上午", hour
    elif hour == 12:
        period, h12 = "中午", 12
    elif hour < 19:
        period, h12 = "下午", hour - 12
    else:
        period, h12 = "晚上", hour - 12
    if minute == 0:
        return f"{period}{h12}点"
    if minute == 30:
        return f"{period}{h12}点半"
    return f"{period}{h12}点{minute}分"


def normalize_spoken_times(text):
    """Rewrite leftover English / numeric clock times into spoken Chinese.

    Idempotent on already-normalized strings such as ``下午3点``.
    Requires two-digit minutes in ``H:MM`` so ratios like ``3:1`` are left alone.
    """
    if not text:
        return text

    def hour_token(m, g):
        return _parse_hour_token(m.group(g))

    def repl_half(m):
        return _zh_clock(hour_token(m, 1), 30)

    def repl_q_past(m):
        return _zh_clock(hour_token(m, 1), 15)

    def repl_q_to(m):
        return _zh_clock((hour_token(m, 1) - 1) % 12 or 12, 45)

    def repl_hmm_mer(m):
        minute = int(m.group(2))
        if minute > 59:
            return m.group(0)
        return _zh_clock(int(m.group(1)), minute, m.group(3))

    def repl_h_mer(m):
        return _zh_clock(hour_token(m, 1), 0, m.group(2))

    def repl_oclock(m):
        return f"{hour_token(m, 1)}点"

    def repl_24h(m):
        hour, minute = int(m.group(1)), int(m.group(2))
        if hour > 23 or minute > 59:
            return m.group(0)
        return _zh_clock(hour, minute)

    text = re.sub(
        rf"(?i)\bhalf\s+past\s+(\d{{1,2}}|{_HOUR_WORD_ALT})\b",
        repl_half, text,
    )
    text = re.sub(
        rf"(?i)\b(?:a\s+)?quarter\s+past\s+(\d{{1,2}}|{_HOUR_WORD_ALT})\b",
        repl_q_past, text,
    )
    text = re.sub(
        rf"(?i)\b(?:a\s+)?quarter\s+to\s+(\d{{1,2}}|{_HOUR_WORD_ALT})\b",
        repl_q_to, text,
    )
    text = re.sub(
        rf"(?i)\b(\d{{1,2}}):(\d{{2}})\s*({_MERIDIEM})(?!\w)",
        repl_hmm_mer, text,
    )
    text = re.sub(
        rf"(?i)\b(\d{{1,2}}|{_HOUR_WORD_ALT})\s*({_MERIDIEM})(?!\w)",
        repl_h_mer, text,
    )
    text = re.sub(
        rf"(?i)\b(\d{{1,2}}|{_HOUR_WORD_ALT})\s+o['’]clock\b",
        repl_oclock, text,
    )
    text = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", repl_24h, text)
    return text


_NUM = r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
_MILLION = r"(?:million|mm?)"
_BILLION = r"(?:billion|bn?)"


def _parse_spoken_num(token):
    return float(token.replace(",", ""))


def _fmt_spoken_num(n):
    if abs(n - round(n)) < 1e-6:
        return str(int(round(n)))
    return f"{n:.4f}".rstrip("0").rstrip(".")


def _scale_wan_yi(n, scale):
    if scale == "million":
        return _fmt_spoken_num(n * 100) + "万"
    return _fmt_spoken_num(n * 10) + "亿"


def _scale_name(token):
    t = token.lower().rstrip(".")
    if t.startswith("b") or t == "bn":
        return "billion"
    return "million"


def normalize_spoken_numbers(text):
    """Rewrite leftover money / scale / percent into spoken Chinese.

    Idempotent on already-normalized strings such as ``100美元`` / ``百分之50``.
    """
    if not text:
        return text

    def repl_cur_scale(m):
        return _scale_wan_yi(_parse_spoken_num(m.group(1)), _scale_name(m.group(2))) + "美元"

    def repl_scale_dollars(m):
        return _scale_wan_yi(_parse_spoken_num(m.group(1)), _scale_name(m.group(2))) + "美元"

    def repl_scale(m):
        return _scale_wan_yi(_parse_spoken_num(m.group(1)), _scale_name(m.group(2)))

    def repl_usd(m):
        return _fmt_spoken_num(_parse_spoken_num(m.group(1))) + "美元"

    def repl_eur(m):
        return _fmt_spoken_num(_parse_spoken_num(m.group(1))) + "欧元"

    def repl_gbp(m):
        return _fmt_spoken_num(_parse_spoken_num(m.group(1))) + "英镑"

    def repl_cny(m):
        return _fmt_spoken_num(_parse_spoken_num(m.group(1))) + "元"

    def repl_pct(m):
        return "百分之" + _fmt_spoken_num(_parse_spoken_num(m.group(1)))

    def repl_comma(m):
        return m.group(1).replace(",", "")

    # $1.5 million / USD 1.5m
    text = re.sub(
        rf"(?i)(?:us\$|usd\s*|\$)\s*{_NUM}\s*({_MILLION}|{_BILLION})\b",
        repl_cur_scale, text,
    )
    # 1.5 million dollars / 1 billion USD
    text = re.sub(
        rf"(?i){_NUM}\s*({_MILLION}|{_BILLION})\s*(?:us\s*)?(?:dollars?|usd)\b",
        repl_scale_dollars, text,
    )
    # 1.5 million / 2 billion (no currency)
    text = re.sub(
        rf"(?i){_NUM}\s*({_MILLION}|{_BILLION})\b",
        repl_scale, text,
    )
    text = re.sub(rf"(?i)(?:us\$|usd\s*|\$)\s*{_NUM}", repl_usd, text)
    text = re.sub(rf"(?i){_NUM}\s*(?:us\s*)?(?:dollars?|usd)\b", repl_usd, text)
    text = re.sub(rf"(?i)(?:€|eur\s*)\s*{_NUM}", repl_eur, text)
    text = re.sub(rf"(?i){_NUM}\s*(?:euros?|eur)\b", repl_eur, text)
    text = re.sub(rf"(?i)(?:£|gbp\s*)\s*{_NUM}", repl_gbp, text)
    text = re.sub(rf"(?i){_NUM}\s*(?:pounds?|gbp)\b", repl_gbp, text)
    text = re.sub(rf"(?i)(?:¥|￥|rmb\s*|cny\s*)\s*{_NUM}", repl_cny, text)
    text = re.sub(rf"(?i){_NUM}\s*(?:yuan|rmb|cny)\b", repl_cny, text)
    text = re.sub(rf"(?i)(?<!百分之){_NUM}\s*(?:%|percent\b|per\s*cent\b)", repl_pct, text)
    text = re.sub(r"\b(\d{1,3}(?:,\d{3})+)\b", repl_comma, text)
    return text


_EN_WEEKDAYS = {
    "monday": "星期一", "mon": "星期一",
    "tuesday": "星期二", "tue": "星期二", "tues": "星期二",
    "wednesday": "星期三", "wed": "星期三",
    "thursday": "星期四", "thu": "星期四", "thur": "星期四", "thurs": "星期四",
    "friday": "星期五", "fri": "星期五",
    "saturday": "星期六", "sat": "星期六",
    "sunday": "星期日", "sun": "星期日",
}
_WEEKDAY_FULL_ALT = "|".join(
    sorted((k for k in _EN_WEEKDAYS if len(k) > 3), key=len, reverse=True)
)
_WEEKDAY_ABBR_ALT = "|".join(
    sorted((k for k in _EN_WEEKDAYS if len(k) <= 3), key=len, reverse=True)
)


def normalize_spoken_weekdays(text):
    """Rewrite leftover English weekdays into 星期X. Bare ``Sun`` is left alone."""
    if not text:
        return text

    def repl_full(m):
        return _EN_WEEKDAYS[m.group(1).lower()]

    def repl_abbr(m):
        return _EN_WEEKDAYS[m.group(1).lower().rstrip(".")]

    text = re.sub(rf"(?i)\b({_WEEKDAY_FULL_ALT})\b", repl_full, text)
    text = re.sub(rf"(?i)\b({_WEEKDAY_ABBR_ALT})\.", repl_abbr, text)
    text = re.sub(r"(?i)\bweekends?\b", "周末", text)
    return text


def normalize_spoken_datetime(text):
    """Normalize leftover English dates, times, numbers, and weekdays for TTS."""
    text = normalize_spoken_dates(text)
    text = normalize_spoken_times(text)
    text = normalize_spoken_numbers(text)
    text = normalize_spoken_weekdays(text)
    return text


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

class _CheckpointEncoder(json.JSONEncoder):
    """JSON encoder that skips numpy arrays (e.g. speaker embeddings)."""
    def default(self, obj):
        if type(obj).__name__ == "ndarray":
            return None  # drop numpy arrays — recomputed on resume
        return super().default(obj)


def _save_checkpoint(work_dir, step, segments=None, **extra):
    """Save pipeline progress after each step."""
    data = {"step": step, "segments": segments}
    data.update(extra)
    path = os.path.join(work_dir, CHECKPOINT_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, cls=_CheckpointEncoder)
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
    from indextts.infer_v2_5 import IndexTTS2
    return IndexTTS2(
        cfg_path=os.path.join(model_dir, "config.yaml"),
        model_dir=model_dir,
        use_bf16=bool(use_fp16),
    )


def _offload_tts(tts):
    """Move TTS model to CPU to free GPU VRAM for other steps."""
    import torch
    if tts is None or not torch.cuda.is_available():
        return
    for name in ("gpt", "s2mel", "bigvgan", "campplus_model",
                 "semantic_model", "semantic_codec"):
        model = getattr(tts, name, None)
        if model is not None:
            model.cpu()
    torch.cuda.empty_cache()


def _reload_tts(tts):
    """Move TTS model back to GPU for inference."""
    import torch
    if tts is None or not torch.cuda.is_available():
        return
    device = tts.device
    for name in ("gpt", "s2mel", "bigvgan", "campplus_model",
                 "semantic_model", "semantic_codec"):
        model = getattr(tts, name, None)
        if model is not None:
            model.to(device)


def transcribe_and_diarize(
    vocals_path, hf_token, num_speakers=None, whisper_model="large-v2",
):
    """Transcribe with word-level timestamps and speaker labels.

    ``num_speakers=1`` skips pyannote and labels every segment SPEAKER_00.
    """
    import whisperx

    device = _detect_device()
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"  Loading Whisper model ({whisper_model})...")
    model = whisperx.load_model(whisper_model, device, compute_type=compute_type)
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

    if num_speakers == 1:
        print("  Skipping speaker diarization (num_speakers=1)")
        segments = result["segments"]
        for seg in segments:
            seg["speaker"] = "SPEAKER_00"
        print(f"  Transcribed {len(segments)} segments, 1 speaker")
        return segments

    if not hf_token:
        raise ValueError("hf_token is required for speaker diarization (num_speakers != 1)")

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


def find_loudest_window(audio_path, window_sec=SKIP_ASR_REF_SECONDS, ranges=None, hop_sec=0.05):
    """Return (start, end) of the highest-RMS window, optionally inside ranges."""
    audio, sr = sf.read(audio_path, dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    n = int(len(audio))
    if n <= 0 or sr <= 0:
        return 0.0, 0.0
    duration = n / sr
    hop = max(1, int(hop_sec * sr))
    win = max(hop, int(float(window_sec) * sr))
    if win >= n:
        return 0.0, duration

    n_hops = max(1, n // hop)
    energy = np.empty(n_hops, dtype=np.float64)
    for i in range(n_hops):
        chunk = audio[i * hop:(i + 1) * hop]
        energy[i] = float(np.mean(chunk * chunk)) if len(chunk) else 0.0

    def hop_ok(i):
        if not ranges:
            return True
        t = (i + 0.5) * hop / sr
        return any(a <= t < b for a, b in ranges)

    hops_per_win = max(1, win // hop)
    csum = np.concatenate([[0.0], np.cumsum(energy)])
    best_i, best_e = 0, -1.0
    last = len(energy) - hops_per_win + 1
    for i in range(max(0, last)):
        inside = sum(1 for j in range(i, i + hops_per_win) if hop_ok(j))
        if ranges and inside < hops_per_win * 0.4:
            continue
        e = float(csum[i + hops_per_win] - csum[i])
        if e > best_e:
            best_e = e
            best_i = i
    if best_e < 0:
        if ranges:
            a, b = ranges[0]
            return float(a), float(min(duration, max(b, a + min(window_sec, duration))))
        return 0.0, min(float(window_sec), duration)
    start = best_i * hop / sr
    end = min(duration, start + win / sr)
    return float(start), float(end)


def _compute_speaker_embedding(audio_path):
    """Compute a 192-dim speaker embedding using CAMPPlus.

    Loads the model on first call (CPU only, very lightweight).
    Returns a numpy vector of shape (192,).
    """
    import torch
    import torchaudio
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus

    if not hasattr(_compute_speaker_embedding, "_model"):
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
        model = CAMPPlus(feat_dim=80, embedding_size=192)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()
        _compute_speaker_embedding._model = model

    model = _compute_speaker_embedding._model
    audio, sr = torchaudio.load(audio_path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)
    feat = torchaudio.compliance.kaldi.fbank(audio, num_mel_bins=80, dither=0,
                                              sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    with torch.no_grad():
        emb = model(feat.unsqueeze(0))  # [1, 192]
    return emb.squeeze(0).numpy()


def _cosine_similarity(a, b):
    """Cosine similarity between two 1-D numpy vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm > 0 else 0.0


SPEAKER_SIM_THRESHOLD = 0.7  # below this, diarization label is likely wrong


def extract_speaker_refs(segments, vocals_path, work_dir, fallback_refs=None):
    """Extract the best (longest) reference audio per speaker with embeddings."""
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
        ranges = [(s["start"], s["end"]) for s in segs if s.get("end", 0) > s.get("start", 0)]
        try:
            start, end = find_loudest_window(
                vocals_path, window_sec=SKIP_ASR_REF_SECONDS, ranges=ranges or None,
            )
        except Exception:
            best_seg = max(segs, key=lambda s: s["end"] - s["start"])
            start, end = best_seg["start"], best_seg["end"]
        if end - start < 0.3:
            best_seg = max(segs, key=lambda s: s["end"] - s["start"])
            start, end = best_seg["start"], best_seg["end"]
        ref_path = os.path.join(refs_dir, f"ref_{spk}.wav")
        extract_audio_segment(vocals_path, start, end, ref_path)

        if fallback_refs and spk in fallback_refs:
            fallback = fallback_refs[spk]
        else:
            fallback = ref_path

        # Pre-compute speaker embedding for voice consistency verification
        embedding = _compute_speaker_embedding(ref_path)

        speaker_refs[spk] = {
            "fallback": fallback,
            "best_auto": ref_path,
            "embedding": embedding,
        }
        print(f"  {spk}: ref {start:.1f}-{end:.1f}s ({end - start:.1f}s), "
              f"fallback={'user-provided' if fallback_refs and spk in fallback_refs else 'auto'}")

    return speaker_refs


def _find_best_speaker(seg_embedding, speaker_refs):
    """Find the speaker whose embedding is most similar to the segment."""
    best_spk = None
    best_sim = -1.0
    for spk, info in speaker_refs.items():
        sim = _cosine_similarity(seg_embedding, info["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_spk = spk
    return best_spk, best_sim


def get_ref_for_segment(seg, speaker_refs, vocals_path, seg_ref_dir):
    """Select reference audio with speaker embedding verification for short segments."""
    duration = seg["end"] - seg["start"]
    spk = seg.get("speaker", "UNKNOWN")

    if duration >= MIN_REF_DURATION:
        seg_ref = os.path.join(seg_ref_dir, f"seg_{seg['start']:.2f}_{seg['end']:.2f}.wav")
        if not os.path.exists(seg_ref):
            extract_audio_segment(vocals_path, seg["start"], seg["end"], seg_ref)
        return seg_ref

    # Short segment: verify voice matches assigned speaker via embedding
    seg_ref = os.path.join(seg_ref_dir, f"seg_{seg['start']:.2f}_{seg['end']:.2f}.wav")
    if not os.path.exists(seg_ref):
        extract_audio_segment(vocals_path, seg["start"], seg["end"], seg_ref)

    seg_embedding = _compute_speaker_embedding(seg_ref)
    assigned_sim = _cosine_similarity(seg_embedding, speaker_refs[spk]["embedding"])

    if assigned_sim >= SPEAKER_SIM_THRESHOLD:
        return speaker_refs[spk]["fallback"]

    # Diarization likely wrong — find the best matching speaker
    best_spk, best_sim = _find_best_speaker(seg_embedding, speaker_refs)
    if best_spk and best_spk != spk:
        print(f"  Voice mismatch: seg {seg['start']:.1f}-{seg['end']:.1f}s "
              f"labeled {spk} (sim={assigned_sim:.2f}) → using {best_spk} (sim={best_sim:.2f})")
        seg["speaker"] = best_spk  # correct the label for downstream use
        return speaker_refs[best_spk]["fallback"]

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


def _build_translation_prompt(to_translate, context, keep_original_str, prev_lines):
    """Build the translation prompt for a set of (idx, seg) pairs."""
    batch_lines = []
    for idx, seg in to_translate:
        duration = seg["end"] - seg["start"]
        target_chars = max(4, int(duration * CHARS_PER_SECOND))
        batch_lines.append(
            f"#{idx} [{seg.get('speaker', '?')}] ({duration:.1f}s, ~{target_chars}字) {seg['text']}"
        )

    return f"""你是一位专业的中文配音翻译。

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
7. 日期一律译成配音口播格式，不要保留英文月份或斜杠数字：
   - January 15, 2024 / Jan. 15th, 2024 / 15 January 2024 → 2024年1月15日
   - the 3rd of March / March 3rd → 3月3日
   - March 2020 → 2020年3月
   - 2024-01-15、01/15/2024 → 2024年1月15日
   - 不要写成 January、1/15、15th
8. 时刻一律译成配音口播格式，不要保留 AM/PM 或冒号：
   - 3:00 PM / 3 p.m. / 15:00 → 下午3点
   - 3:30 PM / half past 3 → 下午3点半
   - 10:15 AM → 上午10点15分
   - 12:00 PM → 中午12点；12:00 AM → 凌晨12点
   - 3 o'clock → 3点
   - quarter to 5 → 4点45分
   - 不要写成 3:00、3PM、15:00
9. 数字、金额、百分比译成配音口播格式：
   - $100 / 100 dollars / USD 100 → 100美元
   - $1.5 million / 1.5 million dollars → 150万美元
   - 1 million → 100万；1 billion → 10亿
   - 50% / 50 percent → 百分之50
   - €50 → 50欧元；£20 → 20英镑；¥100 / 100 yuan → 100元
10. 星期一律译成「星期X」：Monday → 星期一；weekend → 周末
11. 每句末尾用 || 附上 8 维情绪（0-1，happy,angry,sad,afraid,disgusted,melancholic,surprised,calm），总和不超过 0.8

返回格式（每行一句，#号对应原句编号）：
#{to_translate[0][0]} 翻译结果 || 0.1,0,0,0,0,0,0,0.4
#{to_translate[-1][0]} 翻译结果 || 0,0,0,0,0,0,0,0.4
..."""


MAX_TRANSLATION_RETRIES = 3


def translate_with_context(segments, context, llm_client, batch_size=12,
                           skip_translated=False):
    """Phase B: Translate in context-aware batches with sliding window and retry."""
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

        # Retry loop for this batch
        pending = list(need_translation)
        for attempt in range(MAX_TRANSLATION_RETRIES):
            if not pending:
                break

            prompt = _build_translation_prompt(
                pending, context, keep_original_str, prev_lines
            )

            try:
                response = llm_client.chat(prompt)
            except Exception as e:
                print(f"  Warning: LLM API error (attempt {attempt + 1}/{MAX_TRANSLATION_RETRIES}): {e}")
                continue

            parsed, emos = _parse_translation_response(response)

            still_pending = []
            for idx, seg in pending:
                zh_text = parsed.get(idx)
                if zh_text is not None:
                    zh_text = normalize_spoken_datetime(zh_text)
                    seg["zh_text"] = zh_text
                    if idx in emos:
                        seg["emo_vector"] = emos[idx]
                    translated.append({"id": idx, "zh_text": zh_text})
                else:
                    still_pending.append((idx, seg))

            pending = still_pending
            if pending and attempt < MAX_TRANSLATION_RETRIES - 1:
                missing_ids = [idx for idx, _ in pending]
                print(f"  Retrying {len(pending)} missing segments {missing_ids} "
                      f"(attempt {attempt + 2}/{MAX_TRANSLATION_RETRIES})...")

        # Do not fall back to English — leftover source would be spoken as Chinese.
        for idx, seg in pending:
            print(f"  Warning: segment #{idx} translation failed after "
                  f"{MAX_TRANSLATION_RETRIES} attempts; leaving zh_text empty")
            seg.pop("zh_text", None)

        translated_ids = [idx for idx, _ in need_translation]
        print(f"  Translated segments {translated_ids[0]}-{translated_ids[-1]}")

    return segments


def _parse_emo_vector(text):
    parts = [p.strip() for p in text.replace("，", ",").split(",") if p.strip()]
    if len(parts) != 8:
        return None
    try:
        vec = [max(0.0, min(1.0, float(p))) for p in parts]
    except ValueError:
        return None
    total = sum(vec)
    if total > 0.8 and total > 0:
        vec = [v * 0.8 / total for v in vec]
    return vec


def _split_zh_and_emotion(rest):
    """Split ``译文 || 0.1,0,...`` into (zh_text, emo_vector_or_None)."""
    if "||" in rest:
        left, right = rest.rsplit("||", 1)
        vec = _parse_emo_vector(right)
        if vec is not None:
            return left.strip(), vec
    return rest.strip(), None


def _parse_translation_response(response):
    """Parse numbered translation lines. Returns (zh_by_id, emo_by_id)."""
    result = {}
    emos = {}
    for line in response.strip().split("\n"):
        line = line.strip()
        match = re.match(r"#(\d+)\s+(.+)", line)
        if match:
            idx = int(match.group(1))
            zh, emo = _split_zh_and_emotion(match.group(2).strip())
            if zh:
                result[idx] = zh
            if emo is not None:
                emos[idx] = emo
    return result, emos


def estimate_emotion_from_text(text):
    """Rule-of-thumb 8-dim vector, or None when the line looks neutral."""
    if not text:
        return None
    vec = [0.0] * 8  # happy angry sad afraid disgusted melancholic surprised calm
    if re.search(r"[哈嘻乐笑太棒真棒太好]", text) or text.count("！") >= 2 or text.count("!") >= 2:
        vec[0] = 0.25
    if re.search(r"生气|气愤|愤怒|怒骂", text):
        vec[1] = 0.25
    if re.search(r"伤心|难过|哭泣|悲伤", text):
        vec[2] = 0.25
    if re.search(r"害怕|恐惧|震惊", text) or "？！" in text or "?!" in text:
        vec[3] = 0.15
        vec[6] = 0.15
    if "？" in text or "?" in text:
        vec[6] = max(vec[6], 0.1)
    if sum(vec) == 0:
        return None
    vec[7] = 0.15
    total = sum(vec)
    if total > 0.8:
        vec = [v * 0.8 / total for v in vec]
    return vec


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


def untranslated_segment_ids(segments):
    """Ids that have source text but no Chinese translation."""
    return [
        i for i, seg in enumerate(segments)
        if (seg.get("text") or "").strip() and not (seg.get("zh_text") or "").strip()
    ]


# ---------------------------------------------------------------------------
# Step 4.5: External subtitle loading (skip LLM when subs exist)
# ---------------------------------------------------------------------------

# Non-dialogue patterns to strip from external subtitles
_CREDIT_RE = re.compile(
    r"^(翻译|译者|译制|听译|审核|校对|校订|字幕|时间轴|压制|后期|特效|制作)"
    r"(组|人员|者)?[：:]\s*\S+",
    re.MULTILINE,
)
_CREDIT_EN_RE = re.compile(
    r"^(Translated|Reviewed|Subtitl(?:es|ed)?|Timing|Encoded|Transcript(?:ed)?)\s+by\b",
    re.IGNORECASE | re.MULTILINE,
)
_CREDIT_INLINE_RE = re.compile(
    r"[（(\[]\s*(?:翻译|译者|译制|字幕|校对|Translated\s+by|Subtitles?\s+by)[^)\]）]*[)\]）]",
    re.IGNORECASE,
)
_CREDIT_WHOLE_RE = re.compile(
    r"^(?:字幕组|字幕制作|听写稿|感谢观看|请不吝点赞|"
    r"thanks?\s+(?:you\s+)?for\s+watching|please\s+subscribe)\b",
    re.IGNORECASE,
)
# Speaker label: "比拉瓦尔·西杜（BS）:" or "克里斯·安德森（Chris Anderson）：" at start of cue.
# Full-name form always matches; bare abbreviation form collected dynamically.
# Group 1 captures the parenthesized part — may be abbreviation ("BS") or full name ("Chris Anderson").
_SPEAKER_FULL_RE = re.compile(
    r"^(?:[\w\s·\-]+)[（(]([A-Za-z][A-Za-z\s.\-']*?)[)）]\s*[：:]\s*"
)
_SPEAKER_BARE_RE_TMPL = r"^(?:{})\s*[：:]\s*"
# Generic bare uppercase label: "BS:", "CA：", "SA :" etc. — always stripped.
_SPEAKER_GENERIC_BARE_RE = re.compile(r"^[A-Z]{1,5}\s*[：:]\s*")
_SPEAKER_EN_NAME_RE = re.compile(
    r"^(?:>>\s*)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s*[：:]\s*"
)
_SPEAKER_ROLE_RE = re.compile(
    r"^(?:主持人|嘉賓|嘉宾|旁白|解说|記者|记者|觀眾|观众|采访者|受访者)"
    r"[ABC甲乙丙丁0-9]{0,2}\s*[：:]\s*"
)
# 2–4 CJK chars + colon is often a name; skip discourse connectives.
_SPEAKER_ZH_NAME_RE = re.compile(r"^[\u4e00-\u9fff]{2,4}\s*[：:]\s*")
_ZH_NOT_SPEAKER = frozenset(
    "所以 但是 因为 因為 如果 然后 然後 接着 接著 而且 不过 不過 可是 于是 於是 "
    "因此 其实 其實 当然 當然 那么 那麼 就是 不是 没有 沒有 这个 這個 那个 那個 "
    "什么 什麼 怎么 怎麼 为什么 為什麼 现在 現在 今天 我们 我們 他们 他們 你们 你們 "
    "大家 其实 还有 還有 或者 以及 虽然 雖然 尽管 儘管 除了 关于 關於".split()
)


def peek_speaker_label(raw_text):
    """Return a speaker/role label from the start of a raw cue, or None."""
    if not raw_text:
        return None
    line = _HTML_TAG_RE.sub("", str(raw_text).replace("&nbsp;", " "))
    line = re.split(r"[\n\\N]+", line.strip())[0].strip()
    if not line:
        return None
    m = _SPEAKER_FULL_RE.match(line)
    if m:
        return m.group(1).strip() or None
    m = _SPEAKER_GENERIC_BARE_RE.match(line)
    if m:
        return line[: m.end()].rstrip("：: ").strip() or None
    m = _SPEAKER_EN_NAME_RE.match(line)
    if m:
        return line[: m.end()].lstrip("> ").rstrip("：: ").strip() or None
    m = _SPEAKER_ROLE_RE.match(line)
    if m:
        return line[: m.end()].rstrip("：: ").strip() or None
    m = _SPEAKER_ZH_NAME_RE.match(line)
    if m:
        name = line[: m.end()].rstrip("：: ").strip()
        if name and name not in _ZH_NOT_SPEAKER:
            return name
    return None
_HTML_TAG_RE = re.compile(r"</?[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
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


def _strip_speaker_prefix(line, speaker_re):
    """Strip one speaker/role label from the start of a line; keep the dialogue."""
    text = speaker_re.sub("", line)
    text = _SPEAKER_GENERIC_BARE_RE.sub("", text)
    text = _SPEAKER_EN_NAME_RE.sub("", text)
    text = _SPEAKER_ROLE_RE.sub("", text)
    zh = _SPEAKER_ZH_NAME_RE.match(text)
    if zh:
        name = text[: zh.end()].split("：")[0].split(":")[0].strip()
        if name not in _ZH_NOT_SPEAKER:
            text = text[zh.end() :]
    return text


def clean_subtitle_cues(cues):
    """Remove credits, speaker labels, and non-speech marks so cues can be spoken."""
    # Pass 1: collect known speaker abbreviations from full-name labels
    # e.g. "比拉瓦尔·西杜（BS）:" → "BS"
    # e.g. "克里斯·安德森（Chris Anderson）:" → derive "CA" from initials
    speaker_abbrevs = set()
    for cue in cues:
        m = _SPEAKER_FULL_RE.match(cue["text"])
        if m:
            name_or_abbrev = m.group(1).strip()
            if " " in name_or_abbrev:
                # Full name like "Chris Anderson" → derive initials "CA"
                initials = "".join(
                    w[0].upper() for w in name_or_abbrev.split() if w and w[0].isalpha()
                )
                if initials:
                    speaker_abbrevs.add(initials)
            else:
                speaker_abbrevs.add(name_or_abbrev)

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
        raw = cue["text"]
        raw = _HTML_TAG_RE.sub("", raw)
        raw = raw.replace("&nbsp;", " ").replace("{", "").replace("}", "")
        raw = _URL_RE.sub("", raw)
        raw = _CREDIT_INLINE_RE.sub("", raw)

        lines = []
        for line in re.split(r"[\n\\N]+", raw):
            line = line.strip()
            if not line:
                continue
            if _CREDIT_RE.search(line) or _CREDIT_EN_RE.search(line) or _CREDIT_WHOLE_RE.search(line):
                continue
            line = _strip_speaker_prefix(line, speaker_re)
            line = line.strip()
            if not line:
                continue
            if _CREDIT_RE.search(line) or _CREDIT_EN_RE.search(line) or _CREDIT_WHOLE_RE.search(line):
                continue
            line = _SOUND_BRACKET_RE.sub(
                lambda m: "" if _is_sound_description(m) else m.group(0), line
            )
            line = _MUSIC_RE.sub("", line).strip()
            if line:
                lines.append(line)

        text = " ".join(lines).strip()
        if not text or _CREDIT_WHOLE_RE.search(text):
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


def discover_subtitle(video_path, quiet=False):
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

    if not quiet:
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


def load_cleaned_subtitle_cues(video_path, external_subs=None, quiet=False):
    """Discover, parse, and clean subtitle cues.

    Returns ``(cues, sub_path, lang_hint)``. ``cues`` is empty when nothing usable
    is found; ``sub_path`` / ``lang_hint`` may still be set for an empty file.
    """
    sub_path, lang_hint = discover_subtitle(external_subs or video_path, quiet=quiet)
    if sub_path is None:
        return [], None, None

    parser = parse_ass if sub_path.endswith(".ass") else parse_srt
    cues = parser(sub_path)
    if not cues:
        if not quiet:
            print(f"  Warning: subtitle file is empty: {sub_path}")
        return [], sub_path, lang_hint

    for cue in cues:
        cue["speaker_label"] = peek_speaker_label(cue.get("text", ""))
    cues = clean_subtitle_cues(cues)
    if not cues:
        if not quiet:
            print("  Warning: all subtitle cues were non-dialogue, skipping")
        return [], sub_path, lang_hint

    if lang_hint == "unknown":
        sample = " ".join(c["text"] for c in cues[:50])
        lang_hint = detect_subtitle_language(sample)
    return cues, sub_path, lang_hint


def should_skip_asr(num_speakers, video_path, no_external_subs=False, external_subs=None):
    """True when cleaned EN/ZH subtitles exist (any speaker count)."""
    if no_external_subs:
        return False
    cues, _, _ = load_cleaned_subtitle_cues(
        video_path, external_subs=external_subs, quiet=True,
    )
    return bool(cues)


def dummy_single_speaker_segments(vocals_path, clip_sec=SKIP_ASR_REF_SECONDS):
    """Placeholder ASR row from the loudest vocal window."""
    try:
        start, end = find_loudest_window(vocals_path, window_sec=clip_sec)
    except Exception:
        start, end = 0.0, float(clip_sec)
    if end - start < 0.3:
        start, end = 0.0, max(float(clip_sec), 0.5)
    return [{
        "start": start,
        "end": end,
        "text": "",
        "speaker": "SPEAKER_00",
    }]


def diarize_only(vocals_path, hf_token, num_speakers=None):
    """Speaker turns from pyannote only — no Whisper transcription."""
    from whisperx.diarize import DiarizationPipeline

    device = _detect_device()
    model = DiarizationPipeline(token=hf_token, device=device)
    kwargs = {}
    if num_speakers is not None:
        kwargs["min_speakers"] = max(1, num_speakers - 1)
        kwargs["max_speakers"] = num_speakers + 1
    raw = model(vocals_path, **kwargs)
    del model
    turns = []
    if hasattr(raw, "iterrows"):
        for _, row in raw.iterrows():
            turns.append({
                "start": float(row["start"]),
                "end": float(row["end"]),
                "speaker": str(row.get("speaker", "SPEAKER_00")),
            })
    elif hasattr(raw, "itertracks"):
        for turn, _, speaker in raw.itertracks(yield_label=True):
            turns.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            })
    print(f"  Diarization-only: {len(turns)} turns, "
          f"{len({t['speaker'] for t in turns})} speakers")
    return turns


def _speaker_from_turns(cue, turns):
    best, best_ov = None, 0.0
    for turn in turns:
        ov = min(cue["end"], turn["end"]) - max(cue["start"], turn["start"])
        if ov > best_ov:
            best_ov = ov
            best = turn["speaker"]
    return best


def assign_speakers_to_cues(cues, vocals_path=None, num_speakers=None, hf_token=None):
    """Attach SPEAKER_XX using subtitle labels, else diarization, else SPEAKER_00."""
    labels = [c.get("speaker_label") for c in cues]
    labeled = [x for x in labels if x]
    unique = []
    for name in labeled:
        if name not in unique:
            unique.append(name)

    enough = len(labeled) >= max(1, int(len(cues) * 0.3)) if cues else False
    if enough and unique:
        mapping = {name: f"SPEAKER_{i:02d}" for i, name in enumerate(unique)}
        default = mapping[unique[0]]
        for cue in cues:
            cue["speaker"] = mapping.get(cue.get("speaker_label"), default)
        print(f"  Speakers from subtitle labels: {mapping}")
        return cues

    if num_speakers == 1 or not hf_token:
        for cue in cues:
            cue["speaker"] = "SPEAKER_00"
        if num_speakers != 1 and not hf_token:
            print("  Warning: no speaker labels and no HF token; all cues SPEAKER_00")
        return cues

    turns = diarize_only(vocals_path, hf_token, num_speakers)
    for cue in cues:
        cue["speaker"] = _speaker_from_turns(cue, turns) or "SPEAKER_00"
    return cues


def cues_to_segments(cues, lang_hint):
    """Turn cleaned cues into pipeline segments."""
    is_english = lang_hint == "en"
    segments = []
    for cue in sorted(cues, key=lambda c: c["start"]):
        speaker = cue.get("speaker") or "SPEAKER_00"
        if is_english:
            segments.append({
                "start": cue["start"],
                "end": cue["end"],
                "text": cue["text"],
                "speaker": speaker,
            })
        else:
            segments.append({
                "start": cue["start"],
                "end": cue["end"],
                "text": "",
                "zh_text": cue["text"],
                "speaker": speaker,
            })
    return segments


def build_segments_from_subtitles(
    video_path, vocals_path, num_speakers=None, hf_token=None, external_subs=None,
):
    """Build synthesis units from cleaned subtitles without Whisper."""
    cues, sub_path, lang_hint = load_cleaned_subtitle_cues(
        video_path, external_subs=external_subs,
    )
    if not cues or not sub_path:
        return None, None
    source_desc = lang_hint
    if lang_hint == "zht":
        print("  Converting Traditional Chinese → Simplified Chinese...")
        cues = convert_traditional_to_simplified(cues)
        source_desc = "zht→zh"
    cues = assign_speakers_to_cues(
        cues, vocals_path=vocals_path, num_speakers=num_speakers, hf_token=hf_token,
    )
    text_lang = "en" if lang_hint == "en" else "zh"
    segments = cues_to_segments(cues, text_lang)
    print(f"  Subtitle-driven ({text_lang}): {len(segments)} cues, Whisper skipped")
    return segments, f"{source_desc}:{Path(sub_path).name}"


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


def _cue_speaker_and_text(cue, sorted_asr):
    """Pick the speaker with the most time-overlap with this cue, and collect
    the overlapping ASR (original-language) text for the EN subtitle output.

    sorted_asr must be sorted by 'start'. Returns (speaker_or_None, en_text).
    """
    overlap_by_spk = {}
    texts = []
    for seg in sorted_asr:
        if seg["start"] >= cue["end"]:
            break  # sorted by start: nothing after this can overlap
        ov = min(cue["end"], seg["end"]) - max(cue["start"], seg["start"])
        if ov <= 0:
            continue
        spk = seg.get("speaker", "UNKNOWN")
        overlap_by_spk[spk] = overlap_by_spk.get(spk, 0.0) + ov
        if seg.get("text", "").strip():
            texts.append((seg["start"], seg["text"].strip()))
    spk = max(overlap_by_spk, key=overlap_by_spk.get) if overlap_by_spk else None
    texts.sort(key=lambda x: x[0])
    return spk, " ".join(t for _, t in texts)


def build_subtitle_driven_segments(video_path, asr_segments, external_subs=None):
    """Use each external subtitle cue directly as a synthesis unit.

    When a complete Chinese subtitle is provided, the cue's own (start, end, text)
    becomes a segment, and the speaker is assigned from the ASR diarization by
    time overlap. This avoids grafting whole-cue text onto ASR segments — which
    duplicates text when cues are coarser than ASR segments and drops text when
    cues are finer (see match_srt_to_segments).

    Returns (segments, source_desc) with zh_text pre-filled on every segment,
    or (None, None) to fall back to the ASR + LLM translation path.
    """
    cues, sub_path, lang_hint = load_cleaned_subtitle_cues(
        video_path, external_subs=external_subs,
    )
    if not cues or not sub_path:
        return None, None

    source_desc = lang_hint
    is_english = lang_hint == "en"
    if lang_hint == "zht":
        print("  Converting Traditional Chinese → Simplified Chinese...")
        cues = convert_traditional_to_simplified(cues)
        source_desc = "zht→zh"

    # Default speaker for cues with no ASR overlap = most frequent ASR speaker.
    spk_counts = {}
    for s in asr_segments:
        spk = s.get("speaker", "SPEAKER_00")
        spk_counts[spk] = spk_counts.get(spk, 0) + 1
    default_spk = max(spk_counts, key=spk_counts.get) if spk_counts else "SPEAKER_00"

    sorted_asr = sorted(asr_segments, key=lambda s: s["start"])
    segments = []
    for cue in sorted(cues, key=lambda c: c["start"]):
        spk, asr_en = _cue_speaker_and_text(cue, sorted_asr)
        if is_english:
            segments.append({
                "start": cue["start"],
                "end": cue["end"],
                "text": cue["text"],
                "speaker": spk or default_spk,
            })
        else:
            segments.append({
                "start": cue["start"],
                "end": cue["end"],
                "text": asr_en or cue["text"],
                "zh_text": cue["text"],
                "speaker": spk or default_spk,
            })

    if is_english:
        print(f"  Subtitle-driven (EN): {len(segments)} cleaned cues will be LLM-translated "
              f"(ASR had {len(asr_segments)} segments)")
    else:
        print(f"  Subtitle-driven: {len(segments)} cues used as segments "
              f"(ASR had {len(asr_segments)} segments)")
    return segments, f"{source_desc}:{Path(sub_path).name}"


# ---------------------------------------------------------------------------
# Pronunciation glossary (IndexTTS-2.5 <字|PINYIN> / <word|CMU> tags)
# ---------------------------------------------------------------------------

_DEFAULT_PRON_PATH = Path(__file__).resolve().parent / "pronunciation.yaml"
_PRON_TAG_RE = re.compile(r"<[^<>|]+\|[^<>]+>")
_EN_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9][A-Za-z0-9.\-]*\b")
_G2P = None
_G2P_UNAVAILABLE = False


def load_pronunciation_glossary(path=None, work_dir=None):
    """Load ``term: replacement`` maps from YAML. work_dir overrides the default."""
    merged = {}
    if path:
        candidates = [path]
    else:
        candidates = [str(_DEFAULT_PRON_PATH)]
        if work_dir:
            candidates.append(os.path.join(work_dir, "pronunciation.yaml"))
    for candidate in candidates:
        if not candidate or not os.path.isfile(candidate):
            continue
        try:
            import yaml
            with open(candidate, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"  Warning: failed to load pronunciation glossary {candidate}: {e}")
            continue
        if not isinstance(data, dict):
            continue
        terms = data["terms"] if isinstance(data.get("terms"), dict) else data
        for key, value in terms.items():
            if key == "terms" and isinstance(value, dict):
                continue
            if key and isinstance(value, str):
                merged[str(key)] = value
    return merged


def _annotate_english_g2p(text):
    """Wrap leftover English words in ``<word|CMU PHONEMES>`` via g2p-en."""
    global _G2P, _G2P_UNAVAILABLE
    if _G2P_UNAVAILABLE:
        return text
    if _G2P is None:
        try:
            import nltk
            nltk.data.find("corpora/cmudict")
            from g2p_en import G2p
            _G2P = G2p()
        except Exception:
            _G2P_UNAVAILABLE = True
            return text

    def repl(m):
        word = m.group(0)
        try:
            phones = [p for p in _G2P(word) if p and p.strip() and p != " "]
        except Exception:
            return word
        if not phones:
            return word
        return f"<{word}|{' '.join(phones)}>"

    return _EN_WORD_RE.sub(repl, text)


def annotate_tts_text(text, glossary=None, use_g2p=True):
    """Apply pronunciation.yaml then optional g2p-en. Existing tags are kept."""
    if not text:
        return text
    if glossary is None:
        glossary = load_pronunciation_glossary()

    placeholders = {}

    def protect(m):
        key = f"\x00P{len(placeholders)}\x00"
        placeholders[key] = m.group(0)
        return key

    protected = _PRON_TAG_RE.sub(protect, text)
    for term in sorted(glossary, key=len, reverse=True):
        if term and term in protected:
            protected = protected.replace(term, glossary[term])
    protected = _PRON_TAG_RE.sub(protect, protected)
    if use_g2p:
        protected = _annotate_english_g2p(protected)
    for key, value in placeholders.items():
        protected = protected.replace(key, value)
    return protected


# ---------------------------------------------------------------------------
# Step 6: TTS generation with IndexTTS2
# ---------------------------------------------------------------------------

def get_audio_duration(path):
    """Get audio duration in seconds."""
    info = sf.info(path)
    return info.duration


def tts_duration_factor(zh_text, target_sec):
    """Map target slot vs natural length onto IndexTTS-2.5 duration_factor."""
    n = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", zh_text or ""))
    if n < 1 or target_sec <= 0:
        return 1.0
    natural = n / CHARS_PER_SECOND
    if natural <= 0:
        return 1.0
    factor = target_sec / natural
    return max(DURATION_FACTOR_MIN, min(DURATION_FACTOR_MAX, factor))


def generate_speech(
    segments, speaker_refs, vocals_path, work_dir, tts,
    checkpoint_cb=None, checkpoint_every=TTS_CHECKPOINT_EVERY,
):
    """Generate Chinese speech for each segment using IndexTTS2."""
    tts_dir = os.path.join(work_dir, "tts_output")
    seg_ref_dir = os.path.join(work_dir, "seg_refs")
    os.makedirs(tts_dir, exist_ok=True)
    os.makedirs(seg_ref_dir, exist_ok=True)
    glossary = load_pronunciation_glossary(work_dir=work_dir)
    synthesized = 0

    for i, seg in enumerate(segments):
        zh_text = normalize_spoken_datetime(seg.get("zh_text", "") or "")
        if zh_text:
            seg["zh_text"] = zh_text
        if not zh_text.strip():
            seg["wav_path"] = None
            seg["actual_duration"] = 0
            continue

        output_path = os.path.join(tts_dir, f"tts_{i:04d}.wav")
        target_dur = seg["end"] - seg["start"]

        # Skip if already generated (for resume)
        if os.path.exists(output_path):
            seg["wav_path"] = output_path
            seg["actual_duration"] = get_audio_duration(output_path)
        else:
            ref_audio = get_ref_for_segment(seg, speaker_refs, vocals_path, seg_ref_dir)
            tts_text = annotate_tts_text(zh_text, glossary=glossary)
            emo = seg.get("emo_vector") or estimate_emotion_from_text(zh_text)
            infer_kwargs = dict(
                spk_audio_prompt=ref_audio,
                text=tts_text,
                output_path=output_path,
                lang="zh",
                emo_audio_prompt=ref_audio,
                duration_factor=tts_duration_factor(zh_text, target_dur),
                verbose=False,
            )
            if emo:
                infer_kwargs["emo_vector"] = emo
            tts.infer(**infer_kwargs)
            seg["wav_path"] = output_path
            seg["actual_duration"] = get_audio_duration(output_path)
            synthesized += 1
            if checkpoint_cb and checkpoint_every and synthesized % checkpoint_every == 0:
                checkpoint_cb(segments)

        ratio = seg["actual_duration"] / target_dur if target_dur > 0 else 1.0
        print(f"  [{i:3d}] {seg.get('speaker', '?')} | "
              f"target={target_dur:.1f}s actual={seg['actual_duration']:.1f}s "
              f"ratio={ratio:.2f} | {zh_text[:30]}...")

    if checkpoint_cb and synthesized:
        checkpoint_cb(segments)
    return segments


# ---------------------------------------------------------------------------
# Step 7: Duration alignment (with gap absorption)
# ---------------------------------------------------------------------------

def align_durations(segments, work_dir, audio_only_align=False):
    """Align generated audio with optional video slowdown.

    When audio_only_align=False (default):
    - ratio <= AUDIO_COMFORT_LIMIT (1.2): audio-only speedup
    - ratio > AUDIO_COMFORT_LIMIT: audio caps at 1.2x, video slows down the rest

    When audio_only_align=True:
    - Audio absorbs all time difference (no video re-encoding needed)
    - Much faster assembly via -c:v copy, but speech may sound faster
    """
    import pyrubberband as pyrb

    aligned_dir = os.path.join(work_dir, "aligned")
    os.makedirs(aligned_dir, exist_ok=True)

    if audio_only_align:
        print("  Audio-only alignment mode: no video slowdown")

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
            if audio_only_align:
                audio_rate = ratio
                seg["video_slowdown"] = 1.0
            elif ratio <= AUDIO_COMFORT_LIMIT:
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
            # TTS shorter than target: slow down within comfort limit, then pad remainder
            seg["video_slowdown"] = 1.0
            slowdown_rate = max(ratio, SLOWDOWN_COMFORT_LIMIT)
            if slowdown_rate < 0.99:
                stretched = pyrb.time_stretch(audio, sr, rate=slowdown_rate)
            else:
                stretched = audio
            # If still shorter after slowdown, pad with silence
            target_samples = int(target_dur * sr)
            if len(stretched) < target_samples:
                padded = np.zeros(target_samples, dtype=stretched.dtype)
                padded[: len(stretched)] = stretched
                stretched = padded
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
    """Rebuild video with per-segment slowdowns using ffmpeg filter_complex.

    Merges consecutive normal-speed regions (gaps + segments with slowdown=1.0)
    into single trim operations to keep the filter graph small. Uses a script
    file instead of a command-line argument to avoid OS arg-length limits.
    """
    # Build a flat list of (start, end, slowdown) covering the full video.
    # Normal-speed entries use slowdown=1.0.
    raw_entries = []
    prev_end = 0.0
    for (orig_start, orig_end, _, _, slowdown) in timeline:
        if orig_start > prev_end + 0.01:
            raw_entries.append((prev_end, orig_start, 1.0))
        raw_entries.append((orig_start, orig_end, slowdown if slowdown > 1.01 else 1.0))
        prev_end = orig_end
    if prev_end < video_duration - 0.01:
        raw_entries.append((prev_end, video_duration, 1.0))

    # Merge consecutive normal-speed (1.0) entries into single trims.
    merged = []
    for entry in raw_entries:
        if (merged and entry[2] == 1.0
                and merged[-1][2] == 1.0
                and abs(entry[0] - merged[-1][1]) < 0.02):
            merged[-1] = (merged[-1][0], entry[1], 1.0)
        else:
            merged.append(entry)

    n_original = len(raw_entries)
    print(f"  Timeline: {n_original} raw entries → {len(merged)} after merging "
          f"({sum(1 for _, _, s in merged if s > 1.01)} slowdowns)")

    filter_parts = []
    concat_inputs = []
    for idx, (start, end, slowdown) in enumerate(merged):
        label = f"v{idx}"
        if slowdown > 1.01:
            filter_parts.append(
                f"[0:v]trim={start:.3f}:{end:.3f},setpts={slowdown:.4f}*(PTS-STARTPTS)[{label}]"
            )
        else:
            if end >= video_duration - 0.01:
                filter_parts.append(
                    f"[0:v]trim={start:.3f},setpts=PTS-STARTPTS[{label}]"
                )
            else:
                filter_parts.append(
                    f"[0:v]trim={start:.3f}:{end:.3f},setpts=PTS-STARTPTS[{label}]"
                )
        concat_inputs.append(f"[{label}]")

    n = len(merged)
    concat_str = "".join(concat_inputs)
    filter_parts.append(f"{concat_str}concat=n={n}:v=1:a=0[vout]")
    filter_complex = ";".join(filter_parts)

    # Write filter to a script file to avoid OS command-line length limits.
    filter_script = os.path.join(os.path.dirname(output_path) or ".", "_filter.txt")
    with open(filter_script, "w") as f:
        f.write(filter_complex)

    _run_ffmpeg(
        "-i", video_only_path,
        "-filter_complex_script", filter_script,
        "-map", "[vout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an",
        output_path,
    )
    os.remove(filter_script)
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


SRT_CHARS_PER_LINE = 16
SRT_MAX_LINES = 2
_PRON_TAG_STRIP_RE = re.compile(r"<([^<>|]+)\|[^<>]+>")
_CAPTION_PUNCT = "，。！？、；：,.!?;:"


def _strip_pron_tags(text):
    return _PRON_TAG_STRIP_RE.sub(r"\1", text or "")


def _display_len(text):
    return len(re.sub(r"\s+", "", text or ""))


def _caption_weight(text):
    n = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text or ""))
    return max(1, n)


def _layout_caption(text):
    """Fit one caption into at most two on-screen lines."""
    compact = re.sub(r"\s+", "", text or "").strip()
    if not compact:
        return ""
    if len(compact) <= SRT_CHARS_PER_LINE:
        return compact
    window = compact[:SRT_CHARS_PER_LINE]
    best = -1
    for i, ch in enumerate(window):
        if ch in _CAPTION_PUNCT:
            best = i
    if best >= max(4, SRT_CHARS_PER_LINE // 3):
        return compact[: best + 1] + "\n" + compact[best + 1 :]
    return compact[:SRT_CHARS_PER_LINE] + "\n" + compact[SRT_CHARS_PER_LINE:]


def _hard_wrap_caption(text, max_chars):
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return []
    return [
        _layout_caption(compact[i:i + max_chars])
        for i in range(0, len(compact), max_chars)
    ]


def split_zh_captions(text, max_chars=None):
    """Split spoken Chinese into short on-screen captions.

    TTS keeps the full sentence; only the exported SRT is sliced.
    Each caption is at most two lines of ``SRT_CHARS_PER_LINE`` characters.
    """
    max_chars = max_chars or (SRT_CHARS_PER_LINE * SRT_MAX_LINES)
    text = _strip_pron_tags(text).strip()
    if not text:
        return []

    units = [p.strip() for p in re.split(rf"(?<=[{re.escape(_CAPTION_PUNCT)}])", text) if p.strip()]
    if not units:
        units = [text]

    captions = []
    current = ""
    for unit in units:
        if _display_len(unit) > max_chars:
            if current:
                captions.append(_layout_caption(current))
                current = ""
            captions.extend(_hard_wrap_caption(unit, max_chars))
            continue
        trial = current + unit
        if current and _display_len(trial) > max_chars:
            captions.append(_layout_caption(current))
            current = unit
        else:
            current = trial
    if current:
        captions.append(_layout_caption(current))
    return [c for c in captions if c]


def _caption_spans(captions, start, end):
    """Spread captions across [start, end] by readable-character weight."""
    if not captions:
        return []
    weights = [_caption_weight(c) for c in captions]
    total_w = sum(weights) or 1
    dur = max(0.0, float(end) - float(start))
    t = float(start)
    spans = []
    for i, (cap, weight) in enumerate(zip(captions, weights)):
        if i == len(captions) - 1:
            spans.append((t, float(end), cap))
        else:
            piece = dur * (weight / total_w)
            spans.append((t, t + piece, cap))
            t += piece
    return spans


def generate_srt(segments, output_path, lang="zh"):
    """Generate SRT subtitle file aligned to actual dubbed audio timing.

    Uses new_start (shifted timeline) and aligned_duration (post-stretch).
    Chinese cues are split so each screen shows at most two short lines.
    """
    text_key = "zh_text" if lang == "zh" else "text"
    with open(output_path, "w", encoding="utf-8") as f:
        idx = 0
        for seg in segments:
            text = seg.get(text_key, "") or ""
            if not text.strip():
                continue
            srt_start = seg.get("new_start", seg["start"])
            aligned_dur = seg.get("aligned_duration")
            if aligned_dur is not None:
                srt_end = srt_start + aligned_dur
            else:
                srt_end = srt_start + (seg["end"] - seg["start"])
            if lang == "zh":
                pieces = _caption_spans(split_zh_captions(text), srt_start, srt_end)
            else:
                pieces = [(srt_start, srt_end, text.strip())]
            for a, b, cap in pieces:
                if not cap:
                    continue
                idx += 1
                f.write(
                    f"{idx}\n{_format_srt_time(a)} --> {_format_srt_time(b)}\n{cap}\n\n"
                )
    print(f"  Saved SRT: {output_path}")


# ---------------------------------------------------------------------------
# Patch / re-dub specific sentences (no full pipeline rerun)
# ---------------------------------------------------------------------------

def _video_work_dir(work_dir, video_path):
    return os.path.join(work_dir, Path(video_path).stem)


def _load_work_state(work_dir, video_path):
    video_work_dir = _video_work_dir(work_dir, video_path)
    ckpt = _load_checkpoint(video_work_dir)
    if not ckpt or not ckpt.get("segments"):
        raise FileNotFoundError(
            f"No usable checkpoint in {video_work_dir}. Run dub_video first."
        )
    return video_work_dir, ckpt


def list_dub_segments(video_path, work_dir="dub_workspace"):
    """Print [id] speaker start-end zh_text from a previous dub_video run."""
    _video_work_dir_path, ckpt = _load_work_state(work_dir, video_path)
    segments = ckpt["segments"]
    print(f"  {len(segments)} segments in {_video_work_dir_path}")
    rows = []
    for i, seg in enumerate(segments):
        zh = (seg.get("zh_text") or "").replace("\n", " ")
        print(
            f"[{i:3d}] {str(seg.get('speaker', '?')):12} "
            f"{seg['start']:7.1f}-{seg['end']:7.1f}  {zh}"
        )
        rows.append({"id": i, "speaker": seg.get("speaker"), "start": seg["start"],
                     "end": seg["end"], "zh_text": seg.get("zh_text", ""),
                     "text": seg.get("text", "")})
    return rows


_SUSPICIOUS_MONTH_RE = re.compile(rf"(?i)\b({_MONTH_ALT})\b")
_SUSPICIOUS_AMPM_RE = re.compile(rf"(?i)\b{_MERIDIEM}\b")
_SUSPICIOUS_CLOCK_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
_SUSPICIOUS_WEEKDAY_RE = re.compile(
    rf"(?i)\b({_WEEKDAY_FULL_ALT}|weekend)\b"
)
_SUSPICIOUS_MONEY_RE = re.compile(
    r"[$€£¥￥]|\b(?:USD|EUR|GBP|RMB|CNY)\b|\b(?:dollars?|euros?|pounds?|yuan)\b",
    re.I,
)


def _ascii_letter_ratio(text):
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    letters = sum(1 for c in chars if ("A" <= c <= "Z") or ("a" <= c <= "z"))
    return letters / len(chars)


def find_suspicious_segments(segments):
    """Return segments that likely need redub (empty zh, leftover EN, pace)."""
    rows = []
    for i, seg in enumerate(segments):
        zh = (seg.get("zh_text") or "").strip()
        en = (seg.get("text") or "").strip()
        reasons = []
        if not zh:
            if en:
                reasons.append("empty_zh")
        else:
            if zh == en and re.search(r"[A-Za-z]{3,}", zh):
                reasons.append("leftover_english")
            elif _ascii_letter_ratio(zh) >= 0.35 and len(re.findall(r"[A-Za-z]", zh)) >= 4:
                reasons.append("leftover_english")
            if _SUSPICIOUS_MONTH_RE.search(zh) or _SUSPICIOUS_AMPM_RE.search(zh) or _SUSPICIOUS_CLOCK_RE.search(zh):
                reasons.append("leftover_datetime")
            if _SUSPICIOUS_MONEY_RE.search(zh):
                reasons.append("leftover_money")
            if _SUSPICIOUS_WEEKDAY_RE.search(zh):
                reasons.append("leftover_weekday")
            duration = float(seg.get("end", 0) or 0) - float(seg.get("start", 0) or 0)
            n_chars = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", zh))
            if duration >= 1.0 and n_chars:
                cps = n_chars / duration
                if cps > CHARS_PER_SECOND * 1.6:
                    reasons.append("too_fast")
                elif cps < CHARS_PER_SECOND * 0.35 and n_chars >= 2:
                    reasons.append("too_slow")
        if not reasons:
            continue
        rows.append({
            "id": i,
            "speaker": seg.get("speaker"),
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": en,
            "zh_text": zh,
            "reasons": reasons,
        })
    return rows


def list_suspicious_segments(video_path, work_dir="dub_workspace"):
    """Print checkpoint sentences that likely need a redub pass."""
    _video_work_dir_path, ckpt = _load_work_state(work_dir, video_path)
    rows = find_suspicious_segments(ckpt["segments"])
    print(f"  {len(rows)} suspicious / {len(ckpt['segments'])} segments in {_video_work_dir_path}")
    for row in rows:
        zh = (row.get("zh_text") or "").replace("\n", " ")
        print(
            f"[{row['id']:3d}] {','.join(row['reasons']):28} "
            f"{row['start']:7.1f}-{row['end']:7.1f}  {zh}"
        )
    return rows


def parse_redub_file(path):
    """Parse `id<TAB>zh_text` lines into {id: zh_text}."""
    patches = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                idx_s, text = line.split("\t", 1)
            else:
                idx_s, text = line.split(None, 1)
            patches[int(idx_s)] = text
    return patches


def redub_segments(
    video_path,
    ids,
    texts=None,
    output_path=None,
    work_dir="dub_workspace",
    model_dir="checkpoints",
    use_fp16=True,
    tts=None,
    audio_only_align=True,
):
    """Re-TTS selected sentence ids and remux, using checkpoint artifacts.

    ``texts`` is an optional ``{id: new_zh_text}`` map (supports
    ``<行|HANG2>`` / ``<word|CMU PHONEMES>``). Omitted ids keep existing zh_text.
    """
    video_path = str(video_path)
    video_work_dir, ckpt = _load_work_state(work_dir, video_path)
    segments = ckpt["segments"]
    paths = ckpt.get("paths") or {}

    ids = sorted({int(i) for i in ids})
    for i in ids:
        if i < 0 or i >= len(segments):
            raise IndexError(f"segment id {i} out of range 0..{len(segments) - 1}")

    texts = texts or {}
    tts_dir = os.path.join(video_work_dir, "tts_output")
    aligned_dir = os.path.join(video_work_dir, "aligned")
    for i in ids:
        if i in texts:
            segments[i]["zh_text"] = texts[i]
        for folder, prefix in ((tts_dir, "tts"), (aligned_dir, "aligned")):
            wav = os.path.join(folder, f"{prefix}_{i:04d}.wav")
            if os.path.isfile(wav):
                os.remove(wav)
        segments[i].pop("wav_path", None)
        segments[i].pop("aligned_path", None)

    vocals_path = paths.get("vocals_path")
    bg_path = paths.get("bg_path")
    video_only_path = paths.get("video_only_path")
    if not vocals_path or not os.path.isfile(vocals_path):
        raise FileNotFoundError("vocals track missing; cannot pick speaker reference")
    if not bg_path or not video_only_path:
        raise FileNotFoundError("cached video/background paths missing from checkpoint")

    speaker_refs = paths.get("speaker_refs") or {}
    if not speaker_refs:
        speaker_refs = extract_speaker_refs(segments, vocals_path, video_work_dir)
        paths["speaker_refs"] = speaker_refs
    else:
        for info in speaker_refs.values():
            if info.get("embedding") is None and os.path.exists(info.get("best_auto", "")):
                info["embedding"] = _compute_speaker_embedding(info["best_auto"])

    if tts is None:
        tts = _init_tts(model_dir, use_fp16)
    else:
        _reload_tts(tts)

    print(f"\n[redub] Re-synthesizing {len(ids)} segment(s): {ids}")
    segments = generate_speech(segments, speaker_refs, vocals_path, video_work_dir, tts)
    _save_checkpoint(video_work_dir, 7, segments=segments, paths=paths)

    print("[redub] Aligning durations...")
    segments = align_durations(segments, video_work_dir, audio_only_align=audio_only_align)
    _save_checkpoint(video_work_dir, 8, segments=segments, paths=paths)

    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(video_path).parent / f"{stem}_cn.mp4")

    print("[redub] Assembling video...")
    assemble_final(segments, bg_path, video_only_path, output_path, video_work_dir)
    _save_checkpoint(video_work_dir, 10, segments=segments, paths=paths)

    srt_base = output_path.rsplit(".", 1)[0]
    generate_srt(segments, f"{srt_base}.srt", lang="zh")
    generate_srt(segments, f"{srt_base}_en.srt", lang="en")

    translations_path = os.path.join(video_work_dir, "translations.json")
    with open(translations_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"id": i, "speaker": s.get("speaker"), "start": s["start"], "end": s["end"],
              "en": s.get("text", ""), "zh": s.get("zh_text", "")}
             for i, s in enumerate(segments)],
            f, ensure_ascii=False, indent=2,
        )

    print(f"\nRedub done! Output: {output_path}")
    return output_path


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
    audio_only_align=False,
    whisper_model="large-v2",
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
        audio_only_align: Audio absorbs all time difference (no video slowdown).
            Much faster assembly (uses -c:v copy), but speech may be faster.
        whisper_model: WhisperX model id (e.g. large-v2, large-v3-turbo).
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

    # Free GPU for demucs/whisperx steps (TTS not needed until Step 6)
    if done < 7:
        _offload_tts(tts)

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
        skip_asr = should_skip_asr(
            num_speakers, video_path,
            no_external_subs=no_external_subs, external_subs=external_subs,
        )
        if skip_asr:
            print("\n[Step 3/11] Skipping Whisper (cleaned subtitles)...")
            segments, sub_source = build_segments_from_subtitles(
                video_path, vocals_path,
                num_speakers=num_speakers, hf_token=hf_token,
                external_subs=external_subs,
            )
            if segments:
                paths["skipped_asr"] = True
                paths["subtitle_driven"] = True
                paths["subtitle_source"] = sub_source
            else:
                print("  Subtitle path empty, falling back to Whisper")
                segments = transcribe_and_diarize(
                    vocals_path, hf_token, num_speakers, whisper_model=whisper_model,
                )
        else:
            print("\n[Step 3/11] Transcribing and diarizing...")
            segments = transcribe_and_diarize(
                vocals_path, hf_token, num_speakers, whisper_model=whisper_model,
            )
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
        # Recompute embeddings when resuming from checkpoint
        for spk, info in speaker_refs.items():
            if info.get("embedding") is None and os.path.exists(info.get("best_auto", "")):
                info["embedding"] = _compute_speaker_embedding(info["best_auto"])
        print(f"\n[Step 4/11] Skipped (cached)")

    # --- Step 5: Translation (with external subtitle support) ---
    if done < 6:
        print("\n[Step 5/11] Translating to Chinese...")

        # Subtitle-driven mode: when a complete external subtitle is provided,
        # use each cue directly as a synthesis unit (cue timing + text + speaker
        # from ASR overlap) instead of grafting cue text onto ASR segments.
        sub_source = paths.get("subtitle_source") if paths.get("subtitle_driven") else None
        if paths.get("subtitle_driven") and segments:
            print(f"  Using subtitle-driven segments from step 3 ({sub_source})")
        elif not no_external_subs:
            sub_segments, sub_source = build_subtitle_driven_segments(
                video_path, segments, external_subs=external_subs
            )
            if sub_segments is not None:
                segments = absorb_gaps(sub_segments)  # rebuild end_padded for cue units
                print(f"  Using subtitle-driven segments ({sub_source})")

        untranslated = sum(1 for s in segments if not s.get("zh_text") and s.get("text", "").strip())

        if sub_source and untranslated == 0:
            print(f"  All segments from external subtitles ({sub_source})")
        else:
            if sub_source:
                print(f"  {len(segments) - untranslated} from external subs, {untranslated} need LLM")
            llm_client = LLMClient(api_key=llm_api_key, api_base=llm_api_base, model=llm_model)
            segments, _ = translate_segments(segments, llm_client, skip_translated=True)
        translations_path = os.path.join(video_work_dir, "translations.json")
        with open(translations_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"id": i, "speaker": s.get("speaker"), "start": s["start"], "end": s["end"],
                  "en": s["text"], "zh": s.get("zh_text", "")}
                 for i, s in enumerate(segments)],
                f, ensure_ascii=False, indent=2,
            )
        print(f"  Saved translations: {translations_path}")
        failed = untranslated_segment_ids(segments)
        if failed:
            _save_checkpoint(video_work_dir, 5, segments=segments, paths=paths)
            preview = failed[:20]
            more = f" (+{len(failed) - 20} more)" if len(failed) > 20 else ""
            raise RuntimeError(
                f"Translation failed for {len(failed)} segment(s) {preview}{more}. "
                "Not falling back to English. Re-run to retry; already-translated "
                "sentences are kept."
            )
        _save_checkpoint(video_work_dir, 6, segments=segments, paths=paths)
    else:
        print(f"\n[Step 5/11] Skipped (cached)")

    # --- Step 6: TTS generation ---
    if done < 7:
        print("\n[Step 6/11] Generating Chinese speech with IndexTTS2...")
        _reload_tts(tts)
        if tts is None:
            tts = _init_tts(model_dir, use_fp16)

        def _tts_ckpt(segs):
            _save_checkpoint(video_work_dir, 6, segments=segs, paths=paths)

        segments = generate_speech(
            segments, speaker_refs, vocals_path, video_work_dir, tts,
            checkpoint_cb=_tts_ckpt,
        )
        _save_checkpoint(video_work_dir, 7, segments=segments, paths=paths)
    else:
        print(f"\n[Step 6/11] Skipped (cached)")

    # --- Step 7: Duration alignment (with 50/50 video slowdown) ---
    if done < 8:
        print("\n[Step 7/11] Aligning durations (audio + video)...")
        segments = align_durations(segments, video_work_dir, audio_only_align=audio_only_align)
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

    # Keep a finished checkpoint so list/redub still work after success.
    _save_checkpoint(
        video_work_dir, 10, segments=segments, paths=paths,
        timeline=timeline, new_total=new_total,
    )

    # Cleanup if requested
    if cleanup:
        _clear_checkpoint(video_work_dir)
        import shutil
        shutil.rmtree(video_work_dir)
        print(f"  Cleaned up: {video_work_dir}")

    print(f"\nDone! Output: {output_path}")
    return output_path


def dub_batch(
    input_dir,
    output_dir=None,
    video_extensions=(".mp4", ".mkv", ".mov", ".avi"),
    recursive=False,
    **kwargs,
):
    """
    Batch process all videos in a directory.

    Args:
        input_dir: Input directory containing video files.
        output_dir: Output directory. Defaults to {input_dir}_cn/.
        video_extensions: Tuple of video file extensions to process.
        recursive: Recurse into subdirectories, preserving their structure in
            the output (and per-video work dir) so same-named videos in
            different folders don't collide.
        **kwargs: All other arguments passed to dub_video().
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir.parent / f"{input_dir.name}_cn"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_resolved = output_dir.resolve()
    base_work_dir = kwargs.pop("work_dir", "dub_workspace")

    if recursive:
        candidates = sorted(input_dir.rglob("*"))
    else:
        candidates = sorted(input_dir.iterdir())

    videos = []
    for f in candidates:
        if f.suffix.lower() not in video_extensions:
            continue
        # Skip our own outputs (when output_dir lives inside input_dir)
        if f.stem.endswith("_cn") or out_resolved in f.resolve().parents:
            continue
        videos.append(f)

    if not videos:
        where = "recursively in" if recursive else "in"
        print(f"No video files found {where} {input_dir}")
        return

    print(f"Found {len(videos)} videos to process{' (recursive)' if recursive else ''}")
    print(f"Output directory: {output_dir}")

    tts = _init_tts(kwargs.get("model_dir", "checkpoints"), kwargs.get("use_fp16", True))

    results = []
    for i, video in enumerate(videos):
        # Relative path (without suffix) preserves subdir structure in outputs
        rel = video.relative_to(input_dir).with_suffix("")
        print(f"\n{'#' * 60}")
        print(f"[{i + 1}/{len(videos)}] {rel}")
        print(f"{'#' * 60}")

        out_path = output_dir / f"{rel}_cn.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path = str(out_path)
        # Mirror subdir structure in the work dir to avoid same-stem collisions
        video_work_dir = os.path.join(base_work_dir, *rel.parent.parts) if rel.parent.parts else base_work_dir
        try:
            dub_video(str(video), out_path, tts=tts, work_dir=video_work_dir, **kwargs)
            results.append((str(rel), "OK", out_path))
        except Exception as e:
            print(f"FAILED: {rel} — {e}")
            results.append((str(rel), "FAILED", str(e)))
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
    parser.add_argument("--recursive", action="store_true", help="Batch mode: recurse into subdirectories (preserves folder structure in output)")
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
    parser.add_argument("--audio-only-align", action="store_true",
                        help="Audio absorbs all time difference (no video slowdown/re-encoding, much faster)")
    parser.add_argument(
        "--whisper-model", default="large-v2",
        help="WhisperX model name (large-v2, large-v3-turbo, distil-large-v3)",
    )
    parser.add_argument(
        "--list-segments", action="store_true",
        help="List checkpoint sentence ids (after a previous dub) and exit",
    )
    parser.add_argument(
        "--list-suspicious", action="store_true",
        help="List checkpoint sentences that likely need redub and exit",
    )
    parser.add_argument(
        "--redub", default=None,
        help="Comma-separated sentence ids to re-TTS and remux (e.g. 12,15)",
    )
    parser.add_argument(
        "--redub-text", default=None,
        help="Replacement zh_text for a single --redub id (supports <行|HANG2>)",
    )
    parser.add_argument(
        "--redub-file", default=None,
        help="Patch file with lines: id<TAB>zh_text",
    )

    args = parser.parse_args()

    if args.list_segments:
        list_dub_segments(args.input, work_dir=args.work_dir)
        return

    if args.list_suspicious:
        list_suspicious_segments(args.input, work_dir=args.work_dir)
        return

    if args.redub or args.redub_file:
        texts = parse_redub_file(args.redub_file) if args.redub_file else {}
        if args.redub:
            ids = [int(x) for x in args.redub.split(",") if x.strip() != ""]
        else:
            ids = list(texts)
        if args.redub_text is not None:
            if len(ids) != 1:
                print("Error: --redub-text requires exactly one --redub id")
                sys.exit(1)
            texts[ids[0]] = args.redub_text
        use_fp16 = args.fp16 and not args.no_fp16
        redub_segments(
            args.input,
            ids,
            texts=texts or None,
            output_path=args.output,
            work_dir=args.work_dir,
            model_dir=args.model_dir,
            use_fp16=use_fp16,
            audio_only_align=True,
        )
        return

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    llm_api_key = args.llm_api_key or os.environ.get("LLM_API_KEY")
    use_fp16 = args.fp16 and not args.no_fp16

    needs_diarization = args.num_speakers is None or args.num_speakers != 1
    if needs_diarization and not hf_token:
        print("Error: HuggingFace token required for diarization. Use --hf-token or set HF_TOKEN, or pass --num-speakers 1.")
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
        audio_only_align=args.audio_only_align,
        whisper_model=args.whisper_model,
    )

    if args.batch:
        dub_batch(args.input, args.output, recursive=args.recursive, **common_kwargs)
    else:
        dub_video(args.input, args.output, **common_kwargs)


if __name__ == "__main__":
    main()
