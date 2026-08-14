"""
Novel audiobook pipeline: novel text → multi-speaker chapter WAVs via IndexTTS-2.5.

Task 1 provides text ingest, chapter splitting, and silent WAV concatenation.
Later steps add dialogue split, LLM analysis, voice cards, and TTS.
"""

from __future__ import annotations

import re
import wave
from pathlib import Path

# Chinese chapter headings: 第N章 / 第N节 / 第N回 / 第N卷
CHAPTER_RE = re.compile(
    r"(?m)^(第[零一二三四五六七八九十百千万0-9]+[章节回卷][^\n]*)"
)
# English: Chapter 1 / Chapter 12 Title
_CHAPTER_RE_EN = re.compile(r"(?m)^(Chapter\s+\d+[^\n]*)", re.IGNORECASE)
# Short numbered headings: "1 Title" / "1.2 Title" with line length < 40
_CHAPTER_RE_NUM = re.compile(r"(?m)^(\d+(?:\.\d+)*\s+\S[^\n]*)$")


def ingest_text(raw: str) -> str:
    """Normalize novel text: strip UTF-8 BOM and unify newlines to \\n."""
    if raw.startswith("\ufeff"):
        raw = raw[1:]
    return raw.replace("\r\n", "\n").replace("\r", "\n")


def split_chapters(text: str, source_name: str) -> list[dict]:
    """
    Split text into chapters by heading rules.

    Returns list of dicts: {id, index, title, start_char, end_char}.
    id is c{index:02d}. No match → single chapter titled from source_name.
    """
    matches = list(CHAPTER_RE.finditer(text))
    if not matches:
        matches = list(_CHAPTER_RE_EN.finditer(text))
    if not matches:
        matches = [
            m
            for m in _CHAPTER_RE_NUM.finditer(text)
            if len(m.group(0)) < 40
        ]

    if not matches:
        title = Path(source_name).stem if source_name else "chapter"
        return [
            {
                "id": "c01",
                "index": 1,
                "title": title,
                "start_char": 0,
                "end_char": len(text),
            }
        ]

    chapters: list[dict] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        index = i + 1
        chapters.append(
            {
                "id": f"c{index:02d}",
                "index": index,
                "title": match.group(1).strip(),
                "start_char": start,
                "end_char": end,
            }
        )
    return chapters


def _concat_wavs(segments: list[dict], output_path: str) -> str:
    """
    Concatenate WAV segments with optional trailing silence.

    Each segment dict needs:
      - audio_path or wav_path: path to a 22050 Hz mono 16-bit WAV
      - silence_after_ms: non-negative silence after this segment (default 0)

    Raises ValueError if formats differ or are not 22050/mono/16-bit.
    Returns output_path.
    """
    if not segments:
        raise ValueError("segments must not be empty")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _path(seg: dict) -> str:
        p = seg.get("audio_path") or seg.get("wav_path")
        if not p:
            raise ValueError("segment missing audio_path/wav_path")
        return str(p)

    first_path = _path(segments[0])
    with wave.open(first_path, "rb") as first:
        channels = first.getnchannels()
        sample_width = first.getsampwidth()
        frame_rate = first.getframerate()

    if channels != 1 or sample_width != 2 or frame_rate != 22050:
        raise ValueError(
            f"expected 22050 Hz mono 16-bit WAV, got "
            f"rate={frame_rate} channels={channels} width={sample_width}"
        )

    with wave.open(str(out), "wb") as output_wav:
        output_wav.setnchannels(channels)
        output_wav.setsampwidth(sample_width)
        output_wav.setframerate(frame_rate)

        for segment in segments:
            path = _path(segment)
            with wave.open(path, "rb") as input_wav:
                if (
                    input_wav.getnchannels() != channels
                    or input_wav.getsampwidth() != sample_width
                    or input_wav.getframerate() != frame_rate
                ):
                    raise ValueError(
                        f"format mismatch in {path}: "
                        f"rate={input_wav.getframerate()} "
                        f"channels={input_wav.getnchannels()} "
                        f"width={input_wav.getsampwidth()}"
                    )
                output_wav.writeframes(input_wav.readframes(input_wav.getnframes()))

            silence_ms = int(segment.get("silence_after_ms") or 0)
            if silence_ms < 0:
                raise ValueError("silence_after_ms must be non-negative")
            silence_frames = frame_rate * silence_ms // 1000
            if silence_frames:
                output_wav.writeframes(
                    b"\0" * channels * sample_width * silence_frames
                )

    return str(out)
