"""
Novel audiobook pipeline: novel text → multi-speaker chapter WAVs via IndexTTS-2.5.

Task 1 provides text ingest, chapter splitting, and silent WAV concatenation.
Task 2 adds dialogue/narration split and IndexTTS 2.5 sentence-length limits.
Later steps add LLM analysis, voice cards, and TTS.
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

# IndexTTS pronunciation tags: <字|PINYIN>
_PRON_TAG_RE = re.compile(r"<[^|>]+\|[^>]+>")

# Quote pairs for dialogue detection (open → close)
_QUOTE_PAIRS = {
    "「": "」",
    "『": "』",
    "\u201c": "\u201d",  # “ ”
    "\u2018": "\u2019",  # ‘ ’
    '"': '"',
}
_QUOTE_CLOSE_TO_OPEN = {v: k for k, v in _QUOTE_PAIRS.items() if k != v}
_QUOTE_CHARS = set(_QUOTE_PAIRS) | set(_QUOTE_PAIRS.values())

# Preferred split points when enforcing TTS char limits
_SPLIT_PUNCT = set("。！？；….!?;\n，、,")


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


def _strip_wrapping_quotes(s: str) -> str:
    """Remove a single layer of matching wrapping quotes from dialogue text."""
    s = s.strip()
    if len(s) < 2:
        return s
    first, last = s[0], s[-1]
    if first in _QUOTE_PAIRS and _QUOTE_PAIRS[first] == last:
        return s[1:-1].strip()
    return s


def split_utterances(chapter_text: str, chapter_id: str) -> list[dict]:
    """
    Split chapter text into narration / dialogue utterances.

    Quote pairing uses a stack; supports 「」『』“”‘’ and ASCII \".
    Dialogue tts_text strips wrapping quotes; text keeps original including quotes.
    Each utterance: id, chapter_id, seq, kind, text, tts_text, lang.
    """
    raw_segments: list[tuple[str, str]] = []  # (kind, text)
    stack: list[str] = []  # expected closers
    buf: list[str] = []
    in_dialogue = False

    def flush(kind: str) -> None:
        text = "".join(buf)
        buf.clear()
        if text.strip():
            raw_segments.append((kind, text))

    i = 0
    n = len(chapter_text)
    while i < n:
        ch = chapter_text[i]

        if ch in _QUOTE_CHARS:
            # Prefer closing if stack expects this char
            if stack and ch == stack[-1]:
                buf.append(ch)
                stack.pop()
                if not stack:
                    flush("dialogue")
                    in_dialogue = False
                i += 1
                continue

            if ch in _QUOTE_PAIRS:
                # Opening quote
                if not in_dialogue:
                    flush("narration")
                    in_dialogue = True
                buf.append(ch)
                stack.append(_QUOTE_PAIRS[ch])
                i += 1
                continue

            if ch in _QUOTE_CLOSE_TO_OPEN:
                # Stray closer — treat as plain text
                buf.append(ch)
                i += 1
                continue

        buf.append(ch)
        i += 1

    if buf:
        flush("dialogue" if in_dialogue else "narration")

    utterances: list[dict] = []
    seq = 0
    for kind, text in raw_segments:
        seq += 1
        tts_text = _strip_wrapping_quotes(text) if kind == "dialogue" else text.strip()
        # Keep original text for dialogue (incl. quotes); strip edges for narration
        stored_text = text if kind == "dialogue" else text.strip()
        utterances.append(
            {
                "id": f"{chapter_id}_{seq:04d}",
                "chapter_id": chapter_id,
                "seq": seq,
                "kind": kind,
                "text": stored_text,
                "tts_text": tts_text,
                "lang": "zh",
            }
        )
    return utterances


def enforce_tts_limits(text: str, max_chars: int = 80) -> list[str]:
    """
    Split text to respect IndexTTS ~2.5 sentence length limits.

    Pronunciation tags ``<字|PINYIN>`` are replaced with placeholders before
    splitting by punctuation / char count, then restored so tags stay intact.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not text:
        return []

    placeholders: list[str] = []

    def _protect(m: re.Match[str]) -> str:
        placeholders.append(m.group(0))
        return f"\0P{len(placeholders) - 1}\0"

    protected = _PRON_TAG_RE.sub(_protect, text)

    def _restore(s: str) -> str:
        for idx, tag in enumerate(placeholders):
            s = s.replace(f"\0P{idx}\0", tag)
        return s

    parts: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush_buf() -> None:
        nonlocal buf_len
        if not buf:
            return
        chunk = "".join(buf).strip()
        buf.clear()
        buf_len = 0
        if chunk:
            parts.append(_restore(chunk))

    for ch in protected:
        buf.append(ch)
        # Placeholders count as 1 logical unit for length? Use raw char len of
        # protected string; placeholders are short (\0Pn\0) so tags won't split.
        buf_len += 1
        if ch in _SPLIT_PUNCT and buf_len >= 1:
            # Split after punctuation when buffer is getting long, or always
            # on strong sentence-end punctuation when over limit.
            if buf_len >= max_chars or ch in "。！？….!?\n":
                flush_buf()
        elif buf_len >= max_chars:
            # Hard split: prefer last punctuation inside buffer
            joined = "".join(buf)
            split_at = -1
            for j in range(len(joined) - 1, 0, -1):
                if joined[j] in _SPLIT_PUNCT:
                    split_at = j + 1
                    break
            if split_at > 0:
                head = joined[:split_at].strip()
                tail = list(joined[split_at:])
                if head:
                    parts.append(_restore(head))
                buf = tail
                buf_len = len(buf)
            else:
                # No punctuation: split at max_chars but not inside placeholder
                cut = max_chars
                # Avoid cutting mid-placeholder \0Pn\0
                ph_start = joined.rfind("\0", 0, cut)
                if ph_start >= 0:
                    ph_end = joined.find("\0", ph_start + 1)
                    if ph_end >= 0 and ph_end >= cut - 1:
                        # cut lands inside or just after placeholder start
                        if ph_start > 0:
                            cut = ph_start
                        else:
                            cut = ph_end + 1
                head = joined[:cut].strip()
                tail = list(joined[cut:])
                if head:
                    parts.append(_restore(head))
                buf = tail
                buf_len = len(buf)

    flush_buf()
    return parts if parts else [_restore(protected.strip())] if protected.strip() else []


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
