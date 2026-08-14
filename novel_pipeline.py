"""
Novel audiobook pipeline: novel text → multi-speaker chapter WAVs via IndexTTS-2.5.

Task 1 provides text ingest, chapter splitting, and silent WAV concatenation.
Task 2 adds dialogue/narration split and IndexTTS 2.5 sentence-length limits.
Task 3 adds LLM style/character analysis and validation helpers.
Task 4 adds seed voice bank matching and IndexTTS-2.5 character voice cards.
Task 5 builds per-chapter reading scripts (speaker, emotion, glossary, silence).
Task 6 synthesizes lines in order and merges chapter WAVs.
Task 7 orchestrates steps with checkpoint resume and a CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import wave
from pathlib import Path
from typing import Any

import yaml

from dub_pipeline import (
    annotate_tts_text,
    estimate_emotion_from_text,
    load_pronunciation_glossary,
)
from highlight_pipeline import (
    LLMClient,
    _clear_checkpoint,
    _free_vram,
    _init_tts,
    _load_checkpoint,
    _parse_json_response,
    _save_checkpoint,
)

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


# ---------------------------------------------------------------------------
# Task 3: style / character LLM analysis and validation
# ---------------------------------------------------------------------------

_VALID_LANGS = frozenset({"zh", "en", "ja", "es", "ar"})
_STYLE_DURATION_MIN = 0.8
_STYLE_DURATION_MAX = 1.3
_EMO_DIMS = 8
_DEFAULT_NARRATOR = {
    "id": "narrator",
    "name": "旁白",
    "aliases": [],
    "role": "narrator",
    "gender": "male",
    "age": "middle",
    "personality": "沉稳",
    "voice_traits": "低沉",
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _normalize_base_emo(raw: Any) -> list[float]:
    """Pad/truncate to 8 floats clamped to [0, 1]."""
    if raw is None:
        values: list[Any] = []
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        values = []
    out: list[float] = []
    for i in range(_EMO_DIMS):
        if i < len(values):
            try:
                v = float(values[i])
            except (TypeError, ValueError):
                v = 0.0
            out.append(_clamp(v, 0.0, 1.0))
        else:
            out.append(0.0)
    return out


def validate_style(data: dict) -> dict:
    """
    Normalize and clamp a style analysis dict.

    lang → lowercase in {zh,en,ja,es,ar} (default zh);
    duration_factor → clamp 0.8–1.3;
    base_emo → 8 floats in [0,1].
    """
    if not isinstance(data, dict):
        data = {}
    out = dict(data)

    lang = str(out.get("lang") or "zh").strip().lower()
    # Accept prefixes like "zh-CN"
    if lang not in _VALID_LANGS:
        for code in _VALID_LANGS:
            if lang.startswith(code):
                lang = code
                break
        else:
            lang = "zh"
    out["lang"] = lang

    try:
        df = float(out.get("duration_factor", 1.0))
    except (TypeError, ValueError):
        df = 1.0
    out["duration_factor"] = _clamp(df, _STYLE_DURATION_MIN, _STYLE_DURATION_MAX)
    out["base_emo"] = _normalize_base_emo(out.get("base_emo"))

    for key in (
        "title",
        "genre",
        "narrative_pov",
        "era",
        "tone",
        "pacing",
        "narrator_style",
    ):
        if key not in out or out[key] is None:
            out[key] = ""
        else:
            out[key] = str(out[key])
    return out


def validate_character(data: dict) -> dict:
    """
    Normalize a single character record.

    Ensures id/name/aliases/role/gender/age/personality/voice_traits and
    optional duration_factor / base_emo when present.
    """
    if not isinstance(data, dict):
        data = {}
    out = dict(data)

    name = str(out.get("name") or "").strip() or "unknown"
    out["name"] = name

    cid = str(out.get("id") or "").strip()
    if not cid:
        # slug from name: keep alnum / underscore; Chinese kept as-is for readability
        slug = re.sub(r"\s+", "_", name)
        slug = re.sub(r"[^\w\u4e00-\u9fff-]", "", slug) or "char"
        cid = slug
    out["id"] = cid

    aliases = out.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    cleaned_aliases: list[str] = []
    seen_a: set[str] = set()
    for a in aliases:
        a_s = str(a).strip()
        if a_s and a_s != name and a_s not in seen_a:
            seen_a.add(a_s)
            cleaned_aliases.append(a_s)
    out["aliases"] = cleaned_aliases

    role = str(out.get("role") or "dialogue").strip().lower()
    if role not in {"narrator", "dialogue", "crowd"}:
        role = "dialogue"
    if cid == "narrator" or name in {"旁白", "narrator", "Narrator"}:
        role = "narrator"
        out["id"] = "narrator"
        if not name or name == "unknown":
            out["name"] = "旁白"
    out["role"] = role

    out["gender"] = str(out.get("gender") or "unknown").strip().lower() or "unknown"
    out["age"] = str(out.get("age") or "unknown").strip().lower() or "unknown"
    out["personality"] = str(out.get("personality") or "").strip()
    out["voice_traits"] = str(out.get("voice_traits") or "").strip()

    if "duration_factor" in out and out["duration_factor"] is not None:
        try:
            out["duration_factor"] = _clamp(float(out["duration_factor"]), 0.5, 2.0)
        except (TypeError, ValueError):
            out.pop("duration_factor", None)

    if "base_emo" in out and out["base_emo"] is not None:
        out["base_emo"] = _normalize_base_emo(out["base_emo"])

    return out


def _character_name_keys(char: dict) -> set[str]:
    keys = set()
    name = str(char.get("name") or "").strip()
    if name:
        keys.add(name)
    for a in char.get("aliases") or []:
        a_s = str(a).strip()
        if a_s:
            keys.add(a_s)
    return keys


def _merge_two_characters(a: dict, b: dict) -> dict:
    """Merge b into a (prefer a for primary fields; union aliases)."""
    merged = dict(a)
    # Prefer longer / more informative personality and voice_traits
    for field in ("personality", "voice_traits"):
        av = str(a.get(field) or "")
        bv = str(b.get(field) or "")
        if len(bv) > len(av):
            merged[field] = bv
    for field in ("gender", "age"):
        if (not a.get(field) or a.get(field) == "unknown") and b.get(field):
            merged[field] = b[field]
    # Union name keys into aliases; keep a's name as primary
    names = _character_name_keys(a) | _character_name_keys(b)
    primary = str(merged.get("name") or "").strip()
    aliases = sorted(n for n in names if n and n != primary)
    merged["aliases"] = aliases
    # Prefer non-narrator id that looks stable; narrator handled separately
    if a.get("role") == "narrator" or b.get("role") == "narrator":
        merged["role"] = "narrator"
        merged["id"] = "narrator"
        if not primary or primary in {"unknown", "旁白"}:
            merged["name"] = a.get("name") if a.get("role") == "narrator" else b.get("name")
            if not merged.get("name"):
                merged["name"] = "旁白"
    return validate_character(merged)


def merge_character_lists(batches: list[list[dict]]) -> list[dict]:
    """
    Merge character lists from chapter batches via undirected name/alias components.

    Always returns exactly one narrator (id=narrator). Guarantees a default
    narrator if none appears in input.
    """
    flat: list[dict] = []
    for batch in batches or []:
        if not batch:
            continue
        for item in batch:
            if isinstance(item, dict):
                flat.append(validate_character(item))

    if not flat:
        return [dict(_DEFAULT_NARRATOR)]

    # Union-Find over name keys
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # Each character's name keys form a clique; shared keys link characters
    char_keys: list[set[str]] = []
    for ch in flat:
        keys = _character_name_keys(ch)
        if not keys:
            keys = {ch["id"]}
        char_keys.append(keys)
        keys_list = list(keys)
        for i in range(1, len(keys_list)):
            union(keys_list[0], keys_list[i])

    components: dict[str, list[int]] = {}
    for idx, keys in enumerate(char_keys):
        root = find(next(iter(keys)))
        components.setdefault(root, []).append(idx)

    merged_chars: list[dict] = []
    narrators: list[dict] = []

    for indices in components.values():
        group = [flat[i] for i in indices]
        # Prefer first occurrence as base; fold rest
        acc = group[0]
        for other in group[1:]:
            acc = _merge_two_characters(acc, other)
        if acc.get("role") == "narrator" or acc.get("id") == "narrator":
            acc["id"] = "narrator"
            acc["role"] = "narrator"
            if not acc.get("name") or acc["name"] == "unknown":
                acc["name"] = "旁白"
            narrators.append(acc)
        else:
            merged_chars.append(acc)

    # Exactly one narrator
    if narrators:
        acc = narrators[0]
        for other in narrators[1:]:
            acc = _merge_two_characters(acc, other)
        acc["id"] = "narrator"
        acc["role"] = "narrator"
        narrator = validate_character(acc)
    else:
        narrator = dict(_DEFAULT_NARRATOR)

    # De-dupe dialogue chars that might still equal narrator by name
    final: list[dict] = [narrator]
    narrator_keys = _character_name_keys(narrator) | {"narrator", "旁白"}
    seen_ids: set[str] = {"narrator"}
    for ch in merged_chars:
        keys = _character_name_keys(ch)
        if keys & narrator_keys:
            continue
        if ch["id"] in seen_ids:
            # rename collision
            ch = dict(ch)
            ch["id"] = f"{ch['id']}_{len(seen_ids)}"
        seen_ids.add(ch["id"])
        final.append(validate_character(ch))

    return final


def _style_sample_text(text: str, chapters: list[dict]) -> str:
    """Build a bounded sample for style analysis (not full novel)."""
    parts: list[str] = []
    head = text[:2500]
    if head:
        parts.append("【开头】\n" + head)
    for ch in (chapters or [])[:8]:
        start = int(ch.get("start_char") or 0)
        end = int(ch.get("end_char") or start)
        snippet = text[start : min(start + 200, end)]
        title = ch.get("title") or ch.get("id") or ""
        if snippet.strip():
            parts.append(f"【{title}】\n{snippet}")
    tail = text[-800:] if len(text) > 800 else ""
    if tail and tail != head:
        parts.append("【结尾】\n" + tail)
    return "\n\n".join(parts)


def analyze_style(text: str, chapters: list[dict], llm: Any) -> dict:
    """
    Ask LLM for overall novel style; return validated style dict.

    Prompt requires JSON only (no markdown). Uses chapter samples, not full text.
    """
    sample = _style_sample_text(text or "", chapters or [])
    prompt = (
        "你是小说有声书风格分析助手。根据下列文本抽样，分析整体风格。\n"
        "只输出 JSON，不要 markdown，不要解释。\n"
        "字段：title, genre, narrative_pov, era, tone, pacing, lang, "
        "narrator_style, duration_factor, base_emo。\n"
        "lang 必须是 zh/en/ja/es/ar 之一。\n"
        "duration_factor 为 0.8–1.3 的小数（旁白语速，1.0 正常）。\n"
        "base_emo 为长度 8 的数组，顺序 "
        "[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]，"
        "每项 0–1。\n\n"
        f"文本抽样：\n{sample}"
    )
    raw = llm.chat(prompt)
    parsed = _parse_json_response(raw)
    if not isinstance(parsed, dict):
        parsed = {}
    return validate_style(parsed)


def extract_characters(text: str, chapters: list[dict], llm: Any) -> list[dict]:
    """
    Extract characters per chapter (map) then merge (reduce).

    For each chapter, if body > 6000 chars, split into 4000-char windows.
    Always returns a list with exactly one narrator after merge.
    """
    batches: list[list[dict]] = []
    chapters = chapters or []
    text = text or ""

    windows: list[tuple[str, str]] = []
    if not chapters:
        windows.append(("full", text[:4000] if text else ""))
    else:
        for ch in chapters:
            start = int(ch.get("start_char") or 0)
            end = int(ch.get("end_char") or start)
            body = text[start:end]
            cid = ch.get("id") or "c"
            if len(body) <= 6000:
                windows.append((cid, body))
            else:
                step = 4000
                for i in range(0, len(body), step):
                    windows.append((f"{cid}_{i // step}", body[i : i + step]))

    for win_id, body in windows:
        if not body.strip():
            continue
        prompt = (
            "你是小说人物分析助手。从本章文本中提取说话人与旁白特征。\n"
            "只输出 JSON 数组，不要 markdown，不要解释。\n"
            "每个元素字段：id, name, aliases, role, gender, age, personality, voice_traits。\n"
            "role 为 narrator 或 dialogue；旁白 id 必须为 narrator，name 为旁白。\n"
            "id 用英文 snake_case；aliases 为字符串数组；gender 为 male/female/unknown；\n"
            "age 为 child/young_adult/middle/elder/unknown。\n\n"
            f"章节/窗口：{win_id}\n文本：\n{body}"
        )
        raw = llm.chat(prompt)
        parsed = _parse_json_response(raw)
        batch: list[dict] = []
        if isinstance(parsed, list):
            batch = [c for c in parsed if isinstance(c, dict)]
        elif isinstance(parsed, dict):
            # Allow {"characters": [...]} wrapper
            inner = parsed.get("characters") or parsed.get("people") or []
            if isinstance(inner, list):
                batch = [c for c in inner if isinstance(c, dict)]
            else:
                batch = [parsed]
        if batch:
            batches.append(batch)

    return merge_character_lists(batches)


# ---------------------------------------------------------------------------
# Task 4: seed voice bank matching + character voice cards (IndexTTS-2.5)
# ---------------------------------------------------------------------------

# Timbre tags → keywords that may appear in character voice_traits
_TIMBRE_SYNONYMS: dict[str, list[str]] = {
    "deep": ["deep", "低沉", "低", "沉", "浑厚"],
    "firm": ["firm", "硬", "刚", "坚定", "有力"],
    "bright": ["bright", "亮", "清亮", "偏亮", "清脆"],
    "warm": ["warm", "暖", "温和", "柔"],
    "soft": ["soft", "软", "柔和", "轻"],
    "clear": ["clear", "清晰", "干净"],
}

_CARD_TAIL = (
    "今天天气不错，山上的风从松树林里穿过来，溪水轻轻响着。一二三四五，金木水火土。"
)


def load_voice_bank(path: str) -> list[dict]:
    """
    Load seed voice entries from a YAML bank file.

    Expects ``{voices: [{id, path, gender, age, timbre, languages}, ...]}``
    or a bare list of the same dicts. Returns a list of plain dicts.
    """
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return []
    if isinstance(data, list):
        voices = data
    elif isinstance(data, dict):
        voices = data.get("voices") or data.get("bank") or []
    else:
        return []

    out: list[dict] = []
    for item in voices:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        if "id" not in entry and "path" in entry:
            entry["id"] = Path(str(entry["path"])).stem
        out.append(entry)
    return out


def _is_narrator(char: dict) -> bool:
    return char.get("role") == "narrator" or char.get("id") == "narrator"


def _seed_match_score(
    char: dict,
    voice: dict,
    used_ids: set[str],
    *,
    require_gender: bool = True,
) -> float | None:
    """
    Deterministic score for assigning ``voice`` to ``char``.

    Returns None if gender is incompatible under require_gender.
    Higher is better; used seeds get a large penalty (collision avoidance).
    """
    char_g = str(char.get("gender") or "unknown").strip().lower() or "unknown"
    voice_g = str(voice.get("gender") or "unknown").strip().lower() or "unknown"

    if require_gender:
        if char_g not in {"unknown", ""} and voice_g not in {"unknown", ""}:
            if char_g != voice_g:
                return None

    score = 0.0
    if char_g == voice_g and char_g not in {"unknown", ""}:
        score += 100.0
    elif char_g in {"unknown", ""} or voice_g in {"unknown", ""}:
        score += 10.0

    char_age = str(char.get("age") or "unknown").strip().lower() or "unknown"
    voice_age = str(voice.get("age") or "unknown").strip().lower() or "unknown"
    if char_age not in {"unknown", ""} and char_age == voice_age:
        score += 50.0

    traits = str(char.get("voice_traits") or "")
    traits_l = traits.lower()
    timbre = str(voice.get("timbre") or "").strip().lower()
    if timbre:
        if timbre in traits_l or timbre in traits:
            score += 30.0
        synonyms = _TIMBRE_SYNONYMS.get(timbre, [])
        for kw in synonyms:
            if not kw:
                continue
            if kw.lower() in traits_l or kw in traits:
                score += 20.0
                break

    vid = voice.get("id")
    if vid is not None and vid in used_ids:
        score -= 1000.0

    return score


def _pick_seed_voice(
    char: dict,
    bank: list[dict],
    used_ids: set[str],
) -> dict | None:
    """Pick best bank entry: gender match, age, timbre; unused preferred; yaml order ties."""
    if not bank:
        return None

    def best(require_gender: bool) -> dict | None:
        winner: dict | None = None
        best_score = float("-inf")
        best_idx = -1
        for idx, voice in enumerate(bank):
            s = _seed_match_score(char, voice, used_ids, require_gender=require_gender)
            if s is None:
                continue
            if s > best_score or (s == best_score and (winner is None or idx < best_idx)):
                winner = voice
                best_score = s
                best_idx = idx
        return winner

    return best(True) or best(False)


def assign_seed_voices(
    characters: list[dict],
    bank: list[dict],
    narrator_audio: str | None = None,
) -> list[dict]:
    """
    Assign each character a seed voice from the bank.

    Narrator is assigned first. ``narrator_audio`` overrides the narrator seed
    path without consuming a bank entry. Gender must match when possible; used
    seeds are heavily penalized to avoid collisions.
    """
    chars = [dict(c) for c in (characters or [])]
    bank_list = list(bank or [])
    used: set[str] = set()

    order: list[int] = []
    for i, c in enumerate(chars):
        if _is_narrator(c):
            order.append(i)
    for i in range(len(chars)):
        if i not in order:
            order.append(i)

    result: list[dict | None] = [None] * len(chars)
    for i in order:
        char = dict(chars[i])
        if _is_narrator(char) and narrator_audio:
            char["seed_path"] = narrator_audio
            if not char.get("seed_voice_id"):
                char["seed_voice_id"] = "narrator_custom"
        else:
            voice = _pick_seed_voice(char, bank_list, used)
            if voice is not None:
                vid = voice.get("id")
                char["seed_voice_id"] = vid
                char["seed_path"] = voice.get("path")
                if vid is not None:
                    used.add(vid)
        result[i] = char

    return [c if c is not None else dict(chars[i]) for i, c in enumerate(result)]


def build_card_text(character: dict) -> str:
    """
    Build TTS text for a character voice card.

    Prefers existing ``card_text``; otherwise a fixed template with name +
    personality and a neutral phonetically-rich tail for timbre coverage.
    """
    existing = character.get("card_text")
    if existing:
        return str(existing)

    name = str(character.get("name") or character.get("id") or "角色").strip() or "角色"
    personality = str(
        character.get("personality") or character.get("voice_traits") or "沉稳"
    ).strip() or "沉稳"
    personality = personality.rstrip("。.!！?？")
    return f"我是{name}。{personality}。{_CARD_TAIL}"


def generate_character_voices(
    characters: list[dict],
    tts: Any,
    work_dir: str,
    lang: str,
    ref_mode: str,
) -> list[dict]:
    """
    Produce per-character reference WAVs under ``{work_dir}/voices/``.

    - ``ref_mode=="card"``: synthesize via ``tts.infer`` (IndexTTS-2.5 kwargs).
    - ``ref_mode=="seed"``: copy seed audio to the card path.
    Existing card WAVs are skipped. Writes ``voices/manifest.json``.
    Characters may supply ``seed_path`` directly or after ``assign_seed_voices``.
    """
    voices_dir = Path(work_dir) / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)

    out: list[dict] = []
    mode = (ref_mode or "card").strip().lower()

    for raw in characters or []:
        char = dict(raw)
        cid = str(char.get("id") or "char").strip() or "char"
        ref_wav = str(voices_dir / f"{cid}.wav")
        seed_path = char.get("seed_path")

        if Path(ref_wav).is_file():
            char["ref_wav"] = ref_wav
            out.append(char)
            continue

        if not seed_path:
            char["ref_wav"] = None
            out.append(char)
            continue

        if mode == "seed":
            shutil.copy2(str(seed_path), ref_wav)
        else:
            card_text = char.get("card_text") or build_card_text(char)
            char["card_text"] = card_text
            base_emo = char.get("base_emo")
            if base_emo is None:
                base_emo = [0.0] * _EMO_DIMS
            else:
                base_emo = _normalize_base_emo(base_emo)
            emo_vector = tts.normalize_emo_vec(base_emo)
            tts.infer(
                spk_audio_prompt=seed_path,
                text=card_text,
                lang=lang,
                output_path=ref_wav,
                emo_vector=emo_vector,
                duration_factor=1.0,
                interval_silence=200,
                max_text_tokens_per_segment=120,
                use_random=False,
                verbose=False,
            )

        char["ref_wav"] = ref_wav
        out.append(char)

    manifest_path = voices_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out


# ---------------------------------------------------------------------------
# Task 5: chapter reading script — speaker attribution, emotion, glossary
# ---------------------------------------------------------------------------

_SILENCE_SPEAKER_CHANGE_MS = 420
_SILENCE_SAME_SPEAKER_MS = 280
_SILENCE_DIALOGUE_TO_NARRATION_MS = 350
_LLM_UTTERANCE_BATCH = 40
_DURATION_FACTOR_MIN = 0.5
_DURATION_FACTOR_MAX = 2.0


def prepare_tts_text(text: str, work_dir: str) -> str:
    """
    Apply pronunciation glossary (repo default + work_dir override) to TTS text.

    Uses ``annotate_tts_text`` with glossary from
    ``load_pronunciation_glossary(work_dir=work_dir)``.
    """
    if not text:
        return text or ""
    glossary = load_pronunciation_glossary(work_dir=work_dir)
    return annotate_tts_text(text, glossary=glossary)


def assign_silence(utterances: list[dict]) -> list[dict]:
    """
    Set ``silence_after_ms`` on each utterance from speaker/kind transitions.

    Rules (priority: dialogue→narration > speaker change > same speaker):
      - dialogue → narration: 350 ms
      - speaker change: 420 ms
      - same speaker: 280 ms
    Last utterance uses same-speaker default (280).
    """
    if not utterances:
        return []

    out: list[dict] = []
    n = len(utterances)
    for i, raw in enumerate(utterances):
        utt = dict(raw)
        if i >= n - 1:
            utt["silence_after_ms"] = _SILENCE_SAME_SPEAKER_MS
            out.append(utt)
            continue

        nxt = utterances[i + 1]
        cur_spk = str(utt.get("speaker_id") or "")
        next_spk = str(nxt.get("speaker_id") or "")
        cur_kind = str(utt.get("kind") or "")
        next_kind = str(nxt.get("kind") or "")

        if cur_kind == "dialogue" and next_kind == "narration":
            utt["silence_after_ms"] = _SILENCE_DIALOGUE_TO_NARRATION_MS
        elif cur_spk != next_spk:
            utt["silence_after_ms"] = _SILENCE_SPEAKER_CHANGE_MS
        else:
            utt["silence_after_ms"] = _SILENCE_SAME_SPEAKER_MS
        out.append(utt)
    return out


def _character_lookup(characters: list[dict] | None) -> dict[str, dict]:
    """Map character id → character dict; always includes narrator if missing."""
    lookup: dict[str, dict] = {}
    for ch in characters or []:
        if not isinstance(ch, dict):
            continue
        cid = str(ch.get("id") or "").strip()
        if cid:
            lookup[cid] = ch
    if "narrator" not in lookup:
        lookup["narrator"] = dict(_DEFAULT_NARRATOR)
    return lookup


def _normalize_emo_vector(raw: Any) -> list[float] | None:
    """Return 8-dim emotion vector or None if unusable."""
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        return None
    if len(raw) == 0:
        return None
    return _normalize_base_emo(raw)


def _fallback_emo(text: str, character: dict | None) -> list[float]:
    """LLM-failure emotion: estimate_emotion_from_text, else character base_emo, else calm."""
    estimated = estimate_emotion_from_text(text or "")
    if estimated is not None:
        return _normalize_base_emo(estimated)
    if character and character.get("base_emo") is not None:
        return _normalize_base_emo(character.get("base_emo"))
    return [0.0] * _EMO_DIMS


def _resolve_duration_factor(character: dict | None, style: dict | None) -> float:
    """Prefer character duration_factor, else style, else 1.0; clamp 0.5–2.0."""
    for source in (character, style):
        if not source:
            continue
        if source.get("duration_factor") is None:
            continue
        try:
            return _clamp(float(source["duration_factor"]), _DURATION_FACTOR_MIN, _DURATION_FACTOR_MAX)
        except (TypeError, ValueError):
            continue
    return 1.0


def _expand_utterances_for_tts(utterances: list[dict], chapter_id: str) -> list[dict]:
    """Apply enforce_tts_limits to each utterance; re-sequence ids."""
    expanded: list[dict] = []
    for utt in utterances:
        parts = enforce_tts_limits(utt.get("tts_text") or "", max_chars=80)
        if not parts:
            text = (utt.get("tts_text") or utt.get("text") or "").strip()
            if not text:
                continue
            parts = [text]
        for part in parts:
            nu = dict(utt)
            nu["tts_text"] = part
            if utt.get("kind") != "dialogue":
                nu["text"] = part
            expanded.append(nu)

    seq = 0
    out: list[dict] = []
    for utt in expanded:
        seq += 1
        nu = dict(utt)
        nu["seq"] = seq
        nu["id"] = f"{chapter_id}_{seq:04d}"
        nu["chapter_id"] = chapter_id
        out.append(nu)
    return out


def _llm_attribute_batch(
    batch: list[dict],
    characters: list[dict],
    style: dict | None,
    llm: Any,
) -> list[dict] | None:
    """
    Ask LLM for speaker_id / emo_vector / tts_text for a batch.

    Returns list of attribution dicts aligned by index, or None on failure.
    """
    char_brief = [
        {
            "id": c.get("id"),
            "name": c.get("name"),
            "aliases": c.get("aliases") or [],
            "role": c.get("role"),
        }
        for c in (characters or [])
        if isinstance(c, dict)
    ]
    items = [
        {
            "index": i,
            "id": u.get("id"),
            "kind": u.get("kind"),
            "text": u.get("text"),
            "tts_text": u.get("tts_text"),
        }
        for i, u in enumerate(batch)
    ]
    prompt = (
        "你是小说有声书朗读脚本助手。为每条 utterance 指定说话人与情绪。\n"
        "只输出 JSON 数组，不要 markdown，不要解释。\n"
        "每个元素字段：index (与输入相同), speaker_id, emo_vector, tts_text。\n"
        "speaker_id 必须是人物表中的 id；叙述/旁白用 narrator；无法判断也用 narrator。\n"
        "emo_vector 为长度 8 的数组，顺序 "
        "[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]，每项 0–1。\n"
        "tts_text：口语化，保留原意，不扩写剧情，不发明旁白；"
        "可对数字/英文专有名词加 <字|发音> 标注；单条不超过 80 字；不删信息点。\n"
        "dialogue 的 speaker 应为人物；narration 通常为 narrator。\n\n"
        f"人物表：{json.dumps(char_brief, ensure_ascii=False)}\n"
        f"风格 lang={((style or {}).get('lang') or 'zh')}\n"
        f"utterances：{json.dumps(items, ensure_ascii=False)}"
    )
    try:
        raw = llm.chat(prompt)
    except Exception:
        return None

    parsed = _parse_json_response(raw)
    if parsed is None:
        return None

    rows: list[dict] = []
    if isinstance(parsed, list):
        rows = [r for r in parsed if isinstance(r, dict)]
    elif isinstance(parsed, dict):
        inner = parsed.get("utterances") or parsed.get("items") or parsed.get("results")
        if isinstance(inner, list):
            rows = [r for r in inner if isinstance(r, dict)]
        else:
            return None
    else:
        return None

    if not rows:
        return None

    # Align by index when present; else by order
    by_index: dict[int, dict] = {}
    ordered: list[dict] = []
    for r in rows:
        if "index" in r:
            try:
                by_index[int(r["index"])] = r
            except (TypeError, ValueError):
                ordered.append(r)
        else:
            ordered.append(r)

    result: list[dict] = []
    oi = 0
    for i in range(len(batch)):
        if i in by_index:
            result.append(by_index[i])
        elif oi < len(ordered):
            result.append(ordered[oi])
            oi += 1
        else:
            result.append({})
    return result


def build_chapter_script(
    chapter_text: str,
    chapter_id: str,
    characters: list[dict],
    style: dict | None,
    llm: Any,
    work_dir: str,
) -> list[dict]:
    """
    Build a full reading script for one chapter.

    Pipeline:
      split_utterances → enforce_tts_limits → LLM attribution (batches of 40)
      → validate speaker_id ∈ character table (else narrator)
      → re-enforce_tts_limits after LLM rewrite (keep speaker/emo on pieces)
      → prepare_tts_text (glossary) → character duration_factor → assign_silence

    LLM failure falls back to ``estimate_emotion_from_text`` (and narrator / original text).
    """
    style = style or {}
    lang = str(style.get("lang") or "zh").strip().lower() or "zh"
    if lang not in _VALID_LANGS:
        lang = "zh"

    lookup = _character_lookup(characters)
    char_list = list(lookup.values())

    raw_utts = split_utterances(chapter_text or "", chapter_id)
    utterances = _expand_utterances_for_tts(raw_utts, chapter_id)

    # LLM attribution in batches of 40
    attributions: list[dict | None] = [None] * len(utterances)
    for start in range(0, len(utterances), _LLM_UTTERANCE_BATCH):
        batch = utterances[start : start + _LLM_UTTERANCE_BATCH]
        try:
            batch_attrs = _llm_attribute_batch(batch, char_list, style, llm)
        except Exception:
            batch_attrs = None
        if batch_attrs is None:
            for j in range(len(batch)):
                attributions[start + j] = None
        else:
            for j, attr in enumerate(batch_attrs):
                attributions[start + j] = attr

    built: list[dict] = []
    for utt, attr in zip(utterances, attributions):
        item = dict(utt)
        item["lang"] = lang
        item["wav_path"] = None

        kind = str(item.get("kind") or "narration")
        default_speaker = "narrator"

        speaker_id = default_speaker
        tts_text = item.get("tts_text") or item.get("text") or ""
        emo: list[float] | None = None

        if attr and isinstance(attr, dict):
            sid = str(attr.get("speaker_id") or "").strip()
            if sid and sid in lookup:
                speaker_id = sid
            if attr.get("tts_text"):
                candidate = str(attr["tts_text"]).strip()
                if candidate:
                    tts_text = candidate
            emo = _normalize_emo_vector(attr.get("emo_vector"))

        # Unknown speaker → narrator
        if speaker_id not in lookup:
            speaker_id = "narrator"

        character = lookup.get(speaker_id) or lookup["narrator"]
        if emo is None:
            emo = _fallback_emo(item.get("text") or tts_text, character)

        item["speaker_id"] = speaker_id
        item["emo_vector"] = emo
        item["duration_factor"] = _resolve_duration_factor(character, style)

        # LLM rewrite may exceed 80 chars; re-split and keep speaker/emo on each piece
        parts = enforce_tts_limits(str(tts_text or ""), max_chars=80)
        if not parts:
            fallback = str(tts_text or item.get("text") or "").strip()
            if not fallback:
                continue
            parts = [fallback]
        for part in parts:
            nu = dict(item)
            nu["tts_text"] = prepare_tts_text(part, work_dir)
            if kind != "dialogue":
                nu["text"] = part
            built.append(nu)

    seq = 0
    resequenced: list[dict] = []
    for utt in built:
        seq += 1
        nu = dict(utt)
        nu["seq"] = seq
        nu["id"] = f"{chapter_id}_{seq:04d}"
        nu["chapter_id"] = chapter_id
        resequenced.append(nu)

    return assign_silence(resequenced)


# ---------------------------------------------------------------------------
# Task 6: sequential synthesis and per-chapter merge
# ---------------------------------------------------------------------------

_TTS_SAMPLE_RATE = 22050


def _characters_to_map(characters: Any) -> dict[str, dict]:
    """Normalize characters to id → dict (accepts list or id-keyed dict)."""
    if not characters:
        return {}
    if isinstance(characters, dict):
        # id-keyed map (tests) vs single character dict with "id"
        if "id" in characters and not any(
            isinstance(v, dict) and ("ref_wav" in v or "id" in v)
            for v in characters.values()
        ):
            cid = str(characters.get("id") or "").strip()
            return {cid: characters} if cid else {}
        out: dict[str, dict] = {}
        for k, v in characters.items():
            if isinstance(v, dict):
                out[str(k)] = v
        return out
    if isinstance(characters, list):
        out = {}
        for ch in characters:
            if not isinstance(ch, dict):
                continue
            cid = str(ch.get("id") or "").strip()
            if cid:
                out[cid] = ch
        return out
    return {}


def _write_silence_wav(path: str | Path, duration_sec: float, sample_rate: int = _TTS_SAMPLE_RATE) -> None:
    """Write mono 16-bit PCM silence WAV of the given duration."""
    nframes = max(1, int(duration_sec * sample_rate))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * nframes)


def _emo_for_infer(tts: Any, emo_vector: Any) -> list[float] | None:
    """Normalize emo_vector for infer; None when all dims are zero / missing."""
    if emo_vector is None:
        return None
    if not isinstance(emo_vector, (list, tuple)):
        return None
    vals = [float(x) for x in emo_vector]
    if not vals or not any(v > 0 for v in vals):
        return None
    return tts.normalize_emo_vec(vals)


def synthesize_chapter(
    utterances: list[dict],
    characters: Any,
    tts: Any,
    work_dir: str,
    strict: bool = False,
) -> list[dict]:
    """
    Synthesize each utterance WAV under ``{work_dir}/tts/{chapter_id}/{seq:04d}.wav``.

    Skips lines whose output already exists. On infer failure with ``strict=False``,
    writes silence of ``max(1.2, 0.15 * len(tts_text))`` seconds (22050 Hz mono 16-bit).
    Returns utterances with ``wav_path`` set.
    """
    char_map = _characters_to_map(characters)
    out: list[dict] = []

    for raw in utterances or []:
        utt = dict(raw)
        chapter_id = str(utt.get("chapter_id") or "c01")
        try:
            seq = int(utt.get("seq", 0))
        except (TypeError, ValueError):
            seq = 0

        wav_path = Path(work_dir) / "tts" / chapter_id / f"{seq:04d}.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        utt["wav_path"] = str(wav_path)

        if wav_path.is_file():
            out.append(utt)
            continue

        speaker_id = str(utt.get("speaker_id") or "").strip()
        character = char_map.get(speaker_id) or {}
        ref_wav = character.get("ref_wav")
        tts_text = str(utt.get("tts_text") or utt.get("text") or "")
        lang = str(utt.get("lang") or "zh").strip().lower() or "zh"

        try:
            df = float(utt.get("duration_factor", character.get("duration_factor", 1.0)))
        except (TypeError, ValueError):
            df = 1.0
        duration_factor = _clamp(df, _DURATION_FACTOR_MIN, _DURATION_FACTOR_MAX)
        emo_vector = _emo_for_infer(tts, utt.get("emo_vector"))

        try:
            tts.infer(
                spk_audio_prompt=ref_wav,
                text=tts_text,
                output_path=str(wav_path),
                lang=lang,
                emo_vector=emo_vector,
                duration_factor=duration_factor,
                interval_silence=200,
                max_text_tokens_per_segment=120,
                use_random=False,
            )
        except Exception:
            if strict:
                raise
            silence_sec = max(1.2, 0.15 * len(tts_text))
            _write_silence_wav(wav_path, silence_sec)

        out.append(utt)

    return out


def merge_chapter(utterances: list[dict], output_path: str) -> str:
    """Concatenate utterance WAVs with per-line ``silence_after_ms`` via ``_concat_wavs``."""
    return _concat_wavs(list(utterances or []), output_path)


# ---------------------------------------------------------------------------
# Task 7: orchestration, checkpoint resume, CLI
# ---------------------------------------------------------------------------

_STOP_AFTER_STEPS = {
    "chapters": 1,
    "style": 2,
    "characters": 3,
    "voices": 4,
    "script": 5,
    "tts": 6,
    "merge": 7,
}
_DEFAULT_WORK_DIR = "novel_workspace"
_DEFAULT_VOICE_BANK = "examples/voice_bank.yaml"
_BOOK_CHAPTER_GAP_MS = 1500


def _write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_novel_checkpoint(novel_dir: str | Path, step: int, paths: dict | None = None) -> None:
    """Wrap highlight ``_save_checkpoint`` so novel state is ``{step, paths}``."""
    _save_checkpoint(str(novel_dir), int(step), paths=paths or {})


def _strip_markdown_hashes(text: str) -> str:
    """Drop Markdown heading markers but keep the title text."""
    return re.sub(r"(?m)^#{1,6}\s*", "", text)


def _target_step(stop_after: str | None) -> int:
    if not stop_after:
        return 7
    key = str(stop_after).strip().lower()
    if key not in _STOP_AFTER_STEPS:
        raise ValueError(
            f"invalid stop_after={stop_after!r}; "
            f"expected one of {sorted(_STOP_AFTER_STEPS)}"
        )
    return _STOP_AFTER_STEPS[key]


def _require_llm_key(llm_api_key: str | None, target: int) -> str | None:
    if target < 2:
        return llm_api_key
    key = llm_api_key or os.environ.get("LLM_API_KEY")
    if not key:
        raise ValueError(
            "LLM API key required from style onward. "
            "Use --llm-api-key or set LLM_API_KEY."
        )
    return key


def _default_paths(novel_dir: Path) -> dict[str, str]:
    return {
        "source": str(novel_dir / "source.txt"),
        "chapters": str(novel_dir / "chapters.json"),
        "style": str(novel_dir / "style.json"),
        "characters": str(novel_dir / "characters.json"),
        "script": str(novel_dir / "script"),
        "voices": str(novel_dir / "voices"),
        "tts": str(novel_dir / "tts"),
        "chapters_audio": str(novel_dir / "chapters"),
    }


def _filter_chapters(chapters: list[dict], chapter: int | None) -> list[dict]:
    if chapter is None:
        return list(chapters)
    n = int(chapter)
    cid = f"c{n:02d}"
    selected = [
        c
        for c in chapters
        if int(c.get("index") or 0) == n or c.get("id") == cid
    ]
    if not selected:
        raise ValueError(f"chapter {chapter} not found")
    return selected


def _resolve_voice_bank(voice_bank: str | None) -> list[dict]:
    p = Path(voice_bank) if voice_bank else Path(_DEFAULT_VOICE_BANK)
    if p.is_dir():
        yamls = sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml"))
        if not yamls:
            return []
        p = yamls[0]
    if not p.is_file():
        return []
    bank = load_voice_bank(str(p))
    base = p.parent
    for entry in bank:
        path = entry.get("path")
        if not path:
            continue
        raw = Path(path)
        if not raw.is_file():
            alt = base / path
            if alt.is_file():
                entry["path"] = str(alt)
    return bank


def _all_scripts_exist(novel_dir: Path, chapters: list[dict]) -> bool:
    script_dir = novel_dir / "script"
    return bool(chapters) and all(
        (script_dir / f"{c['id']}.json").is_file() for c in chapters
    )


def _ensure_wav_paths(utterances: list[dict], novel_dir: Path) -> list[dict]:
    out: list[dict] = []
    for raw in utterances:
        utt = dict(raw)
        if not utt.get("wav_path"):
            cid = str(utt.get("chapter_id") or "c01")
            try:
                seq = int(utt.get("seq", 0))
            except (TypeError, ValueError):
                seq = 0
            utt["wav_path"] = str(novel_dir / "tts" / cid / f"{seq:04d}.wav")
        out.append(utt)
    return out


def _apply_style_voice_defaults(characters: list[dict], style: dict) -> list[dict]:
    out: list[dict] = []
    style_emo = style.get("base_emo") or [0.0] * _EMO_DIMS
    style_df = style.get("duration_factor")
    for raw in characters:
        char = dict(raw)
        if char.get("base_emo") is None:
            char["base_emo"] = list(style_emo)
        if char.get("duration_factor") is None and style_df is not None:
            try:
                char["duration_factor"] = _clamp(float(style_df), 0.5, 2.0)
            except (TypeError, ValueError):
                pass
        out.append(char)
    return out


def _export_chapter_wavs(
    chapter_wavs: list[tuple[str, str]],
    output: str | None,
    stem: str,
    concat_book: bool,
) -> list[str]:
    """Copy chapter WAVs to ``--output`` and optionally concat ``book.wav``."""
    if not chapter_wavs:
        return []

    dest = Path(output) if output else Path(f"{stem}_audiobook")
    exported: list[str] = []

    if dest.suffix.lower() == ".wav":
        if len(chapter_wavs) == 1:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(chapter_wavs[0][1], dest)
            exported.append(str(dest))
            dest_dir = dest.parent
        else:
            dest_dir = dest.with_name(f"{stem}_chapters")
            dest_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Multi-chapter output file → directory {dest_dir}")
            for cid, src in chapter_wavs:
                target = dest_dir / f"{stem}_{cid}.wav"
                shutil.copy2(src, target)
                exported.append(str(target))
    else:
        dest_dir = dest
        dest_dir.mkdir(parents=True, exist_ok=True)
        for cid, src in chapter_wavs:
            target = dest_dir / f"{stem}_{cid}.wav"
            shutil.copy2(src, target)
            exported.append(str(target))

    print("  Output chapter WAVs:")
    for path in exported:
        print(f"    {path}")

    if concat_book:
        book_path = dest if dest.suffix.lower() == ".wav" and len(chapter_wavs) == 1 else dest_dir / "book.wav"
        segs = [
            {"wav_path": src, "silence_after_ms": _BOOK_CHAPTER_GAP_MS}
            for _, src in chapter_wavs
        ]
        segs[-1]["silence_after_ms"] = 0
        merge_chapter(segs, str(book_path))
        print(f"  Concatenated book: {book_path}")
        if str(book_path) not in exported:
            exported.append(str(book_path))

    return exported


def run_novel_pipeline(
    input_path: str,
    output: str,
    work_dir: str = _DEFAULT_WORK_DIR,
    stop_after: str | None = None,
    llm_api_key: str | None = None,
    llm_api_base: str = "https://api.openai.com/v1",
    llm_model: str = "gpt-4o-mini",
    model_dir: str = "checkpoints",
    use_fp16: bool = False,
    ref_audio: str | None = None,
    narrator_audio: str | None = None,
    voice_bank: str | None = None,
    ref_mode: str = "card",
    lang: str | None = None,
    chapter: int | None = None,
    force_tts: bool = False,
    strict: bool = False,
    concat_book: bool = False,
    cleanup: bool = False,
) -> dict:
    """
    Run the novel-to-audiobook pipeline with file-backed resume.

    Work dir is ``{work_dir}/{stem}/``. ``stop_after="chapters"`` writes
    ``chapters.json`` and returns ``{"step": 1, ...}`` without loading TTS.
    LLM key is required from style onward. TTS is loaded only at voices/tts.
    """
    src_path = Path(input_path)
    if not src_path.is_file():
        raise FileNotFoundError(f"input not found: {input_path}")

    stem = src_path.stem
    novel_dir = Path(work_dir) / stem
    novel_dir.mkdir(parents=True, exist_ok=True)

    target = _target_step(stop_after)
    llm_api_key = _require_llm_key(llm_api_key, target)

    ckpt = _load_checkpoint(str(novel_dir))
    done = int((ckpt or {}).get("step") or 0)
    paths = _default_paths(novel_dir)
    paths.update((ckpt or {}).get("paths") or {})

    source_path = Path(paths["source"])
    chapters_path = Path(paths["chapters"])
    style_path = Path(paths["style"])
    characters_path = Path(paths["characters"])
    script_dir = Path(paths["script"])
    voices_dir = Path(paths["voices"])
    tts_root = Path(paths["tts"])
    chapters_audio_dir = Path(paths["chapters_audio"])

    print(f"\n{'=' * 60}")
    print("Novel Audiobook Pipeline")
    print(f"Input:    {src_path}")
    print(f"Output:   {output}")
    print(f"Work dir: {novel_dir}")
    print(f"{'=' * 60}")

    text = ""
    chapters: list[dict] = []
    style: dict = {}
    characters: list[dict] = []
    llm: Any = None
    tts: Any = None
    result: dict = {"step": done, "paths": paths}

    def ensure_llm() -> Any:
        nonlocal llm
        if llm is None:
            llm = LLMClient(api_key=llm_api_key, api_base=llm_api_base, model=llm_model)
        return llm

    def ensure_tts() -> Any:
        nonlocal tts
        if tts is None:
            print("  Loading IndexTTS-2.5...")
            tts = _init_tts(model_dir, use_fp16)
        return tts

    def finish(step: int) -> dict:
        result["step"] = step
        result["paths"] = paths
        return result

    try:
        # Step 0 ingest + Step 1 chapters
        if done < 1 or not source_path.is_file() or not chapters_path.is_file():
            print("\n[Step 0-1] Ingest and split chapters...")
            raw = src_path.read_text(encoding="utf-8")
            text = _strip_markdown_hashes(ingest_text(raw))
            source_path.write_text(text, encoding="utf-8")
            chapters = split_chapters(text, src_path.name)
            _write_json(chapters_path, chapters)
            paths["source"] = str(source_path)
            paths["chapters"] = str(chapters_path)
            done = 1
            _save_novel_checkpoint(novel_dir, done, paths)
            print(f"  {len(chapters)} chapter(s) → {chapters_path}")
        else:
            text = source_path.read_text(encoding="utf-8")
            chapters = _read_json(chapters_path)
            print(f"[Step 0-1] Skipped (cached, {len(chapters)} chapters)")

        if target <= 1:
            return finish(1)

        # Step 2 style
        if done < 2 or not style_path.is_file():
            print("\n[Step 2] Analyzing style...")
            style = analyze_style(text, chapters, ensure_llm())
            if lang:
                style = validate_style({**style, "lang": lang})
            _write_json(style_path, style)
            paths["style"] = str(style_path)
            done = 2
            _save_novel_checkpoint(novel_dir, done, paths)
        else:
            style = _read_json(style_path)
            if lang:
                style = validate_style({**style, "lang": lang})
            print("[Step 2] Skipped (cached style)")

        if target <= 2:
            return finish(2)

        # Step 3 characters
        if done < 3 or not characters_path.is_file():
            print("\n[Step 3] Extracting characters...")
            characters = extract_characters(text, chapters, ensure_llm())
            _write_json(characters_path, characters)
            paths["characters"] = str(characters_path)
            done = 3
            _save_novel_checkpoint(novel_dir, done, paths)
            print(f"  {len(characters)} character(s)")
        else:
            characters = _read_json(characters_path)
            print(f"[Step 3] Skipped (cached, {len(characters)} characters)")

        if target <= 3:
            return finish(3)

        selected = _filter_chapters(chapters, chapter)
        style_lang = str((lang or style.get("lang") or "zh")).strip().lower() or "zh"

        # Step 4 voice cards (load TTS)
        manifest_path = voices_dir / "manifest.json"
        if done < 4 or not manifest_path.is_file():
            print("\n[Step 4] Assigning seeds and generating voice cards...")
            bank = _resolve_voice_bank(voice_bank)
            narr_src = narrator_audio or ref_audio
            if not bank and not narr_src:
                raise ValueError(
                    "voice bank or --ref-audio/--narrator-audio is required "
                    "to generate character voices"
                )
            characters = _apply_style_voice_defaults(characters, style)
            characters = assign_seed_voices(characters, bank, narrator_audio=narr_src)
            characters = generate_character_voices(
                characters, ensure_tts(), str(novel_dir), style_lang, ref_mode,
            )
            _write_json(characters_path, characters)
            paths["voices"] = str(voices_dir)
            paths["characters"] = str(characters_path)
            done = 4
            _save_novel_checkpoint(novel_dir, done, paths)
        else:
            characters = _read_json(characters_path)
            print("[Step 4] Skipped (cached voice cards)")

        if target <= 4:
            return finish(4)

        # Step 5 per-chapter scripts (no TTS)
        script_dir.mkdir(parents=True, exist_ok=True)
        need_script = done < 5 or chapter is not None or not _all_scripts_exist(novel_dir, chapters)
        if need_script:
            print("\n[Step 5] Building chapter scripts...")
            for ch in selected:
                script_path = script_dir / f"{ch['id']}.json"
                if script_path.is_file():
                    print(f"  {ch['id']}: existing script")
                    continue
                body = text[int(ch.get("start_char") or 0) : int(ch.get("end_char") or 0)]
                utts = build_chapter_script(
                    body, ch["id"], characters, style, ensure_llm(), str(novel_dir),
                )
                _write_json(script_path, utts)
                print(f"  {ch['id']}: {len(utts)} utterances")
            if _all_scripts_exist(novel_dir, chapters):
                done = 5
                paths["script"] = str(script_dir)
                _save_novel_checkpoint(novel_dir, done, paths)
        else:
            print("[Step 5] Skipped (cached scripts)")

        if target <= 5:
            return finish(5)

        # Step 6 sequential TTS
        if force_tts:
            for ch in selected:
                tdir = tts_root / ch["id"]
                if tdir.is_dir():
                    shutil.rmtree(tdir)
                    print(f"  Removed {tdir} (--force-tts)")

        need_tts = done < 6 or force_tts or chapter is not None
        if need_tts:
            print("\n[Step 6] Synthesizing chapter lines...")
            for ch in selected:
                script_path = script_dir / f"{ch['id']}.json"
                utts = _read_json(script_path)
                utts = synthesize_chapter(
                    utts, characters, ensure_tts(), str(novel_dir), strict=strict,
                )
                _write_json(script_path, utts)
                print(f"  {ch['id']}: {len(utts)} lines")
            if chapter is None:
                done = 6
                paths["tts"] = str(tts_root)
                _save_novel_checkpoint(novel_dir, done, paths)
        else:
            print("[Step 6] Skipped (cached TTS)")

        if target <= 6:
            return finish(6)

        # Step 7 merge + export
        print("\n[Step 7] Merging chapter WAVs...")
        chapters_audio_dir.mkdir(parents=True, exist_ok=True)
        merged: list[tuple[str, str]] = []
        for ch in selected:
            script_path = script_dir / f"{ch['id']}.json"
            utts = _ensure_wav_paths(_read_json(script_path), novel_dir)
            out_wav = chapters_audio_dir / f"{ch['id']}.wav"
            merge_chapter(utts, str(out_wav))
            merged.append((ch["id"], str(out_wav)))
            print(f"  {out_wav}")

        exported = _export_chapter_wavs(merged, output, stem, concat_book)
        paths["chapters_audio"] = str(chapters_audio_dir)
        paths["output"] = exported
        result["output"] = exported
        if chapter is None:
            done = 7
            _save_novel_checkpoint(novel_dir, done, paths)

        if cleanup:
            _clear_checkpoint(str(novel_dir))
            shutil.rmtree(novel_dir, ignore_errors=True)
            print(f"Cleaned up: {novel_dir}")

        return finish(7)
    finally:
        if tts is not None:
            del tts
            _free_vram()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Turn a novel text file into a multi-speaker audiobook "
            "(per-chapter WAV) via IndexTTS-2.5"
        )
    )
    parser.add_argument("input", help="Input novel .txt or .md")
    parser.add_argument(
        "-o", "--output", default="audiobook_out",
        help="Output directory or .wav file (directory lists {stem}_c01.wav)",
    )
    parser.add_argument(
        "--work-dir", default=_DEFAULT_WORK_DIR,
        help="Working directory (artifacts under {work_dir}/{stem}/)",
    )
    parser.add_argument("--model-dir", default="checkpoints", help="IndexTTS-2.5 model directory")
    parser.add_argument(
        "--fp16", action="store_true", default=True,
        help="Use bf16 for TTS (default: True)",
    )
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable bf16")
    parser.add_argument("--ref-audio", default=None, help="Narrator / fallback seed audio")
    parser.add_argument("--narrator-audio", default=None, help="Override narrator seed audio")
    parser.add_argument(
        "--voice-bank", default=None,
        help="voice_bank.yaml or a directory containing yaml + wav",
    )
    parser.add_argument(
        "--ref-mode", choices=["card", "seed"], default="card",
        help="card: synthesize 2.5 voice cards; seed: use seed wavs directly",
    )
    parser.add_argument("--lang", default=None, help="Override language (zh/en/ja/es/ar)")
    parser.add_argument("--llm-api-key", default=None, help="LLM API key (or LLM_API_KEY env)")
    parser.add_argument("--llm-api-base", default="https://api.openai.com/v1")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument(
        "--stop-after",
        choices=["chapters", "style", "characters", "voices", "script", "tts", "merge"],
        default=None,
        help="Stop after this stage (chapters does not require an LLM key)",
    )
    parser.add_argument(
        "--chapter", type=int, default=None,
        help="Only process this 1-based chapter for script/tts/merge",
    )
    parser.add_argument(
        "--force-tts", action="store_true",
        help="Delete target chapter tts/ then resynthesize",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Abort on TTS failure instead of writing silence",
    )
    parser.add_argument(
        "--concat-book", action="store_true",
        help="Also write book.wav (1500 ms silence between chapters)",
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Delete work dir after success; keep --output chapter wavs",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    llm_key = args.llm_api_key or os.environ.get("LLM_API_KEY")
    if args.stop_after not in {"chapters"} and not llm_key:
        parser.error(
            "LLM API key required for this run. "
            "Use --llm-api-key or set LLM_API_KEY "
            "(not required with --stop-after chapters)."
        )
    run_novel_pipeline(
        input_path=args.input,
        output=args.output,
        work_dir=args.work_dir,
        stop_after=args.stop_after,
        llm_api_key=llm_key,
        llm_api_base=args.llm_api_base,
        llm_model=args.llm_model,
        model_dir=args.model_dir,
        use_fp16=args.fp16,
        ref_audio=args.ref_audio,
        narrator_audio=args.narrator_audio,
        voice_bank=args.voice_bank,
        ref_mode=args.ref_mode,
        lang=args.lang,
        chapter=args.chapter,
        force_tts=args.force_tts,
        strict=args.strict,
        concat_book=args.concat_book,
        cleanup=args.cleanup,
    )


if __name__ == "__main__":
    main()

