"""
Novel audiobook pipeline: novel text → multi-speaker chapter WAVs via IndexTTS-2.5.

Task 1 provides text ingest, chapter splitting, and silent WAV concatenation.
Task 2 adds dialogue/narration split and IndexTTS 2.5 sentence-length limits.
Task 3 adds LLM style/character analysis and validation helpers.
Task 4 adds seed voice bank matching and IndexTTS-2.5 character voice cards.
"""

from __future__ import annotations

import json
import re
import shutil
import wave
from pathlib import Path
from typing import Any

import yaml

from highlight_pipeline import LLMClient, _parse_json_response  # noqa: F401

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

