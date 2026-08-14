from novel_pipeline import (
    enforce_tts_limits,
    ingest_text,
    merge_character_lists,
    split_chapters,
    split_utterances,
    validate_character,
    validate_style,
)


def test_split_chapters_chinese_heading():
    text = open("tests/fixtures/novel_sample.txt", encoding="utf-8").read()
    chapters = split_chapters(ingest_text(text), "novel_sample.txt")
    assert [c["id"] for c in chapters] == ["c01", "c02"]
    assert chapters[0]["title"].startswith("第一章")
    assert "张三站在巷口" in text[chapters[0]["start_char"]:chapters[0]["end_char"]]
    assert "雨停了" in text[chapters[1]["start_char"]:chapters[1]["end_char"]]


def test_ingest_strips_bom_and_crlf():
    assert ingest_text("\ufeffhello\r\nworld\r\n") == "hello\nworld\n"


def test_split_utterances_quotes():
    utts = split_utterances("张三说：「今晚别跟过来。」巷子里很静。", "c01")
    kinds = [u["kind"] for u in utts]
    assert "dialogue" in kinds and "narration" in kinds
    dialogue = next(u for u in utts if u["kind"] == "dialogue")
    assert "今晚别跟过来" in dialogue["text"]
    assert "「" not in dialogue["tts_text"] and "」" not in dialogue["tts_text"]


def test_enforce_tts_limits_protects_pron_tags():
    text = "他在银<行|HANG2>办了一件非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常长的业务。"
    parts = enforce_tts_limits(text, max_chars=20)
    assert all("<行|HANG2>" in p or "<行|HANG2>" not in text for p in parts) or any(
        "<行|HANG2>" in p for p in parts
    )
    assert all("<行|" not in p or "|HANG2>" in p for p in parts)


def test_merge_characters_aliases_and_narrator():
    merged = merge_character_lists([
        [{"id": "zhang_san", "name": "张三", "aliases": ["老张"], "role": "dialogue",
          "gender": "male", "age": "young_adult", "personality": "急", "voice_traits": "亮"}],
        [{"id": "lao_zhang", "name": "老张", "aliases": ["张三"], "role": "dialogue",
          "gender": "male", "age": "young_adult", "personality": "急躁", "voice_traits": "偏亮"}],
        [{"id": "narrator", "name": "旁白", "aliases": [], "role": "narrator",
          "gender": "male", "age": "middle", "personality": "沉稳", "voice_traits": "低"}],
    ])
    names = {c["name"] for c in merged}
    assert "张三" in names or "老张" in names
    assert sum(1 for c in merged if c["role"] == "narrator") == 1
    assert len([c for c in merged if c["name"] in {"张三", "老张"}]) == 1


def test_validate_style_clamps_duration_and_emo():
    style = validate_style({
        "title": "x", "genre": "y", "narrative_pov": "第三人称", "era": "当代",
        "tone": "冷", "pacing": "慢", "lang": "ZH", "narrator_style": "沉",
        "duration_factor": 3.0, "base_emo": [1, 1, 1],
    })
    assert style["lang"] == "zh"
    assert 0.8 <= style["duration_factor"] <= 1.3
    assert len(style["base_emo"]) == 8

