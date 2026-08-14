from novel_pipeline import (
    enforce_tts_limits,
    ingest_text,
    split_chapters,
    split_utterances,
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

