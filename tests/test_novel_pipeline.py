from novel_pipeline import ingest_text, split_chapters


def test_split_chapters_chinese_heading():
    text = open("tests/fixtures/novel_sample.txt", encoding="utf-8").read()
    chapters = split_chapters(ingest_text(text), "novel_sample.txt")
    assert [c["id"] for c in chapters] == ["c01", "c02"]
    assert chapters[0]["title"].startswith("第一章")
    assert "张三站在巷口" in text[chapters[0]["start_char"]:chapters[0]["end_char"]]
    assert "雨停了" in text[chapters[1]["start_char"]:chapters[1]["end_char"]]


def test_ingest_strips_bom_and_crlf():
    assert ingest_text("\ufeffhello\r\nworld\r\n") == "hello\nworld\n"
