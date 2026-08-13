from dub_pipeline import clean_subtitle_cues, build_subtitle_driven_segments


def _cues(*texts):
    return [{"start": i, "end": i + 1, "text": t} for i, t in enumerate(texts)]


def test_strips_translator_credits_and_keeps_dialogue():
    cleaned = clean_subtitle_cues(_cues(
        "翻译：张三",
        "Translated by Alice",
        "Subtitles by TED",
        "大家好，欢迎来到今天的演讲",
        "Thank you for watching",
    ))
    texts = [c["text"] for c in cleaned]
    assert texts == ["大家好，欢迎来到今天的演讲"]


def test_strips_speaker_names_not_connectives():
    cleaned = clean_subtitle_cues(_cues(
        "Chris Anderson: Welcome to TED.",
        "BS: I agree.",
        "主持人：下一个问题",
        "张三：我觉得可以",
        "所以：我们继续往下看",
        "比拉瓦尔·西杜（BS）：大家好",
    ))
    texts = [c["text"] for c in cleaned]
    assert texts[0] == "Welcome to TED."
    assert texts[1] == "I agree."
    assert texts[2] == "下一个问题"
    assert texts[3] == "我觉得可以"
    assert any(t.startswith("所以") for t in texts)
    assert "大家好" in texts[-1]


def test_strips_sound_marks_and_html():
    cleaned = clean_subtitle_cues(_cues(
        "<i>Hello</i> [Applause]",
        "♪ music ♪",
        "（掌声）谢谢大家",
        "Visit https://example.com now",
    ))
    texts = [c["text"] for c in cleaned]
    assert texts[0] == "Hello"
    assert all("♪" not in t for t in texts)
    assert "谢谢大家" in texts[1]
    assert "https" not in texts[-1]


def test_english_subtitle_driven_leaves_text_for_llm(tmp_path):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"")
    srt = tmp_path / "talk.en.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nChris Anderson: Hello everyone\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\nTranslated by Bob\n\n",
        encoding="utf-8",
    )
    asr = [{"start": 0.0, "end": 2.0, "text": "hello", "speaker": "SPEAKER_00"}]
    segs, source = build_subtitle_driven_segments(str(video), asr)
    assert source and source.startswith("en:")
    assert len(segs) == 1
    assert segs[0]["text"] == "Hello everyone"
    assert "zh_text" not in segs[0]


def test_chinese_subtitle_driven_fills_zh_text(tmp_path):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"")
    srt = tmp_path / "talk.zh.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\n主持人：大家好\n\n",
        encoding="utf-8",
    )
    asr = [{"start": 0.0, "end": 2.0, "text": "hello", "speaker": "SPEAKER_00"}]
    segs, source = build_subtitle_driven_segments(str(video), asr)
    assert source and source.startswith("zh:")
    assert segs[0]["zh_text"] == "大家好"
