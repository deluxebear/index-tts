from pathlib import Path

from dub_pipeline import generate_srt, split_zh_captions, _caption_spans


def test_short_caption_stays_one():
    assert split_zh_captions("大家好") == ["大家好"]
    assert split_zh_captions("银<行|HANG2>办理业务") == ["银行办理业务"]


def test_splits_on_punctuation_without_wrap():
    text = "这是第一句，里面还有逗号所以会比较长。这是第二句也一样很长需要再拆一次！"
    caps = split_zh_captions(text)
    assert len(caps) >= 2
    for cap in caps:
        assert "\n" not in cap
        assert len(cap) <= 32


def test_hard_wraps_without_punctuation():
    text = "这是一段完全没有标点的超长中文用来确认硬切不会挤在同一屏上面继续往下写"
    caps = split_zh_captions(text)
    assert len(caps) >= 2
    assert "".join(caps) == text
    for cap in caps:
        assert "\n" not in cap
        assert len(cap) <= 32


def test_screenshot_sentence_stays_one_line():
    text = "项目宪章只是形式化这些项目级细节的一种方式。"
    assert split_zh_captions(text) == [text]


def test_caption_spans_are_proportional():
    caps = ["短", "这是比较长的一句"]
    spans = _caption_spans(caps, 10.0, 20.0)
    assert spans[0][0] == 10.0
    assert abs(spans[-1][1] - 20.0) < 1e-6
    d0 = spans[0][1] - spans[0][0]
    d1 = spans[1][1] - spans[1][0]
    assert d1 > d0


def test_zh_source_fills_english_sidecar(tmp_path):
    from dub_pipeline import build_segments_from_subtitles, generate_srt

    video = tmp_path / "00-introduction.mp4"
    video.write_bytes(b"")
    (tmp_path / "00-introduction.zh.srt").write_text(
        "1\n00:00:00,000 --> 00:00:02,000\n大家好\n\n",
        encoding="utf-8",
    )
    (tmp_path / "00-introduction.en.srt").write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nHello everyone\n\n",
        encoding="utf-8",
    )
    # leftover empty sidecar must not be used as the English source
    (tmp_path / "00-introduction_cn_en.srt").write_text("", encoding="utf-8")
    segs, source = build_segments_from_subtitles(str(video), "vocals.wav", num_speakers=1)
    assert source.startswith("zh:")
    assert segs[0]["zh_text"] == "大家好"
    assert segs[0]["text"] == "Hello everyone"
    en_out = tmp_path / "00-introduction_cn_en.srt"
    generate_srt(segs, str(en_out), lang="en")
    body = en_out.read_text(encoding="utf-8")
    assert en_out.stat().st_size > 0
    assert "Hello everyone" in body


def test_generate_srt_splits_zh_keeps_en(tmp_path):
    segs = [{
        "start": 0.0, "end": 10.0, "new_start": 0.0, "aligned_duration": 10.0,
        "text": "Hello everyone this is a long English line",
        "zh_text": "大家好，欢迎来到今天的演讲，我们一起来看看后面会发生什么。接下来还有更多内容需要慢慢展开说明。",
    }]
    zh_path = tmp_path / "out.srt"
    en_path = tmp_path / "out_en.srt"
    generate_srt(segs, str(zh_path), lang="zh")
    generate_srt(segs, str(en_path), lang="en")
    zh = zh_path.read_text(encoding="utf-8")
    en = en_path.read_text(encoding="utf-8")
    assert zh.count("-->") >= 2
    assert en.count("-->") == 1
    assert "Hello everyone" in en
    assert "<" not in zh
