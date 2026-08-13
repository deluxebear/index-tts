from unittest.mock import patch

from dub_pipeline import (
    build_segments_from_subtitles,
    dummy_single_speaker_segments,
    load_cleaned_subtitle_cues,
    should_skip_asr,
)


def _write_srt(path, text="大家好"):
    path.write_text(
        f"1\n00:00:00,000 --> 00:00:02,000\n{text}\n\n",
        encoding="utf-8",
    )


def test_should_skip_asr_when_cleaned_subs_exist(tmp_path):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"")
    _write_srt(tmp_path / "talk.zh.srt")
    assert should_skip_asr(1, str(video)) is True
    assert should_skip_asr(2, str(video)) is True
    assert should_skip_asr(None, str(video)) is True
    assert should_skip_asr(1, str(video), no_external_subs=True) is False


def test_should_not_skip_without_usable_subs(tmp_path):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"")
    assert should_skip_asr(1, str(video)) is False
    _write_srt(tmp_path / "talk.zh.srt", "翻译：张三")
    assert should_skip_asr(1, str(video)) is False


def test_build_segments_from_labeled_subs(tmp_path):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"")
    (tmp_path / "talk.en.srt").write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nChris Anderson: Hello\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\nBS: Thanks\n\n",
        encoding="utf-8",
    )
    segs, source = build_segments_from_subtitles(
        str(video), "vocals.wav", num_speakers=2, hf_token=None,
    )
    assert source.startswith("en:")
    assert [s["speaker"] for s in segs] == ["SPEAKER_00", "SPEAKER_01"]
    assert segs[0]["text"] == "Hello"
    assert "zh_text" not in segs[0]


def test_load_cleaned_subtitle_cues(tmp_path):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"")
    _write_srt(tmp_path / "talk.en.srt", "Chris Anderson: Hello")
    cues, path, lang = load_cleaned_subtitle_cues(str(video))
    assert lang == "en"
    assert path.endswith("talk.en.srt")
    assert cues[0]["text"] == "Hello"


def test_dummy_single_speaker_clips_ref(tmp_path):
    with patch("dub_pipeline.find_loudest_window", return_value=(12.0, 20.0)):
        segs = dummy_single_speaker_segments("vocals.wav")
    assert len(segs) == 1
    assert segs[0]["speaker"] == "SPEAKER_00"
    assert segs[0]["start"] == 12.0
    assert segs[0]["end"] == 20.0
    assert segs[0]["text"] == ""
