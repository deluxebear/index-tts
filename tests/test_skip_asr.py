from unittest.mock import patch

from dub_pipeline import (
    dummy_single_speaker_segments,
    load_cleaned_subtitle_cues,
    should_skip_asr,
)


def _write_srt(path, text="大家好"):
    path.write_text(
        f"1\n00:00:00,000 --> 00:00:02,000\n{text}\n\n",
        encoding="utf-8",
    )


def test_should_skip_asr_single_speaker_with_cleaned_subs(tmp_path):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"")
    _write_srt(tmp_path / "talk.zh.srt")
    assert should_skip_asr(1, str(video)) is True
    assert should_skip_asr(2, str(video)) is False
    assert should_skip_asr(None, str(video)) is False
    assert should_skip_asr(1, str(video), no_external_subs=True) is False


def test_should_not_skip_without_usable_subs(tmp_path):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"")
    assert should_skip_asr(1, str(video)) is False
    _write_srt(tmp_path / "talk.zh.srt", "翻译：张三")
    assert should_skip_asr(1, str(video)) is False


def test_load_cleaned_subtitle_cues(tmp_path):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"")
    _write_srt(tmp_path / "talk.en.srt", "Chris Anderson: Hello")
    cues, path, lang = load_cleaned_subtitle_cues(str(video))
    assert lang == "en"
    assert path.endswith("talk.en.srt")
    assert cues[0]["text"] == "Hello"


def test_dummy_single_speaker_clips_ref(tmp_path):
    with patch("dub_pipeline.get_audio_duration", return_value=120.0):
        segs = dummy_single_speaker_segments("vocals.wav")
    assert len(segs) == 1
    assert segs[0]["speaker"] == "SPEAKER_00"
    assert segs[0]["start"] == 0.0
    assert segs[0]["end"] == 8.0
    assert segs[0]["text"] == ""
