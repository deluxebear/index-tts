from unittest.mock import patch

import numpy as np
import soundfile as sf

from dub_pipeline import (
    _looks_like_untranslated_english,
    _split_zh_and_emotion,
    deoverlap_same_speaker_segments,
    find_loudest_window,
    subtitle_has_usable_speaker_labels,
    translation_needs_llm,
)


def test_deoverlap_merges_same_speaker_keeps_crosstalk():
    segs = [
        {"start": 10.0, "end": 13.5, "text": "First", "zh_text": "第一句", "speaker": "SPEAKER_00"},
        {"start": 12.0, "end": 15.0, "text": "Second", "zh_text": "第二句", "speaker": "SPEAKER_00"},
        {"start": 12.2, "end": 14.0, "text": "Hi", "zh_text": "你好", "speaker": "SPEAKER_01"},
    ]
    out = deoverlap_same_speaker_segments(segs)
    same = [s for s in out if s["speaker"] == "SPEAKER_00"]
    other = [s for s in out if s["speaker"] == "SPEAKER_01"]
    assert len(same) == 1
    assert same[0]["start"] == 10.0
    assert same[0]["end"] == 15.0
    assert "第一句" in same[0]["zh_text"] and "第二句" in same[0]["zh_text"]
    assert same[0]["start"] < same[0]["end"]
    assert len(other) == 1
    assert other[0]["start"] == 12.2
    assert other[0]["end"] == 14.0
    # same-speaker windows no longer overlap each other
    assert same[0]["end"] <= 15.0
    assert not (same[0]["start"] < 12.0 < same[0]["end"] and same[0]["zh_text"] == "第一句")


def test_reject_english_translation_and_strip_bad_emo():
    assert _looks_like_untranslated_english("Hello everyone")
    assert not _looks_like_untranslated_english("大家好")
    assert not _looks_like_untranslated_english("我们用 ChatGPT")
    zh, emo = _split_zh_and_emotion("大家好 || 0.1,0,0,0,0,0,0,0.4")
    assert zh == "大家好" and emo[0] == 0.1
    zh, emo = _split_zh_and_emotion("大家好 || 开心")
    assert zh == "大家好" and emo is None


def test_loudest_window_stays_inside_speaker_range(tmp_path):
    sr = 8000
    audio = np.zeros(sr * 6, dtype=np.float32)
    audio[0:sr] = 0.9  # loud but outside the speaker range
    audio[sr * 3: sr * 3 + sr // 2] = 0.2
    path = tmp_path / "v.wav"
    sf.write(path, audio, sr)
    start, end = find_loudest_window(
        str(path), window_sec=8.0, ranges=[(3.0, 3.5)], hop_sec=0.05,
    )
    assert start >= 2.99
    assert end <= 3.51


def test_translation_needs_llm_and_labels(tmp_path):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"")
    (tmp_path / "talk.zh.srt").write_text(
        "1\n00:00:00,000 --> 00:00:02,000\n大家好\n\n",
        encoding="utf-8",
    )
    assert translation_needs_llm(str(video)) is False
    assert subtitle_has_usable_speaker_labels([
        {"speaker_label": "CA"}, {"speaker_label": "BS"}, {"speaker_label": "CA"},
    ])
    assert not subtitle_has_usable_speaker_labels([
        {"speaker_label": None}, {"speaker_label": None},
    ])
