import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from dub_pipeline import (
    _parse_translation_response,
    _split_zh_and_emotion,
    assign_speakers_to_cues,
    cues_to_segments,
    estimate_emotion_from_text,
    find_loudest_window,
    generate_speech,
    peek_speaker_label,
    tts_duration_factor,
)


def test_keep_finished_checkpoint_helper_exists():
    import inspect
    from dub_pipeline import dub_video
    src = inspect.getsource(dub_video)
    assert "_clear_checkpoint(video_work_dir)" in src
    assert "if cleanup:" in src
    # finished runs must save step 10 before optionally clearing
    assert "_save_checkpoint(video_work_dir, 10" in src


def test_peek_and_assign_speakers_from_labels():
    assert peek_speaker_label("Chris Anderson: Hello") == "Chris Anderson"
    assert peek_speaker_label("BS: I agree") == "BS"
    assert peek_speaker_label("主持人：下一个问题") == "主持人"
    assert peek_speaker_label("张三：我觉得可以") == "张三"
    assert peek_speaker_label("所以：我们继续") is None
    cues = [
        {"start": 0.0, "end": 1.0, "text": "Hello", "speaker_label": "Chris Anderson"},
        {"start": 1.0, "end": 2.0, "text": "Hi", "speaker_label": "BS"},
        {"start": 2.0, "end": 3.0, "text": "ok", "speaker_label": "Chris Anderson"},
    ]
    out = assign_speakers_to_cues(cues, vocals_path=None, num_speakers=2, hf_token=None)
    assert out[0]["speaker"] == "SPEAKER_00"
    assert out[1]["speaker"] == "SPEAKER_01"
    assert out[2]["speaker"] == "SPEAKER_00"


def test_cues_to_segments_zh_and_en():
    cues = [{"start": 0.0, "end": 1.0, "text": "大家好", "speaker": "SPEAKER_00"}]
    zh = cues_to_segments(cues, "zh")
    assert zh[0]["zh_text"] == "大家好"
    assert zh[0]["text"] == ""
    en = cues_to_segments(cues, "en")
    assert en[0]["text"] == "大家好"
    assert "zh_text" not in en[0]


def test_find_loudest_window(tmp_path):
    sr = 8000
    audio = np.zeros(sr * 4, dtype=np.float32)
    audio[sr * 2 : sr * 2 + sr] = 0.8  # loud second at t=2s
    path = tmp_path / "v.wav"
    sf.write(path, audio, sr)
    start, end = find_loudest_window(str(path), window_sec=1.0, hop_sec=0.05)
    assert 1.6 <= start <= 2.2
    assert 0.9 <= (end - start) <= 1.1


def test_duration_factor_and_emotion_parse():
    factor = tts_duration_factor("这是一句十个汉字对吧啊", 1.0)
    assert 0.5 <= factor <= 2.0
    assert tts_duration_factor("", 1.0) == 1.0
    zh, emo = _split_zh_and_emotion("大家好 || 0.2,0,0,0,0,0,0,0.4")
    assert zh == "大家好"
    assert emo[0] == 0.2
    assert emo[7] == 0.4
    parsed, emos = _parse_translation_response("#0 大家好 || 0.2,0,0,0,0,0,0,0.4\n#1 谢谢")
    assert parsed[0] == "大家好"
    assert parsed[1] == "谢谢"
    assert emos[0][0] == 0.2
    assert 1 not in emos
    assert estimate_emotion_from_text("哈哈太棒了！！")[0] > 0
    assert estimate_emotion_from_text("今天天气不错") is None


def test_generate_speech_passes_duration_and_emotion(tmp_path, monkeypatch):
    recorded = []

    class FakeTTS:
        def infer(self, **kwargs):
            recorded.append(kwargs)
            Path(kwargs["output_path"]).write_bytes(b"")

    monkeypatch.setattr("dub_pipeline.get_ref_for_segment", lambda *a, **k: "ref.wav")
    monkeypatch.setattr("dub_pipeline.get_audio_duration", lambda p: 1.0)
    monkeypatch.setattr("dub_pipeline.annotate_tts_text", lambda t, **k: t)

    segs = [
        {"zh_text": "哈哈太棒了", "start": 0.0, "end": 2.0, "speaker": "A",
         "emo_vector": [0.3, 0, 0, 0, 0, 0, 0, 0.2]},
        {"zh_text": "下一句", "start": 2.0, "end": 3.0, "speaker": "A"},
    ]
    ckpt_calls = []
    generate_speech(
        segs, {}, "vocals.wav", str(tmp_path), FakeTTS(),
        checkpoint_cb=ckpt_calls.append, checkpoint_every=1,
    )
    assert recorded[0]["lang"] == "zh"
    assert 0.5 <= recorded[0]["duration_factor"] <= 2.0
    assert recorded[0]["emo_vector"][0] == 0.3
    assert len(ckpt_calls) >= 1
