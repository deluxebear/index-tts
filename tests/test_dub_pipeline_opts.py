import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dub_pipeline import (
    dub_batch,
    dub_video,
    lookup_num_speakers,
    parse_num_speakers_map,
    transcribe_and_diarize,
)


def test_dub_video_accepts_new_speed_kwargs():
    sig = inspect.signature(dub_video)
    assert "whisper_model" in sig.parameters
    assert "audio_only_align" in sig.parameters
    assert "external_subs" in sig.parameters


def test_single_speaker_skips_diarization():
    fake_result = {
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.2, "end": 2.0, "text": "world"},
        ],
    }
    whisperx = MagicMock()
    whisperx.load_model.return_value.transcribe.return_value = fake_result
    whisperx.load_align_model.return_value = (MagicMock(), {})
    whisperx.align.return_value = fake_result

    with patch.dict("sys.modules", {"whisperx": whisperx}):
        segs = transcribe_and_diarize(
            "vocals.wav", hf_token=None, num_speakers=1, whisper_model="large-v3-turbo",
        )

    whisperx.load_model.assert_called_once()
    assert whisperx.load_model.call_args.args[0] == "large-v3-turbo"
    assert not hasattr(whisperx, "diarize") or whisperx.diarize.DiarizationPipeline.call_count == 0
    assert all(s["speaker"] == "SPEAKER_00" for s in segs)
    assert len(segs) == 2


def test_parse_num_speakers_map_accepts_dict_and_strings():
    assert parse_num_speakers_map(None) == {}
    assert parse_num_speakers_map({"talk.mp4": 1, "qna": 3}) == {"talk.mp4": 1, "qna": 3}
    assert parse_num_speakers_map("talk.mp4=1,qna:3") == {"talk.mp4": 1, "qna": 3}
    assert parse_num_speakers_map(["interviews/qna.mp4=4"]) == {"interviews/qna.mp4": 4}
    with pytest.raises(ValueError):
        parse_num_speakers_map("talk")


def test_lookup_num_speakers_prefers_specific_keys(tmp_path):
    root = tmp_path / "in"
    video = root / "interviews" / "qna.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    overrides = {"qna.mp4": 3, "interviews/qna.mp4": 4, "talk": 1}
    assert lookup_num_speakers(video, root, default=2, overrides=overrides) == 4
    assert lookup_num_speakers(root / "talk.mp4", root, default=2, overrides=overrides) == 1
    assert lookup_num_speakers(root / "other.mp4", root, default=2, overrides=overrides) == 2


def test_dub_batch_passes_per_video_speaker_override(tmp_path, monkeypatch):
    root = tmp_path / "in"
    root.mkdir()
    (root / "solo.mp4").write_bytes(b"x")
    (root / "panel.mp4").write_bytes(b"x")
    seen = []

    def _fake_dub(video_path, output_path, tts=None, work_dir=None, **kwargs):
        seen.append((Path(video_path).name, kwargs.get("num_speakers")))
        return output_path

    monkeypatch.setattr("dub_pipeline._init_tts", lambda *a, **k: object())
    monkeypatch.setattr("dub_pipeline.dub_video", _fake_dub)
    dub_batch(
        str(root),
        output_dir=str(tmp_path / "out"),
        num_speakers=2,
        num_speakers_map={"solo.mp4": 1},
    )
    assert ("solo.mp4", 1) in seen
    assert ("panel.mp4", 2) in seen
