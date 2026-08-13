import inspect
from unittest.mock import MagicMock, patch

from dub_pipeline import dub_video, transcribe_and_diarize


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
