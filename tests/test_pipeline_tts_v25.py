import sys
import types
from pathlib import Path

from dub_pipeline import _init_tts as dub_init_tts
from dub_pipeline import generate_speech
from highlight_pipeline import _init_tts as highlight_init_tts
from highlight_pipeline import generate_narration
from novel_pipeline import synthesize_chapter


def _install_fake_v25(monkeypatch):
    captured = {}

    class FakeIndexTTS2:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake = types.ModuleType("indextts.infer_v2_5")
    fake.IndexTTS2 = FakeIndexTTS2
    monkeypatch.setitem(sys.modules, "indextts.infer_v2_5", fake)
    return captured


def test_dub_init_tts_uses_v25_bf16(monkeypatch):
    captured = _install_fake_v25(monkeypatch)
    imported = []
    real_import = __import__

    def _spy(name, *args, **kwargs):
        imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _spy)
    dub_init_tts("/ckpt", True)
    assert captured.get("use_bf16") is True
    assert "use_fp16" not in captured
    assert captured.get("model_dir") == "/ckpt"
    assert captured.get("cfg_path") == "/ckpt/config.yaml"
    assert "indextts.infer_v2" not in imported
    assert "indextts.infer_v2_5" in imported


def test_highlight_init_tts_uses_v25_bf16(monkeypatch):
    captured = _install_fake_v25(monkeypatch)
    imported = []
    real_import = __import__

    def _spy(name, *args, **kwargs):
        imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _spy)
    highlight_init_tts("/ckpt", True)
    assert captured.get("use_bf16") is True
    assert "use_fp16" not in captured
    assert "indextts.infer_v2" not in imported
    assert "indextts.infer_v2_5" in imported


def test_generate_speech_passes_lang_zh(tmp_path, monkeypatch):
    recorded = {}

    class FakeTTS:
        def infer(self, **kwargs):
            recorded.update(kwargs)

    monkeypatch.setattr("dub_pipeline.get_ref_for_segment", lambda *a, **k: "ref.wav")
    monkeypatch.setattr("dub_pipeline.get_audio_duration", lambda p: 1.0)

    segments = [{"zh_text": "你好", "start": 0.0, "end": 1.0, "speaker": "A"}]
    generate_speech(segments, {}, "vocals.wav", str(tmp_path), FakeTTS())
    assert recorded.get("lang") == "zh"


def test_generate_narration_passes_lang_zh(tmp_path, monkeypatch):
    recorded = {}

    class FakeTTS:
        def normalize_emo_vec(self, vec):
            return vec

        def infer(self, **kwargs):
            recorded.update(kwargs)

    monkeypatch.setattr("highlight_pipeline._init_tts", lambda *a, **k: FakeTTS())
    monkeypatch.setattr("highlight_pipeline.get_audio_duration", lambda p: 1.0)
    monkeypatch.setattr("highlight_pipeline._free_vram", lambda: None)

    script = [{"narration": "你好", "emo_vector": None}]
    generate_narration(script, "ref.wav", str(tmp_path))
    assert recorded.get("lang") == "zh"


def test_novel_synthesize_uses_lang_zh(tmp_path, monkeypatch):
    recorded = {}

    class FakeTTS:
        def normalize_emo_vec(self, vec):
            return vec

        def infer(self, **kwargs):
            recorded.update(kwargs)
            p = kwargs["output_path"]
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            Path(p).write_bytes(b"RIFF")

    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    utts = [{
        "chapter_id": "c01", "seq": 0, "speaker_id": "narrator",
        "tts_text": "你好", "lang": "zh", "emo_vector": [0] * 8,
        "duration_factor": 1.0, "silence_after_ms": 200,
    }]
    synthesize_chapter(utts, {"narrator": {"ref_wav": str(ref)}}, FakeTTS(), str(tmp_path))
    assert recorded.get("lang") == "zh"
