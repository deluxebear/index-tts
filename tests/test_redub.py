import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import soundfile as sf

from dub_pipeline import list_dub_segments, parse_redub_file, redub_segments


def _wav(path, seconds=0.4):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(int(22050 * seconds), dtype=np.float32), 22050)
    return str(path)


def _ckpt(work, video_stem, n=3):
    vwd = Path(work) / video_stem
    vwd.mkdir(parents=True)
    vocals = _wav(vwd / "vocals.wav")
    bg = _wav(vwd / "bg.wav")
    video = vwd / "video_only.mp4"
    video.write_bytes(b"not-a-real-mp4")
    video = str(video)
    segs = []
    for i in range(n):
        tts = _wav(vwd / "tts_output" / f"tts_{i:04d}.wav")
        segs.append({
            "start": float(i), "end": float(i) + 0.5,
            "text": f"en{i}", "zh_text": f"中文{i}",
            "speaker": "SPEAKER_00", "wav_path": tts,
        })
    refs = {
        "SPEAKER_00": {
            "fallback": vocals, "best_auto": vocals,
            "embedding": np.ones(192, dtype=np.float32),
        }
    }
    data = {
        "step": 10,
        "segments": segs,
        "paths": {
            "vocals_path": vocals, "bg_path": bg,
            "video_only_path": video, "speaker_refs": refs,
        },
    }
    (vwd / "checkpoint.json").write_text(json.dumps(data, cls=_Enc), encoding="utf-8")
    return vwd


class _Enc(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def test_parse_redub_file(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("# c\n12\t银<行|HANG2>\n18 ChatGPT\n", encoding="utf-8")
    assert parse_redub_file(p) == {12: "银<行|HANG2>", 18: "ChatGPT"}


def test_list_dub_segments(tmp_path, capsys):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"x")
    _ckpt(tmp_path / "ws", "talk")
    rows = list_dub_segments(str(video), work_dir=str(tmp_path / "ws"))
    assert len(rows) == 3
    assert "[  0]" in capsys.readouterr().out


def test_redub_only_resynthesizes_selected_id(tmp_path, monkeypatch):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"x")
    vwd = _ckpt(tmp_path / "ws", "talk")
    old = vwd / "tts_output" / "tts_0001.wav"
    old_mtime = old.stat().st_mtime
    called = []

    tts = MagicMock()

    def _infer(**kw):
        called.append(kw["text"])
        _wav(kw["output_path"], 0.3)

    tts.infer.side_effect = lambda **kw: _infer(**kw)

    def _align(segs, work_dir, audio_only_align=False):
        for s in segs:
            s["aligned_path"] = s.get("wav_path")
            s["aligned_duration"] = 0.4
            s["video_slowdown"] = 1.0
        return segs

    monkeypatch.setattr(
        "dub_pipeline.get_ref_for_segment",
        lambda *a, **k: str(vwd / "vocals.wav"),
    )
    monkeypatch.setattr("dub_pipeline.align_durations", _align)
    monkeypatch.setattr("dub_pipeline.assemble_final", lambda *a, **k: None)
    monkeypatch.setattr("dub_pipeline.generate_srt", lambda *a, **k: None)

    redub_segments(
        str(video), [1], texts={1: "银<行|HANG2>"},
        work_dir=str(tmp_path / "ws"), tts=tts,
        output_path=str(tmp_path / "out.mp4"),
    )

    assert called == ["银<行|HANG2>"]
    assert old.exists()
    assert old.stat().st_mtime >= old_mtime
    ckpt = json.loads((vwd / "checkpoint.json").read_text(encoding="utf-8"))
    assert ckpt["segments"][1]["zh_text"] == "银<行|HANG2>"
    assert ckpt["segments"][0]["zh_text"] == "中文0"
