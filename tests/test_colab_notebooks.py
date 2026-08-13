import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _nb_text(name: str) -> str:
    data = json.loads((ROOT / name).read_text(encoding="utf-8"))
    parts: list[str] = []
    for cell in data["cells"]:
        src = cell.get("source", "")
        parts.append("".join(src) if isinstance(src, list) else src)
    return "\n".join(parts)


def test_main_notebook_is_webui_path():
    text = _nb_text("IndexTTS2_Colab.ipynb")
    assert "from tools.colab import start" in text
    assert "index-tts-cache" in text
    assert "IndexTeam/IndexTTS-2.5" in text
    assert "INDEX_TTS_MODEL_DIR" in text
    assert "dub_video" not in text
    assert "IndexTeam/IndexTTS-2\n" not in text
    assert "IndexTeam/IndexTTS-2 " not in text
    assert 'IndexTeam/IndexTTS-2"' not in text
    assert "numpy<2.0" not in text
    assert "pip uninstall -y torch" not in text
    assert "Path(LOCAL_FALLBACK).mkdir" not in text
    assert "os.makedirs(LOCAL_FALLBACK" not in text
    assert 'raise SystemExit("pip install failed")' in text
    assert "_exit_code" in text
    assert "setup_colab.sh" in text


def test_pipeline_notebooks_share_cache_and_v25():
    for name in (
        "DubbingPipeline_Colab.ipynb",
        "HighlightPipeline_Colab.ipynb",
        "IntentPipeline_Colab.ipynb",
    ):
        text = _nb_text(name)
        assert "index-tts-cache" in text, name
        assert "numpy<2.0" not in text, name
        assert "IndexTeam/IndexTTS-2.5" in text, name
        assert "models_cache" not in text, name


def test_voxcpm_notebook_cache_root_only():
    text = _nb_text("VoxCPM_DubbingPipeline_Colab.ipynb")
    assert "index-tts-cache" in text
    assert "numpy<2.0" not in text
    assert "models_cache" not in text
