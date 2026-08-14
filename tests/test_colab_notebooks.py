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
        "NovelPipeline_Colab.ipynb",
    ):
        text = _nb_text(name)
        assert "index-tts-cache" in text, name
        assert "numpy<2.0" not in text, name
        assert "IndexTeam/IndexTTS-2.5" in text, name
        assert "models_cache" not in text, name


def test_dubbing_notebook_uses_drive_work_dir():
    text = _nb_text("DubbingPipeline_Colab.ipynb")
    assert 'WORK_DIR = f"{DRIVE_CACHE}/dub_workspace"' in text
    assert 'os.listdir(WORK_DIR)' in text
    assert 'os.listdir("dub_workspace")' not in text
    assert "list_suspicious_segments" in text
    assert "跳过 Whisper" in text
    assert "{DRIVE_CACHE}/output/" in text
    assert "/content/{stem}_cn.mp4" not in text
    assert "FORCE_UPLOAD" in text
    assert "except Exception:" in text
    assert "NUM_SPEAKERS_FOR" in text
    assert "num_speakers_map" in text


def test_novel_notebook_uses_drive_work_dir():
    text = _nb_text("NovelPipeline_Colab.ipynb")
    assert 'WORK_DIR = f"{DRIVE_CACHE}/novel_workspace"' in text
    assert "run_novel_pipeline" in text
    assert "from novel_pipeline import run_novel_pipeline" in text
    assert "IndexTeam/IndexTTS-2.5" in text
    assert "setup_colab.sh --extra novel" in text
    assert "{DRIVE_CACHE}/output/" in text
    assert "FORCE_UPLOAD" in text
    assert "except Exception:" in text
    assert "userdata.get('LLM_API_KEY')" in text
    assert "os.listdir(WORK_DIR)" in text
    assert 'os.listdir("novel_workspace")' not in text
    assert "numpy<2.0" not in text
    assert "models_cache" not in text
    assert "HF_TOKEN" not in text
    assert "whisperx" not in text
    assert "cleanup=CLEANUP" in text
    assert "ref_mode=REF_MODE" in text
    assert "force_tts=True" in text
    assert "REF_AUDIO =" in text
    assert "examples/voice_05.wav" in text
    assert "BATCH_REF  = ref_audio" not in text


def test_voxcpm_notebook_cache_root_only():
    text = _nb_text("VoxCPM_DubbingPipeline_Colab.ipynb")
    assert "index-tts-cache" in text
    assert "numpy<2.0" not in text
    assert "models_cache" not in text
