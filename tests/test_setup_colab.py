import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "setup_colab.sh"


def test_setup_colab_script_exists_and_parses():
    assert SCRIPT.is_file()
    proc = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_setup_colab_script_installs_matching_torch_stack():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "torchvision" in text
    assert "torchaudio" in text
    assert "download.pytorch.org/whl/cu128" in text
    assert "pip uninstall -y torch" not in text
    assert "tensorflow" in text
    assert "keras" in text
    assert "torchvision.ops import nms" in text
    assert "--extra" in text
    assert "novel" in text
    assert "openai" in text
