from pathlib import Path


def _webui_source() -> str:
    webui_path = Path(__file__).resolve().parents[1] / "webui.py"
    return webui_path.read_text(encoding="utf-8")


def test_webui_python_source_parses():
    source = _webui_source()
    assert compile(source, "webui.py", "exec") is not None


def test_webui_declares_share_flag():
    source = _webui_source()
    assert '"--share"' in source or "'--share'" in source


def test_webui_launch_passes_share():
    source = _webui_source()
    assert "share=cmd_args.share" in source
