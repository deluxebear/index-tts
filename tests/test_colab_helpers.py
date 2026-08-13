import os
from unittest.mock import MagicMock, patch

import pytest

from tools.colab import (
    build_webui_argv,
    is_colab,
    is_webui_healthy,
    parse_cloudflare_url,
    parse_gradio_share_url,
    port_busy_with_foreign_process,
    resolve_model_dir,
    wants_cloudflare,
)
import tools.colab as colab


def test_parse_cloudflare_url_extracts_https_host():
    text = "2026-08-13 INF | https://abc-def.trycloudflare.com\n"
    assert parse_cloudflare_url(text) == "https://abc-def.trycloudflare.com"


def test_parse_cloudflare_url_returns_none_when_missing():
    assert parse_cloudflare_url("no tunnel yet") is None


def test_parse_gradio_share_url():
    text = "Running on public URL: https://deadbeef.gradio.live\n"
    assert parse_gradio_share_url(text) == "https://deadbeef.gradio.live"


def test_resolve_model_dir_prefers_argument(monkeypatch):
    monkeypatch.setenv("INDEX_TTS_MODEL_DIR", "/from-env")
    assert resolve_model_dir("/explicit") == "/explicit"


def test_resolve_model_dir_uses_env_then_default(monkeypatch):
    monkeypatch.setenv("INDEX_TTS_MODEL_DIR", "/from-env")
    assert resolve_model_dir(None) == "/from-env"
    monkeypatch.delenv("INDEX_TTS_MODEL_DIR", raising=False)
    assert resolve_model_dir(None) == "./checkpoints"


def test_build_webui_argv_includes_fp16_and_optional_share():
    argv = build_webui_argv(
        port=7860, version="2.5", model_dir="/ckpt", fp16=True, share=False
    )
    assert argv[1].endswith("webui.py")
    assert argv[argv.index("--port") + 1] == "7860"
    assert argv[argv.index("--version") + 1] == "2.5"
    assert argv[argv.index("--model_dir") + 1] == "/ckpt"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert "--fp16" in argv
    assert "--share" not in argv
    argv_share = build_webui_argv(
        port=7861, version="2", model_dir="/ckpt", fp16=False, share=True
    )
    assert "--fp16" not in argv_share
    assert "--share" in argv_share
    assert argv_share[argv_share.index("--port") + 1] == "7861"


def test_is_colab_true_when_env_set(monkeypatch):
    monkeypatch.setenv("COLAB_RELEASE_TAG", "test")
    assert is_colab() is True


def test_is_colab_false_without_env_or_module(monkeypatch):
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.delenv("COLAB_BACKEND_VERSION", raising=False)
    import builtins
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "google.colab" or name.startswith("google.colab"):
            raise ImportError("no colab")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    assert is_colab() is False


def test_wants_cloudflare_none_follows_is_colab():
    with patch("tools.colab.is_colab", return_value=True):
        assert wants_cloudflare(None) is True
    with patch("tools.colab.is_colab", return_value=False):
        assert wants_cloudflare(None) is False
    assert wants_cloudflare(True) is True
    assert wants_cloudflare(False) is False


def test_is_webui_healthy_requires_gradio_in_body():
    colab._webui_proc = None
    colab._webui_port = None
    with patch("tools.colab._http_get", return_value="<html>gradio app</html>"):
        assert is_webui_healthy(7860) is True
    with patch("tools.colab._http_get", return_value="<html>nginx</html>"):
        assert is_webui_healthy(7860) is False
    with patch("tools.colab._http_get", return_value=None):
        assert is_webui_healthy(7860) is False


def test_is_webui_healthy_true_when_owned_proc_answers():
    proc = MagicMock()
    proc.poll.return_value = None
    colab._webui_proc = proc
    colab._webui_port = 7860
    try:
        with patch("tools.colab._http_get", return_value="<html>loading</html>"):
            assert is_webui_healthy(7860) is True
        with patch("tools.colab._http_get", return_value=None):
            assert is_webui_healthy(7860) is False
    finally:
        colab._webui_proc = None
        colab._webui_port = None


def test_port_busy_with_foreign_process():
    colab._webui_proc = None
    colab._webui_port = None
    with patch("tools.colab._tcp_open", return_value=False):
        assert port_busy_with_foreign_process(7860) is False
    with patch("tools.colab._tcp_open", return_value=True), patch(
        "tools.colab.is_webui_healthy", return_value=True
    ):
        assert port_busy_with_foreign_process(7860) is False
    with patch("tools.colab._tcp_open", return_value=True), patch(
        "tools.colab.is_webui_healthy", return_value=False
    ):
        assert port_busy_with_foreign_process(7860) is True
