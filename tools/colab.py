"""Colab helpers for launching IndexTTS WebUI with a public URL."""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-linux-amd64"
)
CLOUDFLARED_PATH = Path("/tmp/cloudflared")
CLOUDFLARE_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
GRADIO_SHARE_RE = re.compile(r"https://[a-zA-Z0-9-]+\.gradio\.live")

_webui_proc: subprocess.Popen[str] | None = None
_webui_port: int | None = None
_tunnel_proc: subprocess.Popen[str] | None = None


def is_colab() -> bool:
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION"):
        return True
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_model_dir(model_dir: str | None) -> str:
    if model_dir:
        return model_dir
    return os.environ.get("INDEX_TTS_MODEL_DIR", "./checkpoints")


def wants_cloudflare(cloudflare: bool | None) -> bool:
    if cloudflare is not None:
        return bool(cloudflare)
    return is_colab()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_webui_argv(
    *,
    port: int,
    version: str,
    model_dir: str,
    fp16: bool,
    share: bool = False,
    host: str = "127.0.0.1",
) -> list[str]:
    argv = [
        sys.executable,
        str(repo_root() / "webui.py"),
        "--host",
        host,
        "--port",
        str(port),
        "--version",
        version,
        "--model_dir",
        model_dir,
    ]
    if fp16:
        argv.append("--fp16")
    if share:
        argv.append("--share")
    return argv


def parse_cloudflare_url(text: str) -> str | None:
    match = CLOUDFLARE_URL_RE.search(text)
    return match.group(0) if match else None


def parse_gradio_share_url(text: str) -> str | None:
    match = GRADIO_SHARE_RE.search(text)
    return match.group(0) if match else None


def _http_get(url: str, timeout: float) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return None


def _tcp_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


def is_webui_healthy(port: int, timeout: float = 2.0) -> bool:
    body = _http_get(f"http://127.0.0.1:{port}/", timeout)
    if body is None:
        return False
    if "gradio" in body.lower():
        return True
    owned = (
        _webui_proc is not None
        and _webui_port == port
        and _webui_proc.poll() is None
    )
    return bool(owned)


def port_busy_with_foreign_process(port: int) -> bool:
    if not _tcp_open(port):
        return False
    return not is_webui_healthy(port)
