"""Colab helpers for launching IndexTTS WebUI with a public URL."""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-linux-amd64"
)
CLOUDFLARED_PATH = Path("/tmp/cloudflared")
CLOUDFLARE_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
GRADIO_SHARE_RE = re.compile(r"https://[a-zA-Z0-9-]+\.gradio\.live")

HEALTH_POLL_SECONDS = 20 * 60
TUNNEL_WAIT_SECONDS = 30
KEEPALIVE_INTERVAL = 300

_webui_proc: subprocess.Popen[str] | None = None
_webui_port: int | None = None
_tunnel_proc: subprocess.Popen[str] | None = None
_webui_log: deque[str] = deque(maxlen=200)


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


def setup(*, extra: str = "", verbose: bool = False) -> None:
    """Run tools/setup_colab.sh (Colab torch stack + optional pipeline extras)."""
    cmd = ["bash", str(repo_root() / "tools" / "setup_colab.sh")]
    if extra:
        cmd.extend(["--extra", extra])
    if verbose:
        cmd.append("--verbose")
    code = subprocess.call(cmd)
    if code != 0:
        raise RuntimeError(f"tools/setup_colab.sh failed with exit code {code}")


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
        "-u",
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


def ready_card_html(url: str, *, kind: str) -> str:
    notes = {
        "cloudflare": "This Cloudflare HTTPS link works from any device.",
        "gradio": "This Gradio share link works from any device. It expires when the runtime stops.",
        "local": "Local only — this URL will not open from outside this runtime.",
    }
    kind_label = {"cloudflare": "Cloudflare", "gradio": "Gradio", "local": "Local"}.get(
        kind, kind
    )
    note = notes.get(kind, "")
    return f"""
    <div style="display:inline-block;padding:20px;background:#fff;border:2px solid #000;
                border-radius:12px;margin:10px 0;font-family:system-ui,-apple-system,sans-serif;">
        <h2 style="margin:0 0 12px 0;font-size:24px;font-weight:800;">IndexTTS WebUI is ready</h2>
        <p style="margin:0 0 12px 0;font-size:14px;font-weight:700;">{note} ({kind_label})</p>
        <a href="{url}" onclick="var w=window.open(this.href,'_blank');if(!w){{return true;}}return false;"
           style="display:inline-flex;align-items:center;gap:8px;padding:12px 24px;
                  background:#000;color:#fff;text-decoration:none;border-radius:8px;
                  font-weight:800;font-size:16px;">Open IndexTTS WebUI</a>
        <p style="margin:16px 0 0 0;font-size:13px;font-family:monospace;font-weight:700;">{url}</p>
    </div>
    """


def show_ready_card(url: str, *, kind: str) -> None:
    html = ready_card_html(url, kind=kind)
    try:
        from IPython.display import HTML, display

        display(HTML(html))
    except Exception:
        print(f"IndexTTS WebUI is ready ({kind}): {url}")


def ensure_cloudflared() -> Path:
    if CLOUDFLARED_PATH.is_file() and os.access(CLOUDFLARED_PATH, os.X_OK):
        return CLOUDFLARED_PATH
    urllib.request.urlretrieve(CLOUDFLARED_URL, CLOUDFLARED_PATH)
    CLOUDFLARED_PATH.chmod(0o755)
    return CLOUDFLARED_PATH


def _stop_cloudflare_tunnel() -> None:
    global _tunnel_proc
    proc = _tunnel_proc
    _tunnel_proc = None
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def start_cloudflare_tunnel(port: int) -> str | None:
    global _tunnel_proc
    _stop_cloudflare_tunnel()
    try:
        binary = ensure_cloudflared()
    except OSError as exc:
        print(f"Could not download cloudflared ({exc})")
        return None
    try:
        proc = subprocess.Popen(
            [str(binary), "tunnel", "--url", f"http://127.0.0.1:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        print(f"Could not start cloudflared ({exc})")
        return None
    _tunnel_proc = proc
    found: list[str] = []

    def _pump() -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            print(line, end="")
            url = parse_cloudflare_url(line)
            if url and not found:
                found.append(url)

    thread = threading.Thread(target=_pump, daemon=True)
    thread.start()
    deadline = time.time() + TUNNEL_WAIT_SECONDS
    while time.time() < deadline:
        if found:
            return found[0]
        if proc.poll() is not None:
            break
        time.sleep(0.2)
    print("cloudflared did not produce a trycloudflare.com URL in time")
    _stop_cloudflare_tunnel()
    return None


def _spawn_webui(argv: list[str]) -> subprocess.Popen[str]:
    global _webui_proc, _webui_port, _webui_log
    _webui_log = deque(maxlen=200)
    proc = subprocess.Popen(
        argv,
        cwd=str(repo_root()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    _webui_proc = proc

    def _pump() -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            print(line, end="")
            _webui_log.append(line)

    threading.Thread(target=_pump, daemon=True).start()
    return proc


def _last_logs(n: int = 80) -> str:
    return "".join(list(_webui_log)[-n:])


def _wait_until_healthy(proc: subprocess.Popen[str], port: int) -> bool:
    deadline = time.time() + HEALTH_POLL_SECONDS
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        if is_webui_healthy(port):
            return True
        time.sleep(2)
    return False


def _terminate_owned_webui() -> None:
    global _webui_proc
    proc = _webui_proc
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=15)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    _webui_proc = None


def _public_url_from_logs() -> str | None:
    return parse_gradio_share_url("".join(_webui_log))


def start(
    port: int = 7860,
    *,
    version: str = "2.5",
    model_dir: str | None = None,
    fp16: bool = True,
    cloudflare: bool | None = None,
    share: bool | None = None,
) -> None:
    global _webui_port
    model_dir = resolve_model_dir(model_dir)
    use_cloudflare = wants_cloudflare(cloudflare)
    force_share = share is True
    allow_share_fallback = share is not False

    if port_busy_with_foreign_process(port):
        print(
            f"Port {port} is in use by another process. "
            f"Try start(port={port + 1}) instead. Not killing the other process."
        )
        return

    owned_live = (
        _webui_proc is not None
        and _webui_port == port
        and _webui_proc.poll() is None
    )
    if not owned_live and not is_webui_healthy(port):
        argv = build_webui_argv(
            port=port, version=version, model_dir=model_dir, fp16=fp16, share=force_share
        )
        print("Starting IndexTTS WebUI:", " ".join(argv))
        proc = _spawn_webui(argv)
        _webui_port = port
        if not _wait_until_healthy(proc, port):
            print("WebUI failed to become healthy. Last logs:")
            print(_last_logs(80))
            return
    else:
        print(f"Reusing existing WebUI on port {port}")
        _webui_port = port

    public_url: str | None = None
    kind = "local"

    if force_share:
        deadline = time.time() + 60
        while time.time() < deadline:
            public_url = _public_url_from_logs()
            if public_url:
                break
            time.sleep(1)
        kind = "gradio" if public_url else "local"
    elif use_cloudflare:
        public_url = start_cloudflare_tunnel(port)
        if public_url:
            kind = "cloudflare"
        elif allow_share_fallback and _webui_proc is not None:
            print("Cloudflare tunnel failed; restarting WebUI with --share")
            _stop_cloudflare_tunnel()
            _terminate_owned_webui()
            argv = build_webui_argv(
                port=port, version=version, model_dir=model_dir, fp16=fp16, share=True
            )
            proc = _spawn_webui(argv)
            _webui_port = port
            if not _wait_until_healthy(proc, port):
                print("WebUI --share restart failed. Last logs:")
                print(_last_logs(80))
                public_url = None
                kind = "local"
            else:
                deadline = time.time() + 60
                while time.time() < deadline and public_url is None:
                    public_url = _public_url_from_logs()
                    if public_url:
                        break
                    time.sleep(1)
                kind = "gradio" if public_url else "local"
        elif allow_share_fallback and _webui_proc is None:
            print(
                "Cloudflare failed and this WebUI was not started by start(), "
                "so --share fallback cannot restart it. Using the local URL."
            )
            kind = "local"

    if public_url is None:
        public_url = f"http://127.0.0.1:{port}"
        kind = "local"
        print(
            "No public URL available. The local address will not open from "
            "outside this runtime."
        )

    show_ready_card(public_url, kind=kind)
    print("Keepalive on — interrupt the cell to stop the tunnel (WebUI stays up).")
    try:
        while True:
            time.sleep(KEEPALIVE_INTERVAL)
            print(".", end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopping Cloudflare tunnel (WebUI process left running).")
    finally:
        _stop_cloudflare_tunnel()
