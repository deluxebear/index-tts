# Colab WebUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch IndexTTS WebUI on Google Colab with one `start()` call that yields a clickable public URL, and align existing Colab notebooks to Drive-backed IndexTTS-2.5.

**Architecture:** `tools/colab.py` owns Colab detection, `webui.py` subprocess launch, `cloudflared` tunnel, Gradio `--share` fallback, ready card, and keepalive. `webui.py` only gains `--share`. The main notebook is rewritten as the WebUI path; pipeline notebooks only change cache root, drop `numpy<2.0`, and download 2.5.

**Tech Stack:** Python 3.12, Gradio 5, `cloudflared` quick tunnels, Colab notebooks, pytest (no live tunnel in CI).

## Global Constraints

- No Unsloth login, bootstrap password, credential cards, or in-cell iframe.
- Do not change Gradio UI layout or inference code.
- Do not merge dub / highlight / intent into `IndexTTS2_Colab.ipynb`.
- Do not open a Cloudflare tunnel by default outside Colab (`cloudflare=None` → tunnel only when `is_colab()`).
- Do not kill an unrelated process that owns the chosen port.
- Local `python webui.py` stays private unless `--share`.
- Drive cache root is exactly `/content/drive/MyDrive/index-tts-cache/` with `hf_home/`, `torch_home/`, `checkpoints-2.5/`.
- Default model is IndexTTS-2.5 (`IndexTeam/IndexTTS-2.5`).
- Do not `pip uninstall` torch/vision/audio in the main notebook.
- Remove `pip install "numpy<2.0"` from pipeline notebooks.
- Commands use `uv run` for local tests: `PYTHONPATH="$PYTHONPATH:." uv run pytest …`

## File map

| File | Responsibility |
|---|---|
| `webui.py` | Add `--share`; pass to `demo.launch` |
| `tools/__init__.py` | Make `tools` a regular package |
| `tools/colab.py` | `start()` and all helpers |
| `tests/test_colab_helpers.py` | Unit tests (parse, health, argv, detect; mock network) |
| `tests/test_colab_notebooks.py` | Static checks on notebook JSON |
| `tests/test_webui_syntax.py` | Extend: `--share` present, `share=cmd_args.share` |
| `IndexTTS2_Colab.ipynb` | Main WebUI notebook |
| `DubbingPipeline_Colab.ipynb` | Cache + 2.5 + drop numpy pin |
| `HighlightPipeline_Colab.ipynb` | Same |
| `IntentPipeline_Colab.ipynb` | Same |
| `VoxCPM_DubbingPipeline_Colab.ipynb` | Cache root + drop numpy pin only |

---

### Task 1: `webui.py --share`

**Files:**
- Modify: `webui.py` (argparse block ~L19–36 and launch ~L1358–1360)
- Modify: `tests/test_webui_syntax.py`

**Interfaces:**
- Consumes: existing `cmd_args` / `demo.launch`
- Produces: `cmd_args.share: bool` (default `False`); `demo.launch(..., share=cmd_args.share)`

- [ ] **Step 1: Extend the syntax test so it fails**

Replace `tests/test_webui_syntax.py` with:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_webui_syntax.py -v`

Expected: `test_webui_python_source_parses` PASS; the two new tests FAIL (`assert '--share'` / `share=cmd_args.share`).

- [ ] **Step 3: Add `--share` to `webui.py`**

After the `--gui_seg_tokens` argument (currently line 35), insert:

```python
parser.add_argument("--share", action="store_true", default=False, help="Create a public Gradio share link")
```

Replace the launch call:

```python
if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port, share=cmd_args.share)
```

Do not change any other `webui.py` behavior.

- [ ] **Step 4: Re-run tests**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_webui_syntax.py -v`

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add webui.py tests/test_webui_syntax.py
git commit -m "Add --share flag to Gradio WebUI launch"
```

---

### Task 2: Colab helper — pure functions and health checks

**Files:**
- Create: `tools/__init__.py` (empty; `tools/` currently has no package init)
- Create: `tools/colab.py`
- Create: `tests/test_colab_helpers.py`

**Interfaces:**
- Consumes: nothing from Task 1 except that `webui.py` accepts `--share`
- Produces:
  - `is_colab() -> bool`
  - `resolve_model_dir(model_dir: str | None) -> str`
  - `wants_cloudflare(cloudflare: bool | None) -> bool`
  - `repo_root() -> Path`
  - `build_webui_argv(*, port: int, version: str, model_dir: str, fp16: bool, share: bool = False, host: str = "127.0.0.1") -> list[str]`
  - `parse_cloudflare_url(text: str) -> str | None`
  - `parse_gradio_share_url(text: str) -> str | None`
  - `is_webui_healthy(port: int, timeout: float = 2.0) -> bool`
  - `port_busy_with_foreign_process(port: int) -> bool`
  - module state `_webui_proc`, `_webui_port` (set by later `start()`)

- [ ] **Step 1: Write failing tests**

Create `tests/test_colab_helpers.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_colab_helpers.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'tools.colab'` (or import error). There is already a `tools/` package (`tools/i18n/`), so `tools` is importable; only `colab.py` is missing.

- [ ] **Step 3: Implement `tools/colab.py` (helpers only; `start()` comes in Task 3)**

Create empty `tools/__init__.py`. Create `tools/colab.py` with exactly these helpers (leave `start` / tunnel / card for Task 3, but you may add the module-level state now):

```python
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
```

- [ ] **Step 4: Re-run tests**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_colab_helpers.py -v`

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add tools/__init__.py tools/colab.py tests/test_colab_helpers.py
git commit -m "Add Colab helper URL, argv, and health checks"
```

---

### Task 3: Tunnel, ready card, and `start()`

**Files:**
- Modify: `tools/colab.py`
- Modify: `tests/test_colab_helpers.py`

**Interfaces:**
- Consumes: all Task 2 functions; `webui.py --share`
- Produces:
  - `start(port: int = 7860, *, version: str = "2.5", model_dir: str | None = None, fp16: bool = True, cloudflare: bool | None = None, share: bool | None = None) -> None`
  - `start_cloudflare_tunnel(port: int) -> str | None`
  - `show_ready_card(url: str, *, kind: str) -> None`
  - `_stop_cloudflare_tunnel() -> None`
  - `ensure_cloudflared() -> Path`
  - constants: `HEALTH_POLL_SECONDS = 20 * 60`, `TUNNEL_WAIT_SECONDS = 30`, `KEEPALIVE_INTERVAL = 300`

- [ ] **Step 1: Add failing tests for parse-from-log and start() decision helpers**

Append to `tests/test_colab_helpers.py`:

```python
from tools.colab import ready_card_html


def test_ready_card_html_contains_url_and_kind():
    html = ready_card_html("https://abc.trycloudflare.com", kind="cloudflare")
    assert "https://abc.trycloudflare.com" in html
    assert "IndexTTS WebUI is ready" in html
    assert "Cloudflare" in html
    assert "<iframe" not in html.lower()


def test_ready_card_html_gradio_and_local_notes():
    g = ready_card_html("https://x.gradio.live", kind="gradio")
    assert "Gradio" in g
    loc = ready_card_html("http://127.0.0.1:7860", kind="local")
    assert "Local" in loc
```

Do **not** add a live `cloudflared` or live `webui.py` test.

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_colab_helpers.py::test_ready_card_html_contains_url_and_kind tests/test_colab_helpers.py::test_ready_card_html_gradio_and_local_notes -v`

Expected: FAIL (`ImportError: cannot import name 'ready_card_html'`).

- [ ] **Step 3: Implement tunnel, card, and `start()` in `tools/colab.py`**

Add these imports at the top of `tools/colab.py` (merge with existing):

```python
import time
from collections import deque
import threading
```

Add constants next to the existing ones:

```python
HEALTH_POLL_SECONDS = 20 * 60
TUNNEL_WAIT_SECONDS = 30
KEEPALIVE_INTERVAL = 300
_webui_log: deque[str] = deque(maxlen=200)
```

Append the following functions (do not remove Task 2 helpers):

```python
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
        public_url = _public_url_from_logs()
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
```

If `force_share` is True and we just spawned with `--share`, `start()` already launched with share; the `force_share` branch only reads logs. When `force_share` and we had to spawn, `_wait_until_healthy` already ran; after that, wait up to 60s for the Gradio URL before falling back to local — add this immediately after the reuse/spawn block when `force_share` and `public_url` is still unset:

After `if force_share:` set:

```python
    if force_share:
        deadline = time.time() + 60
        while time.time() < deadline:
            public_url = _public_url_from_logs()
            if public_url:
                break
            time.sleep(1)
        kind = "gradio" if public_url else "local"
```

Use that block instead of the one-liner `_public_url_from_logs()` in the `force_share` branch above.

- [ ] **Step 4: Run helper tests**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_colab_helpers.py tests/test_webui_syntax.py -v`

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add tools/colab.py tests/test_colab_helpers.py
git commit -m "Add Colab start() with Cloudflare tunnel and Gradio share fallback"
```

---

### Task 4: Rewrite `IndexTTS2_Colab.ipynb`

**Files:**
- Modify: `IndexTTS2_Colab.ipynb` (replace entire notebook)
- Create: `tests/test_colab_notebooks.py`

**Interfaces:**
- Consumes: `from tools.colab import start; start(model_dir=...)`
- Produces: main notebook cells 0–5 as specified in the spec

- [ ] **Step 1: Write failing static notebook tests**

Create `tests/test_colab_notebooks.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_colab_notebooks.py::test_main_notebook_is_webui_path -v`

Expected: FAIL (no `tools.colab import start`, still has `dub_video` / `IndexTTS-2`).

- [ ] **Step 3: Rewrite `IndexTTS2_Colab.ipynb`**

Overwrite the notebook with these cells (valid nbformat 4, GPU accelerator, Python 3). Use the Write tool. Exact cell sources:

**markdown**

```markdown
# IndexTTS 2.5 — Colab WebUI

Run the Gradio WebUI on Google Colab (Python 3.12 + GPU).

**Steps**
1. Check GPU
2. Mount Drive (weights persist in `MyDrive/index-tts-cache`)
3. Clone `py3.12` and install
4. Download IndexTTS-2.5 if missing
5. `start()` — Cloudflare public link, Gradio share as fallback

Dub / highlight / intent pipelines stay in their own notebooks (see the last cell).
```

**markdown:** `## 0. Check GPU`

**code**

```python
import shutil
import subprocess
import sys

print(f"Python {sys.version}")
if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"], check=False)
else:
    print("WARNING: no GPU detected. CPU inference will be very slow.")
```

**markdown:** `## 1. Mount Google Drive (model cache)`

**code**

```python
import os
from pathlib import Path

CACHE_ROOT = "/content/drive/MyDrive/index-tts-cache"
LOCAL_FALLBACK = "/content/index-tts/checkpoints"

try:
    from google.colab import drive

    drive.mount("/content/drive")
    Path(CACHE_ROOT, "hf_home").mkdir(parents=True, exist_ok=True)
    Path(CACHE_ROOT, "torch_home").mkdir(parents=True, exist_ok=True)
    Path(CACHE_ROOT, "checkpoints-2.5").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = f"{CACHE_ROOT}/hf_home"
    os.environ["TORCH_HOME"] = f"{CACHE_ROOT}/torch_home"
    os.environ["INDEX_TTS_MODEL_DIR"] = f"{CACHE_ROOT}/checkpoints-2.5"
    print(f"Cache: {CACHE_ROOT}")
    print(f"INDEX_TTS_MODEL_DIR={os.environ['INDEX_TTS_MODEL_DIR']}")
except Exception as exc:
    Path(LOCAL_FALLBACK).mkdir(parents=True, exist_ok=True)
    os.environ["INDEX_TTS_MODEL_DIR"] = LOCAL_FALLBACK
    print(
        f"WARNING: Drive mount failed ({exc}). "
        f"Weights will not persist. Using {LOCAL_FALLBACK}"
    )
```

**markdown:** `## 2. Clone repo and install`

**code**

```python
import os

os.chdir("/content")
if os.path.isdir("/content/index-tts/.git"):
    os.chdir("/content/index-tts")
    get_ipython().system("git fetch origin")
    get_ipython().system("git checkout py3.12")
    get_ipython().system("git pull --ff-only origin py3.12")
else:
    get_ipython().system("git clone -b py3.12 https://github.com/deluxebear/index-tts.git")
    os.chdir("/content/index-tts")
print("cwd:", os.getcwd())
```

**code**

```python
import os

os.chdir("/content/index-tts")
get_ipython().system("pip uninstall -y tensorflow keras 2>/dev/null")
get_ipython().system("pip install ninja")
get_ipython().system(
    'pip install -e ".[webui]" --extra-index-url https://download.pytorch.org/whl/cu128'
)

try:
    import numba
    import torch

    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"numba={numba.__version__}")
except Exception as exc:
    print(
        f"Import failed after install ({exc}). Restart the runtime, "
        "then re-run the Drive cell and the start cell — do not reinstall."
    )
    import IPython

    IPython.Application.instance().kernel.do_shutdown(True)
```

**markdown:** `## 3. Download IndexTTS-2.5 checkpoints`

**code**

```python
import os
from pathlib import Path

model_dir = os.environ["INDEX_TTS_MODEL_DIR"]
required = [
    "gpt.pth",
    "s2mel.pth",
    "codec.pth",
    "multilingual_zh_ja_yue_char_del.tiktoken",
    "wav2vec2bert_stats.pt",
]
missing = [name for name in required if not Path(model_dir, name).is_file()]
if missing:
    print(f"Downloading IndexTTS-2.5 (missing: {', '.join(missing)})...")
    get_ipython().system(
        f'huggingface-cli download IndexTeam/IndexTTS-2.5 --local-dir "{model_dir}"'
    )
    missing = [name for name in required if not Path(model_dir, name).is_file()]
    if missing:
        raise SystemExit(
            f"Download incomplete, still missing: {missing}. "
            "Download IndexTeam/IndexTTS-2.5 manually."
        )
else:
    print(f"Checkpoints already present at {model_dir}")
```

**markdown:** `## 4. Start WebUI`

**code**

```python
import os
import sys

sys.path.insert(0, "/content/index-tts")
from tools.colab import start

start(model_dir=os.environ["INDEX_TTS_MODEL_DIR"])
```

**markdown**

```markdown
## 5. Other pipelines

These stay in their own notebooks (same Drive cache root):

- [DubbingPipeline_Colab.ipynb](https://github.com/deluxebear/index-tts/blob/py3.12/DubbingPipeline_Colab.ipynb)
- [HighlightPipeline_Colab.ipynb](https://github.com/deluxebear/index-tts/blob/py3.12/HighlightPipeline_Colab.ipynb)
- [IntentPipeline_Colab.ipynb](https://github.com/deluxebear/index-tts/blob/py3.12/IntentPipeline_Colab.ipynb)
```

Notebook metadata:

```json
{
  "accelerator": "GPU",
  "colab": { "gpuType": "T4", "provenance": [] },
  "kernelspec": { "display_name": "Python 3", "name": "python3" },
  "language_info": { "name": "python", "version": "3.12.0" }
}
```

- [ ] **Step 4: Run the main-notebook test**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_colab_notebooks.py::test_main_notebook_is_webui_path -v`

Expected: PASS. The other two tests in that file still FAIL (pipeline notebooks not aligned yet).

- [ ] **Step 5: Commit**

```bash
git add IndexTTS2_Colab.ipynb tests/test_colab_notebooks.py
git commit -m "Rewrite IndexTTS2 Colab notebook as WebUI start() path"
```

---

### Task 5: Align pipeline notebooks

**Files:**
- Modify: `DubbingPipeline_Colab.ipynb`
- Modify: `HighlightPipeline_Colab.ipynb`
- Modify: `IntentPipeline_Colab.ipynb`
- Modify: `VoxCPM_DubbingPipeline_Colab.ipynb`

**Interfaces:**
- Consumes: same cache env names as the main notebook (`HF_HOME`, `TORCH_HOME`, `INDEX_TTS_MODEL_DIR` optional)
- Produces: notebooks that satisfy `test_pipeline_notebooks_share_cache_and_v25` and `test_voxcpm_notebook_cache_root_only`

Do **not** change pipeline APIs, Secrets usage, or run cells beyond path/install/checkpoint updates.

- [ ] **Step 1: Confirm the alignment tests still fail**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_colab_notebooks.py::test_pipeline_notebooks_share_cache_and_v25 tests/test_colab_notebooks.py::test_voxcpm_notebook_cache_root_only -v`

Expected: FAIL (`models_cache` / `numpy<2.0` / missing `IndexTTS-2.5`).

- [ ] **Step 2: Apply the same string replacements in all four notebooks**

In every cell source string:

1. `/content/drive/MyDrive/models_cache` → `/content/drive/MyDrive/index-tts-cache`
2. Delete the two lines (and the comment above them) that install the old numpy pin:
   ```
   # Fix: ... numpy < 2.0
   !pip install "numpy<2.0"
   ```
   In VoxCPM the comment is `# Fix: demucs/whisperx/voxcpm may pull numpy 2.x but numba 0.60 needs numpy < 2.0`.
3. In Dubbing / Highlight / Intent only:
   - `DRIVE_CKPT = f"{DRIVE_CACHE}/indextts2_checkpoints"` → `DRIVE_CKPT = f"{DRIVE_CACHE}/checkpoints-2.5"`
   - `IndexTeam/IndexTTS-2` → `IndexTeam/IndexTTS-2.5` (download command and nearby prose)
   - Existence check may stay `config.yaml` **or** switch to `gpt.pth`; either is fine as long as the download repo is 2.5
   - After setting `DRIVE_CKPT`, also set `os.environ["INDEX_TTS_MODEL_DIR"] = DRIVE_CKPT` so it matches the main notebook
4. Dubbing / Highlight / Intent install cells currently `pip uninstall -y torch torchvision torchaudio tensorflow keras tensorboard protobuf`. Change that uninstall line to `pip uninstall -y tensorflow keras` only (do not uninstall torch). Keep each notebook's extra pipeline deps (`demucs`, `whisperx`, `flash-attn`, etc.).
5. Do **not** add `start()` or WebUI cells to these notebooks.
6. Do **not** edit files under the untracked `index-tts/` directory.

- [ ] **Step 3: Run all Colab-related tests**

Run:

```bash
PYTHONPATH="$PYTHONPATH:." uv run pytest \
  tests/test_colab_helpers.py \
  tests/test_colab_notebooks.py \
  tests/test_webui_syntax.py -v
```

Expected: all passed.

- [ ] **Step 4: Commit**

```bash
git add DubbingPipeline_Colab.ipynb HighlightPipeline_Colab.ipynb IntentPipeline_Colab.ipynb VoxCPM_DubbingPipeline_Colab.ipynb
git commit -m "Align Colab pipeline notebooks to Drive cache and IndexTTS-2.5"
```

---

## Spec coverage

| Spec item | Task |
|---|---|
| `tools/colab.py` `start()` + helpers | 2, 3 |
| `webui.py --share` | 1 |
| Cloudflare then Gradio fallback | 3 |
| Ready card, no iframe, keepalive, Ctrl+C keeps WebUI | 3 |
| Foreign port not killed | 2, 3 |
| Default no tunnel outside Colab | 2 `wants_cloudflare` |
| Main notebook rewrite, drop dubbing cells | 4 |
| Drive cache paths | 4, 5 |
| Pipeline notebook alignment + drop `numpy<2.0` | 5 |
| VoxCPM cache root only | 5 |
| Unit tests without live tunnel | 2, 3 |
| Local `webui.py` unchanged without `--share` | 1 |

## Manual acceptance (not CI)

After push, on a fresh Colab GPU runtime open `IndexTTS2_Colab.ipynb` from the `py3.12` branch and walk the spec acceptance list: public link opens WebUI; second session skips download; re-run start reuses the port; Ctrl+C stops the tunnel only; local CLI has no share by default.
