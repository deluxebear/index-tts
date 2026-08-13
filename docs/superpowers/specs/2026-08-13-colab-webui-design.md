# Colab WebUI — Design

**Date:** 2026-08-13
**Status:** Approved (Approach 2)

## Goal

Make IndexTTS convenient to run on Google Colab: one main notebook that
installs the project, caches IndexTTS-2.5 weights on Drive, and launches
the Gradio WebUI with a clickable public URL. Align the existing pipeline
notebooks to the same cache and 2.5 defaults. Do not merge those pipelines
into the main notebook.

## Non-goals

- Unsloth login, bootstrap password, or credential cards (IndexTTS has no auth)
- In-cell Colab proxy iframe (often blank on current Colab)
- Changing Gradio UI layout or inference code
- Merging dub / highlight / intent pipelines into `IndexTTS2_Colab.ipynb`
- Opening a Cloudflare tunnel by default outside Colab
- Killing an unrelated process that already owns the chosen port

## Approach (2): Colab helper + Cloudflare, Gradio share as fallback

A small `tools/colab.py` exposes `start()`. It launches `webui.py` as a
subprocess, opens a `cloudflared` quick tunnel, and shows a ready card.
If the tunnel fails, it restarts WebUI with `--share` and uses the
`*.gradio.live` link. `webui.py` only gains a `--share` flag; its
argparse-at-import structure stays.

## Architecture

```
IndexTTS2_Colab.ipynb
  → Drive cache + install + download 2.5
  → from tools.colab import start; start(model_dir=...)

tools/colab.py
  start()
    → reuse live WebUI on port, or Popen webui.py
    → poll http://127.0.0.1:{port}/
    → cloudflared tunnel --url http://127.0.0.1:{port}
    → on failure: restart webui.py --share, parse *.gradio.live
    → HTML ready card + keepalive

webui.py
  --share  →  demo.launch(..., share=cmd_args.share)
```

```
笔记本 start()
  → 子进程: python webui.py --fp16 --version 2.5 --model_dir <Drive>/checkpoints-2.5 --port 7860 --host 127.0.0.1
  → 轮询 http://127.0.0.1:7860
  → cloudflared tunnel --url http://127.0.0.1:7860
  → 成功: 显示 trycloudflare.com 卡片
  → 失败: 重启 webui --share，显示 *.gradio.live
  → keepalive（Ctrl+C 拆隧道，WebUI 子进程保留）
```

## `start()` API

```python
def start(
    port: int = 7860,
    *,
    version: str = "2.5",
    model_dir: str | None = None,
    fp16: bool = True,
    cloudflare: bool | None = None,
    share: bool | None = None,
) -> None:
```

| Argument | Default | Meaning |
|---|---|---|
| `port` | `7860` | Local bind port |
| `version` | `"2.5"` | Passed to `webui.py --version` |
| `model_dir` | `INDEX_TTS_MODEL_DIR` env, else `./checkpoints` | Passed to `--model_dir` |
| `fp16` | `True` | Pass `--fp16` when true |
| `cloudflare` | `None` | `None` = enable on Colab only; `True`/`False` force |
| `share` | `None` | `None` = Gradio share only after tunnel failure; `True` skip tunnel; `False` never share |

Helpers live in the same file and stay Colab-only in purpose:

- `is_colab() -> bool` — `google.colab` import or `/content` + Colab env
- `is_webui_healthy(port, timeout=2.0) -> bool` — GET `http://127.0.0.1:{port}/`. True only if the response body contains `gradio` (case-insensitive) **or** this process still holds a live `Popen` we started for that port. Connection refused → False. Gradio has no `/api/health`.
- `port_busy_with_foreign_process(port) -> bool` — TCP accept on the port but `is_webui_healthy` is False
- `start_cloudflare_tunnel(port) -> str | None` — download `cloudflared` once, parse `https://*.trycloudflare.com`
- `show_ready_card(url, *, kind)` — one HTML card, `kind` is `cloudflare` / `gradio` / `local`
- `_stop_cloudflare_tunnel()` — best-effort teardown on KeyboardInterrupt

Module state: `_webui_proc: Popen | None` and `_webui_port: int | None` so reuse and share-fallback know which child we own.

### Launch and reuse

1. If `port_busy_with_foreign_process(port)`: error and suggest `start(port=7861)`. Do not kill strangers.
2. If `is_webui_healthy(port)`: do not rebind. Only (re)open the tunnel and show the card.
3. Else `Popen([sys.executable, "webui.py", ...])` with `--host 127.0.0.1`, no `--share`. Stream child logs into the notebook. Store the handle in `_webui_proc`.
4. Poll health up to **20 minutes** (first 2.5 load is slow). Non-zero exit or timeout: print the last 80 log lines and return. Do not start a tunnel.

Working directory for the child is the repo root (directory that contains `webui.py`).

### Cloudflare

- Binary: `/tmp/cloudflared`, downloaded from
  `https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64`
  on first use; `chmod +x`.
- Command: `cloudflared tunnel --url http://127.0.0.1:{port}`.
- Parse the first `https://<id>.trycloudflare.com` on stderr within ~30s.
- `cloudflare=False` skips this step. `cloudflare=None` skips it when `not is_colab()`.

### Gradio share fallback

Used when the tunnel is skipped or fails, and `share is not False`:

1. If `_webui_proc` is ours, terminate it. If we are reusing a WebUI we did not start, skip restart and show the local URL plus a note that share fallback needs a process we launched.
2. Relaunch with the same args plus `--share`.
3. Parse `*.gradio.live` from stdout. If that also fails, show `http://127.0.0.1:{port}` and say the UI will not open from outside this runtime.

`share=True` skips Cloudflare and goes straight to this path.

### Ready card and keepalive

One card: title “IndexTTS WebUI is ready”, a button that `window.open`s the public URL, the full URL in monospace, and a short note of the link kind (Cloudflare / Gradio / local). No iframe.

Keepalive prints `.` every 5 minutes so Colab does not recycle the kernel. `Ctrl+C` stops the tunnel only; the WebUI subprocess stays so a later `start()` can reuse it.

Outside Colab, with defaults, no tunnel and no share: card shows `http://127.0.0.1:{port}`.

## `webui.py` change

Add:

```python
parser.add_argument("--share", action="store_true", default=False,
                    help="Create a public Gradio share link")
```

Launch becomes:

```python
demo.launch(server_name=cmd_args.host, server_port=cmd_args.port, share=cmd_args.share)
```

No other WebUI behavior changes. Local `python webui.py` stays private.

## Main notebook

Rewrite `IndexTTS2_Colab.ipynb`. Drop the inlined dubbing cells; link out instead.

| Cell | Action |
|---|---|
| 0. GPU | `nvidia-smi` + `sys.version` |
| 1. Drive | Mount; set cache env (below). If mount is cancelled, fall back to `/content/index-tts/checkpoints` and warn that weights will not persist |
| 2. Clone / install | In `/content`, clone `-b py3.12` or `git fetch && git checkout py3.12 && git pull`. Uninstall Colab `tensorflow` / `keras` only (do **not** uninstall torch). `pip install ninja` then `pip install -e ".[webui]" --extra-index-url https://download.pytorch.org/whl/cu128`. Print `torch` / CUDA. If `import torch` or `import numba` fails after install (ABI mismatch), shut down the kernel and tell the user to re-run Drive + start, not install |
| 3. Model | If `INDEX_TTS_MODEL_DIR` is missing 2.5 required files (`gpt.pth`, `s2mel.pth`, `codec.pth`, `multilingual_zh_ja_yue_char_del.tiktoken`, `wav2vec2bert_stats.pt`), download `IndexTeam/IndexTTS-2.5`. Skip when Drive already has them |
| 4. Start | `from tools.colab import start; start(model_dir=os.environ["INDEX_TTS_MODEL_DIR"])` |
| 5. Other entry points | Markdown links to `DubbingPipeline_Colab.ipynb`, `HighlightPipeline_Colab.ipynb`, `IntentPipeline_Colab.ipynb` |

## Drive cache

Root: `/content/drive/MyDrive/index-tts-cache/`

| Path | Role |
|---|---|
| `hf_home/` | `HF_HOME` |
| `torch_home/` | `TORCH_HOME` |
| `checkpoints-2.5/` | `INDEX_TTS_MODEL_DIR` and `webui.py --model_dir` |

Create the directories after a successful mount. Pipeline notebooks switch to this same root so weights and HF caches are shared.

## Pipeline notebook alignment

`DubbingPipeline_Colab.ipynb`, `HighlightPipeline_Colab.ipynb`, `IntentPipeline_Colab.ipynb` (and VoxCPM notebook cache paths if they still use `models_cache`):

- Cache root becomes `/content/drive/MyDrive/index-tts-cache/` (same env vars as the main notebook)
- Remove the outdated `pip install "numpy<2.0"` pin (repo now uses `numpy==2.2.6` / `numba==0.63.0`)
- Checkpoint download target is IndexTTS-2.5 under `checkpoints-2.5/`
- Pipeline behavior (cells, APIs, Secrets) stays the same

`VoxCPM_DubbingPipeline_Colab.ipynb` only changes the shared cache root if it currently uses `models_cache`; it does not switch TTS engine or model.

## Error handling

| Situation | Behavior |
|---|---|
| No GPU | Warn that CPU will be very slow; continue |
| Drive mount failed / cancelled | Local `checkpoints/`; warn weights will not persist |
| Install failed | Stop in that cell; do not call `start()` |
| Model download failed | Stop; tell the user to download `IndexTeam/IndexTTS-2.5` manually |
| WebUI exit or health timeout | Print last 80 log lines; no tunnel |
| `cloudflared` download or handshake failed | Restart with `--share`; if that fails too, local URL + explanation |
| Port owned by another program | Error; suggest `start(port=7861)` |
| Start cell re-run | Reuse live WebUI; refresh tunnel + card |

## Acceptance

1. Empty Colab GPU runtime, run the main notebook top to bottom: ready card shows a clickable `trycloudflare.com` or `gradio.live` link; the browser opens WebUI.
2. Second session with 2.5 already on Drive: download is skipped; start is clearly faster.
3. Re-running the start cell does not rebind the port; the card appears again.
4. `Ctrl+C` stops the tunnel and leaves WebUI up; a later `start()` produces a new link.
5. Local `python webui.py` is unchanged (no share unless `--share`).
6. The three pipeline notebooks still run their own flows; install no longer pins `numpy<2.0`.

## Files

| File | Change |
|---|---|
| `tools/colab.py` | Create: `start()` and helpers |
| `webui.py` | Add `--share`; pass it to `demo.launch` |
| `IndexTTS2_Colab.ipynb` | Rewrite as the WebUI path; drop inlined dubbing |
| `DubbingPipeline_Colab.ipynb` | Cache root, drop `numpy<2.0`, 2.5 checkpoints |
| `HighlightPipeline_Colab.ipynb` | Same alignment |
| `IntentPipeline_Colab.ipynb` | Same alignment |
| `VoxCPM_DubbingPipeline_Colab.ipynb` | Cache root only, if it still uses `models_cache` |
| `tests/test_colab_helpers.py` | Unit tests for URL parse, health heuristic, `start()` arg building, Colab detection (no live tunnel) |
