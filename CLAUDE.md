# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

IndexTTS2 — a zero-shot text-to-speech model by Bilibili with emotion control and duration adaptation. Supports Chinese, English, and mixed-language input. Python 3.10+, licensed under LicenseRef-Bilibili-IndexTTS.

## Commands

```bash
# Install dependencies (uv is required, not pip/conda)
uv sync --all-extras

# Run web UI (Gradio at http://127.0.0.1:7860)
uv run webui.py [--port 7860] [--fp16] [--deepspeed] [--cuda_kernel]

# CLI (IndexTTS1 legacy only)
uv run indextts "TEXT" -v voice.wav [-o output.wav] [--fp16]

# Video dubbing (English → Chinese)
uv run dub_pipeline.py video.mp4 -o output_cn.mp4
uv run dub_pipeline.py --batch /path/to/videos -o /path/to/output

# Video highlight (long video → Douyin short with AI narration)
uv run highlight_pipeline.py video.mp4 --ref-audio voice.wav -o highlight.mp4
uv run highlight_pipeline.py --batch /path/to/videos --ref-audio voice.wav -o /output

# Run scripts (must use uv run, may need PYTHONPATH)
PYTHONPATH="$PYTHONPATH:." uv run <script.py>

# Tests
PYTHONPATH="$PYTHONPATH:." uv run tests/regression_test.py
PYTHONPATH="$PYTHONPATH:." uv run tests/padding_test.py [model_dir]

# GPU check
uv run tools/gpu_check.py
```

## Architecture

```
Input Text + Reference Audio
    → TextNormalizer + TextTokenizer (BPE) → indextts/utils/front.py
    → SemanticCodec (MaskGCT acoustic tokens) → indextts/utils/maskgct/
    → GPT (UnifiedVoice, autoregressive token gen) → indextts/gpt/model_v2.py
    → S2Mel (DiT + Length Regulator → mel spectrogram) → indextts/s2mel/
    → BigVGAN vocoder (mel → waveform) → indextts/s2mel/modules/bigvgan/
    → Output WAV
```

**Emotion system** (in `indextts/infer_v2.py`): 8 dimensions [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]. Three input modes: emotion reference audio, emotion vector (0-1 floats), or text description (via QwenEmotion/Qwen 0.6B). Speaker embeddings via CAMPPlus.

### Key modules

- `indextts/infer_v2.py` — **IndexTTS2** main inference engine (current)
- `indextts/infer.py` — IndexTTS1 inference (legacy)
- `indextts/cli.py` — CLI entry point (legacy, IndexTTS1 only)
- `indextts/gpt/model_v2.py` — UnifiedVoice GPT model (24 layers, 1280 dims, 20 heads)
- `indextts/gpt/model.py` — Original GPT model (IndexTTS1)
- `indextts/s2mel/` — Semantic-to-Mel with DiT, length regulator, style encoder
- `indextts/s2mel/modules/bigvgan/` — BigVGAN v2 vocoder (nvidia/bigvgan_v2_22khz_80band_256x)
- `indextts/s2mel/modules/campplus/` — CAMPPlus speaker embedding
- `indextts/utils/front.py` — TextNormalizer (with glossary support via `glossary.yaml`), TextTokenizer
- `indextts/utils/maskgct/` — MaskGCT semantic codec
- `indextts/accel/` — Inference acceleration (GPT2 accel, KV cache, optional CUDA kernels)
- `webui.py` — Gradio web interface
- `dub_pipeline.py` — Video dubbing pipeline (English → Chinese, WhisperX + LLM translation + IndexTTS2)
- `highlight_pipeline.py` — Video highlight pipeline (long video → Douyin short, Qwen2.5-VL + LLM + IndexTTS2)
- `checkpoints/` — Model weights, config.yaml, bpe.model, emotion/speaker matrices

### Platform-specific text processing

- Linux: `WeTextProcessing`
- macOS/Windows: `wetext`

### Highlight pipeline architecture

```
Input Video (≤60min) + Reference Audio
    → Extract audio (ffmpeg)
    → Whisper transcription (GPU ~4GB → unload)
    → Qwen2.5-VL-7B video analysis + temporal grounding (GPU ~18GB → unload)
    → LLM script generation + emotion vectors + effects choreography (API)
    → IndexTTS2 narration with emotion injection (GPU ~5GB → unload)
    → ffmpeg assembly: portrait, transitions, color grading, ASS subtitles
    → Output: 2-5min Douyin-style MP4 + SRT
```

VRAM managed sequentially (peak ~18GB, fits L4 24GB). Requires: `LLM_API_KEY` env var, `whisperx`, `qwen-vl-utils`, `transformers>=4.52.1`.

### Dubbing pipeline architecture

```
English Video → demucs source separation → WhisperX ASR + pyannote diarization
    → LLM context-aware translation → IndexTTS2 voice cloning + emotion
    → Duration alignment (50/50 audio/video burden split)
    → ffmpeg assembly + SRT output
```

Requires: `HF_TOKEN` and `LLM_API_KEY` env vars.

## Key conventions

- All commands use `uv run` — never activate venvs manually
- Model checkpoints download from HuggingFace (`IndexTeam/IndexTTS-2`) or ModelScope to `checkpoints/`
- Config lives in `checkpoints/config.yaml` (OmegaConf)
- PyTorch CUDA wheels from `cu128` index (Linux/Windows); MPS on macOS
- FP16 is disabled automatically on CPU/MPS devices
- Pinyin annotations in text enable pronunciation control (valid entries in `checkpoints/pinyin.vocab`)
