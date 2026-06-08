# VoxCPM Dubbing Pipeline — Design

**Date:** 2026-06-08
**Status:** Approved (Approach A)

## Goal

Create a new, self-contained pipeline file `voxcpm_dub_pipeline.py` that dubs
English videos into Chinese using **VoxCPM** ([OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM))
as the TTS engine, instead of IndexTTS2. It must reuse `dub_pipeline.py` without
modifying it, expose the same `dub_video()` / `dub_batch()` interface, and run on
Google Colab (imported from a notebook cell).

Non-goals: changing `dub_pipeline.py`; building a notebook (.ipynb); multilingual
target selection (stays English → Chinese, same as `dub_pipeline.py`).

## Approach (A): Import-and-reuse

The new file imports every TTS-engine-agnostic function from `dub_pipeline.py`
and reimplements only the VoxCPM-specific parts plus its own orchestration. This
avoids duplicating ~2000 lines and automatically inherits subtitle-driven mode,
recursive batch, gap absorption, duration alignment, and all current fixes.

Hard constraint satisfied: `dub_pipeline.py` is imported, never edited.

## Architecture

```
voxcpm_dub_pipeline.py
├── imports from dub_pipeline (reused, engine-agnostic):
│   _save_checkpoint, _load_checkpoint, _clear_checkpoint, _detect_device,
│   extract_tracks, separate_vocals, transcribe_and_diarize, absorb_gaps,
│   extract_speaker_refs, get_ref_for_segment, build_subtitle_driven_segments,
│   translate_segments, LLMClient, align_durations, _get_video_duration,
│   compute_shifted_timeline, assemble_final, generate_srt, get_audio_duration
│
├── NEW (VoxCPM-specific):
│   _init_voxcpm(model_id, load_denoiser)        # VoxCPM.from_pretrained(...)
│   _offload_voxcpm(model) / _reload_voxcpm(...) # best-effort VRAM management
│   generate_speech_voxcpm(...)                  # replaces Step 6 (tts.infer → model.generate)
│   dub_video(...)                               # 11-step orchestration calling reused helpers
│   dub_batch(...)                               # incl. recursive=, mirrors dub_pipeline
│   main()                                       # CLI with VoxCPM args
```

## Pipeline steps (unchanged flow, only Step 6 differs)

1. extract_tracks (reused)
2. separate_vocals — demucs (reused)
3. transcribe_and_diarize — whisperx (reused)
3.5 absorb_gaps (reused)
4. extract_speaker_refs (reused)
5. subtitle-driven (`build_subtitle_driven_segments`) or LLM translation (reused)
6. **generate_speech_voxcpm (NEW)** — VoxCPM synthesis
7. align_durations (reused)
7.5 compute_shifted_timeline (reused)
8-9. assemble_final (reused)
10-11. generate_srt ×2 (reused)

Checkpoint step numbers stay identical to `dub_pipeline.py` so resume semantics
match. The new `dub_video` writes the same `checkpoint.json` schema.

## VoxCPM TTS integration (the only new synthesis logic)

```python
from voxcpm import VoxCPM
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
wav = model.generate(
    text=zh_text,
    reference_wav_path=ref_audio,      # voice clone from speaker's vocals segment
    cfg_value=cfg_value,               # default 2.0
    inference_timesteps=inference_timesteps,  # default 10
)
sf.write(output_path, wav, model.tts_model.sample_rate)  # 48 kHz
```

`generate_speech_voxcpm(segments, speaker_refs, vocals_path, work_dir, model, ...)`:
- For each segment with non-empty `zh_text`, pick reference audio via the reused
  `get_ref_for_segment(...)` (long segments → own audio window; short → speaker
  fallback with embedding verification).
- Call `model.generate(...)`, write `tts_output/tts_{i:04d}.wav` at the model's
  sample rate, set `seg["wav_path"]` and `seg["actual_duration"]`.
- Skip segments whose output wav already exists (resume-safe), mirroring the
  reused `generate_speech`.
- Empty `zh_text` → `wav_path=None`, `actual_duration=0`.

### Differences from IndexTTS2 (folded into the design)

- **No emotion vector / no separate emotion prompt.** Drop `emo_audio_prompt`;
  VoxCPM clones timbre and prosody directly from `reference_wav_path`.
- **48 kHz output.** No special handling: `align_durations` operates at each
  file's own sample rate, and `_build_speech_track` resamples aligned audio to
  the background-track sample rate with librosa. Mixed sample rates already work.
- **Voice cloning mode.** Default = simple `reference_wav_path` (robust, no
  transcript needed). Optional `--ultimate-clone`: for long segments where the
  reference is the segment's own audio, additionally pass
  `prompt_wav_path=ref` + `prompt_text=seg["text"]` (the English ASR transcript
  of that audio) for higher fidelity. Default OFF to avoid transcript-mismatch
  risk on fallback references.

## VRAM management

VoxCPM ≈ 8 GB. `_init_voxcpm` is called once per `dub_batch` run. During steps
1–5 the model is offloaded to CPU (`_offload_voxcpm`) and reloaded before step 6
(`_reload_voxcpm`), matching `dub_pipeline`'s `_offload_tts`/`_reload_tts`
pattern. Offload is best-effort and wrapped in try/except: it moves the
underlying torch module (`model.tts_model` and nested module if present) to CPU
and calls `torch.cuda.empty_cache()`. If the internal layout differs and offload
fails, the model stays on GPU (works on L4 24 GB; on T4 16 GB the other stages
already free their models via `del`, so peak is bounded). Sequential design keeps
it within Colab T4/L4 limits.

## CLI / interface

`dub_video()` and `dub_batch()` keep the same keyword arguments as
`dub_pipeline.py` **except** the IndexTTS2-only ones:

- Removed: `use_fp16`, `model_dir`, `tts` (IndexTTS2 instance).
- Added: `model_id="openbmb/VoxCPM2"`, `cfg_value=2.0`,
  `inference_timesteps=10`, `ultimate_clone=False`, and a VoxCPM model handle
  passed through batch (`model=`).
- Kept identical: `output_path`, `work_dir`, `hf_token`, `llm_api_key`,
  `llm_api_base`, `llm_model`, `fallback_refs`, `num_speakers`, `cleanup`,
  `external_subs`, `no_external_subs`, `audio_only_align`, and `recursive` (batch).

`main()` argparse mirrors `dub_pipeline.main()`, swapping `--fp16/--no-fp16/--model-dir`
for `--model-id/--cfg-value/--inference-timesteps/--ultimate-clone`. `HF_TOKEN`
and `LLM_API_KEY` env-var fallbacks and required-key checks are preserved.

Colab usage:
```python
from voxcpm_dub_pipeline import dub_batch
dub_batch(input_dir=..., output_dir=..., work_dir=..., recursive=True,
          hf_token=HF_TOKEN, llm_api_key=LLM_API_KEY, llm_api_base=LLM_API_BASE,
          llm_model=LLM_MODEL, num_speakers=NUM_SPEAKERS)
```

## Dependencies

- `pip install voxcpm` (Python ≥3.10 <3.13, PyTorch ≥2.5, CUDA ≥12).
- `dub_pipeline.py` in the same directory / importable on `PYTHONPATH`.
- Existing: demucs, whisperx, pyrubberband, soundfile, librosa, openai, ffmpeg.

## Testing strategy

Full end-to-end requires GPU + model downloads (not available locally), so:

1. **Import/wiring test** — `import voxcpm_dub_pipeline` resolves; every reused
   name exists in `dub_pipeline` (verified: all present at module level).
2. **Syntax check** — `ast.parse` on the new file.
3. **`generate_speech_voxcpm` unit test** — mock the VoxCPM model with a stub
   whose `.generate()` returns a fixed numpy array and `.tts_model.sample_rate`
   is set; run on synthetic segments + a real reference wav; assert wav files
   written, `wav_path`/`actual_duration` set, resume-skip works, empty-text
   handled. No GPU needed.
4. **Batch path test** — monkeypatch `_init_voxcpm` and `dub_video` to assert
   recursive discovery + path/work-dir uniqueness (same harness used for the
   `--recursive` change).

Real Colab GPU run is the user's acceptance test.

## Risks

- **VoxCPM internal module layout** for offload is unverified → offload is
  defensive and non-fatal.
- **Model id `openbmb/VoxCPM2`** assumed current; exposed as `--model-id` so the
  user can switch to `openbmb/VoxCPM-0.5B` etc. without code changes.
- **48 kHz vs background sample rate** mismatch is handled by existing resample
  logic; no new code.
