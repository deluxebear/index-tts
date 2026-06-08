"""
Video Dubbing Pipeline: English → Chinese using VoxCPM.

A drop-in sibling of dub_pipeline.py that swaps the TTS engine from IndexTTS2 to
VoxCPM (https://github.com/OpenBMB/VoxCPM). Every step except speech synthesis is
reused verbatim from dub_pipeline.py — this file imports those functions and only
reimplements the VoxCPM-specific parts (model load, offload/reload, generate) plus
its own orchestration. dub_pipeline.py is never modified.

Pipeline (identical flow, only Step 6 differs):
    demucs vocal separation → whisperx ASR + diarization → subtitle-driven /
    LLM translation → VoxCPM voice cloning → duration alignment → ffmpeg assembly
    → SRT output, with checkpoint/resume.

Usage:
    # Single video
    python voxcpm_dub_pipeline.py video.mp4 -o output_cn.mp4

    # Batch (optionally recursive)
    python voxcpm_dub_pipeline.py --batch /path/to/videos -o /path/to/out --recursive

    # In Colab
    from voxcpm_dub_pipeline import dub_batch
    dub_batch(input_dir=..., output_dir=..., recursive=True,
              hf_token=HF_TOKEN, llm_api_key=LLM_API_KEY)

Requires: voxcpm, plus dub_pipeline.py on the path and its deps
(demucs, whisperx, pyrubberband, soundfile, librosa, openai, ffmpeg).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# --- Reused, engine-agnostic building blocks (import, never modify) ----------
from dub_pipeline import (
    MIN_REF_DURATION,
    LLMClient,
    _clear_checkpoint,
    _compute_speaker_embedding,
    _detect_device,
    _get_video_duration,
    _load_checkpoint,
    _save_checkpoint,
    absorb_gaps,
    align_durations,
    assemble_final,
    build_subtitle_driven_segments,
    compute_shifted_timeline,
    extract_speaker_refs,
    extract_tracks,
    generate_srt,
    get_audio_duration,
    get_ref_for_segment,
    separate_vocals,
    transcribe_and_diarize,
    translate_segments,
)

# VoxCPM defaults
DEFAULT_MODEL_ID = "openbmb/VoxCPM2"
DEFAULT_CFG_VALUE = 2.0
DEFAULT_INFERENCE_TIMESTEPS = 10


# ---------------------------------------------------------------------------
# VoxCPM model: load / offload / reload
# ---------------------------------------------------------------------------

def _init_voxcpm(model_id=DEFAULT_MODEL_ID, load_denoiser=False):
    """Load a VoxCPM model."""
    from voxcpm import VoxCPM
    print(f"  Loading VoxCPM model: {model_id} ...")
    return VoxCPM.from_pretrained(model_id, load_denoiser=load_denoiser)


def _voxcpm_inner_modules(model):
    """Best-effort: collect the torch modules inside a VoxCPM wrapper."""
    mods = []
    for attr in ("tts_model", "model", "denoiser"):
        m = getattr(model, attr, None)
        if m is not None and hasattr(m, "to"):
            mods.append(m)
    if hasattr(model, "to"):
        mods.append(model)
    return mods


def _voxcpm_sample_rate(model):
    """Resolve the output sample rate across VoxCPM versions."""
    for obj, attr in ((getattr(model, "tts_model", None), "sample_rate"),
                      (model, "sample_rate")):
        if obj is not None:
            sr = getattr(obj, attr, None)
            if sr:
                return int(sr)
    raise RuntimeError("Could not determine VoxCPM sample rate")


def _move_voxcpm(model, device):
    """Move VoxCPM to a device. Best-effort and non-fatal (layout varies)."""
    import torch
    if model is None or not torch.cuda.is_available():
        return
    for m in _voxcpm_inner_modules(model):
        try:
            m.to(device)
        except Exception:
            pass
    if device == "cpu":
        torch.cuda.empty_cache()


def _offload_voxcpm(model):
    """Move VoxCPM to CPU to free VRAM for demucs/whisperx."""
    _move_voxcpm(model, "cpu")


def _reload_voxcpm(model):
    """Move VoxCPM back to GPU for inference."""
    _move_voxcpm(model, _detect_device())


# ---------------------------------------------------------------------------
# Step 6: TTS generation with VoxCPM (replaces dub_pipeline.generate_speech)
# ---------------------------------------------------------------------------

def _to_wav_array(wav):
    """Normalize VoxCPM output to a 1-D float numpy array."""
    if hasattr(wav, "detach"):  # torch tensor
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32)
    return np.squeeze(wav)


def generate_speech_voxcpm(segments, speaker_refs, vocals_path, work_dir, model,
                           cfg_value=DEFAULT_CFG_VALUE,
                           inference_timesteps=DEFAULT_INFERENCE_TIMESTEPS,
                           ultimate_clone=False):
    """Generate Chinese speech for each segment using VoxCPM voice cloning.

    Mirrors dub_pipeline.generate_speech (same outputs, same resume behaviour)
    but calls VoxCPM's generate() instead of IndexTTS2's infer(). VoxCPM clones
    timbre and prosody directly from the reference audio — there is no separate
    emotion prompt or emotion vector.
    """
    tts_dir = os.path.join(work_dir, "tts_output")
    seg_ref_dir = os.path.join(work_dir, "seg_refs")
    os.makedirs(tts_dir, exist_ok=True)
    os.makedirs(seg_ref_dir, exist_ok=True)

    sr = _voxcpm_sample_rate(model)

    for i, seg in enumerate(segments):
        zh_text = seg.get("zh_text", "")
        if not zh_text.strip():
            seg["wav_path"] = None
            seg["actual_duration"] = 0
            continue

        output_path = os.path.join(tts_dir, f"tts_{i:04d}.wav")

        if os.path.exists(output_path):  # resume
            seg["wav_path"] = output_path
            seg["actual_duration"] = get_audio_duration(output_path)
        else:
            ref_audio = get_ref_for_segment(seg, speaker_refs, vocals_path, seg_ref_dir)

            gen_kwargs = dict(
                text=zh_text,
                reference_wav_path=ref_audio,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
            )
            # Ultimate cloning: for long segments the reference IS the segment's
            # own audio, so its English ASR text is a valid prompt transcript.
            if ultimate_clone:
                dur = seg["end"] - seg["start"]
                en_text = seg.get("text", "").strip()
                if dur >= MIN_REF_DURATION and en_text:
                    gen_kwargs["prompt_wav_path"] = ref_audio
                    gen_kwargs["prompt_text"] = en_text

            wav = _to_wav_array(model.generate(**gen_kwargs))
            sf.write(output_path, wav, sr)
            seg["wav_path"] = output_path
            seg["actual_duration"] = get_audio_duration(output_path)

        target_dur = seg["end"] - seg["start"]
        ratio = seg["actual_duration"] / target_dur if target_dur > 0 else 1.0
        print(f"  [{i:3d}] {seg.get('speaker', '?')} | "
              f"target={target_dur:.1f}s actual={seg['actual_duration']:.1f}s "
              f"ratio={ratio:.2f} | {zh_text[:30]}...")

    return segments


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def dub_video(
    video_path,
    output_path=None,
    work_dir="dub_workspace",
    hf_token=None,
    llm_api_key=None,
    llm_api_base="https://api.openai.com/v1",
    llm_model="gpt-4o-mini",
    model_id=DEFAULT_MODEL_ID,
    cfg_value=DEFAULT_CFG_VALUE,
    inference_timesteps=DEFAULT_INFERENCE_TIMESTEPS,
    ultimate_clone=False,
    fallback_refs=None,
    num_speakers=None,
    model=None,
    cleanup=False,
    external_subs=None,
    no_external_subs=False,
    audio_only_align=False,
):
    """
    Dub an English video into Chinese using VoxCPM.

    Args mirror dub_pipeline.dub_video, with VoxCPM-specific replacements:
        model_id: VoxCPM model id (default openbmb/VoxCPM2).
        cfg_value / inference_timesteps: VoxCPM generation params.
        ultimate_clone: use prompt_wav_path+prompt_text cloning on long segments.
        model: pre-initialized VoxCPM instance (for batch mode).
    """
    video_path = str(video_path)
    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(video_path).parent / f"{stem}_cn.mp4")

    video_work_dir = os.path.join(work_dir, Path(video_path).stem)
    os.makedirs(video_work_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Dubbing (VoxCPM): {video_path}")
    print(f"Output:  {output_path}")
    print(f"Work:    {video_work_dir}")
    print(f"{'=' * 60}")

    ckpt = _load_checkpoint(video_work_dir)
    done = ckpt["step"] if ckpt else 0
    segments = ckpt.get("segments") if ckpt else None
    paths = ckpt.get("paths", {}) if ckpt else {}

    # Free GPU for demucs/whisperx steps (TTS not needed until Step 6)
    if done < 7:
        _offload_voxcpm(model)

    # --- Step 1: Extract tracks ---
    if done < 1:
        print("\n[Step 1/11] Extracting audio and video tracks...")
        audio_path, video_only_path = extract_tracks(video_path, video_work_dir)
        paths.update(audio_path=audio_path, video_only_path=video_only_path)
        _save_checkpoint(video_work_dir, 1, paths=paths)
    else:
        audio_path = paths["audio_path"]
        video_only_path = paths["video_only_path"]
        print("\n[Step 1/11] Skipped (cached)")

    # --- Step 2: Source separation ---
    if done < 2:
        print("\n[Step 2/11] Separating vocals from background...")
        vocals_path, bg_path = separate_vocals(audio_path, video_work_dir)
        paths.update(vocals_path=vocals_path, bg_path=bg_path)
        _save_checkpoint(video_work_dir, 2, paths=paths)
    else:
        vocals_path = paths["vocals_path"]
        bg_path = paths["bg_path"]
        print("\n[Step 2/11] Skipped (cached)")

    # --- Step 3: ASR + diarization ---
    if done < 3:
        print("\n[Step 3/11] Transcribing and diarizing...")
        segments = transcribe_and_diarize(vocals_path, hf_token, num_speakers)
        transcript_path = os.path.join(video_work_dir, "transcript.json")
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        print(f"  Saved transcript: {transcript_path}")
        _save_checkpoint(video_work_dir, 3, segments=segments, paths=paths)
    else:
        print(f"\n[Step 3/11] Skipped (cached, {len(segments)} segments)")

    # --- Step 3.5: Gap absorption ---
    if done < 4:
        print("\n[Step 3.5/11] Absorbing inter-segment gaps...")
        segments = absorb_gaps(segments)
        _save_checkpoint(video_work_dir, 4, segments=segments, paths=paths)
    else:
        print("\n[Step 3.5/11] Skipped (cached)")

    # --- Step 4: Speaker references ---
    if done < 5:
        print("\n[Step 4/11] Extracting speaker references...")
        speaker_refs = extract_speaker_refs(segments, vocals_path, video_work_dir, fallback_refs)
        paths["speaker_refs"] = speaker_refs
        _save_checkpoint(video_work_dir, 5, segments=segments, paths=paths)
    else:
        speaker_refs = paths.get("speaker_refs", {})
        for spk, info in speaker_refs.items():
            if info.get("embedding") is None and os.path.exists(info.get("best_auto", "")):
                info["embedding"] = _compute_speaker_embedding(info["best_auto"])
        print("\n[Step 4/11] Skipped (cached)")

    # --- Step 5: Translation (subtitle-driven or LLM) ---
    if done < 6:
        print("\n[Step 5/11] Translating to Chinese...")

        sub_source = None
        if not no_external_subs:
            sub_segments, sub_source = build_subtitle_driven_segments(
                video_path, segments, external_subs=external_subs
            )
            if sub_segments is not None:
                segments = absorb_gaps(sub_segments)
                print(f"  Using subtitle-driven segments ({sub_source})")

        untranslated = sum(1 for s in segments if not s.get("zh_text") and s.get("text", "").strip())

        if sub_source and untranslated == 0:
            print(f"  All segments from external subtitles ({sub_source})")
        else:
            if sub_source:
                print(f"  {len(segments) - untranslated} from external subs, {untranslated} need LLM")
            llm_client = LLMClient(api_key=llm_api_key, api_base=llm_api_base, model=llm_model)
            segments, _ = translate_segments(segments, llm_client,
                                             skip_translated=bool(sub_source))
        translations_path = os.path.join(video_work_dir, "translations.json")
        with open(translations_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"id": i, "speaker": s.get("speaker"), "start": s["start"], "end": s["end"],
                  "en": s["text"], "zh": s.get("zh_text", "")}
                 for i, s in enumerate(segments)],
                f, ensure_ascii=False, indent=2,
            )
        print(f"  Saved translations: {translations_path}")
        _save_checkpoint(video_work_dir, 6, segments=segments, paths=paths)
    else:
        print("\n[Step 5/11] Skipped (cached)")

    # --- Step 6: TTS generation with VoxCPM ---
    if done < 7:
        print("\n[Step 6/11] Generating Chinese speech with VoxCPM...")
        if model is None:
            model = _init_voxcpm(model_id)
        _reload_voxcpm(model)
        segments = generate_speech_voxcpm(
            segments, speaker_refs, vocals_path, video_work_dir, model,
            cfg_value=cfg_value, inference_timesteps=inference_timesteps,
            ultimate_clone=ultimate_clone,
        )
        _save_checkpoint(video_work_dir, 7, segments=segments, paths=paths)
    else:
        print("\n[Step 6/11] Skipped (cached)")

    # --- Step 7: Duration alignment ---
    if done < 8:
        print("\n[Step 7/11] Aligning durations (audio + video)...")
        segments = align_durations(segments, video_work_dir, audio_only_align=audio_only_align)
        _save_checkpoint(video_work_dir, 8, segments=segments, paths=paths)
    else:
        print("\n[Step 7/11] Skipped (cached)")

    # --- Step 7.5: Compute shifted timeline ---
    needs_slowdown = any(seg.get("video_slowdown", 1.0) > 1.01 for seg in segments)
    timeline = None
    new_total = None
    if needs_slowdown:
        if done < 9:
            print("\n[Step 7.5/11] Computing shifted timeline for video slowdown...")
            video_duration = _get_video_duration(video_only_path)
            timeline, new_total = compute_shifted_timeline(segments, video_duration)
            print(f"  Original duration: {video_duration:.1f}s → New duration: {new_total:.1f}s (+{new_total - video_duration:.1f}s)")
            _save_checkpoint(video_work_dir, 9, segments=segments, paths=paths,
                             timeline=timeline, new_total=new_total)
        else:
            timeline = ckpt.get("timeline")
            new_total = ckpt.get("new_total")
            print("\n[Step 7.5/11] Skipped (cached)")
    else:
        for seg in segments:
            seg["new_start"] = seg["start"]

    # --- Step 8-9: Assembly ---
    if done < 10:
        print("\n[Step 8-9/11] Assembling final video...")
        assemble_final(segments, bg_path, video_only_path, output_path, video_work_dir,
                       timeline=timeline, new_total_duration=new_total)
        _save_checkpoint(video_work_dir, 10, segments=segments, paths=paths)
    else:
        print("\n[Step 8-9/11] Skipped (cached)")

    # --- Step 10-11: SRT subtitles ---
    print("\n[Step 10-11/11] Generating subtitles...")
    srt_base = output_path.rsplit(".", 1)[0]
    generate_srt(segments, f"{srt_base}.srt", lang="zh")
    generate_srt(segments, f"{srt_base}_en.srt", lang="en")

    _clear_checkpoint(video_work_dir)

    if cleanup:
        import shutil
        shutil.rmtree(video_work_dir)
        print(f"  Cleaned up: {video_work_dir}")

    print(f"\nDone! Output: {output_path}")
    return output_path


def dub_batch(
    input_dir,
    output_dir=None,
    video_extensions=(".mp4", ".mkv", ".mov", ".avi"),
    recursive=False,
    **kwargs,
):
    """
    Batch process all videos in a directory using VoxCPM.

    Args:
        input_dir: Input directory containing video files.
        output_dir: Output directory. Defaults to {input_dir}_cn/.
        video_extensions: Tuple of video file extensions to process.
        recursive: Recurse into subdirectories, preserving structure in the
            output and per-video work dir so same-named videos don't collide.
        **kwargs: All other arguments passed to dub_video().
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir.parent / f"{input_dir.name}_cn"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_resolved = output_dir.resolve()
    base_work_dir = kwargs.pop("work_dir", "dub_workspace")

    if recursive:
        candidates = sorted(input_dir.rglob("*"))
    else:
        candidates = sorted(input_dir.iterdir())

    videos = []
    for f in candidates:
        if f.suffix.lower() not in video_extensions:
            continue
        if f.stem.endswith("_cn") or out_resolved in f.resolve().parents:
            continue
        videos.append(f)

    if not videos:
        where = "recursively in" if recursive else "in"
        print(f"No video files found {where} {input_dir}")
        return

    print(f"Found {len(videos)} videos to process{' (recursive)' if recursive else ''}")
    print(f"Output directory: {output_dir}")

    model = _init_voxcpm(kwargs.get("model_id", DEFAULT_MODEL_ID))

    results = []
    for i, video in enumerate(videos):
        rel = video.relative_to(input_dir).with_suffix("")
        print(f"\n{'#' * 60}")
        print(f"[{i + 1}/{len(videos)}] {rel}")
        print(f"{'#' * 60}")

        out_path = output_dir / f"{rel}_cn.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path = str(out_path)
        video_work_dir = os.path.join(base_work_dir, *rel.parent.parts) if rel.parent.parts else base_work_dir
        try:
            dub_video(str(video), out_path, model=model, work_dir=video_work_dir, **kwargs)
            results.append((str(rel), "OK", out_path))
        except Exception as e:
            print(f"FAILED: {rel} — {e}")
            results.append((str(rel), "FAILED", str(e)))
            continue

    print(f"\n{'=' * 60}")
    print("BATCH SUMMARY")
    print(f"{'=' * 60}")
    for name, status, info in results:
        print(f"  {'OK' if status == 'OK' else 'FAIL'} | {name} | {info}")
    ok_count = sum(1 for _, s, _ in results if s == "OK")
    print(f"\n  {ok_count}/{len(results)} succeeded")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dub English videos into Chinese using VoxCPM"
    )
    parser.add_argument("input", help="Input video file or directory (with --batch)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file or directory. Default: {input}_cn.mp4 or {input_dir}_cn/")
    parser.add_argument("--batch", action="store_true", help="Batch mode: process all videos in input directory")
    parser.add_argument("--recursive", action="store_true", help="Batch mode: recurse into subdirectories (preserves folder structure in output)")
    parser.add_argument("--work-dir", default="dub_workspace", help="Working directory for intermediate files")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="VoxCPM model id (e.g. openbmb/VoxCPM2, openbmb/VoxCPM-0.5B)")
    parser.add_argument("--cfg-value", type=float, default=DEFAULT_CFG_VALUE, help="VoxCPM guidance scale")
    parser.add_argument("--inference-timesteps", type=int, default=DEFAULT_INFERENCE_TIMESTEPS, help="VoxCPM denoising steps")
    parser.add_argument("--ultimate-clone", action="store_true", help="Use prompt_wav_path+prompt_text cloning on long segments")
    parser.add_argument("--num-speakers", type=int, default=None, help="Hint for number of speakers")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--llm-api-key", default=None, help="LLM API key (or set LLM_API_KEY env var)")
    parser.add_argument("--llm-api-base", default="https://api.openai.com/v1", help="LLM API base URL")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--cleanup", action="store_true", help="Delete intermediate files after completion")
    parser.add_argument("--external-subs", default=None, help="Path to external subtitle file (.srt or .ass)")
    parser.add_argument("--no-external-subs", action="store_true", help="Disable auto-discovery of external subtitles")
    parser.add_argument("--audio-only-align", action="store_true",
                        help="Audio absorbs all time difference (no video slowdown/re-encoding, much faster)")

    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    llm_api_key = args.llm_api_key or os.environ.get("LLM_API_KEY")

    if not hf_token:
        print("Error: HuggingFace token required. Use --hf-token or set HF_TOKEN env var.")
        sys.exit(1)
    if not llm_api_key:
        print("Error: LLM API key required. Use --llm-api-key or set LLM_API_KEY env var.")
        sys.exit(1)

    common_kwargs = dict(
        work_dir=args.work_dir,
        hf_token=hf_token,
        llm_api_key=llm_api_key,
        llm_api_base=args.llm_api_base,
        llm_model=args.llm_model,
        model_id=args.model_id,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        ultimate_clone=args.ultimate_clone,
        num_speakers=args.num_speakers,
        cleanup=args.cleanup,
        external_subs=args.external_subs,
        no_external_subs=args.no_external_subs,
        audio_only_align=args.audio_only_align,
    )

    if args.batch:
        dub_batch(args.input, args.output, recursive=args.recursive, **common_kwargs)
    else:
        dub_video(args.input, args.output, **common_kwargs)


if __name__ == "__main__":
    main()
