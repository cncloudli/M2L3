"""
M2L3 — subtitle generation pipeline with WhisperX + Wav2Vec2 + LLM segmentation.

Pipeline:
  1. Load WhisperX ASR model (faster-whisper-large-v3) with Silero VAD
  2. Transcribe audio to text segments
  3. Wav2Vec2 phoneme-level forced alignment for precise word timestamps
  4. LLM-based punctuation restoration and sentence segmentation (Phi-4)
  5. Export to SRT + TXT
"""

import os, sys, subprocess
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# Proxy / GPU environment — must run before any model download
# ══════════════════════════════════════════════════════════════════════════════
import tools.config  # noqa: F401  (sets HTTP_PROXY, NO_PROXY, CUDA_VISIBLE_DEVICES)

import whisperx
import torch  # noqa: E402  (needed for hub monkey-patch below)

"""
Torch hub offline patch — Silero VAD is already cached locally 
but torch.hub._parse_repo_info on Windows connects to GitHub 
to find the default branch and only catches HTTPError / URLError.
RemoteDisconnected (a ConnectionError, subclass of OSError) falls through and kills the pipeline.  
This wrapper catches all network-level errors and falls back to cache.
"""
_orig_parse_repo_info = getattr(torch.hub, '_parse_repo_info', None)
if _orig_parse_repo_info:
    import os as _os
    def _patched_parse_repo_info(github):
        try:
            return _orig_parse_repo_info(github)
        except OSError as _e:
            # RemoteDisconnected, ConnectionResetError, etc. → check cache
            repo_info, ref = github.split(":") if ":" in github else (github, None)
            repo_owner, repo_name = repo_info.split("/")
            _hub_dir = torch.hub.get_dir()
            for _possible_ref in ("main", "master"):
                if _os.path.exists(f"{_hub_dir}/{repo_owner}_{repo_name}_{_possible_ref}"):
                    return repo_owner, repo_name, _possible_ref
            raise RuntimeError(
                "No internet connection and the repo is not in the torch hub cache "
                f"({_hub_dir})"
            ) from _e
    torch.hub._parse_repo_info = _patched_parse_repo_info

from tools.env_check import check_ffmpeg, check_cuda
from tools.cache import save_words_cache
from tools.export import export_srt, export_txt  # , export_word_level  # (word-level debug, commented out)
from tools.extract import extract_words_from_result, fix_word_timestamps

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
MODEL_PATH = BASE_DIR / "models" / "faster-whisper-large-v3"


# =============================================================================
#  Main pipeline
# =============================================================================

def video_to_srt(video_path=None, output_dir=None, output_stem=None,
                 max_chars=120, max_dur=9.0, max_words=30,
                 min_words=4, min_dur=1.5,
                 words_cache_path=None,
                 llm_model="phi4", gpu_layers=None):
    """Full pipeline: transcribe → align → LLM segment → export SRT+TXT."""
    check_ffmpeg()
    cuda_ok = check_cuda()

    if video_path is None:
        video_path = BASE_DIR / "input" / "input.mp4"
    video_path = Path(video_path)
    if output_dir is None:
        output_dir = BASE_DIR / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_stem is None:
        output_stem = f"{video_path.stem}"
    srt_output = output_dir / f"{output_stem}.srt"

    if not video_path.exists():
        print(f"[ERR] File not found: {video_path}")
        print()
        print("Usage:")
        print(f"  python {Path(__file__).name} -i <input_video> [-o <output_srt>]")
        print()
        print("Examples:")
        print(f"  python {Path(__file__).name} -i input/input.mp4")
        print(f"  python {Path(__file__).name} -i input/input.mp4 -o my_subtitles.srt")
        print(f"  python {Path(__file__).name} -i D:/videos/lecture.mp4 -o D:/output/lecture.srt")
        print()
        print("Supported formats: .mp4, .mkv, .avi, .mov, .wav, .mp3, .m4a, .flac, etc.")
        sys.exit(1)
    if not (MODEL_PATH / "config.json").exists():
        print("[ERR] Model incomplete or config.json missing")
        sys.exit(1)

    device = "cuda" if cuda_ok else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Loading WhisperX model: {MODEL_PATH.name} | {device.upper()} | {compute_type}")

    # ── Step 1: Load Whisper ASR model ──
    model = whisperx.load_model(
        str(MODEL_PATH),
        device=device,
        compute_type=compute_type,
        asr_options={
            "beam_size": 5,
            "temperatures": [0.0],
            "no_speech_threshold": 0.38,
            "compression_ratio_threshold": 2.4,
            "condition_on_previous_text": False,
            "word_timestamps": False,
        },
        language="en",
        vad_method="silero",
        vad_options={
            "chunk_size": 30,
            "vad_onset": 0.500,
            "vad_offset": 0.363,
        },
        download_root=str(BASE_DIR / "models"),
    )

    # ── Step 2: Transcribe ──
    print("Transcribing with batched Whisper...")
    result = model.transcribe(
        str(video_path),
        batch_size=4,
        print_progress=True,
    )
    print(f"  Language: {result['language']} | {len(result['segments'])} segments")

    # ── Step 3: Wav2Vec2 forced alignment ──
    print("Loading Wav2Vec2 alignment model...")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device,
    )

    print("Aligning with Wav2Vec2 (phoneme-level)...")
    result_aligned = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        str(video_path),
        device,
        return_char_alignments=False,
        print_progress=True,
    )
    result = result_aligned

    # Cache word-level data
    raw_words = extract_words_from_result(result)

    # ── Fix inflated word timestamps (common to all pipelines) ──
    if raw_words:
        fix_word_timestamps(raw_words)
        print(f"Fixed word timestamps: {len(raw_words)} words")

    if raw_words and words_cache_path:
        save_words_cache(raw_words, words_cache_path)
        print(f"Cached: {words_cache_path} ({len(raw_words)} words)")

    # # Debug: export word-level SRT/TXT (remove for production)
    # if raw_words:
    #     export_word_level(raw_words, output_stem, str(output_dir))

    # ── Step 4: LLM Segmentation ──
    print("Segmenting with LLM pipeline...")
    from tools.llm_pipeline import LLMPipeline
    _llm = LLMPipeline(model=llm_model, gpu_layers=gpu_layers)
    try:
        result['segments'] = _llm.segment(
            raw_words, max_chars=max_chars, max_dur=max_dur,
            max_words=max_words, min_words=min_words, min_dur=min_dur,
        )
    finally:
        _llm.stop()
    print(f"[OK] LLM pipeline: {len(raw_words)} words -> {len(result['segments'])} segments")

    print(f"Exporting SRT: {srt_output}")
    export_srt(result, str(srt_output))
    print(f"[OK] SRT saved: {srt_output.resolve()}")
    txt_output = srt_output.with_suffix(".txt")
    export_txt(result, str(txt_output), pure_text=False)
    print(f"[OK] TXT saved: {txt_output}")
    return str(srt_output), str(txt_output)


# =============================================================================
#  CLI entry point
# =============================================================================

# ── Backward-compatible aliases for debug scripts ──────────────────────────
from tools.format import format_srt_time  # noqa: F401  (used by debug scripts)


def _print_banner():
    print("=" * 55)
    print("  M2L3 - Subtitle Generation Pipeline")
    print("=" * 55)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="M2L3: subtitle generation with WhisperX + Wav2Vec2 alignment",
    )
    parser.add_argument("video", nargs="?", help="Path to input video file (positional)")
    parser.add_argument("-i", "--input", help="Path to input video file (default: input/input.mp4)")
    parser.add_argument("-o", "--output", help="Output SRT file path (default: ./output/<input_stem>.srt)")
    parser.add_argument("-gpu-layers", type=int, default=None,
                        help="Number of model layers to offload to GPU "
                             "(default: auto-detect based on VRAM; "
                             "0 = CPU only)")
    parser.add_argument("-no-cache", action="store_true",
                        help="Skip word-level caching (enabled by default)")
    parser.add_argument("-translate", action="store_true",
                        help="Translate output to target language after transcription")
    parser.add_argument("-backend", default="local",
                        help="Translation backend for translate.py "
                             "(default: local)")
    parser.add_argument("-model", default=None,
                        help="Model name for the translation backend "
                             "(default: per-backend default)")
    args = parser.parse_args()

    # -backend / -model only make sense with -translate
    if (args.backend != "local" or args.model is not None) and not args.translate:
        print("[ERR] -backend and -model require -translate")
        print()
        parser.print_help()
        sys.exit(1)

    video_path = args.input or args.video
    cache_path = None

    user_max_chars = 120
    user_max_dur = 9.0
    user_max_words = 30
    user_min_words = 4
    user_min_dur = 1.5

    if args.output:
        output_path = Path(args.output)
        if output_path.suffix.lower() in ('.srt', '.txt'):
            stem = output_path.with_suffix('').stem
        else:
            stem = output_path.stem
        parent = output_path.parent
        if str(parent) in ('.', ''):
            kwargs = {"output_stem": stem}
        else:
            kwargs = {"output_stem": stem, "output_dir": str(parent.resolve())}
    else:
        kwargs = {}

    # Default: look in input/ directory if not a full path
    if video_path:
        p = Path(video_path)
        if not p.is_absolute() and not p.parent.exists():
            candidate = BASE_DIR / "input" / p
            if candidate.exists():
                video_path = str(candidate)
    else:
        video_path = str(BASE_DIR / "input" / "input.mp4")

    if not args.no_cache:
        cache_dir = BASE_DIR / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = str(cache_dir / f"{Path(video_path).stem}_words.json")

    _print_banner()
    print(f"  Input:  {video_path}")
    print("=" * 55)
    print()

    srt_path, txt_path = video_to_srt(
        video_path=video_path, words_cache_path=cache_path,
        max_chars=user_max_chars, max_dur=user_max_dur,
        max_words=user_max_words,
        min_words=user_min_words, min_dur=user_min_dur,
        llm_model="phi4", gpu_layers=args.gpu_layers,
        **kwargs)

    # ── Optional: translation ─────────────────────────────────
    if args.translate and srt_path and os.path.exists(srt_path):
        print()
        print("=" * 55)
        print("  Translating output…")
        print("=" * 55)
        translate_script = str((BASE_DIR / "translate.py").resolve())
        translate_cmd = [sys.executable, translate_script, "-i", srt_path]
        if args.backend != "local":
            translate_cmd += ["-backend", args.backend]
            if args.model:
                translate_cmd += ["-model", args.model]
        subprocess.run(translate_cmd, cwd=str(BASE_DIR))

    print()
    print("=" * 55)
    print(f"  [OK] Done!")
    print("=" * 55)
