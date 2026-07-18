"""
transcribe.py — WhisperX transcription + Wav2Vec2 alignment + LLM segmentation.

Pipeline:
  1. Load WhisperX ASR model (faster-whisper-large-v3) with Silero VAD
  2. Transcribe audio to text segments
  3. Wav2Vec2 phoneme-level forced alignment for precise word timestamps
  4. LLM-based punctuation restoration and sentence segmentation
  5. Export to SRT + TXT

Standalone usage:
  python transcribe.py -i input/test.mp4
  python transcribe.py -i input/test.mp4 -o output/test.srt
  python transcribe.py -i input/test.mp4 -seg_backend openai -seg_model gpt-4o
  python transcribe.py -i input/test.mp4 -gpu-layers 0 -no-cache
"""

import os, sys, warnings
from pathlib import Path

# ── Project root — must be on sys.path before tools imports ──────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ══════════════════════════════════════════════════════════════════════════════
# Proxy / GPU environment — must run before any model download
# ══════════════════════════════════════════════════════════════════════════════
import tools.config  # noqa: F401  (sets HTTP_PROXY, NO_PROXY, CUDA_VISIBLE_DEVICES, exposes helpers)
from tools.config import get_app_dir, get_bundle_dir

# ── Patch transformers for frozen env (MUST be before any whisperx/transformers import) ──
import tools._patch_transformers  # noqa: F401

# Suppress spurious torchcodec warning from pyannote.audio (fallback to FFmpeg is fine)
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

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

# ── Redirect torch hub cache to local models/hub/ ──
# Both Silero VAD (35 MB) and Wav2Vec2 checkpoints (360 MB) are
# pre-downloaded there so they ship with the packaged exe.
_HUB_DIR = get_bundle_dir() / "models" / "hub"
torch.hub.set_dir(str(_HUB_DIR))

# ── Redirect NLTK data path ──
# whisperx uses nltk punkt / punkt_tab tokenizers for sentence-aware
# forced alignment.  The data is bundled alongside the hub models.
import nltk
_NLTK_DIR = _HUB_DIR / "nltk_data"
if _HUB_DIR.exists() and _NLTK_DIR.exists():
    nltk.data.path.insert(0, str(_NLTK_DIR))

from tools.env_check import check_ffmpeg, check_cuda
from tools.cache import save_words_cache, load_words_cache
from tools.export import export_srt, export_txt
from tools.extract import extract_words_from_result, fix_word_timestamps
from tools.llm_call import load_api_config
from tools.segment import segment_words

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = get_app_dir()
MODEL_PATH = BASE_DIR / "models" / "faster-whisper-large-v3"


# =============================================================================
#  Main pipeline
# =============================================================================

def transcribe_file(video_path=None, output_dir=None, output_stem=None,
                 max_chars=120, max_dur=9.0, max_words=30,
                 min_words=4,
                 words_cache_path=None,
                 seg_backend="local", seg_model=None,
                 api_key="", base_url="", gpu_layers=None):
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

    # ══════════════════════════════════════════════════════════════════════════
    #  Decide: cache hit → skip transcription + alignment
    # ══════════════════════════════════════════════════════════════════════════
    use_cache = bool(words_cache_path and os.path.exists(words_cache_path))

    if use_cache:
        raw_words = load_words_cache(words_cache_path)
        print(f"[CACHE] Loaded {len(raw_words)} words from cache — "
              f"skipping transcription + alignment")
        result = {}
    else:
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

        # ── Free Whisper model before loading alignment model ──
        print("Freeing Whisper model from GPU...")
        del model
        if device == "cuda":
            torch.cuda.synchronize()
        torch.cuda.empty_cache()

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

        # ── Fix inflated word timestamps ──
        if raw_words:
            fix_word_timestamps(raw_words)
            print(f"Fixed word timestamps: {len(raw_words)} words")

        if raw_words and words_cache_path:
            save_words_cache(raw_words, words_cache_path)
            print(f"Cached: {words_cache_path} ({len(raw_words)} words)")

        # ── Free alignment model before LLM segmentation ──
        print("Freeing alignment model from GPU...")
        del align_model, align_metadata
        if device == "cuda":
            torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════════════
    #  Step 4: LLM Segmentation (shared path — cache hit or full pipeline)
    # ══════════════════════════════════════════════════════════════════════════
    print("Segmenting with LLM pipeline...")
    segments = segment_words(
        raw_words, seg_backend=seg_backend, seg_model=seg_model,
        api_key=api_key, base_url=base_url, gpu_layers=gpu_layers,
        max_chars=max_chars, max_dur=max_dur,
        max_words=max_words, min_words=min_words,
    )
    print(f"[OK] LLM pipeline: {len(raw_words)} words -> {len(segments)} segments")

    # Build result dict for export (cache path starts with empty result)
    result["segments"] = segments

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

def print_banner():
    print("=" * 55)
    print("Eng Transcription & Mlt-Ly LLM-Based Sub Seg Pipeline")
    print("=" * 55)


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description="M2L3: transcribe audio/video to SRT with WhisperX + LLM segmentation",
    )
    parser.add_argument("video", nargs="?", help="Path to input video file (positional)")
    parser.add_argument("-i", "--input", help="Path to input video file (default: input/input.mp4)")
    parser.add_argument("-o", "--output", help="Output SRT file path (default: ./output/<input_stem>.srt)")
    parser.add_argument("-gpu-layers", type=int, default=None,
                        help="Number of model layers to offload to GPU "
                             "(default: auto-detect based on VRAM; 0 = CPU only)")
    parser.add_argument("-no-cache", action="store_true",
                        help="Skip word-level caching (enabled by default)")
    parser.add_argument("-seg_backend", default="local",
                        choices=["local", "deepseek", "openai", "qwen",
                                 "gemini", "anthropic"],
                        help="Segmentation backend (default: local)")
    parser.add_argument("-seg_model", default=None,
                        help="Model name for the segmentation backend "
                             "(default: per-backend default)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    video_path = args.input or args.video
    cache_path = None

    user_max_chars = 120
    user_max_dur = 9.0
    user_max_words = 30
    user_min_words = 4

    if args.output:
        output_path = Path(args.output)
        if output_path.suffix.lower() in ('.srt', '.txt'):
            stem = output_path.with_suffix('').name
        else:
            stem = output_path.name
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

    print_banner()
    print(f"  Input:  {video_path}")
    print("=" * 55)
    print()

    # Load API config for online backends
    api_cfg = load_api_config()
    api_key = api_cfg.get("openai_api_key", "")
    api_base = api_cfg.get("api_base_url", "")

    srt_path, txt_path = transcribe_file(
        video_path=video_path, words_cache_path=cache_path,
        max_chars=user_max_chars, max_dur=user_max_dur,
        max_words=user_max_words,
        min_words=user_min_words,
        seg_backend=args.seg_backend, seg_model=args.seg_model,
        api_key=api_key, base_url=api_base,
        gpu_layers=args.gpu_layers,
        **kwargs)

    print()
    print("=" * 55)
    print(f"  [OK] Done!")
    print("=" * 55)


if __name__ == "__main__":
    main()
