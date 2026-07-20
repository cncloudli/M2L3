"""
M2L3 — Subtitle generation pipeline entry point.

Orchestrates transcription (:mod:`transcribe`) with optional translation
(:mod:`translate`).

Usage:
  python main.py -i input/test.mp4                                    # transcribe only
  python main.py -i input/test.mp4 -translate true                    # transcribe + translate
  python main.py -transcribe false -translate true -i input/test.srt  # translate existing subs
"""

import os, sys
from pathlib import Path

from scripts.transcribe import transcribe_file, print_banner
from scripts.translate import translate_file
from tools.llm_call import load_api_config
from tools.config import get_app_dir

BASE_DIR = get_app_dir()


def _str_to_bool(v):
    """Convert a string to a boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y", "t"):
        return True
    if v.lower() in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(
        f"Expected true/false, got {v!r}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="M2L3: subtitle generation pipeline",
        add_help=False,
    )
    parser.add_argument("input_pos", nargs="?",
                        help="Path to input file (positional)")
    parser.add_argument("-i", "--input",
                        help="Path to input file (default: input/input.mp4)")
    parser.add_argument("-o", "--output",
                        help="Output file or directory path")
    parser.add_argument("-h", "--help", action="help",
                        help="Show this help message and exit")
    # transcribe / translate control flags (take true/false values)
    parser.add_argument("-transcribe", type=_str_to_bool, default=True,
                        help="Enable transcription (default: true)")
    parser.add_argument("-translate", type=_str_to_bool, default=False,
                        help="Enable translation (default: false)")
    # model / backend args
    parser.add_argument("-gpu-layers", type=int, default=None,
                        help="Number of local LLM layers to offload to GPU "
                             "(default: auto-detect based on VRAM; 0 = CPU only)")
    parser.add_argument("-no-cache", action="store_true",
                        help="Disable word-level caching (caching is enabled by default)")
    parser.add_argument("-transl_backend", default="local",
                        choices=["local", "deepseek", "openai", "qwen",
                                 "gemini", "anthropic"],
                        help="Translation backend (default: local)")
    parser.add_argument("-transl_model", default=None,
                        help="Model name for the translation backend "
                             "(default: per-backend default)")
    parser.add_argument("-mode", default=None,
                        choices=["accurate", "flexible"],
                        help="Translation mode (from scripts/translate.py). Affects "
                             "sliding-window size and timecode passing: "
                             "accurate (2-line window, line-by-line precision) "
                             "or flexible (4-line window, timecode-aware, "
                             "for online APIs). Only meaningful when "
                             "-translate is true. (default: accurate)")
    parser.add_argument("-tgt_lang", default=None,
                        help="Target language name (default: from translate_config.json)")
    parser.add_argument("-tgt_lang_code", default=None,
                        help="Language code for output filename suffix "
                             "(e.g. JP, CN; default: from translate_config.json)")
    parser.add_argument("-src_lang", default=None,
                        help="Source language name (default: from translate_config.json) "
                             "(only English supported for now)")
    parser.add_argument("-seg_backend", default="local",
                        choices=["local", "deepseek", "openai", "qwen",
                                 "gemini", "anthropic"],
                        help="Segmentation backend (default: local)")
    parser.add_argument("-seg_model", default=None,
                        help="Model name for the segmentation backend "
                             "(default: per-backend default)")
    args = parser.parse_args()

    # ── Validation ─────────────────────────────────────────────────────────
    if not args.transcribe and not args.translate:
        print()
        print("  M2L3 — subtitle generation pipeline")
        print()
        print("  At least one of -transcribe / -translate must be active.")
        print()
        parser.print_help()
        print()
        print("Examples:")
        print("  python main.py -i input/test.mp4")
        print("  python main.py -i input/test.mp4 -translate true")
        print("  python main.py -transcribe false -translate true -i output/test.srt")
        print()
        sys.exit(1)

    # ── -mode requires -translate true ───────────────────────────────────
    if args.mode is not None and not args.translate:
        print()
        print("  [ERR] -mode only applies when -translate is true.")
        print()
        parser.print_help()
        print()
        sys.exit(1)

    input_path = Path(args.input or args.input_pos or "")
    video_path = str(input_path) if input_path else None

    # ── Load API config once ──────────────────────────────────────────────
    api_cfg = load_api_config()
    api_key = api_cfg.get("openai_api_key", "")
    api_base = api_cfg.get("api_base_url", "")

    # ── Resolve local model ────────────────────────────────────────────────
    seg_model = args.seg_model or ("phi4" if args.seg_backend == "local" else None)
    transl_model = args.transl_model or ("phi4" if args.transl_backend == "local" else None)

    # ── Resolve translation mode ───────────────────────────────────────────
    transl_mode = args.mode or "accurate"

    # ═════════════════════════════════════════════════════════════════════════
    #  Mode 1: Transcribe (with optional translate)
    # ═════════════════════════════════════════════════════════════════════════
    if args.transcribe:
        if not video_path:
            video_path = str(BASE_DIR / "input" / "input.mp4")
        else:
            p = Path(video_path)
            if not p.is_absolute() and not p.parent.exists():
                candidate = BASE_DIR / "input" / p
                if candidate.exists():
                    video_path = str(candidate)

        cache_path = None
        if not args.no_cache:
            cache_dir = BASE_DIR / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = str(cache_dir / f"{Path(video_path).stem}_words.json")

        user_max_chars = 120
        user_max_dur = 9.0
        user_max_words = 30
        user_min_words = 4

        # Resolve output kwargs (same logic as before)
        if args.output:
            output_path = Path(args.output)
            if output_path.suffix.lower() in ('.srt', '.txt'):
                stem = output_path.with_suffix('').name
            else:
                stem = output_path.name
            parent = output_path.parent
            if str(parent) in ('.', ''):
                output_kwargs = {"output_stem": stem}
            else:
                output_kwargs = {"output_stem": stem,
                                 "output_dir": str(parent.resolve())}
        else:
            output_kwargs = {}

        print_banner()
        print(f"  Input:  {video_path}")
        print("=" * 55)
        print()

        srt_path, txt_path = transcribe_file(
            video_path=video_path, words_cache_path=cache_path,
            max_chars=user_max_chars, max_dur=user_max_dur,
            max_words=user_max_words,
            min_words=user_min_words,
            seg_backend=args.seg_backend, seg_model=seg_model,
            api_key=api_key, base_url=api_base,
            gpu_layers=args.gpu_layers,
            **output_kwargs)

        # ── Optional: translation after transcription ─────────────────
        if args.translate and srt_path and os.path.exists(srt_path):
            print()
            print("=" * 55)
            print("  Translating output…")
            print("=" * 55)
            translate_file(
                srt_path,
                output_dir=str(Path(srt_path).parent),
                transl_backend=args.transl_backend, transl_model=transl_model,
                api_key=api_key, base_url=api_base,
                gpu_layers=args.gpu_layers,
                mode=transl_mode,
                tgt_lang=args.tgt_lang,
                tgt_lang_code=args.tgt_lang_code,
                src_lang=args.src_lang,
            )

    # ═════════════════════════════════════════════════════════════════════════
    #  Mode 2: Translate-only (existing subtitles)
    # ═════════════════════════════════════════════════════════════════════════
    elif args.translate:
        if not video_path:
            print()
            print("[ERR] Input file is required when -transcribe is false.")
            parser.print_help()
            sys.exit(1)

        ext = Path(video_path).suffix.lower()
        if ext not in ('.srt', '.txt'):
            print()
            print("[ERR] Input must be a subtitle file (.srt or .txt) "
                  "when -transcribe is false.")
            print()
            parser.print_help()
            sys.exit(1)

        print("=" * 55)
        print("  Translating existing subtitles…")
        print("=" * 55)
        print(f"  Input:  {video_path}")
        print()

        # Resolve output path
        if args.output:
            output_path = Path(args.output)
            stem = output_path.with_suffix("").name
            parent = output_path.parent
            if str(parent) in ('.', ''):
                output_dir = str(BASE_DIR / "output")
            else:
                output_dir = str(parent.resolve())
        else:
            output_dir = None
            stem = None

        srt_result, txt_result = translate_file(
            video_path,
            output_stem=stem,
            output_dir=output_dir,
            transl_backend=args.transl_backend, transl_model=transl_model,
            api_key=api_key, base_url=api_base,
            gpu_layers=args.gpu_layers,
            mode=transl_mode,
            tgt_lang=args.tgt_lang,
            tgt_lang_code=args.tgt_lang_code,
            src_lang=args.src_lang,
        )
        if srt_result is None and txt_result is None:
            print("[ERR] Translation failed")
            sys.exit(1)

    print()
    print("=" * 55)
    print(f"  [OK] Done!")
    print("=" * 55)
