"""
batch.py — Batch subtitle pipeline

Iterates over all files in the input folder and calls ``main.py`` as a subprocess
for each one — either transcribing videos or translating existing subtitles.

Transcribe mode (default):
  python batch.py                                           # all mp4 in input/
  python batch.py -translate true                           # transcribe + translate
  python batch.py -i D:/videos                              # custom input folder

Translate-only mode:
  python batch.py -transcribe false -translate true -ext .srt  # batch-translate subtitles

``-i`` and ``-o`` always point to **folders** (unlike main.py which accepts files).
"""

import subprocess, sys, os
from pathlib import Path


BASE_DIR = Path(sys.executable).parent.resolve() if getattr(sys, "frozen", False) \
    else Path(__file__).parent.resolve()


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


# ── CLI ────────────────────────────────────────────────────────────────────

def build_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description="Batch subtitle pipeline — process all files in a folder",
    )
    parser.add_argument("-i", "--input", default=None,
                        help="Input folder (default: input/)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output folder (default: output/)")
    parser.add_argument("-gpu-layers", type=int, default=None,
                        help="Number of GPU layers for the LLM "
                             "(default: auto-detect; 0 = CPU only)")
    parser.add_argument("-no-cache", action="store_true",
                        help="Skip word-level cache")
    parser.add_argument("-ext", default=None,
                        help="File extension to process "
                             "(default: .mp4 for transcribe, .srt for translate-only)")
    parser.add_argument("-transcribe", type=_str_to_bool, default=True,
                        help="Enable transcription (default: true)")
    parser.add_argument("-translate", type=_str_to_bool, default=False,
                        help="Enable translation (default: false)")
    parser.add_argument("-transl_backend", default="local",
                        choices=["local", "deepseek", "openai", "qwen",
                                 "gemini", "anthropic"],
                        help="Translation backend (default: local)")
    parser.add_argument("-transl_model", default=None,
                        help="Model name for translation backend "
                             "(default: per-backend default)")
    parser.add_argument("-seg_backend", default="local",
                        choices=["local", "deepseek", "openai", "qwen",
                                 "gemini", "anthropic"],
                        help="Segmentation backend (default: local)")
    parser.add_argument("-seg_model", default=None,
                        help="Model name for segmentation backend "
                             "(default: per-backend default)")
    parser.add_argument("-mode", default=None,
                        choices=["accurate", "flexible"],
                        help="Translation mode (from scripts/translate.py). Affects "
                             "sliding-window size and timecode passing: "
                             "accurate (2-line window, line-by-line precision) "
                             "or flexible (4-line window, timecode-aware, "
                             "for online APIs). Only meaningful when "
                             "-translate is true. (default: accurate)")
    return parser


def entry():
    parser = build_parser()
    args = parser.parse_args()

    # ── Validate control flags ─────────────────────────────────────────
    if not args.transcribe and not args.translate:
        print()
        print("  At least one of -transcribe / -translate must be true.")
        print()
        parser.print_help()
        sys.exit(1)

    # ── -mode requires -translate true ─────────────────────────────────
    if args.mode is not None and not args.translate:
        print()
        print("  [ERR] -mode only applies when -translate is true.")
        print()
        parser.print_help()
        sys.exit(1)

    # ── Resolve input folder ───────────────────────────────────────────
    user_specified_input = args.input is not None
    if user_specified_input:
        input_dir = Path(args.input)
        if not input_dir.is_absolute():
            input_dir = BASE_DIR / args.input
    else:
        input_dir = BASE_DIR / "input"

    # ── Resolve extension(s) ──────────────────────────────────────────
    if args.ext is not None:
        exts = [args.ext if args.ext.startswith(".") else f".{args.ext}"]
    else:
        exts = [".srt", ".txt"] if not args.transcribe else [".mp4"]

    # ── Validate input folder existence ────────────────────────────────
    if not input_dir.exists():
        if user_specified_input:
            print(f"[ERR] Input folder not found: {input_dir}")
        else:
            _print_default_help(parser)
            print(f"[ERR] Default input folder {input_dir} not found")
        sys.exit(1)

    # ── Scan for matching files ────────────────────────────────────────
    files = sorted(
        f for ext in exts for f in input_dir.glob(f"*{ext}")
    )
    if not files:
        if not args.transcribe:
            hint = "No subtitle files (.srt or .txt) found"
        else:
            hint = f"No *{exts[0]} files found"
        if user_specified_input:
            print(f"[ERR] {hint} in {input_dir}/")
        else:
            _print_default_help(parser)
            print(f"[ERR] {hint} in default input folder {input_dir}/")
        sys.exit(1)

    # ── Validate translate-only input is subtitle files ────────────────
    if not args.transcribe and args.translate:
        bad = [f for f in files if f.suffix.lower() not in (".srt", ".txt")]
        if bad:
            print(f"[ERR] Translate-only mode requires .srt or .txt files, "
                  f"but found: {', '.join(f.name for f in bad[:5])}")
            sys.exit(1)

    # ── Resolve output folder ──────────────────────────────────────────
    output_dir = args.output
    if output_dir is None:
        output_dir = BASE_DIR / "output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Print summary ──────────────────────────────────────────────────
    mode_label = "Translate subtitles" if not args.transcribe else (
        "Transcribe + translate" if args.translate else "Transcribe video")
    print(f"  Mode:          {mode_label}")
    print(f"  Input folder:  {input_dir}/")
    print(f"  Output folder: {output_dir}/")
    print(f"  Found {len(files)} file(s) to process:\n")
    for f in files:
        print(f"    - {f.name}")

    # ── Process each file ──────────────────────────────────────────────────────
    """
    Each file is processed by a standalone main.py subprocess.
    When the subprocess exits, the CUDA context is fully reclaimed by the driver
    — identical to running ``python main.py -i <file>`` manually for each file.
    """
    main_script = [str((BASE_DIR / "main.exe").resolve())] \
        if getattr(sys, "frozen", False) \
        else [sys.executable or "python", str((BASE_DIR / "main.py").resolve())]

    results = []
    for f in files:
        stem = f.stem
        print(f"\n{'='*60}")
        print(f"  [{stem}]")
        print(f"{'='*60}")

        # Build output path — translate-only uses _CN suffix
        if not args.transcribe:
            out_path = str((output_dir / f"{stem}_{'CN'}.srt").resolve())
        else:
            out_path = str((output_dir / f"{stem}.srt").resolve())

        cmd = main_script + [
            "-i", str(f.resolve()),
            "-o", out_path,
        ]
        if not args.transcribe:
            cmd += ["-transcribe", "false"]
        if args.gpu_layers is not None:
            cmd += ["-gpu-layers", str(args.gpu_layers)]
        if args.no_cache:
            cmd.append("-no-cache")
        cmd += ["-translate", "true" if args.translate else "false"]
        if args.seg_backend != "local":
            cmd += ["-seg_backend", args.seg_backend]
            if args.seg_model:
                cmd += ["-seg_model", args.seg_model]
        if args.transl_backend != "local":
            cmd += ["-transl_backend", args.transl_backend]
            if args.transl_model:
                cmd += ["-transl_model", args.transl_model]
        if args.mode is not None:
            cmd += ["-mode", args.mode]

        result = subprocess.run(cmd)

        if result.returncode == 0:
            if not args.transcribe:
                out_srt = output_dir / f"{stem}_CN.srt"
                out_txt = output_dir / f"{stem}_CN.txt"
            else:
                out_srt = output_dir / f"{stem}.srt"
                out_txt = output_dir / f"{stem}.txt"
            srt_path = str(out_srt) if out_srt.exists() else None
            txt_path = str(out_txt) if out_txt.exists() else None
            if srt_path:
                results.append({"name": stem, "srt": srt_path, "txt": txt_path})
                print(f"  [{stem}] done ✓")
            else:
                print(f"  [{stem}] failed ✗: output file not generated")
        else:
            print(f"  [{stem}] failed ✗ (exit code {result.returncode})")

    # ── Summary ────────────────────────────────────────────────────────────────
    n_ok = len(results)
    n_fail = len(files) - n_ok
    print(f"\n{'='*60}")
    print(f"  Batch complete: {n_ok} succeeded, {n_fail} failed")
    print(f"{'='*60}")
    for r in results:
        print(f"    {r['name']}: {r['srt']}")
    print()


def _print_default_help(parser):
    """Print usage examples when default setup has no work to do."""
    print()
    print("Usage:")
    print(f"  python batch.py [options]")
    print()
    print("Examples:")
    print(f"  python batch.py")
    print(f"  python batch.py -i D:/videos")
    print(f"  python batch.py -o D:/output")
    print(f"  python batch.py -ext .mkv")
    print(f"  python batch.py -translate true")
    print(f"  python batch.py -transcribe false -translate true -ext .srt")
    print()
    print("For details: use -h or --help")
    print()


if __name__ == "__main__":
    print("=" * 55)
    print("  M2L3 - Batch Pipeline")
    print("=" * 55)
    entry()
