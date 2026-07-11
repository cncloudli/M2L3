"""
batch_pipeline.py — Batch subtitle pipeline

Iterates over all MP4 files in the input folder and calls ``main.py`` as a subprocess for each one.  
Works identically to running
``python main.py -i <file>`` manually for each file, except that
``-i`` accepts a folder instead of a single file.
Each file processes in its own subprocess; 
after exit the CUDA context is fully reclaimed by the driver so it does not affect the next file.

Usage:
  python batch_pipeline.py                          # all mp4 in input/, Phi-4 transcribe
  python batch_pipeline.py -i D:/videos              # custom input folder
  python batch_pipeline.py -o D:/output              # custom output folder
  python batch_pipeline.py -gpu-layers 25
  python batch_pipeline.py -no-cache                 # skip word-level cache
"""

import subprocess, sys, os
from pathlib import Path


BASE_DIR = Path(sys.executable).parent.resolve() if getattr(sys, "frozen", False) \
    else Path(__file__).parent.resolve()


# ── CLI ────────────────────────────────────────────────────────────────────

def build_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description="Batch subtitle pipeline — process all MP4 files in a folder",
    )
    parser.add_argument("-i", "--input", default=None,
                        help="Input folder (default: input/)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output folder (default: output/)")
    parser.add_argument("-gpu-layers", type=int, default=None,
                        help="Number of GPU layers for the LLM (default: auto-detect; 0 = CPU only)")
    parser.add_argument("-no-cache", action="store_true",
                        help="Skip word-level cache")
    parser.add_argument("-ext", default=".mp4",
                        help="File extension to process (default: .mp4)")
    parser.add_argument("-dry-run", action="store_true",
                        help="Only list files, do not run")
    parser.add_argument("-translate", action="store_true",
                        help="Translate output after transcription")
    parser.add_argument("-backend", default="local",
                        help="Translation backend (local / deepseek / openai / qwen / ollama / anthropic; default: local)")
    parser.add_argument("-model", default=None,
                        help="Model name for translation backend (default: per-backend default)")
    return parser


def entry():
    parser = build_parser()
    args = parser.parse_args()

    # -backend / -model only make sense with -translate
    if (args.backend != "local" or args.model is not None) and not args.translate:
        print("[ERR] -backend and -model require -translate")
        print()
        parser.print_help()
        sys.exit(1)

    # ── Detect if user explicitly specified -i ──────────────────────
    user_specified_input = args.input is not None
    if user_specified_input:
        input_dir = Path(args.input)
        if not input_dir.is_absolute():
            input_dir = BASE_DIR / args.input
    else:
        input_dir = BASE_DIR / "input"

    # ── Print help if default input dir is missing or empty ─────────
    def _print_help():
        print()
        print("Usage:")
        print(f"  python {Path(__file__).name} [options]")
        print()
        print("Examples:")
        print(f"  python {Path(__file__).name}")
        print(f"  python {Path(__file__).name} -i D:/videos")
        print(f"  python {Path(__file__).name} -o D:/output")
        print(f"  python {Path(__file__).name} -ext .mkv")
        print(f"  python {Path(__file__).name} -dry-run")
        print(f"  python {Path(__file__).name} -translate")
        print()
        print("For details: use -h or --help")
        print()

    if not input_dir.exists():
        if user_specified_input:
            print(f"[ERR] Input folder not found: {input_dir}")
        else:
            print(f"[ERR] Default input folder {input_dir} not found")
            _print_help()
        sys.exit(1)

    output_dir = args.output
    if output_dir is None:
        output_dir = BASE_DIR / "output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scan for matching files (default .mp4; use -ext .mp3 / .wav for audio formats)
    ext = args.ext if args.ext.startswith(".") else f".{args.ext}"
    video_files = sorted(input_dir.glob(f"*{ext}"))
    if not video_files:
        if user_specified_input:
            print(f"[ERR] No *{ext} files found in {input_dir}/")
        else:
            print(f"[ERR] No *{ext} files found in default input folder {input_dir}/")
            _print_help()
        sys.exit(1)

    print(f"  Input folder:  {input_dir}/")
    print(f"  Output folder: {output_dir}/")
    print(f"  Found {len(video_files)} file(s) to process:\n")

    for vf in video_files:
        print(f"    - {vf.name}")

    if args.dry_run:
        print("\n  [DRY-RUN] Exiting (no action taken)")
        return

    # ── Process each file ──────────────────────────────────────────────────
    """
    Each file is processed by a standalone main.py subprocess.
    When the subprocess exits, the CUDA context is fully reclaimed by the driver 
    — identical to running ``python main.py -i <file>`` manually for each file.
    """
    main_script = str((BASE_DIR / "main.exe").resolve()) if getattr(sys, "frozen", False) \
        else str((BASE_DIR / "main.py").resolve())

    results = []
    for vf in video_files:
        stem = vf.stem
        print(f"\n{'='*60}")
        print(f"  [{stem}]")
        print(f"{'='*60}")

        # Build the subprocess command
        cmd = [
            main_script,
            "-i", str(vf.resolve()),
            "-o", str((output_dir / f"{stem}.srt").resolve()),
        ]
        if args.gpu_layers is not None:
            cmd += ["-gpu-layers", str(args.gpu_layers)]
        if args.no_cache:
            cmd.append("-no-cache")

        result = subprocess.run(cmd)

        if result.returncode == 0:
            srt_path = str(output_dir / f"{stem}.srt")
            txt_path = str(output_dir / f"{stem}.txt")
            if os.path.exists(srt_path):
                results.append({"name": stem, "srt": srt_path, "txt": txt_path})
                print(f"  [{stem}] done ✓")
                # ── Translation ────────────────────────────────────────────
                if args.translate and os.path.exists(srt_path):
                    translate_script = str((BASE_DIR / "translate.exe").resolve()) \
                        if getattr(sys, "frozen", False) \
                        else str((BASE_DIR / "translate.py").resolve())
                    translate_cmd = [
                        translate_script, "-i", srt_path,
                    ]
                    if args.backend != "local":
                        translate_cmd += ["-backend", args.backend]
                        if args.model:
                            translate_cmd += ["-model", args.model]
                    subprocess.run(translate_cmd, cwd=str(BASE_DIR))
            else:
                print(f"  [{stem}] failed ✗: SRT file not generated")
        else:
            print(f"  [{stem}] failed ✗ (exit code {result.returncode})")

    # ── Summary ────────────────────────────────────────────────────────────
    n_ok = len(results)
    n_fail = len(video_files) - n_ok
    print(f"\n{'='*60}")
    print(f"  Batch complete: {n_ok} succeeded, {n_fail} failed")
    print(f"{'='*60}")
    for r in results:
        print(f"    {r['name']}: {r['srt']}")
    print()


if __name__ == "__main__":
    print("=" * 55)
    print("  M2L3 - Batch Pipeline")
    print("=" * 55)
    entry()
