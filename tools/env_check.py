"""Environment checks: FFmpeg availability & CUDA readiness."""

import subprocess
import sys

import torch


def check_ffmpeg():
    """Verify FFmpeg is installed and callable."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        print("[OK] FFmpeg ready")
    except Exception:
        print("[ERR] FFmpeg not found!  Run: winget install Gyan.FFmpeg")
        sys.exit(1)


def check_cuda():
    """Verify CUDA GPU is available and usable (kernel-launch test).

    Returns True if CUDA is usable, False otherwise.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        try:
            t = torch.tensor([1.0, 2.0]).cuda()
            t.sum()  # force a real kernel launch
            del t
            print(f"[OK] CUDA ready | GPU: {gpu_name} | VRAM: {gpu_mem:.1f} GB")
        except Exception as e:
            print(f"[WARN] GPU {gpu_name} detected but not usable ({e})")
            print("[WARN] Falling back to CPU (slower)")
            return False
    else:
        print("[WARN] CUDA not detected, using CPU (slower)")
    return torch.cuda.is_available()
