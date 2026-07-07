"""
Pre-download all models needed by M2L3 to local cache,
so the pipeline can run fully offline.

Usage:
    python tools/download_models.py

This will download (if not already cached):
  1. Silero VAD        →  C:/Users/<USER>/.cache/torch/hub/snakers4_silero-vad_master/
  2. Wav2Vec2 align    →  C:/Users/<USER>/.cache/torch/hub/checkpoints/  (360 MB)
  3. NLTK punkt_tab    →  C:/Users/<USER>/AppData/Roaming/nltk_data/
"""
import io
import os
import shutil
import sys
import urllib.request
import warnings
import zipfile
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Bypass proxy for all downloads ──────────────────────────────────────
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "*"

TORCH_CACHE = Path.home() / ".cache" / "torch" / "hub"
NLTK_DIR = Path.home() / "AppData" / "Roaming" / "nltk_data"


def download_silero_vad() -> bool:
    target = TORCH_CACHE / "snakers4_silero-vad_master"
    if target.exists():
        print(f"[SKIP] Silero VAD already cached at {target}")
        return True

    url = "https://github.com/snakers4/silero-vad/archive/master.zip"
    zip_path = TORCH_CACHE / "silero-vad-master.zip"
    print(f"Downloading Silero VAD ({url}) ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(zip_path, "wb") as f:
                f.write(resp.read())
        print(f"  Downloaded ({zip_path.stat().st_size / 1024:.0f} KB)")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(TORCH_CACHE)
        extracted = TORCH_CACHE / "silero-vad-master"
        if extracted.exists():
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(extracted), str(target))
        zip_path.unlink()
        print(f"[OK] Silero VAD → {target}")
        return True
    except Exception as e:
        print(f"[ERR] Silero VAD download failed: {e}")
        return False


def download_wav2vec2() -> bool:
    import torchaudio

    chk_dir = TORCH_CACHE / "checkpoints"
    chk_dir.mkdir(parents=True, exist_ok=True)

    model_file = chk_dir / "wav2vec2_fairseq_base_ls960_asr_ls960.pth"
    if model_file.exists():
        print(f"[SKIP] Wav2Vec2 already cached ({model_file.stat().st_size / 1024 / 1024:.0f} MB)")
        return True

    print("Downloading Wav2Vec2 alignment model (WAV2VEC2_ASR_BASE_960H, 360 MB) ...")
    try:
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        bundle.get_model()
        print(f"[OK] Wav2Vec2 → {model_file}")
        return True
    except Exception as e:
        print(f"[ERR] Wav2Vec2 download failed: {e}")
        return False


def download_nltk_data() -> bool:
    import nltk

    # Clean broken downloads
    punkt_zip = NLTK_DIR / "tokenizers" / "punkt.zip"
    if punkt_zip.exists():
        try:
            punkt_zip.unlink()
        except PermissionError:
            pass

    print("Downloading NLTK punkt_tab ...")
    try:
        nltk.download("punkt_tab", quiet=True)
        nltk.download("punkt", quiet=True)
        # Verify
        nltk.data.find("tokenizers/punkt")
        print("[OK] NLTK punkt ready")
        return True
    except Exception as e:
        print(f"[ERR] NLTK download failed: {e}")
        return False


def main():
    print("=" * 55)
    print("  M2L3 — Model Pre-downloader")
    print("=" * 55)
    print()

    ok = 0
    fail = 0

    print("── Silero VAD ───────────────────────────────────")
    if download_silero_vad():
        ok += 1
    else:
        fail += 1

    print()
    print("── Wav2Vec2 Alignment ───────────────────────────")
    if download_wav2vec2():
        ok += 1
    else:
        fail += 1

    print()
    print("── NLTK punkt_tab ───────────────────────────────")
    if download_nltk_data():
        ok += 1
    else:
        fail += 1

    print()
    print("── Summary ──────────────────────────────────────")
    print(f"  {ok} / {ok + fail} models ready")
    if fail > 0:
        print("  Some downloads failed — check errors above")
    else:
        print("  All models pre-downloaded! Run `python main.py -i <video>` offline.")
    print("=" * 55)


if __name__ == "__main__":
    main()
