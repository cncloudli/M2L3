# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller packaging spec for M2L3.

Produces a single ``dist/M2L3/`` directory containing all three executables
that share one ``_internal/`` dependency folder::

    dist/M2L3/
    ├── main.exe              # Transcription + translation entry point
    ├── translate.exe          # Standalone translation entry point
    ├── batch.exe              # Batch processing entry point
    ├── _internal/             # Shared dependencies
    │   ├── tools/             # tools Python modules + tools/llama/ binaries
    │   ├── torch/ / whisperx/ / …
    │   └── python3*.dll
    ├── models/                # (keep as-is)
    ├── input/                 # (keep as-is)
    ├── output/                # (keep as-is)
    ├── cache/                 # (keep as-is)
    ├── docs/                  # (keep as-is)
    └── translate_config.json  # (keep as-is)

Usage:
    cd M2L3/
    pyinstaller packaging.spec
"""

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

HOMEPATH = SPECPATH


def _collect_llama():
    """Collect every file in tools/llama/ (except .gitkeep) as bundle data.

    PyInstaller copies each entry into the bundle at ``tools/llama/<filename>``,
    preserving the relative path so that frozen-aware code can find them via
    ``_BUNDLE_DIR / "tools" / "llama" / "llama-server.exe"``.
    """
    src = os.path.join(HOMEPATH, "tools", "llama")
    if not os.path.isdir(src):
        return []
    return [
        (os.path.join(src, name), "tools/llama")
        for name in os.listdir(src)
        if name != ".gitkeep"
    ]


def _collect_hub_models():
    """Recursively collect ``models/hub/`` as bundle data.

    Contains pre-downloaded Silero VAD (~35 MB), Wav2Vec2 checkpoint
    (~360 MB), and ``nltk_data/`` (~6 MB) — all small enough to ship
    with the executable so users don't need to download them separately.
    """
    hub_src = os.path.join(HOMEPATH, "models", "hub")
    if not os.path.isdir(hub_src):
        return []
    result = []
    for root, dirs, files in os.walk(hub_src):
        rel = os.path.relpath(root, HOMEPATH)
        result.extend(
            (os.path.join(root, f), rel)
            for f in files
        )
    return result


block_cipher = None

# ---------------------------------------------------------------------------
#  Analysis — trace all imports from all three entry points
# ---------------------------------------------------------------------------
a = Analysis(
    ["main.py", "scripts/translate.py", "batch.py"],
    pathex=[HOMEPATH],
    binaries=[],
    datas=_collect_llama() + _collect_hub_models() + collect_data_files("pyannote.audio") + collect_data_files("whisperx"),
    hiddenimports=[
        # tools modules used by main / translate / batch
        "tools.llm_call",
        "tools.segment",
        "tools.download_models",     # standalone utility (not imported by entry points)
        # WhisperX internals that PyInstaller may miss
        "whisperx.alignment",
        "whisperx.asr",
        "whisperx.vad",
        "whisperx.utils",
    ],
    hookspath=[],
    hooksconfig={},
    # Exclude large GUI / data-science libs that are never used
    excludes=[
        "benchmark",       # benchmark scripts — not for distribution
        "tkinter",
        "matplotlib",
        "notebook",
        "jupyter",
        "jupyter_client",
        "ipython",
        "IPython",
        "distributed",
        "tornado",
    ],
    runtime_hooks=[],
    noarchive=False,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, cipher=block_cipher)

# ---------------------------------------------------------------------------
#  EXE — one thin launcher per entry point
# ---------------------------------------------------------------------------
# ``exclude_binaries=True`` tells each EXE that its binaries live in the
# shared COLLECT directory, not embedded inside the .exe itself.
_COMMON_EXE_OPTS = dict(
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,              # upx True adds ~10 min build time for marginal gain
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory="_internal",
)

exe_main = EXE(
    pyz,
    [s for s in a.scripts if s[0] == "main"],
    a.binaries,
    a.datas,
    [],
    exclude_binaries=True,
    name="main",
    **_COMMON_EXE_OPTS,
)

exe_translate = EXE(
    pyz,
    [s for s in a.scripts if "translate" in str(s[0])],
    a.binaries,
    a.datas,
    [],
    exclude_binaries=True,
    name="translate",
    **_COMMON_EXE_OPTS,
)

exe_batch = EXE(
    pyz,
    [s for s in a.scripts if s[0] == "batch"],
    a.binaries,
    a.datas,
    [],
    exclude_binaries=True,
    name="batch",
    **_COMMON_EXE_OPTS,
)

# ---------------------------------------------------------------------------
#  COLLECT — one shared directory with all three launchers + dependencies
# ---------------------------------------------------------------------------
coll = COLLECT(
    exe_main,
    exe_translate,
    exe_batch,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="M2L3",
)
