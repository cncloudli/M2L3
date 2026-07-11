"""
System configuration — proxy bypass & environment variables.

Runs at import time to ensure proxy settings are correct
before any download (PyTorch, HuggingFace, Silero VAD, etc.) happens.

Also exposes ``get_app_dir()`` and ``get_bundle_dir()`` for PyInstaller-aware
path resolution.  Use these instead of ``Path(__file__).parent`` in entry-point
scripts so that user-facing paths (models/, input/, output/, cache/,
translate_config.json) stay alongside the .exe while bundled internal files
(tools/llama/) are found inside the package.
"""

import os
import sys
from pathlib import Path

# Local FlClash proxy (127.0.0.1:7890) is set at the Windows level.
# Some CDNs (download.pytorch.org, huggingface.co) don't work through it,
# so exclude them from proxying.
_HTTP_PROXY = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy') or \
    'http://127.0.0.1:7890'
_NO_PROXY = os.environ.get('NO_PROXY') or os.environ.get('no_proxy') or ''
_NO_PROXY += ',*.pytorch.org,*.huggingface.co,huggingface.co,github.com,snakers4,*.huggingface.co,s3.amazonaws.com'

os.environ.setdefault('HTTP_PROXY', _HTTP_PROXY)
os.environ.setdefault('HTTPS_PROXY', _HTTP_PROXY)
os.environ['NO_PROXY'] = _NO_PROXY
os.environ['no_proxy'] = _NO_PROXY

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ══════════════════════════════════════════════════════════════════════════════
#  PyInstaller-aware path resolution helpers
# ══════════════════════════════════════════════════════════════════════════════


def get_app_dir():
    """Directory for **user-visible** data (models/, input/, output/, cache/, config).

    When packaged (PyInstaller ``--onedir``), this is the directory containing
    the .exe — so ``models/``, ``input/``, ``output/``, ``cache/`` and
    ``translate_config.json`` sit alongside the executable.

    When running as a plain Python script, this is the project root.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent.resolve()
    # In development: __file__ = tools/config.py → parent.parent = project root
    return Path(__file__).resolve().parent.parent


def get_bundle_dir():
    """Directory containing **bundled internal files** (tools/llama/*, compiled modules).

    * ``--onedir`` mode → the ``_internal/`` folder next to the .exe.
    * ``--onefile`` mode → the MEIPASS temporary extraction directory.
    * Plain script mode → the project root (same as :func:`get_app_dir`).
    """
    if getattr(sys, "frozen", False):
        internal = Path(sys.executable).parent / "_internal"
        if internal.is_dir():
            return internal
        if hasattr(sys, "_MEIPASS"):
            return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent
