"""
System configuration — proxy bypass & environment variables.

Runs at import time to ensure proxy settings are correct 
before any download (PyTorch, HuggingFace, Silero VAD, etc.) happens.
"""

import os

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
