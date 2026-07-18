# Multi-Layer LLM-Based Subtitle Segmentation (M2L3)

All-in-one subtitle generation tool integrating English transcription + subtitle segmentation + translation. All processing supports **fully offline operation**.

| Feature | Approach | Used in |
|---------|----------|---------|
| **ASR** | WhisperX (faster-whisper-large-v3) + Silero VAD | Transcription ([scripts/transcribe.py](scripts/transcribe.py)) |
| **Alignment** | Wav2Vec2 phoneme-level forced alignment | Transcription ([scripts/transcribe.py](scripts/transcribe.py)) |
| **Segment** | Local LLM via llama-server or online API | Transcription ([scripts/transcribe.py](scripts/transcribe.py)) |
| **Translate** | Local LLM via llama-server or online API | Translation ([scripts/translate.py](scripts/translate.py)) |

---

## File Reference

### Root directory

| File | Purpose |
|------|---------|
| [main.py](main.py) | **Orchestrator entry point.** Calls `scripts/transcribe.py` for transcription and/or `scripts/translate.py` for translation in a single run. |
| [batch.py](batch.py) | **Batch processing.** Iterates all files in an input folder, spawning one `main.py` subprocess per file (each gets a clean CUDA context). |
| [translate_config.json](translate_config.json) | **Translation config.** Language, punctuation, glossary, API keys — edited by the user, read by `scripts/translate.py` at startup. **Note: API keys also apply to subtitle segmentation.** |
| [README_CN.md](README_CN.md) | Chinese (中文) documentation — setup guide, usage, and file reference. |
| [README.md](README.md) | This file — English documentation. |

### `scripts/` directory

| Script | Purpose |
|--------|---------|
| [scripts/transcribe.py](scripts/transcribe.py) | **Transcription pipeline.** WhisperX ASR → Wav2Vec2 alignment → LLM segmentation → SRT + TXT export. Can run standalone. |
| [scripts/translate.py](scripts/translate.py) | **Translation.** Translates existing SRT (preserving timecodes) or TXT files using a local or API-based LLM. Can run standalone. |

### `tools/` directory

| Module | Purpose |
|--------|---------|
| [tools/\_\_init\_\_.py](tools/__init__.py) | Package marker — module docstring only. |
| [tools/config.py](tools/config.py) | **System configuration.** Sets `HTTP_PROXY` / `NO_PROXY` (bypasses proxy for PyTorch/HuggingFace CDNs) and `CUDA_VISIBLE_DEVICES`. Runs at import time, before any model download. |
| [tools/env_check.py](tools/env_check.py) | **Environment verification.** `check_ffmpeg()` — verifies FFmpeg is on PATH. `check_cuda()` — GPU detection + real kernel-launch test, returns `True`/`False`. |
| [tools/extract.py](tools/extract.py) | **Word extraction from ASR output.** `extract_words_from_result()` — pulls `(start, end, text)` word tuples from WhisperX result dicts. `fix_word_timestamps()` — post-processing to correct inflated timestamps. |
| [tools/cache.py](tools/cache.py) | **Word-level caching.** `save_words_cache()` / `load_words_cache()` — persist/restore extracted word data as JSON for re-split debugging. |
| [tools/format.py](tools/format.py) | **SRT time formatting.** `format_srt_time()` — converts float seconds to `HH:MM:SS,mmm` SRT format. |
| [tools/export.py](tools/export.py) | **Subtitle export.** `export_srt()` — writes SRT with index + timestamps. `export_txt()` — writes plain text or SRT-format text. `export_word_level()` — debug-only word-level SRT/TXT _(currently commented out)_. |
| [tools/download_models.py](tools/download_models.py) | **Model pre-downloader.** One-shot script to download Silero VAD, Wav2Vec2, and NLTK data to local cache so the pipeline runs fully offline. |
| [tools/segment.py](tools/segment.py) | **LLM segmentation engine.** `segment_words()` function — runs the full 10-phase segmentation algorithm (see [docs/segmentation_pipeline.md](docs/segmentation_pipeline.md) for full algorithm chart). |
| [tools/non_split_bigrams.py](tools/non_split_bigrams.py) | **Phrase protection.** `NON_SPLIT_BIGRAMS` set (~5700 entries) of fixed expressions, phrasal verbs, and collocations that must not be split across subtitle segments. `would_break_phrase()` look-up used by all segmentation phases. |
| [tools/llama/](tools/llama/) | **llama.cpp binaries.** Pre-compiled `llama-server.exe` + CUDA 12 DLLs for local LLM inference. Excluded from git — see [Setup → llama.cpp Binary](#llamacpp-binary). |

### `docs/` directory

| File | Purpose |
|------|---------|
| [docs/segmentation_pipeline.md](docs/segmentation_pipeline.md) | **10-phase LLM segmentation algorithm doc (English).** Full algorithm flow chart and detailed description of all 10 phases, guard mechanisms, and LLM call summary. |
| [docs/segmentation_pipeline_CN.md](docs/segmentation_pipeline_CN.md) | **10-phase LLM segmentation algorithm doc (Chinese).** Same content as above in Chinese. |
| [docs/transcription_n_translation.md](docs/transcription_n_translation.md) | **Transcription & translation technical implementation (English).** Design rationale, parameter choices, and code walkthrough for the ASR-to-alignment-to-segmentation pipeline and the translation factory backend. |
| [docs/transcription_n_translation_CN.md](docs/transcription_n_translation_CN.md) | **Transcription & translation technical implementation (Chinese).** Same content as above in Chinese. |
| [docs/models_n_performance.md](docs/models_n_performance.md) | **Models & performance (English).** Hardware requirements, LLM selection guide, and inference benchmarks. |
| [docs/models_n_performance_CN.md](docs/models_n_performance_CN.md) | **Models & performance (Chinese).** Same content as above in Chinese. |

---

## Setup

### Prerequisites

| Category | Examples | Install method |
|----------|----------|---------------|
| **System tools** | NVIDIA driver (≥ CUDA 12.8), FFmpeg, Python venv | Manual (one-time) |
| **llama.cpp binary** | `llama-server.exe` | Download zip → extract to `tools/llama/` (see variant notes below) |
| **Pre-downloaded models** | faster-whisper-large-v3, Phi-4 GGUF | Manual download to `models/` |
| **Auto-downloaded models** | Silero VAD, Wav2Vec2, NLTK | `python tools/download_models.py` |

> **GPU vs CPU**: The entire pipeline (ASR + alignment + LLM) **auto-detects CUDA and falls back to CPU** if no GPU is available. No special flags needed — just choose the right llama.cpp variant below.

### NVIDIA Driver & CUDA

This project depends on PyTorch CUDA 12.8 (see `torch==2.8.0+cu128` in `requirements.txt`), which **requires a compatible NVIDIA driver** for GPU acceleration.

| Requirement | Notes |
|-------------|-------|
| **NVIDIA GPU** | Architecture ≥ Maxwell (GTX 900 series or newer), minimum 4 GB VRAM (8 GB+ recommended) |
| **NVIDIA Driver** | Version ≥ **R570** (Final driver support for CUDA 12.8 was released July 2026; update to the latest Game Ready / Studio Driver) |
| **CUDA Toolkit** | **Not required.** PyTorch ships its own CUDA runtime — `torch==2.8.0+cu128` includes the necessary `cudart` and `cublas` libraries. A sufficiently recent NVIDIA driver is all you need. |
| **No GPU / CPU mode** | The pipeline auto-detects CUDA and falls back to CPU when unavailable (slower). No additional configuration needed. |

To verify, run `nvidia-smi` and check the **CUDA Version** line at the top. If it's below 12.8 or `nvidia-smi` produces no output, update your driver from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx).

### FFmpeg

Required by Whisper for audio extraction.

```powershell
winget install FFmpeg
# or: choco install ffmpeg
# or download from https://ffmpeg.org/download.html and add to PATH
# CN mirror: https://mirrors.tuna.tsinghua.edu.cn/ffmpeg/
```

Verify: `ffmpeg -version`

### Python Virtual Environment

```powershell
# Create .venv (Python 3.10)
python -m venv .venv
# Activate
.venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
```

### llama.cpp Binary

[llama.cpp](https://github.com/ggerganov/llama.cpp) provides the local LLM inference engine. Choose the variant matching your hardware:

| Your setup | Download zip |
|-----------|--------------|
| **NVIDIA GPU (CUDA 12.x)** | `llama-bNNNN-bin-win-cuda12.4-x64.zip` (e.g. [b9888](https://github.com/ggml-org/llama.cpp/releases/tag/b9888)) |
| **CPU only / no GPU** | `llama-bNNNN-bin-win-cpu-x64.zip` (same release page) |

1. Go to the [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases) page
   - CN accelerator proxy: `https://ghproxy.com/https://github.com/ggml-org/llama.cpp/releases`
2. Download the zip for your setup
3. Extract **all files** into `tools/llama/`


> **Note**: The release page also lists a **"CUDA 12.4 DLLs"** package (`cudart-llama-bin-win-cuda-12.4-x64.zip`, ~750 MB). This is an optional supplement — it provides NVIDIA's official cuBLAS libraries which can give slightly better GPU performance than `ggml-cuda.dll`'s built-in implementation. You don't need it; the standard CUDA build works fine on its own. If you do download it, extract the 3 DLLs into the same `tools/llama/` folder.


### Models

Place these in the `models/` directory:

```
models/
├── faster-whisper-large-v3/     # ~3 GB, from HuggingFace
└── *.gguf                       # LLM models (Phi-4, etc.) — see [docs/models_n_performance.md](docs/models_n_performance.md) for download links
```

**faster-whisper-large-v3:**

```powershell
pip install huggingface-hub

# Method A — default download
hf download guillaumek64/faster-whisper-large-v3 --local-dir models/faster-whisper-large-v3

# Method B — CN mirror (hf-mirror.com), faster in China
# Windows PowerShell:
$env:HF_ENDPOINT = "https://hf-mirror.com"
hf download guillaumek64/faster-whisper-large-v3 --local-dir models/faster-whisper-large-v3

# Or manually download via mirror direct link:
# https://hf-mirror.com/guillaumek64/faster-whisper-large-v3/tree/main
```

### Auto-download (for other small models)

```powershell
python tools/download_models.py
```

> If download is slow, set the mirror before running: `$env:HF_ENDPOINT = "https://hf-mirror.com"`

Downloads to `models/hub/` :

| Model | Size | Used by | Description |
|-------|------|---------|-------------|
| **Silero VAD** | ~35 MB | whisperx (VAD) | Voice Activity Detection — finds speech segments in audio |
| **Wav2Vec2 (base 960h)** | ~360 MB | WhisperX align | Phoneme-level word timestamp alignment |
| **NLTK punkt_tab** | ~64 MB | WhisperX alignment (internal) | Sentence boundary detection for phoneme-level forced alignment |

---

## Quick Start

```powershell
# 1. Transcribe a single file (accepts mp4/mkv/avi/mov/wav/mp3/m4a/flac)
python scripts/transcribe.py -i input/input.mp4

# 2. Translate existing subtitles (auto-detects SRT or TXT)
python scripts/translate.py -i output/input.srt

# 3. Transcribe + translate in one command
python main.py -i input/input.mp4 -translate true

# 4. Transcribe + translate with different backends
python main.py -i input/input.mp4 -translate true -seg_backend local -transl_backend deepseek

# 5. Process all videos in a folder
python batch.py

# 6. Batch with translation
python batch.py -translate true
```

---

## Usage

### 1. `scripts/transcribe.py` — Transcription pipeline

```powershell
python scripts/transcribe.py -i <input> [options]
```

Transcribes audio/video to aligned word-level timestamps via WhisperX + Wav2Vec2, then segments the words into subtitle blocks via LLM. Produces SRT + TXT.
On re-run loads a word-level JSON cache (`cache/<stem>_words.json`) to skip the transcription + alignment step. For more details see [transcription_n_translation.md](docs/transcription_n_translation.md) - "Transcription Pipeline".

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-i, --input <path>` | `input/input.mp4` | Input video or audio file (.mp4/.mkv/.avi/.mov/.wav/.mp3/.m4a/.flac) |
| `-o, --output <path>` | `output/<stem>.srt` | Output SRT file path |
| `-gpu-layers <N>` | auto-detect | GPU layers for LLM (0 = CPU only) |
| `-no-cache` | — | Skip word-level `.json` cache |
| `-seg_backend <name>` | `local` | Segmentation backend (`local`, `deepseek`, `openai`, `qwen`, `gemini`, `anthropic`) |
| `-seg_model <name>` | per-backend | Model for segmentation (e.g. `phi4`, `gpt-5.6-terra`; default per backend) |

#### Pipeline

```
Input video
  │
  ├─ 1. WhisperX ASR (faster-whisper-large-v3 + Silero VAD)
  ├─ 2. Wav2Vec2 phoneme-level forced alignment
  ├─ 3. LLM punctuation & segmentation (configured via -seg_backend / -seg_model)
  │      Phase 1-4:   LLM fill missing punctuation (chunked + context)
  │      Phase 5:     Build segments from LLM-inferred sentence boundaries
  │      Phase 6:     Comma-based force-split of overlength segments
  │      Phase 7:     Conjunction split (LLM classifier for ambiguous "and")
  │      Phase 8:     Run-on re-punctuation (recursive LLM split)
  │      Phase 9:     Conjunction fragment merge (LLM classifier)
  │      Phase 10:    Emergency character-count split
  └─ 4. Export SRT + TXT
```

#### Output files

| File | Description |
|------|-------------|
| `output/<stem>.srt` | Subtitle file with timestamps |
| `output/<stem>.txt` | Plain text (with timestamp markers) |
| `cache/<stem>_words.json` | Cached ASR word data (reused on re-run) |

#### Examples

```powershell
# Default
python scripts/transcribe.py -i input/input.mp4

# Custom output path
python scripts/transcribe.py -i input/input.mp4 -o D:/output/lecture.srt

# GPU-free LLM segmentation (CPU only)
python scripts/transcribe.py -i input/input.mp4 -gpu-layers 0

# Use an online LLM for segmentation
python scripts/transcribe.py -i input/input.mp4 -seg_backend openai -seg_model gpt-5.6-terra

# No caching
python scripts/transcribe.py -i input/input.mp4 -no-cache
```

---

### 2. `scripts/translate.py` — Language translation

Translates SRT (preserving timecodes) or plain TXT files using a **local LLM** (via llama-server) or any of **5 online API backends**.
Auto-detects input format by content — SRT files keep their timecodes, TXT files are treated as plain text lines.

Two translation modes are available:
- **`accurate`** (default): **Recommended for local models**.
- **`flexible`**: **Recommended for online API backends**. `allow_flexible_word_order` and `allow_simplify_wording` are only effective in this mode.

For detailed mechanics and window parameters see [transcription_n_translation.md](docs/transcription_n_translation.md) - "3.3.1 Translation Mode".

```powershell
python scripts/translate.py -i <input> [options]
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-i, --input <path>` | **(required)** | Input `.srt` or `.txt` file (auto-detects format by content) |
| `-o, --output <path>` | auto | Output file path |
| `-tgt-lang <name>` | `Chinese` | Target language name |
| `-tgt-lang-code <code>` | `CN` | Language code for filename suffix |
| `-src-lang <name>` | `English` | Source language name |
| `-transl_backend <name>` | `local` | Translation backend (`local`, `deepseek`, `openai`, `qwen`, `gemini`, `anthropic`) |
| `-transl_model <name>` | per-backend | Model for the selected backend (see table below) |
| `-gpu-layers <N>` | auto-detect | GPU layers (local backend only) |
| `-mode <name>` | `accurate` | Translation mode: `accurate` (2 lines/batch) or `flexible` (4 lines/batch, with timecodes) |
| `-local_model <name>` | `phi4` | Local model name (applied when `-transl_backend` is `local` and no `-transl_model` given) |

#### Translation backends

| Backend | API format | Default model | Auth |
|---------|-----------|---------------|------|
| `local` | llama-server subprocess | `phi4` | — |
| `deepseek` | Auto-detect: OpenAI `/chat/completions` **or** Anthropic Messages | `deepseek-v4-flash` | `openai_api_key` |
| `openai` | OpenAI-compatible | `gpt-5.6-terra` | `openai_api_key` |
| `qwen` | OpenAI-compatible | `qwen3.5-plus` | `openai_api_key` |
| `gemini` | OpenAI-compatible | `gemini-3.5-flash` | `openai_api_key` |
| `anthropic` | Anthropic Messages | `claude-opus-4-8` | `anthropic_api_key` |

#### User configuration

Edit [translate_config.json](translate_config.json) in the project root:

```json
{
    "target_lang": "Chinese",
    "target_lang_code": "CN",
    "source_lang": "English",
    "add_punctuation": false,
    "allow_flexible_word_order": false,
    "allow_simplify_wording": false,
    "number_mode": "auto",
    "space_between_cjk_and_latin": false,
    "glossary": ["EXAMPLE1", "EXAMPLE2", "VRAM", "CUDA"],
    "custom_system_prompt": null,
    "cache_prompt": false,
    "drift_threshold": false,
    "openai_api_key": "",
    "anthropic_api_key": "",
    "api_base_url": ""
}
```

For detailed explanations of each field see [transcription_n_translation.md](docs/transcription_n_translation.md) - "3.1 Translation Configuration".

Or use the `.env` file for API keys (place at project root as `.env`): **only takes effect when the corresponding key fields in `translate_config.json` are left empty**

```ini
# ── Translation backend API keys (correspond to translate_config.json fields) ──

OPENAI_API_KEY=sk-your-key-here

# Anthropic-specific (only needed when backend is anthropic)
ANTHROPIC_API_KEY=sk-ant-your-key-here

```

#### Examples

```powershell
# Local Phi-4 — Translate TXT to Chinese (default)
python scripts/translate.py -i output/input.txt
# → output/input_CN.txt

# Local Phi-4 — Translate SRT (preserves timecodes)
python scripts/translate.py -i output/input.srt
# → output/input_CN.srt + output/input_CN.txt

# Online backends
python scripts/translate.py -i output/input.srt -transl_backend deepseek
python scripts/translate.py -i output/input.srt -transl_backend openai -transl_model gpt-5.6-terra
python scripts/translate.py -i output/input.srt -transl_backend anthropic -transl_model claude-opus-4-8

# Translate to Japanese using DeepSeek (flexible mode)
python scripts/translate.py -i output/input.txt -transl_backend deepseek -tgt-lang Japanese -tgt-lang-code JP -mode flexible

# Custom output path
python scripts/translate.py -i output/input.txt -o D:/output/lecture_JP.srt

# Override source language
python scripts/translate.py -i output/input.txt -src-lang English -tgt-lang Chinese
```

---

### 3. `main.py` — Orchestrator (transcribe + translate)

```powershell
python main.py -i <input> [options]
```

`main.py` is the **orchestrator** that imports and calls `scripts.transcribe.transcribe_file()` and/or `scripts.translate.translate_file()` in a single run.
It does **not** re-implement the pipelines — it calls the same functions the standalone scripts expose.

| `-transcribe` | `-translate` | Behaviour |
| ------------ | ----------- | --------- |
| `true` (default) | `false` (default) | Transcribe only (ASR + segment → SRT/TXT) |
| `true` | `true` | Transcribe, then translate the SRT |
| `false` | `true` | Translate-only (input must be `.srt` or `.txt`) |
| `false` | `false` | Error: at least one must be enabled |

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-i, --input <path>` | `input/input.mp4` | Input file (video/audio for transcribe, `.srt`/`.txt` for translate-only) |
| `-o, --output <path>` | `output/<stem>.srt` | Output SRT file path |
| `-transcribe` | `true` | Enable transcription (`true`/`false`) |
| `-translate` | `false` | Enable translation (`true`/`false`) |
| `-seg_backend <name>` | `local` | Segmentation backend (`local`, `deepseek`, `openai`, `qwen`, `gemini`, `anthropic`) |
| `-seg_model <name>` | per-backend | Model for segmentation (e.g. `phi4`, `gpt-5.6-terra`) |
| `-transl_backend <name>` | `local` | Translation backend (same options) |
| `-transl_model <name>` | per-backend | Model for translation |
| `-mode <name>` | `accurate` | Translation mode: `accurate` or `flexible` (only when `-translate true`) |
| `-local_model <name>` | `phi4` | Default local model (overridable individually by `-seg_model`/`-transl_model`) |
| `-gpu-layers <N>` | auto-detect | GPU layers for LLM (0 = CPU only) |
| `-no-cache` | — | Skip word-level `.json` cache |

> **Note:** `-mode` requires `-translate true`. Passing it without translation prints an error and exits.

#### Examples

```powershell
# Transcribe only (default)
python main.py -i input/input.mp4

# Transcribe + translate to Chinese (local Phi-4)
python main.py -i input/input.mp4 -translate true

# Transcribe + translate with online API (flexible mode)
python main.py -i input/input.mp4 -translate true -transl_backend deepseek -mode flexible

# Translate-only (existing subtitles)
python main.py -transcribe false -translate true -i output/lecture.srt

# Different models for segmentation vs translation
python main.py -i input/input.mp4 -translate true -seg_backend local -seg_model phi4 -transl_backend openai -transl_model gpt-5.6-terra
```

---

### 4. `batch.py` — Batch processing

Process all files in a folder, one at a time in isolated subprocesses (each file gets its own CUDA context — no VRAM leak between files).

Supports both video (.mp4, .mkv, .avi, .mov) and audio (.wav, .mp3, .m4a, .flac) inputs via the `-ext` argument.

```powershell
python batch.py [options]
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-i, --input <dir>` | `input/` | Input folder |
| `-o, --output <dir>` | `output/` | Output folder |
| `-ext <ext>` | `.mp4` | File extension to search for (e.g. `.mp3`, `.wav`, `.mkv`) |
| `-gpu-layers <N>` | auto-detect | GPU layers for LLM (0 = CPU only) |
| `-no-cache` | — | Skip word-level cache |
| `-transcribe` | `true` | Enable transcription (`true`/`false`) |
| `-translate` | `false` | Translate each file after transcription (`true`/`false`) |
| `-seg_backend <name>` | `local` | Segmentation backend (`local`, `deepseek`, `openai`, `qwen`, `gemini`, `anthropic`) |
| `-seg_model <name>` | per-backend | Model for segmentation (default per backend) |
| `-transl_backend <name>` | `local` | Translation backend (same options) |
| `-transl_model <name>` | per-backend | Model for translation (default per backend) |
| `-mode <name>` | `accurate` | Translation mode: `accurate` or `flexible` (only when `-translate true`) |

#### Examples

```powershell
# Process all MP4s in input/
python batch.py

# Custom folders
python batch.py -i D:/videos -o D:/output

# Handle MKV files
python batch.py -ext .mkv

# Process audio files (MP3, WAV)
python batch.py -ext .mp3
python batch.py -ext .wav

# Batch + translate all files (local Phi-4)
python batch.py -translate true

# Batch + translate with online API (flexible mode)
python batch.py -translate true -transl_backend deepseek -transl_model deepseek-v4-flash -mode flexible

# Batch translate-only (existing subtitles)
python batch.py -transcribe false -translate true -i D:/subtitles/ -o D:/subtitles_translated/ -transl_backend deepseek

# Online segmentation + local translation (not recommended)
python batch.py -seg_backend openai -seg_model gpt-5.6-terra -translate true
```

## Supported input formats

`.mp4`, `.mkv`, `.avi`, `.mov`, `.wav`, `.mp3`, `.m4a`, `.flac`

---

## Directory structure

```
M2L3/
├── main.py                 # Orchestrator — transcribe + translate in one command
├── batch.py                # Batch processing entry
├── translate_config.json   # Translation user configuration
├── README.md               # English documentation
├── README_CN.md            # Chinese documentation (This file)
├── requirements.txt
├── scripts/
│   ├── transcribe.py       # Transcription pipeline (ASR → align → segment → export)
│   └── translate.py        # Translation pipeline (6 backends, accurate/flexible modes)
├── models/
│   ├── hub/                        # Local model cache
│   │   ├── checkpoints/            # Wav2Vec2 alignment model (~360 MB)
│   │   ├── nltk_data/              # NLTK punkt tokenizers (~64 MB)
│   │   └── snakers4_silero-vad_master/  # Silero VAD (~35 MB)
│   ├── faster-whisper-large-v3/    # ASR model (~3 GB)
│   └── *.gguf           # LLM models for segmentation & translation (Phi-4, etc.)
├── docs/
│   ├── segmentation_pipeline.md           # 10-phase LLM segmentation algorithm (English)
│   ├── segmentation_pipeline_CN.md        # 10 阶段 LLM 分割算法文档（中文）
│   ├── transcription_n_translation.md     # Transcription & translation technical implementation (English)
│   ├── transcription_n_translation_CN.md  # 转录与翻译技术实现文档（中文）
│   ├── models_n_performance.md            # Models & performance (English)
│   └── models_n_performance_CN.md         # 模型与性能（中文）
├── tools/
│   ├── llama/
│   │   └── llama-server.exe        # llama.cpp inference server (CUDA)
│   ├── __init__.py                 # Package marker
│   ├── config.py                   # Proxy & GPU environment config
│   ├── env_check.py                # FFmpeg & CUDA verification
│   ├── extract.py                  # Word extraction from ASR output
│   ├── cache.py                    # Word-level JSON cache
│   ├── format.py                   # SRT time formatting
│   ├── export.py                   # SRT / TXT export
│   ├── download_models.py          # Auto-download script
│   ├── segment.py                  # LLM segmentation engine (10-phase)
│   ├── non_split_bigrams.py        # Phrase protection bigrams
│   └── _patch_transformers.py      # Transformers offline patch (internal)
├── input/                  # Default input folder
├── output/                 # Default output folder
└── cache/                  # Word-level ASR cache
```

---

## License

This project is licensed under the [MIT License](LICENSE).
