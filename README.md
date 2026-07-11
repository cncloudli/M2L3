# Multi-Layer LLM-Based Subtitle Segmentation (M2L3)

Local, GPU-accelerated subtitle generation with WhisperX + Wav2Vec2 alignment + LLM-enhanced segmentation and translation.

| Feature | Approach                                          | When to use |
|---------|---------------------------------------------------|-------------|
| **ASR** | WhisperX (faster-whisper-large-v3) + Silero VAD   | Always |
| **Alignment** | Wav2Vec2 phoneme-level forced alignment           | Always |
| **Segment** | Phi-4 via llama-server (10-phase LLM refinement)  | All content types |
| **Translate** | Phi-4 via llama-server or Online LLM (API needed) | Post-transcription EN → any language |

All processing runs **fully offline** — no data leaves your machine. (Online LLM Translation is **Optional**)

---

## File Reference

### Root directory

| File | Purpose                                                                                                                                                                                      |
|------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [main.py](main.py) | **Single-file pipeline entry point.** Transcribe a video/audio file → aligned word timestamps → LLM segmentation → SRT + TXT export.                                                         |
| [batch_pipeline.py](batch_pipeline.py) | **Batch processing.** Iterates all files in an input folder, spawning one `main.py` subprocess per file (each gets a clean CUDA context).                                                    |
| [translate.py](translate.py) | **Translation.** Translates existing SRT (preserving timecodes) or TXT files using a local or API-based LLM. Supports 7 backends (local, OpenAI, DeepSeek, Qwen, Gemini, Ollama, Anthropic). |
| [translate_config.json](translate_config.json) | **Translation config.** Language, punctuation, glossary, API keys — edited by the user, read by `translate.py` at startup.                                                                   |
| [README.md](README.md) | This file — English documentation, merged usage guide, setup instructions, and file reference.                                                                                               |
| [README_CN.md](README_CN.md) | Chinese (中文) documentation — Chinese translation with mirror download links.                                                                                                                 |

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
| [tools/llm_pipeline.py](tools/llm_pipeline.py) | **LLM segmentation engine.** `LLMPipeline` class — manages llama-server subprocess lifecycle and runs the 10-phase segmentation algorithm (see [docs/segmentation_pipeline.md](docs/segmentation_pipeline.md) for full algorithm chart). `segment_with_llm()` convenience function for one-shot CLI use. |
| [tools/non_split_bigrams.py](tools/non_split_bigrams.py) | **Phrase protection.** `NON_SPLIT_BIGRAMS` set (~5700 entries) of fixed expressions, phrasal verbs, and collocations that must not be split across subtitle segments. `would_break_phrase()` look-up used by all segmentation phases. |
| [tools/llama/](tools/llama/) | **llama.cpp binaries.** Pre-compiled `llama-server.exe` + CUDA 12 DLLs for local LLM inference. Excluded from git — see [Setup → llama.cpp Binary](#llamacpp-binary). |

### `docs/` directory

| File | Purpose                                                                                                                                                                                                              |
|------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [docs/segmentation_pipeline.md](docs/segmentation_pipeline.md) | **10-phase LLM segmentation algorithm doc.** Full algorithm flow chart and detailed description of all 10 phases, guard mechanisms, and LLM call summary (English).                                                  |
| [docs/segmentation_pipeline_CN.md](docs/segmentation_pipeline_CN.md) | **10-phase LLM segmentation algorithm doc.** Same content as above in Chinese (中文).                                                                                                                                  |
| [docs/transcription_n_translation.md](docs/transcription_n_translation.md) | **Transcription & translation technical implementation.** Design rationale, parameter choices, and code walkthrough for the ASR-to-alignment-to-segmentation pipeline and the translation factory backend (English). |
| [docs/transcription_n_translation_CN.md](docs/transcription_n_translation_CN.md) | **Transcription & translation technical implementation.** Same content as above in Chinese (中文).                                                                                                                     |
| [docs/models_n_hardware_reqs.md](docs/models_n_hardware_reqs.md) | **Models & hardware requirements.** (TODO — will cover LLM models used, VRAM/RAM/disk requirements, and performance guidance.)                                                                                       |
| [docs/models_n_hardware_reqs_CN.md](docs/models_n_hardware_reqs_CN.md) | **Models & hardware requirements.** (TODO — same content as above in Chinese.)                                                                                                                                       |

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

The project depends on PyTorch CUDA 12.8 (see `torch==2.8.0+cu128` in `requirements.txt`), which **requires a compatible NVIDIA driver** for GPU acceleration.

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
2. Download the zip for your setup
3. Extract **all files** into `tools/llama/`


> **Note**: The release page also lists a **"CUDA 12.4 DLLs"** package (`cudart-llama-bin-win-cuda-12.4-x64.zip`, ~750 MB). This is an optional supplement — it provides NVIDIA's official cuBLAS libraries which can give slightly better GPU performance than `ggml-cuda.dll`'s built-in implementation. You don't need it; the standard CUDA build works fine on its own. If you do download it, extract the 3 DLLs into the same `tools/llama/` folder.


### Models

Place these in the `models/` directory:

```
models/
├── faster-whisper-large-v3/     # ~3 GB, from HuggingFace
└── phi-4-Q4_K_M.gguf            # ~8.5 GB, Phi-4 (required for segmentation & -translate)
```

**faster-whisper-large-v3:**

```powershell
pip install huggingface-hub
hf download guillaumek64/faster-whisper-large-v3 --local-dir models/faster-whisper-large-v3
```

**Phi-4 (14B, GGUF):**

Single-file download — use either method:

```powershell
# Option A — browser / curl (direct link, ~8.5 GB)
curl -L -o models\phi-4-Q4_K_M.gguf ^
  https://huggingface.co/unsloth/phi-4-GGUF/resolve/main/phi-4-Q4_K_M.gguf

# Option B — huggingface_hub (supports resume, no rename needed)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/phi-4-GGUF', 'phi-4-Q4_K_M.gguf', local_dir='models')
"
```

### Auto-download (for other small models)

```powershell
python tools/download_models.py
```

Downloads to `models/hub/` :

| Model | Size | Used by | Description |
|-------|------|---------|-------------|
| **Silero VAD** | ~35 MB | whisperx (VAD) | Voice Activity Detection — finds speech segments in audio |
| **Wav2Vec2 (base 960h)** | ~360 MB | WhisperX align | Phoneme-level word timestamp alignment |
| **NLTK punkt_tab** | ~64 MB | WhisperX alignment (internal) | Sentence boundary detection for phoneme-level forced alignment |

---

## Quick Start

```powershell
# Transcribe a single video
python main.py -i input/input.mp4

# Transcribe + translate to Chinese
python main.py -i input/input.mp4 -translate

# Process all videos in a folder
python batch_pipeline.py

# Batch with translation
python batch_pipeline.py -translate

# Translate an existing TXT file
python translate.py -i output/input.txt
```

---

## Usage

### 1. `main.py` — Single-file transcription

```powershell
python main.py -i <input> [options]
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-i, --input <path>` | `input/input.mp4` | Input video or audio file (.mp4/.mkv/.avi/.mov/.wav/.mp3/.m4a/.flac) |
| `-o, --output <path>` | `output/<stem>.srt` | Output SRT file path |
| `-gpu-layers <N>` | auto-detect | GPU layers for LLM (0 = CPU only) |
| `-no-cache` | — | Skip word-level `.json` cache |
| `-translate` | — | Translate output after transcription |
| `-backend <name>` | `local` | Translation backend (`local`, `deepseek`, `openai`, `qwen`, `gemini`, `ollama`, `anthropic`) |
| `-model <name>` | per-backend | Translation model (default per backend; only valid with `-translate`) |

#### Pipeline

```
Input video
  │
  ├─ 1. WhisperX ASR (faster-whisper-large-v3 + Silero VAD)
  ├─ 2. Wav2Vec2 phoneme-level forced alignment
  ├─ 3. LLM punctuation & segmentation (Phi-4 via llama-server)
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
| `output/<stem>_wl.srt` | Word-level subtitle (debug — currently commented out) |
| `output/<stem>_wl.txt` | Word-level text (debug — currently commented out) |
| `cache/<stem>_words.json` | Cached ASR word data (reused on re-run) |

#### Examples

```powershell
# Default
python main.py -i input/input.mp4

# If the path includes spaces or special characters, please put it in quotes 
python main.py -i input/"input.mp4"

# Custom output path
python main.py -i input/input.mp4 -o D:/output/lecture.srt

# Transcribe + translate to Chinese
python main.py -i input/input.mp4 -translate

# No caching
python main.py -i input/input.mp4 -no-cache
```

### 2. `batch_pipeline.py` — Batch processing

Process all files in a folder, one at a time in isolated subprocesses (each file gets its own CUDA context — no VRAM leak between files).

Supports both video (.mp4, .mkv, .avi, .mov) and audio (.wav, .mp3, .m4a, .flac) inputs via the `-ext` argument.

```powershell
python batch_pipeline.py [options]
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-i, --input <dir>` | `input/` | Input folder |
| `-o, --output <dir>` | `output/` | Output folder |
| `-gpu-layers <N>` | auto-detect | GPU layers for LLM |
| `-no-cache` | — | Skip word-level cache |
| `-ext <ext>` | `.mp4` | File extension to search for (e.g. `.mp3`, `.wav`, `.mkv`) |
| `-dry-run` | — | List files without processing |
| `-translate` | — | Translate each file after transcription |
| `-backend <name>` | `local` | Translation backend (`local`, `deepseek`, `openai`, `qwen`, `gemini`, `ollama`, `anthropic`) |
| `-model <name>` | per-backend | Translation model (default per backend; only valid with `-translate`) |

#### Examples

```powershell
# Process all MP4s in input/
python batch_pipeline.py

# Custom folders
python batch_pipeline.py -i D:/videos -o D:/output

# Handle MKV files
python batch_pipeline.py -ext .mkv

# Process audio files (MP3, WAV)
python batch_pipeline.py -ext .mp3
python batch_pipeline.py -ext .wav

# Dry run to see what would be processed
python batch_pipeline.py -dry-run

# Batch + translate all files
python batch_pipeline.py -translate
```

### 3. `translate.py` — Language translation

Translates SRT (preserving timecodes) or plain TXT files using a **local LLM** (Phi-4 via llama-server) or any of **6 online API backends**.
Auto-detects input format by content — SRT files keep their timecodes, TXT files are treated as plain text lines.

Two translation modes are available:
- **`accurate`** (default): Small sliding window (2 lines per batch, context 4 lines), strictly line-by-line translation with minimal timeline misalignment risk. **Recommended for local Phi-4 models** — slower local inference benefits from the smaller window for stable latency and lower resource usage.
- **`flexible`**: Larger sliding window (4 lines per batch, context 8 lines), with timecode hints, allows cross-line rephrasing for more natural-sounding output. **Better suited for online API backends** — high-speed APIs can take full advantage of the larger context window.

```powershell
python translate.py -i <input> [options]
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-i, --input <path>` | **(required)** | Input `.srt` or `.txt` file (auto-detects format by content) |
| `-o, --output <path>` | auto | Output file path |
| `-tgt-lang <name>` | `Chinese` | Target language name |
| `-tgt-lang-code <code>` | `CN` | Language code for filename suffix |
| `-src-lang <name>` | `English` | Source language name |
| `-backend <name>` | `local` | Translation backend: `local`, `deepseek`, `openai`, `qwen`, `gemini`, `ollama`, `anthropic` |
| `-model <name>` | per-backend default | Model name for the selected backend (see table below) |
| `-gpu-layers <N>` | auto-detect | GPU layers (local backend only) |
| `-mode <name>` | `accurate` | Translation mode: `accurate` (strict, 2 lines/batch) or `flexible` (relaxed, 4 lines/batch, with timecodes) |

#### Translation backends

| Backend | API format | Default model | Auth |
|---------|-----------|---------------|------|
| `local` | llama-server subprocess | `phi-4` | — |
| `deepseek` | Auto-detect: OpenAI `/chat/completions` **or** Anthropic Messages (see footnote) | `deepseek-v4-flash` | `openai_api_key` |
| `openai` | OpenAI-compatible | `gpt-5.6-terra` | `openai_api_key` |
| `qwen` | OpenAI-compatible | `qwen3.5-plus` | `openai_api_key` |
| `gemini` | OpenAI-compatible | `gemini-3.5-flash` | `openai_api_key` |
| `ollama` | OpenAI-compatible | `llama4` | — |
| `anthropic` | Anthropic Messages | `claude-opus-4-8` | `anthropic_api_key` |

> **OpenAI-compatible backends** (openai, qwen, gemini, ollama) use the same `OPENAI_API_KEY` config field. Ollama ignores the key (runs locally).
> **DeepSeek** supports two API formats, auto-detected from `api_base_url`:
> - `"https://api.deepseek.com"` (default) → OpenAI `/chat/completions`
> - `"https://api.deepseek.com/anthropic"` → Anthropic Messages API (more reliable)
> The choice is made per-request by checking if the URL contains `anthropic`. Authenticate with `openai_api_key`.
> The default model for each backend reflects the latest available as of July 2026.

#### User configuration

Edit [translate_config.json](translate_config.json) in the project root, or use the `.env` file for API keys (see below):

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
    "openai_api_key": "",
    "anthropic_api_key": "",
    "api_base_url": ""
}
```

#### Configuration fields

| Field | Type | Default     | Description                                                                                                                                                                                                                                                              |
|-------|------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `target_lang` | string | `"Chinese"` | Target language name (used in prompts)                                                                                                                                                                                                                                   |
| `target_lang_code` | string | `"CN"`      | Language code appended to output filename                                                                                                                                                                                                                                |
| `source_lang` | string | `"English"` | Source language name (used in prompts)                                                                                                                                                                                                                                   |
| `add_punctuation` | bool | `true`      | Add sentence-ending punctuation (。？！) in translation. When `false`, sentence-ending punctuation is stripped via post-processing — the LLM is not burdened with this constraint.                                                                                          |
| `allow_flexible_word_order` | bool | `false`     | Allow content to be redistributed across adjacent lines for more natural phrasing. **⚠ Only available in `flexible` mode** (not passed in `accurate` mode). **Strongly recommended for API backends only** — local models may produce unstable line index shifts.                  |
| `allow_simplify_wording` | bool | `false`     | Allow condensing colloquial/verbose expressions within each line (filler words, redundancies, rambling). **⚠ Only available in `flexible` mode** (not passed in `accurate` mode). **Strongly recommended for API backends only** — less effective with local models.                   |
| `number_mode` | string | `"auto"`    | Number handling mode: `"auto"` (LLM decides), `"src_lang"` (keep source format), `"digits"` (all Arabic digits), `"tgt_lang"` (target language's native numerals). **Recommended for online API backends** — works with local models but may be less consistent. Usable in both modes.          |
| `space_between_cjk_and_latin` | bool | `false`     | Insert space between CJK and Latin characters                                                                                                                                                                                                                            |
| `glossary` | string[] | `[]`        | Terms to keep untranslated (preserve exactly as written). **Recommended for online API backends** — works with local models but may be less consistent. Usable in both modes.                                                                                 |
| `custom_system_prompt` | string\|null | `none`      | Override the entire LLM system prompt (disables all other prompt flags)                                                                                                                                                                                                  |
| `cache_prompt` | bool | `false`     | Enable KV-cache reuse across batches (`local` backend only). Reduces latency but may cause instruction drift in long translation sessions. **Strongly recommended for API backends only** — set to `false` for local models. Usable in both modes.                                                     |
| `openai_api_key` | string | `""`        | API key for OpenAI-compatible backends. **Recommended:** set via `OPENAI_API_KEY` in `.env` instead.                                                                                                                                                                     |
| `anthropic_api_key` | string | `""`        | API key for Anthropic backend. **Recommended:** set via `ANTHROPIC_API_KEY` in `.env` instead.                                                                                                                                                                           |
| `api_base_url` | string | `""`        | Override base URL for the selected API backend. For DeepSeek: `"https://api.deepseek.com"` (OpenAI format) or `"https://api.deepseek.com/anthropic"` (Anthropic Messages format). The code auto-detects which format to use by checking if the URL contains `anthropic`. |

> **API keys via `.env`** (recommended): Create a `.env` file in the project root (it's in `.gitignore`, so it stays out of git):
> ```
> OPENAI_API_KEY=sk-...
> ANTHROPIC_API_KEY=sk-ant-...
> ```
> Keys set in `translate_config.json` still take priority, but leaving them empty and using `.env` keeps secrets out of version control.

#### Examples

```powershell
# Local Phi-4 — Translate TXT to Chinese (default)
python translate.py -i output/input.txt 
python translate.py -i output/"input.txt"
# → output/input_CN.txt

# Local Phi-4 — Translate SRT (preserves timecodes)
python translate.py -i output/input.srt
# → output/input_CN.srt + output/input_CN.txt

# Online backends
python translate.py -i output/input.srt -backend openai -model gpt-5.6-terra
python translate.py -i output/input.srt -backend deepseek -model deepseek-v4-flash
python translate.py -i output/input.srt -backend gemini -model gemini-3.5-flash
python translate.py -i output/input.srt -backend qwen -model qwen3.5-plus
python translate.py -i output/input.srt -backend ollama -model llama4
python translate.py -i output/input.srt -backend anthropic -model claude-opus-4-8

# Translate to Japanese using DeepSeek
python translate.py -i output/input.txt -backend deepseek -tgt-lang Japanese -tgt-lang-code JP

# Custom output path
python translate.py -i output/input.txt -o D:/output/lecture_JP.srt

# Override source language
python translate.py -i output/input.txt -src-lang English -tgt-lang Chinese
```

### Integration: transcription → translation

Add `-translate` to any `main.py` or `batch_pipeline.py` command to run translation automatically after transcription.

```powershell
# Single file — local translation (default)
python main.py -i input/input.mp4 -translate

# Batch — local translation
python batch_pipeline.py -translate

# Single file — online backend translation
python main.py -i input/input.mp4 -translate -backend openai -model gpt-5.6-terra

# Batch — online backend translation
python batch_pipeline.py -translate -backend anthropic -model claude-opus-4-8
```

This runs `translate.py` as a subprocess on the generated `.srt` file (preserving timecodes throughout), producing `output/<stem>_CN.srt` and `output/<stem>_CN.txt`.

---

## Supported input formats

`.mp4`, `.mkv`, `.avi`, `.mov`, `.wav`, `.mp3`, `.m4a`, `.flac`

---

## Directory structure

```
M2L3/
├── main.py                 # Single-file transcription entry
├── batch_pipeline.py       # Batch processing entry
├── translate.py            # Translation entry
├── translate_config.json   # Translation user configuration
├── README.md               # English documentation (This file)
├── README_CN.md            # Chinese documentation
├── requirements.txt
├── models/
│   ├── hub/                        # Local model cache 
│   │   ├── checkpoints/            # Wav2Vec2 alignment model (~360 MB)
│   │   ├── nltk_data/              # NLTK punkt tokenizers (~64 MB)
│   │   └── snakers4_silero-vad_master/  # Silero VAD (~35 MB)
│   ├── faster-whisper-large-v3/    # ASR model (~3 GB)
│   └── phi-4-Q4_K_M.gguf           # LLM for segmentation & translation (~8.5 GB, Phi-4 14B)
├── docs/
│   ├── segmentation_pipeline.md           # 10-phase LLM segmentation algorithm (English)
│   ├── segmentation_pipeline_CN.md        # 10 阶段 LLM 分割算法文档（中文）
│   ├── transcription_n_translation.md     # Transcription & translation technical implementation (English)
│   ├── transcription_n_translation_CN.md  # 转录与翻译技术实现文档（中文）
│   ├── models_n_hardware_reqs.md          # (TODO) Models & hardware requirements
│   └── models_n_hardware_reqs_CN.md       # (TODO) 模型与硬件需求说明
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
│   ├── llm_pipeline.py             # LLM segmentation engine (10-phase)
│   └── non_split_bigrams.py        # Phrase protection bigrams
├── input/                  # Default input folder
├── output/                 # Default output folder
└── cache/                  # Word-level ASR cache
```
