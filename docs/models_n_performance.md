# Models & Performance

This document covers three aspects of the M2L3 pipeline: **hardware requirements** for running transcription, segmentation, and translation; **LLM selection** guidance for the LLM-powered tasks; and **inference benchmarks** of all tested models.

---

## Part 1 — Hardware Requirements

### 1.1 WhisperX Transcription (Local ASR + Alignment)

If you only run transcription locally and use online APIs for segmentation and translation, the following models are the only ones that need local compute:

| Model | Disk Size | GPU Requirement | CPU Fallback |
| :----- | :--------- | :--------------- | :------------ |
| **faster-whisper-large-v3** | ~3 GB | NVIDIA GPU, 4 GB+ VRAM recommended | Works on CPU (significantly slower) |
| **Silero VAD** | ~35 MB | Optional (GPU not needed) | CPU is fine |
| **Wav2Vec2 (base 960h)** | ~360 MB | Optional (GPU preferred for speed) | CPU is fine |

> **Recommendation:** A dedicated NVIDIA GPU (4 GB+ VRAM) is recommended for acceptable transcription throughput. The entire transcription pipeline can run on CPU alone, but ASR will be substantially slower (typically 3–5× slower than GPU).

> **API-only scenario:** If you use online APIs for **both** segmentation and translation, no local LLM hardware is needed beyond what transcription requires above.

### 1.2 Local LLM Inference

For users running LLM-based segmentation and/or translation locally (instead of via online APIs), the default model is **Phi-4 (14B, Q4_K_M GGUF ~8.5 GB)**. The following hardware is recommended:

| Scenario | System RAM | GPU Requirement | Notes |
| :------- | :---------- | :--------------- | :----- |
| **CPU only** (no GPU) | 32 GB+ | None | Phi-4 14B (Q4_K_M ~8.5 GB) loads entirely into system memory. Inference is slower, but sufficient for batch processing. |
| **CPU + GPU** (hybrid) | 16 GB+ | NVIDIA GPU, 6 GB+ VRAM | Offload layers to GPU via `-gpu-layers N`. The `auto_gpu_layers()` function queries actual free VRAM via `nvidia-smi` and reserves a 1.2 GB buffer for CUDA context + KV cache, then fits as many LLM layers into the remaining VRAM as possible. |

### 1.3 Test Platform

All benchmark data in this document was collected on the following hardware:

| Component | Spec |
| :-------- | :---- |
| **CPU** | AMD Ryzen AI H350 |
| **RAM** | 32 GB LPDDR5 |
| **GPU** | NVIDIA RTX 5060 Laptop (8 GB VRAM) |
| **OS** | Windows 11 |

Performance data was collected across two independent test runs on this hardware. Results within the same run are directly comparable; cross-run comparisons carry ~±10% systematic variance due to differing system load.

---

## Part 2 — LLM Selection

This section covers the LLM models available for the two LLM-powered tasks in the pipeline:

- **Segmentation** — a 10-phase LLM refinement pipeline that splits raw transcription into subtitle segments (see [segmentation_pipeline.md](segmentation_pipeline.md))
- **Translation** — a sliding-window translation engine that translates SRT/TXT files while preserving timecodes (see [transcription_n_translation.md](transcription_n_translation.md) - "Translation Pipeline")

> The smaller, non-LLM models — **faster-whisper-large-v3** (ASR), **Silero VAD** (voice activity detection), and **Wav2Vec2** (phoneme alignment) — are documented in [transcription_n_translation.md](transcription_n_translation.md) - "Transcription Pipeline".

### 2.1 Local Models

All local models are loaded via llama.cpp (llama-server.exe) as GGUF quantized files (see quantization format in the Size column).

| Model | Size (GGUF) | RAM/VRAM | Model Source & Download Link | Notes |
| :----- | :------------ | :-------- | :--------------------------- | :----- |
| **Phi-4 14B** (default) | 8.5 GB (Q4_K_M) | 8–16 GB | **Source:** [unsloth/phi-4-GGUF](https://huggingface.co/unsloth/phi-4-GGUF)<br>**Mirror:** `hf-mirror.com/unsloth/phi-4-GGUF`<br>**Download:** `phi-4-Q4_K_M.gguf` | Strong overall reasoning, performs well for both segmentation and translation; ideal when using a single LLM for all tasks |
| **Qwen3.5-9B** | 8.9 GB (Q8_0) | 12–16 GB | **Source:** [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)<br>**Mirror:** `hf-mirror.com/unsloth/Qwen3.5-9B-GGUF`<br>**Download:** `Qwen3.5-9B-Q8_0.gguf` | Average translation quality, but worst segmentation performance (not recommended for segmentation) |
| **Ministral-3-8B-Instruct 8B** | 8.5 GB (Q8_0) | 12–16 GB | **Source:** [unsloth/Ministral-3-8B-Instruct-2512-GGUF](https://huggingface.co/unsloth/Ministral-3-8B-Instruct-2512-GGUF)<br>**Mirror:** `hf-mirror.com/unsloth/Ministral-3-8B-Instruct-2512-GGUF`<br>**Download:** `Ministral-3-8B-Instruct-2512-Q8_0.gguf` | Smallest model; average translation speed and quality below Phi-4 and Qwen3.5 (not recommended for translation) |
| **Ministral-3-14B-Instruct 14B** | 9.0 GB (Q5_K_M) | 12–16 GB | **Source:** [mistralai/Ministral-3-14B-Instruct-2512-GGUF](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512-GGUF)<br>**Mirror:** `hf-mirror.com/mistralai/Ministral-3-14B-Instruct-2512-GGUF`<br>**Download:** `Ministral-3-14B-Instruct-2512-Q5_K_M.gguf` | Translation quality nearly identical to the 8B variant, but inference too slow |

> **Additional testing note:** We also evaluated **DeepSeek-R1-Distill-Llama-8B (Q8_0)** and **DeepSeek-Coder-V2-Lite-Instruct (16B, Q4_K_M)**. Both produced unsatisfactory results — the R1-distill variant struggled with instruction following, and the Coder variant exhibited format drift and content repetition during translation. For alternative models, prefer **Instruct-tuned models designed for natural language processing** rather than code-oriented or reasoning-distilled variants.

For the complete list of local model names (used to specify `-seg_model` or `-transl_model` when using the local backend), see [transcription_n_translation.md](transcription_n_translation.md) - "1.2 Local Backend".

### 2.2 API Models

All API backends require an internet connection and API key (configured via `translate_config.json` or `.env`). See [transcription_n_translation.md](transcription_n_translation.md) - "1.3 Online Backends" for the full list of supported backends and their base URLs.

### 2.3 When to Use Which

| Scenario | Recommended Choice |
| :------- | :----------------- |
| **Fully offline, sensitive content** | Local: Phi-4 |
| **Best speed & quality, API budget available** | API: DeepSeek (v4-flash) |
| **Best quality, not cost-sensitive** | API: Anthropic (claude-opus-4-8) |
| **Hybrid — local ASR, API LLM** | Local Phi-4 for segmentation, DeepSeek API for translation (recommended) |
| **Budget hardware (4 GB GPU or less)** | Use APIs for everything |

---

## Part 3 — LLM Inference Benchmark

> **Test source video**: [Learning Dune 3 — Mod Matrix Explained](https://youtu.be/0WD4KMbOWGA?si=UKXTDKPL6kNrq1jD) (12:09)  
> **Reference subtitles**: 208 segments (manually proofread)

### 3.1 Segmentation Performance

Segmentation runs the full ASR → alignment → LLM refinement pipeline to split raw transcription into subtitle segments.

**Destructive split rate**: `destructive_splits / total_segments`  
A *destructive split* cuts a sentence at a non-grammatical boundary, producing a subject-verb fragment or a mid-phrase break. Splits at natural conjunction/clause boundaries are counted as correct.

| Model | Time | Segments | Destructive Splits¹ |
| :---- | :---: | :-------: | :------------------: |
| Phi-4 14B (Q4_K_M) | 202.0s | 199 | 3 (1.5%) |
| Ministral-3-8B-Instruct 8B (Q8_0) | 195.6s | 200 | 3 (1.5%) |
| Ministral-3-14B-Instruct 14B (Q5_K_M) | 254.5s | 206 | 3 (1.5%) |
| Qwen3.5-9B (Q8_0) | 346.1s | 211 | 14 (6.6%) |
| DeepSeek API (deepseek-v4-flash) | 84.9s | 205 | 4 (2.0%) |

> **Notes**:
> - Time is **segmentation-only** (LLM refinement), measured on pre-transcribed & aligned words loaded from cache. All models share the same cached ASR output, so ASR/alignment time is excluded from per-model comparisons.

### 3.2 Translation Performance

Translation runs the reference SRT through LLMs in two modes:
- **Accurate**: 2-line sliding window (best per-segment accuracy)
- **Flexible**: 4-line sliding window (better cross-segment flow, may merge segments)

| Model | Accurate Time | Accurate Quality | Flexible Time | Flexible Quality |
| :---- | :-----------: | :--------------: | :-----------: | :--------------: |
| Phi-4 (14B, Q4_K_M) | 627.3s | B — minor terminology inconsistencies, occasional mistranslations | 675.3s | B — same quality as accurate mode, some segments naturally merged |
| Qwen3.5-9B (Q8_0) | 542.2s | B — quality similar to Phi-4, but faster inference | 561.5s | B — more natural expression than accurate mode, faster inference |
| Ministral-3-8B-Instruct (8B, Q8_0) | 580.3s | C — minor terminology inconsistencies, occasional mistranslations, language less natural | 528.7s | C — terms consistent, a few untranslated words, language less natural |
| Ministral-3-14B-Instruct (14B, Q5_K_M) | 657.4s | C — similar content to 8B variant, slightly slower inference | 1099.1s | C — similar content to 8B variant, too slow |
| DeepSeek API (deepseek-v4-flash) | 207.7s | A — more natural expression, occasional term errors, no other issues | 112.8s | A — more natural expression, almost error-free, more segment merging |

> **Notes**: Time is wall-clock for translating 208 reference segments (SRT). Quality grades reflect terminology consistency, technical accuracy in context, and naturalness of Chinese expression; local models generally lag behind online APIs in Chinese naturalness. Key: A=Excellent, B=Good, C=Average, D=Poor.
