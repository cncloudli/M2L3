# Transcription & Translation Technical Implementation

This document describes the technical implementation details of speech transcription (ASR → alignment → post-processing) and translation (local/online LLM) in this project. Usage instructions already covered in the README are not repeated.

---

## 1. LLM Backend Infrastructure

Both the segmentation pipeline and the translation engine share a unified LLM call layer. This section describes the backends, the call interface, and shared configuration — everything related to how LLM inference is performed, regardless of the consumer.

### 1.1 Overall Architecture

All LLM interaction in the project is delegated to [`tools/llm_call.py`](../tools/llm_call.py), which provides a common `chat()` interface across three backend types:

```
tools/llm_call.py              # Unified LLM call layer
  ├─ LLMCallLocal              # Local llama-server subprocess
  ├─ LLMCallOpenAI             # OpenAI-compatible API (DeepSeek, OpenAI, Qwen, Gemini)
  └─ LLMCallAnthropic          # Anthropic Messages API

tools/segment.py               # Segmentation pipeline (10-phase)
  └─ Segmentor.segment_words() # Punctuation infill + conjunction classification
       └─ llm_call.chat()      # via create_llm_call() factory

scripts/translate.py           # Translation engine
  └─ translate()               # Sliding-window translation
       └─ llm_call.chat()      # via create_llm_call() factory
```

The `LLMCall.chat()` interface accepts `system`, `user`, `max_tokens`, `temperature`, and `cache_prompt` parameters, normalising the differences across backends internally.

[`create_llm_call()`](../tools/llm_call.py) factory selects the appropriate class based on the backend name:

```python
def create_llm_call(backend, model=None, api_key="", base_url="", gpu_layers=None):
    if backend == "local":
        m = model or "phi4"
        return LLMCallLocal(model=m, gpu_layers=gpu_layers)
    if backend not in BACKEND_DEFAULTS:
        raise ValueError(f"Unknown backend: {backend!r}. "
                         f"Available: local, {', '.join(BACKEND_DEFAULTS)}")
    cfg = BACKEND_DEFAULTS[backend]
    m = model or cfg["default_model"]
    if backend == "anthropic":
        if not api_key:
            print("  [WARN] anthropic_api_key is not set")
        return LLMCallAnthropic(model=m, api_key=api_key, base_url=base_url)
    if backend == "deepseek":
        base = base_url or "https://api.deepseek.com"
        if "anthropic" in base.lower():
            return LLMCallAnthropic(model=m, api_key=api_key, base_url=base)
    return LLMCallOpenAI(backend=backend, model=m, api_key=api_key, base_url=base_url)
```

### 1.2 Local Backend

#### 1.2.1 Available Models

| Model Name | Default | GGUF File |
|------------|---------|-----------|
| `phi4` | ✅ (default) | `phi-4-Q4_K_M.gguf` |
| `qwen3.5-9b` | | `Qwen3.5-9B-Q8_0.gguf` |
| `ministral-3-8b` | | `Ministral-3-8B-Instruct-2512-Q8_0.gguf` |
| `ministral-3-14b-instruct` | | `Ministral-3-14B-Instruct-2512-Q5_K_M.gguf` |

#### 1.2.2 llama-server Subprocess Management

`LLMCallLocal` manages the lifecycle of a `llama-server.exe` child process:

**Start** — [`LLMCallLocal.start()`](../tools/llm_call.py):
```python
cmd = [
    self.server_path, "-m", self.model_path,    # GGUF model path
    "--port", str(self.port),
    "-t", str(self.num_threads),                 # CPU threads
    "-c", str(self.context_size),                # Context length
    "--cont-batching",                            # Continuous batching
    "-ngl", str(self.gpu_layers),                 # GPU layers
]
self.server_proc = subprocess.Popen(..., creationflags=subprocess.CREATE_NO_WINDOW)
```

After launch, it polls the `/health` endpoint, waiting up to 120s for the service to become ready.

**Inference** — [`LLMCallLocal.chat()`](../tools/llm_call.py):
```python
r = requests.post(
    f"{self.server_url}/completion",
    json={
        "prompt": prompt,           # Formatted full prompt
        "n_predict": n_pred,        # Dynamic prediction length (mapped from max_tokens param)
        "temperature": temperature,  # Low temperature → deterministic output
        "cache_prompt": cache_prompt,  # KV-cache reuse (optional)
    },
    proxies=NO_PROXY,               # Bypass system proxy for local server
    timeout=120,                    # Single inference timeout
)
```

> **Note:** `temperature` defaults to `0`; consumers pass their own value at call time. `n_predict` falls back to `max(64, len(user) // 4 + 30)` when `max_tokens` is unspecified, but consumers such as the translation engine always pass an explicit value.
>
> **Note (thinking suppression):** When using a reasoning model (a model in `_REASONING_MODELS`, currently only `qwen3.5-9b`), [`LLMCallLocal.chat()`](../tools/llm_call.py) automatically appends a thinking/chain-of-thought suppression instruction to the system prompt **before** template wrapping. This mirrors the API-level `body["thinking"] = {"type": "disabled"}` used for online backends (see §1.3.1), but works at the prompt level since the `/completion` endpoint has no API parameter for thinking control. The same model list also drives `strip_reasoning()` post-processing (see §3.3.4).

**Stop** — [`LLMCallLocal.stop()`](../tools/llm_call.py): Four-level shutdown:
1. `proc.terminate()` + 5s wait
2. If that fails → `proc.kill()` + 3s wait
3. If still alive → `taskkill /F /PID` force terminate
4. If the port is still occupied → `taskkill /F /T /PID` force terminate the process tree

#### 1.2.3 Automatic GPU Layer Detection: [`auto_gpu_layers()`](../tools/llm_call.py)

```python
def auto_gpu_layers(model="phi4"):
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0

    n_layers = _MODEL_LAYERS.get(model, 0)
    if n_layers == 0:
        return 0
    model_file = _MODEL_REGISTRY.get(model)
    if not model_file:
        return 0
    model_path = os.path.join(_APP_DIR, "models", model_file)
    if not os.path.exists(model_path):
        return 0

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        free_mib = int(out.stdout.strip().splitlines()[0].strip())
        available = free_mib / 1024.0 - 1.2   # GB, minus 1.2 GB buffer
    except Exception:
        try:
            props = torch.cuda.get_device_properties(0)
            available = props.total_memory / 1e9 - 1.5
        except Exception:
            return 0
    model_size = os.path.getsize(model_path) / 1e9
    if available <= 0 or model_size <= 0:
        return 0
    per_layer = model_size / n_layers * 1.1
    return max(0, min(n_layers, int(available / per_layer)))
```

First queries actual free VRAM via nvidia-smi (minus 1.2 GB buffer for CUDA context + KV cache), falling back to CUDA total_memory estimate (minus 1.5 GB).

### 1.3 Online Backends

#### 1.3.1 OpenAI-Compatible Backend: [`LLMCallOpenAI`](../tools/llm_call.py)

Uses the `/chat/completions` endpoint uniformly, constructing the standard `system` + `user` message format:

```python
headers = {"Content-Type": "application/json"}
if self.api_key:
    headers["Authorization"] = f"Bearer {self.api_key}"

body = {
    "model": self.model,
    "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ],
    "temperature": 0.1,
    "max_tokens": max_tokens,
}
# DeepSeek defaults to thinking mode which eats the token budget;
# explicitly disable it when using DeepSeek.
if self.backend == "deepseek":
    body["thinking"] = {"type": "disabled"}

r = requests.post(
    f"{self.base_url}/chat/completions",
    json=body,
    headers=headers,
    timeout=120,
    proxies=NO_PROXY,
)
```

**Compatibility handling:** DeepSeek V4 enables thinking mode by default, which consumes the token budget and results in an empty `content` field. It automatically adds `body["thinking"] = {"type": "disabled"}` to disable it.

All supported backends share a single client implementation, differing only in `base_url` and default model:

| Backend | base_url | Default Model |
|---------|-----------|---------------|
| deepseek | `api.deepseek.com` | deepseek-v4-flash |
| openai | `api.openai.com` | gpt-5.6-terra |
| qwen | `dashscope-intl.aliyuncs.com/.../v1` | qwen3.5-plus |
| gemini | `generativelanguage.googleapis.com/.../openai` | gemini-3.5-flash |

#### 1.3.2 Anthropic Backend: [`LLMCallAnthropic`](../tools/llm_call.py)

Uses the Anthropic Messages API format, which differs from the OpenAI-compatible format:

```python
r = requests.post(
    f"{self.base_url}/v1/messages",
    json={
        "model": self.model,
        "system": system,                    # Top-level system parameter
        "messages": [{"role": "user", "content": user}],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    },
    headers={
        "x-api-key": self.api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    },
    timeout=120,
    proxies=NO_PROXY,
)
```

**Special handling:** DeepSeek supports two API formats; the code automatically determines whether to use OpenAI format or Anthropic Messages format based on whether `api_base_url` contains `"anthropic"` (see the [`create_llm_call()`](../tools/llm_call.py) DeepSeek branch).

| Backend | base_url | Default Model |
|---------|-----------|---------------|
| anthropic | `api.anthropic.com` | claude-opus-4-8 |

### 1.4 Prompt Template Formatting

For local inference, the `system` and `user` strings must be wrapped in the model-specific chat template before being sent to `/completion`. This is handled by [`_fmt_prompt()`](../tools/llm_call.py), which dispatches via the `_MODEL_TEMPLATES` dictionary by model name:

```python
_MODEL_TEMPLATES = {
    "phi4":                       "phi4",
    "qwen3.5-9b":                 "qwen",
    "ministral-3-8b":             "ministral",
    "ministral-3-14b-instruct":   "ministral",
}

def _fmt_prompt(system, user, model="phi4"):
    tmpl = _MODEL_TEMPLATES.get(model, "phi4")
    # ...
```

**Phi-4 template** (default fallback):
```
<|system|>
{system}
<|end|>
<|user|>
{user}
<|end|>
<|assistant|>
```

**Qwen ChatML template** (qwen3.5-9b):
```
<|im_start|>system
{system}
<|im_end|>
<|im_start|>user
{user}
<|im_end|>
<|im_start|>assistant
```

**Ministral template** (ministral-3-8b, ministral-3-14b-instruct):
```
[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT]
[INST]{user}[/INST]
```

For online API backends, system/user messages are passed through native API fields and do not go through this function.

### 1.5 Shared Configuration

The following fields in [`translate_config.json`](../translate_config.json) configure the LLM backend connection and are shared across both the segmentation and translation pipelines:

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `openai_api_key` | string | `""` | API key for OpenAI-compatible backends (deepseek, openai, qwen, gemini). Can also be set via `OPENAI_API_KEY` in `.env`. |
| `anthropic_api_key` | string | `""` | API key for the Anthropic backend. Can also be set via `ANTHROPIC_API_KEY` in `.env`. |
| `api_base_url` | string | `""` | Custom API base URL. When empty, uses each backend's official default endpoint. Can also be set via `API_BASE_URL` in `.env`. |

---

## 2. Transcription Pipeline

### 2.1 Overall Flow

```
Input audio/video
  │
  ├─ FFmpeg audio decoding ([check_ffmpeg()](../tools/env_check.py))
  ├─ WhisperX batch transcription (faster-whisper-large-v3 + Silero VAD)
  ├─ Wav2Vec2 phoneme-level forced alignment
  ├─ Timestamp post-processing ([fix_word_timestamps()](../tools/extract.py))
  ├─ LLM segmentation (10 phases, see [segmentation_pipeline.md](segmentation_pipeline.md))
  └─ Export SRT + TXT
```

### 2.2 WhisperX Transcription

[`transcribe_file()`](../scripts/transcribe.py) model loading section — calls `whisperx.load_model()` to load faster-whisper-large-v3:

```python
model = whisperx.load_model(
    str(MODEL_PATH), device=device, compute_type=compute_type,
    asr_options={
        "beam_size": 5,              # Beam search width
        "temperatures": [0.0],        # Deterministic decoding (no temperature sampling)
        "no_speech_threshold": 0.38,  # Skip threshold for silent segments
        "compression_ratio_threshold": 2.4,  # Prevent repetitive output
        "condition_on_previous_text": False,  # Decode each segment independently
        "word_timestamps": False,     # Don't use Whisper's built-in word timestamps (use Wav2Vec2 instead)
    },
    language="en",                   # Fixed to English
    vad_method="silero",             # Use Silero VAD for voice activity detection
    vad_options={
        "chunk_size": 30,
        "vad_onset": 0.500,          # Speech start sensitivity
        "vad_offset": 0.363,         # Speech end sensitivity
    },
    download_root=str(BASE_DIR / "models"),
)
```

**Design Rationale:**

- **Silero VAD** replaces Whisper's built-in VAD. Silero is a lightweight (~5 MB) PyTorch model that slides a window over audio to detect voice activity. WhisperX splits VAD-detected speech segments into chunks and feeds each independently to Whisper, yielding more precise results than processing the entire audio at once.
- **`word_timestamps=False`** — Whisper can output word-level timestamps natively, but accuracy is poor (typically off by hundreds of milliseconds). The project disables built-in timestamps and uses Wav2Vec2 for dedicated forced alignment to achieve phoneme-level precision.
- **`condition_on_previous_text=False`** — Each decoding chunk is processed independently, preventing errors from propagating backward from earlier segments.

### 2.3 Wav2Vec2 Phoneme-Level Forced Alignment

[`transcribe_file()`](../scripts/transcribe.py) alignment section:

```python
align_model, align_metadata = whisperx.load_align_model(
    language_code=result["language"], device=device,
)
result_aligned = whisperx.align(
    result["segments"], align_model, align_metadata,
    str(video_path), device,
    return_char_alignments=False, print_progress=True,
)
```

**How it works:**

Wav2Vec2 (base 960h, ~360 MB) is a self-supervised speech model pre-trained on LibriSpeech 960h. The `align()` function in WhisperX operates as follows:

1. Decomposes each Whisper output text segment into a phoneme sequence
2. Generates a phoneme-level probability matrix over the corresponding audio segment using Wav2Vec2
3. Applies **Viterbi decoding** to find the most likely phoneme boundaries
4. Maps phoneme boundaries back to word boundaries (words are sequences of consecutive phonemes)

The resulting word timestamps are accurate to the phoneme level (~10-30ms precision), far exceeding Whisper's native output.

> **NLTK's role**: Before alignment, WhisperX uses NLTK's `punkt` tokenizer to split each ASR segment into individual sentences, then aligns each sentence as a whole against the audio. Sentence-level alignment provides more phoneme context than word-by-word alignment, improving boundary accuracy.

### 2.4 Timestamp Post-processing: [`fix_word_timestamps()`](../tools/extract.py)

Wav2Vec2 forced alignment has a common issue: when a word is followed by a long pause or breath, the alignment algorithm sometimes includes the pause in the word's duration, inflating its timestamp.

**Algorithm (two-pass):**

```
Pass 1 — Trim abnormally long words:
  For each word i:
    - Take 10 words before and after as the local window
    - Compute the local median word duration in the window
    - Threshold = max(local_median × 5, 1.0s)
    - If current word duration > threshold:
        Clip end time to max(local_median × 3, 0.5s)
        And not beyond the next word's start time

Pass 2 — Resolve overlaps:
  If word i's end time > word i+1's start time:
    word i's end time = word i+1's start time
```

Benefits:
- Uses **local median** instead of a global fixed threshold, adapting to speech rate changes (narrow window during fast speech, wide window during slow speech)
- The 5× local median threshold ensures normal speech rate variation isn't incorrectly clipped
- Pass 2 guarantees monotonically increasing timestamps, meeting SRT format requirements

### 2.5 Segmentation

After timestamp post-processing, the word list undergoes LLM-based segmentation to group words into subtitle segments. The segmentation pipeline implements a 10-phase algorithm that combines rule-based and LLM-driven splitting (using the same `LLMCall` layer described in §1):

```
Phases 1-4:  LLM punctuation infill — detect sentence boundaries
Phase 5:     Build segments from all accumulated break points
Phase 6:     Comma forced split — split at clause commas
Phase 7:     Conjunction split — rule-based + LLM classifier
Phase 8:     LLM run-on repunctuation — recursive split
Phase 9:     LLM-guided conjunction-fragment merge
Phase 10:    Emergency split — last resort rule-based
```

See [segmentation_pipeline.md](segmentation_pipeline.md) for the full algorithm description, guard mechanisms, and parameter details.

Usage parameters (`-seg_backend`, `-seg_model`, `-gpu-layers`) are covered in the README.

### 2.6 Word-Level Data Cache

[`save_words_cache()` / `load_words_cache()`](../tools/cache.py):

```python
def save_words_cache(all_words, cache_path):
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(all_words, f, ensure_ascii=False, indent=2)
```

The ASR + alignment output is deterministic for the same audio file, so caching avoids redundant re-computation. The `-no-cache` flag controls whether to skip caching:
- First run: extracted word data is saved to `cache/<stem>_words.json`
- Subsequent runs (cache hit): skip ASR and alignment, load cached data directly into LLM segmentation

This saves minutes per run when debugging segmentation parameters without re-running ASR.

### 2.7 Export

[`export_srt()` / `export_txt()`](../tools/export.py) and [`format_srt_time()`](../tools/format.py):

- **`export_srt()`**: Iterates over the segments list, using `format_srt_time()` to convert float seconds to `HH:MM:SS,mmm` format
- **`export_txt()`**: When `pure_text=False`, outputs full SRT format (with timecodes); when `pure_text=True`, outputs only the text for each segment
- **`format_srt_time()`**: Converts the fractional part of seconds to milliseconds using `round()` to handle floating-point precision

---

## 3. Translation Pipeline

### 3.1 Translation Configuration

[`translate_config.json`](../translate_config.json) fields that control translation behaviour:

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `target_lang` | string | `"Chinese"` | Target language name (used in LLM prompts) |
| `target_lang_code` | string | `"CN"` | Target language code (used in output filename suffix, e.g. `_CN.srt`) |
| `source_lang` | string | `"English"` | Source language name (used in LLM prompts) |
| `add_punctuation` | bool | `false` | Whether to add sentence-ending punctuation to translations. When `false`, punctuation is stripped by post-processing. |
| `allow_flexible_word_order` | bool | `false` | Whether to allow cross-line word reordering. **Only effective in `flexible` mode** — not passed to the LLM in `accurate` mode (line-by-line locked). Strongly recommended **for online API backends only** — may cause unstable line index shifts with local models. |
| `allow_simplify_wording` | bool | `false` | Whether to allow simplifying colloquial expressions. **Only effective in `flexible` mode** — not passed to the LLM in `accurate` mode. Strongly recommended **for online API backends only** — local models have limited ability to follow this instruction reliably. |
| `number_mode` | string | `"auto"` | Number format control. Options: `"auto"` (no special handling), `"src_lang"` (preserve original number formatting), `"digits"` (all numbers use Arabic digits), `"tgt_lang"` (numbers converted to target language native form). Available in all modes; local models may not always follow this instruction reliably due to performance limitations. |
| `space_between_cjk_and_latin` | bool | `false` | Whether to auto-insert spaces between CJK and Latin characters. Only affects output formatting. |
| `glossary` | array | `["EXAMPLE1", ...]` | Glossary terms — these words **must not** be translated and must be kept as-is. Available in all modes; local models may not always follow this instruction reliably due to performance limitations. |
| `custom_system_prompt` | string or null | `null` | Custom system prompt override. When non-null, fully replaces the auto-built system prompt. |
| `cache_prompt` | bool | `false` | Whether to enable LLM KV-cache reuse. Reduces latency on API backends; may cause instruction drift in local models during long tasks — keep `false` for local. Mode-independent. |
| `drift_threshold` | number or bool | `false` | Content drift detection threshold. `false` skips detection entirely (zero extra computation); a number (e.g. `0.16`) enables detection — when the character-set similarity gap between adjacent translations and their sources exceeds this value, content drift is flagged and single-line re-translation is triggered. Lower values are more sensitive. |

API credential fields (`openai_api_key`, `anthropic_api_key`, `api_base_url`) are documented in §1.5 Shared Configuration.

### 3.2 Input Parsing

[`parse_srt()` / `rebuild_srt()`](../scripts/translate.py)

**Parsing** `parse_srt()`: Scans line by line, matching index → timestamp line (`HH:MM:SS,mmm --> HH:MM:SS,mmm`) → text block, building a list of `{start, end, text}` dictionaries.

### 3.3 Sliding-Window Translation Engine

In [`translate()`](../scripts/translate.py) → [`_translate_impl()`](../scripts/translate.py), to avoid excessive context length, the source text is processed in a sliding window. The translate section uses configurable `tr_size` lines, with surrounding context lines passed as read-only reference.

#### 3.3.1 Translation Mode

[`translate()`](../scripts/translate.py) supports two sliding-window modes, controlled by the `-mode` CLI argument:

| Parameter | `accurate` (default) | `flexible` |
| --------- | ------------------- | ---------- |
| `tr_size` (translate window) | 2 lines | 4 lines |
| `ctx_size` (context window) | 4 lines | 8 lines |
| `first_step` | 1 line | 2 lines |
| `later_step` | 2 lines | 4 lines |
| `show_timecodes` | False | True |

- **`accurate`** — Smaller windows produce strictly line-by-line translations with minimal risk of timeline misalignment. **Recommended for local Phi-4 models** — slower local inference benefits from the smaller window for stable latency and lower resource usage.
- **`flexible`** — Larger windows with timecode context give the LLM more room to produce natural-sounding phrasing, including cross-line word reordering. **Better suited for online API backends** — high-speed APIs can take full advantage of the larger context window.

#### 3.3.2 System Prompt Construction

Dynamically assembles the system prompt from multiple optional instruction blocks based on configuration:

```
"Translate each line of {src_lang} text below to {tgt_lang}."
+ [add_punctuation? "Add appropriate punctuation"]
+ [flexible_word_order? "Allow cross-line word order adjustment" / "Translate each line independently"]
+ [simplify_wording? "Simplify colloquial expressions"]
+ [glossary? "Do not translate the following terms: term1, term2"]
+ [number_mode? "Number formatting rules"]
+ Output format requirement: output lines numbered as `[N]`, e.g. `[1] Translation of source line 1.`
```

#### 3.3.3 Window User Prompt Construction

```python
def _build_window_user_prompt(ctx_lines, tr_start_idx, tr_end_idx, show_timecodes=False):
```

Context-only lines are prefixed with `(BEFORE)` / `(AFTER)`. The translate section numbers its lines **from 1** as `1`, `2`, etc. (bare numbers), with the header instructing the LLM to output `[1]`, `[2]` format.

The model-specific template wrapping (Phi-4 / Qwen / Ministral) is handled by the LLM call layer itself — see §1.4 Prompt Template Formatting.

#### 3.3.4 LLM Call

Each sliding window makes a single LLM call via `llm_call.chat()`:

```python
response = llm_call.chat(system, user,
                          max_tokens=get_min_tokens(
                              getattr(llm_call, 'model', ''),
                              len(user)),
                          temperature=0.1, cache_prompt=CACHE_PROMPT)
if response:
    response = strip_reasoning(response)
```

**Parameters:**

- **`temperature=0.1`** — Low temperature produces stable, consistent translation output. The segmentation engine uses a similar temperature, but the translation engine deliberately keeps it low to prevent hallucination that could shift line indices.
- **`max_tokens=get_min_tokens(model, len(user))`** — Dynamic prediction length. `get_min_tokens()` calculates an appropriate output token limit based on the model and input length, preventing truncation while avoiding wasteful over-allocation.
- **`cache_prompt=CACHE_PROMPT`** — Whether to enable KV-cache reuse (controlled by `cache_prompt` in `translate_config.json`). Reduces latency on API backends; may cause instruction drift in local models during long tasks — keep `false` for local.
- **`strip_reasoning(response)`** — Strips thinking chain preamble. Handles three XML-style thought tags (`<think>`, `<reason>`, `<reasoning>`) as well as prose reasoning preamble before the first structured `[N]` line. Note that DeepSeek's default thinking mode is disabled at the API request layer (`body["thinking"] = {"type": "disabled"}`), not stripped post-hoc. Currently `_REASONING_MODELS` contains only `"qwen3.5-9b"`, whose thinking output is processed by `strip_reasoning()`.

**Retry on empty response:** If the LLM returns `None` (network timeout / server error) or an empty string, retry with exponential backoff (5s → 15s → 30s), up to 3 attempts per sliding window. Falls back to the original text after all retries fail.

#### 3.3.5 Response Parsing

In [`translate()`](../scripts/translate.py) → `_translate_impl()`, the LLM response text is parsed line by line, expecting the format `[N] text` or `[N-M] merged text`:

```python
m = re.match(
    r"\[(\d+)(?:\s*-\s*(\d+))?\]\s*"
    r"(?:\[([\d:,]+\s*-->\s*[\d:,]+)\]\s*)?"
    r"(.+)",
    resp_line,
)
if m:
    n = int(m.group(1)) - 1       # 0-based index
    m_n = int(m.group(2)) - 1 if m.group(2) else n  # merge end
    llm_tc = m.group(3)           # timecode (optional)
    text = m.group(4).strip()
```

When the LLM merges adjacent lines (e.g. `[2-3]`), the timecodes are automatically combined during SRT reconstruction.

#### 3.3.6 CJK Window-Level Auto-Retry

**CJK auto-retry (Layer 1 — per-window):** Some models occasionally echo source text verbatim instead of translating. After each window's LLM response is parsed, the engine checks whether ANY output line contains CJK characters. If none do (all English), it retries with a reinforced system prompt, temperature=0, cache_prompt=False (up to 2 retries per window). This catches wholesale translation failures.

### 3.4 Post-processing Pipeline

[`post_process()`](../scripts/translate.py) and its dependent functions

Each translation result goes through the following post-processing steps:

```
Raw LLM output
  │
  ├─ 1. [_strip_template_tokens()](../scripts/translate.py) — Remove <|end|> and other template tokens
  ├─ 2. [_strip_end_punct()](../scripts/translate.py)       — If add_punct=False, strip sentence-ending punctuation
  ├─ 3. [_normalize_commas()](../scripts/translate.py)      — For CJK languages, replace , with ，
  ├─ 4. [_normalize_quotes()](../scripts/translate.py)      — Replace quotes in CJK content with ""
  └─ 5. [_add_cjk_latin_spacing()](../scripts/translate.py) — Insert spaces between CJK and Latin characters
```

**Punctuation stripping** [`_strip_end_punct()`](../scripts/translate.py): When `add_punct=False`, removes sentence-ending punctuation using `rstrip("。？！.!?")`. The benefit of this approach is that it doesn't impose punctuation constraints on the LLM during generation — translation quality is unaffected, and the user gets punctuation-free output through post-processing.

**Smart quote replacement** [`_normalize_quotes()`](../scripts/translate.py): Determines whether to use full-width or half-width quotes based on whether the quoted content contains CJK characters — English terms within the same text keep half-width quotes, while Chinese content uses full-width quotes.

### 3.5 Overflow Merge Protection

When translating in `accurate` mode with online models (e.g. DeepSeek), the LLM may merge content from beyond the translate window into its output. For example, with `tr_size=2`, the LLM might output `[2-3]`, indicating it has translated line 2 together with content from line 3 (which belongs to the next batch).

This is handled through two complementary mechanisms:

**1. Clamp fix (all backends):**
Instead of discarding output where `m_n >= actual_tr`, the merge range is clamped to `min(m_n, actual_tr - 1)`, ensuring the current window's translation is still stored. The merged content is preserved; overlapping content will be re-translated when its own window arrives.

**2. Overflow re-split (all backends):**
When `m_n >= actual_tr`, the pair `(merged_idx, overflowed_idx)` is recorded. After all batches complete, `_fix_overflow()` sends both source lines and their current translations back to the LLM for a clean re-split. The LLM removes duplicate/overlapping content and returns two independent translations. This ensures no content is duplicated or lost.

```
[Post-process] Fixing 14 overflow merge(s)...
  [Fix] Lines 8 & 9: re-split OK
  [Fix] Lines 10 & 11: re-split OK
  ...
```

---

### 3.6 Content Drift Detection

When an LLM in `accurate` mode merges content from adjacent lines into a single output, "content drift" occurs — the translation has the correct line number, but its content has been copied from or mixed with a neighbouring line's information. This mechanism detects and fixes such issues using Jaccard character-set similarity.

#### Similarity Calculation

String similarity uses **Jaccard character-set similarity** ([`_char_sim()`](../scripts/translate.py)):

```
_char_sim(a, b) = len(set(a) ∩ set(b)) / len(set(a) ∪ set(b))
```

This is the **intersection-over-union** of the two strings' character sets (after deduplication). It cares about *which* characters appear, not their order.

Examples:

- `"你好世界"` (set `{你, 好, 世, 界}`) vs `"你好"` (set `{你, 好}`) = 2/4 = 0.5
- `"filter"` (set `{f, i, l, t, e, r}`) vs `"滤波器"` (set `{滤, 波, 器}`) = 0/9 ≈ 0.0

#### Drift Detection Logic

[`_detect_drift_pairs()`](../scripts/translate.py) checks adjacent line pairs:

1. Compute translation similarity `trans_sim = _char_sim(translation[i], translation[i+1])`
2. Compute source similarity `src_sim = _char_sim(source[i], source[i+1])`
3. Compute the gap `gap = trans_sim - src_sim`
4. If `gap > threshold`, both lines are flagged as drift

**Rationale:** When the source lines talk about different things (`src_sim` is low) but the translations are suspiciously similar (`trans_sim` is high), the LLM likely copied content from one line into the other.

#### Fix Procedure

Each flagged line is sent individually to the LLM with a strict single-line prompt:

```
CRITICAL: Translate ONLY the single line provided below.
Do NOT combine it with any other text.
Output format: [1] <translation>
```

If the isolated re-translation differs from the original, it replaces it; otherwise the original is kept.

#### Configuration

Controlled via the `drift_threshold` field in `translate_config.json`:

| Value   | Behaviour                                                              |
|:--------|:-----------------------------------------------------------------------|
| `false` | Detection skipped entirely (default), no similarity computation        |
| `0.16`  | Detection enabled, `gap > 0.16` triggers drift flagging                |

The threshold of 0.16 was determined empirically: known drift cases have `gap` values between 0.167–0.385, while Phi-4 shows no drift behaviour in `accurate` mode. Different models may have different drift characteristics — adjust based on actual output.

### 3.7 CJK Per-Line Post-Processing

[`_translate_impl()`](../scripts/translate.py) performs a third post-processing pass after overflow merge protection and content drift detection — per-line CJK detection and fix. This catches individual lines that passed the window-level CJK check (Section 3.3.6) because their window-mate contained CJK characters, but are themselves still in English.

**Trigger condition:** The window-level CJK check (3.3.6) only verifies that at least ONE line in the window contains CJK characters. When a window has mixed CJK and non-CJK lines, the window passes, but lines where the LLM echoed English source text or the original was preserved go unnoticed at the window level.

**Detection:** Scans all result lines, flagging those that simultaneously satisfy:
1. Translation result is not `None`
2. Contains no CJK characters (regex `[一-鿿㐀-䶿　-〿＀-￯]`)
3. Either a preserved original text line (`results[i] == text_lines[i]`), or length ≥ 3 words (excludes short noise)

**Fix procedure:** Each flagged line is sent independently to the LLM with a strict context-free single-line prompt:

```
CRITICAL: Translate ONLY the single line below.
Output format: [1] <translation>.
The translation MUST contain Chinese characters.
```

Retry parameters are `temperature=0, cache_prompt=False`, isolated from the original line's context. If the new translation contains CJK characters it replaces the old value; otherwise the original is kept.

```
  [CJK Fix] 3 line(s) lack CJK, re-translating individually...
    [CJK Fix] Line 5: fixed
      Old: This is the original English subtitle text.
      New: 这是原始英文字幕文本。
    [CJK Fix] Line 12: new output also no CJK, keeping original
  [CJK Fix] Done — 3 line(s) processed
```


### 3.8 SRT Reconstruction & File Output

```python
def rebuild_srt(blocks, translated_texts):
    out = []
    i = 0
    seq = 1
    while i < len(blocks):
        # Skip blocks absorbed by a neighbor (None translation)
        if i < len(translated_texts) and translated_texts[i] is None:
            i += 1
            continue
        text = translated_texts[i] if i < len(translated_texts) else blocks[i]["text"]
        start = blocks[i]["start"]
        # Look ahead: absorb trailing Nones into this block's timecode
        end = blocks[i]["end"]
        j = i + 1
        while j < len(blocks):
            if j < len(translated_texts) and translated_texts[j] is not None:
                break
            end = blocks[j]["end"]
            j += 1
        out.append(str(seq))
        seq += 1
        out.append(f"{start} --> {end}")
        out.append(text)
        out.append("")
        i = j
    return "\n".join(out)
```

When the LLM merges adjacent lines (e.g. `[2-3]` means lines 2 and 3 are merged), `translated_texts[2]` (0-based index 1) is set to `None`, and `rebuild_srt()` automatically absorbs that block's timecode into the preceding valid block, ensuring the timeline is preserved.


### 3.9 Integration into main.py

In [`main.py`](../main.py)'s CLI `__main__` block, translation is now a one-liner via [`translate_file()`](../scripts/translate.py):

```python
if args.translate and srt_path and os.path.exists(srt_path):
    translate_file(
        srt_path,
        output_dir=str(Path(srt_path).parent),
        transl_backend=args.transl_backend, transl_model=transl_model,
        api_key=api_key, base_url=api_base,
        gpu_layers=args.gpu_layers,
        mode=transl_mode,
    )
```

[`translate_file()`](../scripts/translate.py) wraps `read_input()` → `translate()` → `write_output()` in a single call, so `main.py` doesn't need to repeat file I/O logic. Translation runs in-process (direct function call, not subprocess), sharing the same `LLMCall` layer used by segmentation.
