# Transcription & Translation Technical Implementation

This document describes the technical implementation details of speech transcription (ASR → alignment → post-processing) and translation (local/online LLM) in this project. Usage instructions already covered in the README are not repeated.

---

## 1. Transcription Pipeline

### 1.1 Overall Flow

```
Input audio/video
  │
  ├─ FFmpeg audio decoding ([check_ffmpeg()](../tools/env_check.py))
  ├─ WhisperX batch transcription (faster-whisper-large-v3 + Silero VAD)
  ├─ Wav2Vec2 phoneme-level forced alignment
  ├─ Timestamp post-processing ([fix_word_timestamps](../tools/extract.py))
  ├─ LLM segmentation (10 phases, see [segmentation_pipeline.md](segmentation_pipeline.md))
  └─ Export SRT + TXT
```

### 1.2 WhisperX Transcription

[`video_to_srt()`](../main.py) model loading section — calls `whisperx.load_model()` to load faster-whisper-large-v3:

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

### 1.3 Wav2Vec2 Phoneme-Level Forced Alignment

[`video_to_srt()`](../main.py) alignment section:

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

### 1.4 Timestamp Post-processing: [`fix_word_timestamps()`](../tools/extract.py)

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

### 1.5 Word-Level Data Cache

[`save_words_cache()` / `load_words_cache()`](../tools/cache.py):

```python
def save_words_cache(all_words, cache_path):
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(all_words, f, ensure_ascii=False, indent=2)
```

ASR + alignment is the most time-consuming part of the pipeline (especially Wav2Vec2 alignment). The `-no-cache` flag controls whether to skip caching:
- First run: extracted word data is saved to `cache/<stem>_words.json`
- Subsequent runs (cache hit): skip ASR and alignment, load cached data directly into LLM segmentation

This saves minutes per run when debugging segmentation parameters without re-running ASR.

### 1.6 Export

[`export_srt()` / `export_txt()`](../tools/export.py) and [`format_srt_time()`](../tools/format.py):

- **`export_srt()`**: Iterates over the segments list, using `format_srt_time()` to convert float seconds to `HH:MM:SS,mmm` format
- **`export_txt()`**: When `pure_text=False`, outputs full SRT format (with timecodes); when `pure_text=True`, outputs only the text for each segment
- **`format_srt_time()`**: Converts the fractional part of seconds to milliseconds using `round()` to handle floating-point precision

---

## 2. Translation Pipeline

### 2.1 Overall Architecture

[translate.py](../translate.py) uses an **abstract backend + factory pattern**, supporting 7 translation backends:

```
translate.py
  ├─ [TranslatorLocal](../translate.py)      # Local llama-server subprocess
  ├─ [TranslatorOpenAI](../translate.py)     # OpenAI-compatible API (DeepSeek, OpenAI, Qwen, Gemini, Ollama)
  └─ [TranslatorAnthropic](../translate.py)  # Anthropic Messages API
```

[`create_translator()`](../translate.py) factory function selects the backend based on the `-backend` argument:

```python
def create_translator(backend, model=None, gpu_layers=None):
    if backend == "local":
        return TranslatorLocal(model=m, gpu_layers=gpu_layers)
    if backend == "anthropic":
        return TranslatorAnthropic(model=m, api_key=api_key, ...)
    # Other backends → TranslatorOpenAI
    return TranslatorOpenAI(backend=backend, model=m, ...)
```

### 2.2 Local Translation: [`TranslatorLocal`](../translate.py)

#### 2.2.1 llama-server Subprocess Management

`TranslatorLocal` manages the lifecycle of a `llama-server.exe` child process:

**Start** — [`TranslatorLocal.start()`](../translate.py):
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

**Inference** — [`TranslatorLocal.translate_batch()`](../translate.py):
```python
r = requests.post(
    f"{self.server_url}/completion",
    json={
        "prompt": prompt,           # Formatted full prompt
        "n_predict": max(200, int(len(user) * 1.5)),  # Dynamic prediction length
        "temperature": 0.1,         # Low temperature → deterministic translation output
        "cache_prompt": CACHE_PROMPT,  # KV-cache reuse (optional)
    },
    timeout=120,                    # Single inference timeout
)
```

**Stop** — [`TranslatorLocal.stop()`](../translate.py): Three-level shutdown strategy:
1. `proc.terminate()` + 5s wait
2. If that fails → `proc.kill()` + 3s wait
3. If still alive → `taskkill /F /PID` force terminate

#### 2.2.2 Automatic GPU Layer Detection: [`auto_gpu_layers()`](../translate.py)

```python
def auto_gpu_layers(model="phi4"):
    available = total_vram - 1.5             # Reserve 1.5 GB for system/ASR models
    per_layer = model_size / n_layers * 1.1  # Size per layer × 1.1 safety factor
    return max(0, min(n_layers, int(available / per_layer)))
```

Estimates how many layers can fit in GPU VRAM based on total VRAM, model file size, and layer count, reserving 1.5 GB for other processes.

### 2.3 Online Translation Backends

#### 2.3.1 OpenAI-Compatible Backend: [`TranslatorOpenAI`](../translate.py)

Uses the `/chat/completions` endpoint uniformly, constructing the standard `system` + `user` message format:

```python
body = {
    "model": self.model,
    "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ],
    "temperature": 0.1,
    "max_tokens": max_tokens,
}
```

**Compatibility handling:** DeepSeek V4 enables thinking mode by default, which consumes the token budget and results in an empty `content` field. It automatically adds `body["thinking"] = {"type": "disabled"}` to disable it.

All supported backends share a single client implementation, differing only in `base_url` and default model:

| Backend | base_url | Default Model |
|---------|-----------|---------------|
| deepseek | `api.deepseek.com` | deepseek-v4-flash |
| openai | `api.openai.com` | gpt-5.6-terra |
| qwen | `dashscope-intl.aliyuncs.com/.../v1` | qwen3.5-plus |
| gemini | `generativelanguage.googleapis.com/.../openai` | gemini-3.5-flash |
| ollama | `localhost:11434/v1` | llama4 |

#### 2.3.2 Anthropic Backend: [`TranslatorAnthropic`](../translate.py)

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
    headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
)
```

**Special handling:** DeepSeek supports two API formats; the code automatically determines whether to use OpenAI format or Anthropic Messages format based on whether `api_base_url` contains `"anthropic"` (see the [`create_translator()`](../translate.py) DeepSeek branch).

### 2.4 Batch Processing & Prompt Design

#### 2.4.1 Batched Translation

In [`translate_texts()`](../translate.py), to avoid excessive context length, the source text is sent in batches of `BATCH_SIZE` (default 8 lines). Each batch constructs its own independent prompt.

#### 2.4.2 System Prompt Construction: [`_build_system_prompt()`](../translate.py)

Dynamically assembles the system prompt from multiple optional instruction blocks based on configuration:

```
"Translate each line of {src_lang} text below to {tgt_lang}."
+ [add_punctuation? "Add appropriate punctuation"]
+ [flexible_word_order? "Allow cross-line word order adjustment" / "Translate each line independently"]
+ [simplify_wording? "Simplify colloquial expressions"]
+ [glossary? "Do not translate the following terms: term1, term2"]
+ [number_mode? "Number formatting rules"]
+ Output format requirement: "Output each line as 'N: translation result'"
```

**User prompt** — [`_build_user_prompt()`](../translate.py):
```python
def _build_user_prompt(lines):
    return "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))
```

**Template formatting** — [`_fmt_prompt()`](../translate.py):
```python
def _fmt_prompt(model, system, user):
    return (
        f"<|system|>\n{system}\n<|end|>\n"
        f"<|user|>\n{user}\n<|end|>\n"
        f"<|assistant|>\n"
    )
```
This is the Phi-4 chat template format. For online API backends, system/user messages are passed through native API fields and do not go through this function.

#### 2.4.3 Response Parsing

In [`translate_texts()`](../translate.py), the LLM response text is parsed line by line, expecting the format `N: translated text`:

```python
m = re.match(r"(?:Line\s*)?(\d+)\s*[:：]\s*(.+)", resp_line, re.IGNORECASE)
if m:
    num = int(m.group(1)) - 1  # → 0-based index
    if 0 <= num < len(batch):
        translated[num] = m.group(2).strip()
```

**Fallback strategy:** If the LLM ignores the numbering format (e.g., some models don't follow instructions), lines are matched in document order one-to-one.

**Empty response retry:** Uses exponential backoff (5s → 15s) to retry empty responses, handling temporary API rate limits.

### 2.5 Post-processing Pipeline

[`post_process()`](../translate.py) and its dependent functions

Each translation result goes through the following post-processing steps:

```
Raw LLM output
  │
  ├─ 1. [_strip_template_tokens()](../translate.py) — Remove <|end|> and other template tokens
  ├─ 2. [_strip_end_punct()](../translate.py)       — If add_punct=False, strip sentence-ending punctuation
  ├─ 3. [_normalize_commas()](../translate.py)      — For CJK languages, replace , with ，
  ├─ 4. [_normalize_quotes()](../translate.py)      — Replace quotes in CJK content with “”
  └─ 5. [_add_cjk_latin_spacing()](../translate.py) — Insert spaces between CJK and Latin characters
```

**Punctuation stripping** [`_strip_end_punct()`](../translate.py): When `add_punct=False`, removes sentence-ending punctuation using `rstrip("。？！.!?")`. The benefit of this approach is that it doesn't impose punctuation constraints on the LLM during generation — translation quality is unaffected, and the user gets punctuation-free output through post-processing.

**Smart quote replacement** [`_normalize_quotes()`](../translate.py): Determines whether to use full-width or half-width quotes based on whether the quoted content contains CJK characters — English terms within the same text keep half-width quotes, while Chinese content uses full-width quotes.

### 2.6 Configuration System

[`_load_env()` / `_load_config()`](../translate.py)

Uses a three-layer configuration priority:

1. **Code defaults** (lowest priority) → `defaults` dictionary in `_load_config()`
2. **`translate_config.json`** (overrides defaults) → user-editable JSON file
3. **`.env` file + environment variables** (API keys only) → fallback when not set in JSON

[`_load_env()`](../translate.py) uses `os.environ.setdefault()`, so keys set in JSON take higher priority, but `.env` automatically applies when they are left empty.

### 2.7 SRT Parsing & Reconstruction

[`parse_srt()` / `rebuild_srt()`](../translate.py)

**Parsing** `parse_srt()`: Scans line by line, matching index → timestamp line (`HH:MM:SS,mmm --> HH:MM:SS,mmm`) → text block, building a list of `{start, end, text}` dictionaries.

**Reconstruction** `rebuild_srt()`: Fills the translated text list back into the original SRT block structure by index:

```python
for i, blk in enumerate(blocks):
    text = translated_texts[i] if i < len(translated_texts) else blk["text"]
    out.append(str(i + 1))
    out.append(f"{blk['start']} --> {blk['end']}")
    out.append(text)
    out.append("")
```

This ensures the translated SRT output preserves the original timecodes, indices, and blank-line separators.

### 2.8 Integration: Transcription → Translation

In [`main.py`](../main.py)'s CLI `__main__` block, the `-translate` implementation:

```python
if args.translate and srt_path and os.path.exists(srt_path):
    translate_script = str((BASE_DIR / "translate.py").resolve())
    translate_cmd = [sys.executable, translate_script, "-i", srt_path]
    if args.backend != "local":
        translate_cmd += ["-backend", args.backend]
        if args.model:
            translate_cmd += ["-model", args.model]
    subprocess.run(translate_cmd, cwd=str(BASE_DIR))
```

Translation runs as a child process, isolated from the main transcription process. This way, even if translation fails (e.g., API timeout), the transcription-generated SRT file remains intact.

---
