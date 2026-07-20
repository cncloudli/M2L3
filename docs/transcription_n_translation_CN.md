# 转录与翻译技术实现

本文档介绍项目中语音转录（ASR → 对齐 → 后处理）和翻译（本地/在线 LLM）的技术实现细节。README 中已有的用法说明不再重复。

---

## 一、LLM 后端基础设施

切分流水线和翻译引擎共享同一个统一的 LLM 调用层。本章介绍后端类型、调用接口和共用配置——所有与 LLM 推理相关的内容，无论消费者是谁。

### 1.1 整体架构

项目的全部 LLM 交互都委托给统一的 [`tools/llm_call.py`](../tools/llm_call.py) 层，提供通用的 `chat()` 接口，支持三种后端类型：

```
tools/llm_call.py              # 统一 LLM 调用层
  ├─ LLMCallLocal              # 本地 llama-server 子进程
  ├─ LLMCallOpenAI             # OpenAI 兼容 API（DeepSeek, OpenAI, Qwen, Gemini）
  └─ LLMCallAnthropic          # Anthropic Messages API

tools/segment.py               # 切分流水线（10 阶段）
  └─ Segmentor.segment_words() # 标点填充 + 连词分类
       └─ llm_call.chat()      # 经由 create_llm_call() 工厂

scripts/translate.py           # 翻译引擎
  └─ translate()               # 滑动窗口翻译
       └─ llm_call.chat()      # 经由 create_llm_call() 工厂
```

`LLMCall.chat()` 接口接受 `system`、`user`、`max_tokens`、`temperature`、`cache_prompt` 参数，内部统一处理后端差异。

[`create_llm_call()`](../tools/llm_call.py) 工厂根据后端名选择对应的类：

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

### 1.2 本地后端

#### 1.2.1 可用模型

| 模型名 | 默认值 | GGUF 文件 |
|----------|----------|-------------|
| `phi4` | ✅（默认） | `phi-4-Q4_K_M.gguf` |
| `qwen3.5-9b` | | `Qwen3.5-9B-Q8_0.gguf` |
| `ministral-3-8b` | | `Ministral-3-8B-Instruct-2512-Q8_0.gguf` |
| `ministral-3-14b-instruct` | | `Ministral-3-14B-Instruct-2512-Q5_K_M.gguf` |

#### 1.2.2 llama-server 子进程管理

`LLMCallLocal` 管理一个 `llama-server.exe` 子进程的生命周期：

**启动** — [`LLMCallLocal.start()`](../tools/llm_call.py)：
```python
cmd = [
    self.server_path, "-m", self.model_path,    # GGUF 模型路径
    "--port", str(self.port),
    "-t", str(self.num_threads),                 # CPU 线程数
    "-c", str(self.context_size),                # 上下文长度
    "--cont-batching",                            # 连续批处理
    "-ngl", str(self.gpu_layers),                 # GPU 层数
]
self.server_proc = subprocess.Popen(..., creationflags=subprocess.CREATE_NO_WINDOW)
```

启动后轮询 `/health` 端点，最多等待 120s 直到服务就绪。

**推理** — [`LLMCallLocal.chat()`](../tools/llm_call.py)：
```python
r = requests.post(
    f"{self.server_url}/completion",
    json={
        "prompt": prompt,           # 已格式化的完整提示词
        "n_predict": n_pred,        # 动态预测长度（max_tokens 参数映射而来）
        "temperature": temperature,  # 低温度 → 确定性输出
        "cache_prompt": cache_prompt,  # KV-cache 复用（可选）
    },
    proxies=NO_PROXY,               # 绕过系统代理直连本地服务
    timeout=120,                    # 单次推理超时
)
```

> **注意：** `temperature` 默认值为 `0`；调用者可在调用时传入自己的值。`n_predict` 在未指定 `max_tokens` 时默认为 `max(64, len(user) // 4 + 30)`，但翻译引擎等调用者始终会传入显式值。
>
> **注意（思考抑制）：** 使用推理模型（即 `_REASONING_MODELS` 中的模型，目前仅为 `qwen3.5-9b`）时，[`LLMCallLocal.chat()`](../tools/llm_call.py) 会在模板包裹**之前**自动在 system prompt 末尾追加思考/思维链抑制指令。这相当于在线后端在 API 层面做的 `body["thinking"] = {"type": "disabled"}`（见 [1.3.1 OpenAI 兼容后端](#131-openai-兼容后端llmcallopenai)），但对于 `/completion` 端点没有 API 参数可以控制思考模式的情况，通过提示词层面实现同一目的。同一模型列表也驱动 `strip_reasoning()` 后处理（见 [3.3.4 LLM 调用](#334-llm-调用)）。

**关闭** — [`LLMCallLocal.stop()`](../tools/llm_call.py)：四级关闭策略：
1. `proc.terminate()` + 5s 等待
2. 如失败 → `proc.kill()` + 3s 等待
3. 如仍存活 → `taskkill /F /PID` 强制终止
4. 如端口仍被占用 → `taskkill /F /T /PID` 强制终止进程树

#### 1.2.3 GPU 层数自动检测：[`auto_gpu_layers()`](../tools/llm_call.py)

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
        available = free_mib / 1024.0 - 1.2   # GB, 预留 1.2 GB buffer
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

优先通过 nvidia-smi 查询实际空闲显存（减去 1.2 GB buffer 给 CUDA 上下文 + KV cache），失败则回退到 CUDA total_memory 估算（减去 1.5 GB）。

### 1.3 在线后端

#### 1.3.1 OpenAI 兼容后端：[`LLMCallOpenAI`](../tools/llm_call.py)

统一使用 `/chat/completions` 端点，构造标准的 `system` + `user` 消息格式：

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
# DeepSeek 默认启用 thinking 模式，会消耗 token 预算；
# 使用 DeepSeek 时显式禁用。
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

**兼容性处理：** DeepSeek V4 默认启用思考（thinking）模式，会消耗 token 预算导致 `content` 为空。代码自动添加 `body["thinking"] = {"type": "disabled"}` 来禁用。

支持的后端共用一个客户端实现，仅在 `base_url` 和默认模型上不同：

| 后端 | base_url | 默认模型 |
|---------|-----------|---------------|
| deepseek | `api.deepseek.com` | deepseek-v4-flash |
| openai | `api.openai.com` | gpt-5.6-terra |
| qwen | `dashscope-intl.aliyuncs.com/compatible-mode/v1` | qwen3.7-max |
| gemini | `generativelanguage.googleapis.com/v1beta/openai` | gemini-3.5-flash |

#### 1.3.2 Anthropic 后端：[`LLMCallAnthropic`](../tools/llm_call.py)

使用 Anthropic Messages API 格式，与 OpenAI 兼容格式不同：

```python
r = requests.post(
    f"{self.base_url}/v1/messages",
    json={
        "model": self.model,
        "system": system,                    # 顶层 system 参数
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

**特殊处理：** DeepSeek 支持两种 API 格式，通过 `api_base_url` 是否包含 `"anthropic"` 自动判断使用 OpenAI 格式还是 Anthropic Messages 格式（见 [`create_llm_call()`](../tools/llm_call.py) 中的 DeepSeek 分支）。

| 后端 | base_url | 默认模型 |
|---------|-----------|---------------|
| anthropic | `api.anthropic.com` | claude-opus-4-8 |

### 1.4 提示词模板格式化

本地推理时，`system` 和 `user` 字符串需要用模型对应的聊天模板包裹后才能发送到 `/completion`。由 [`_fmt_prompt()`](../tools/llm_call.py) 处理，通过 `_MODEL_TEMPLATES` 字典按模型名派发：

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

**Phi-4 模板**（默认回退）：
```
<|system|>
{system}
<|end|>
<|user|>
{user}
<|end|>
<|assistant|>
```

**Qwen ChatML 模板**（qwen3.5-9b）：
```
<|im_start|>system
{system}
<|im_end|>
<|im_start|>user
{user}
<|im_end|>
<|im_start|>assistant
```

**Ministral 模板**（ministral-3-8b、ministral-3-14b-instruct）：
```
[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT]
[INST]{user}[/INST]
```

对于在线 API 后端，system/user 通过 API 原生字段传递，不经过此函数。

### 1.5 共用配置

[`translate_config.json`](../translate_config.json) 中以下字段配置 LLM 后端连接，在切分和翻译流水线之间共用：

| 字段 | 类型 | 默认值 | 说明 |
| ---- | ---- | ------ | ----------- |
| `openai_api_key` | 字符串 | `""` | OpenAI 兼容后端的 API 密钥（deepseek、openai、qwen、gemini）。可在 `.env` 中通过 `OPENAI_API_KEY` 设置。 |
| `anthropic_api_key` | 字符串 | `""` | Anthropic 后端的 API 密钥。可在 `.env` 中通过 `ANTHROPIC_API_KEY` 设置。 |
| `api_base_url` | 字符串 | `""` | 自定义 API 基础地址。为空时使用各后端的官方默认地址。也可在 `.env` 中通过 `API_BASE_URL` 设置。 |

---

## 二、转录流水线

### 2.1 整体流程

```
输入音频/视频
  │
  ├─ FFmpeg 音频解码（[check_ffmpeg()](../tools/env_check.py)）
  ├─ WhisperX 批量转录（faster-whisper-large-v3 + Silero VAD）
  ├─ Wav2Vec2 音素级强制对齐
  ├─ 时间戳后处理（[fix_word_timestamps()](../tools/extract.py)）
  ├─ LLM 切分（10 阶段，参见 [segmentation_pipeline_CN.md](segmentation_pipeline_CN.md)）
  └─ 导出 SRT + TXT
```

### 2.2 WhisperX 转录

[`transcribe_file()`](../scripts/transcribe.py) 中的模型加载部分 — 调用 `whisperx.load_model()` 加载 faster-whisper-large-v3：

```python
model = whisperx.load_model(
    str(MODEL_PATH), device=device, compute_type=compute_type,
    asr_options={
        "beam_size": 5,           # 束搜索宽度
        "temperatures": [0.0],     # 确定性解码（无温度采样）
        "no_speech_threshold": 0.38,  # 静音段跳过阈值
        "compression_ratio_threshold": 2.4,  # 防止重复输出
        "condition_on_previous_text": False,  # 逐段独立解码
        "word_timestamps": False,  # 不使用 Whisper 内置词时间戳（改用 Wav2Vec2）
    },
    language="en",                # 固定英语
    vad_method="silero",          # 使用 Silero VAD 做语音活动检测
    vad_options={
        "chunk_size": 30,
        "vad_onset": 0.500,       # 语音起始敏感度
        "vad_offset": 0.363,      # 语音结束敏感度
    },
    download_root=str(BASE_DIR / "models"),
)
```

**设计思路：**

- **Silero VAD** 代替 Whisper 内置的 VAD。Silero 是一个轻量级（~35 MB）的 PyTorch 模型，在音频上滑动窗口检测语音活动。WhisperX 将 VAD 输出的有声片段切分成块，每个块独立送入 Whisper 解码，比直接处理整段音频更精确。
- **`word_timestamps=False`** Whisper 本身可以输出词级时间戳，但精度不足（通常偏差数百毫秒）。项目选择关掉内置时间戳，改用 Wav2Vec2 做专门的强制对齐以获得音素级精度。
- **`condition_on_previous_text=False`**：每个解码块独立处理，避免前面片段的错误向后传播。

### 2.3 Wav2Vec2 音素级强制对齐

[`transcribe_file()`](../scripts/transcribe.py) 中的对齐部分：

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

**原理：**

Wav2Vec2（base 960h，~360 MB）是一个在 LibriSpeech 960h 上预训练的自监督语音模型。WhisperX 的 `align()` 函数的工作方式：

1. 将每个 Whisper 输出的文本段分解为音素序列
2. 用 Wav2Vec2 在对应的音频片段上生成音素级的概率矩阵
3. 使用**维特比（Viterbi）解码**找到最可能的音素边界
4. 将音素边界映射为词边界（词由连续音素组成）

这样得到的词时间戳精确到音素级别（~10-30ms 精度），远超 Whisper 原生输出。

> **NLTK 的作用**：在对齐之前，WhisperX 使用 NLTK 的 `punkt` tokenizer 将每个 ASR 输出段拆分为单独的句子，然后对每个句子整体做音素级对齐。句子级对齐比逐词对齐提供更多音素上下文，从而提高边界精度。

### 2.4 时间戳后处理：[`fix_word_timestamps()`](../tools/extract.py)

Wav2Vec2 强制对齐有一个常见问题：如果一个词后面跟着较长的停顿或呼吸，对齐算法有时会把停顿也计入该词的持续时间，导致词时间戳膨胀。

**算法（两遍扫描）：**

```
第一遍 — 裁剪异常长词：
  对每个词 i：
    - 取前后各 10 个词作为局部窗口
    - 计算窗口内词持续时间的局部中位数（local_median）
    - 阈值 = max(local_median × 5, 1.0s)
    - 若当前词持续时间 > 阈值：
        将结束时间裁剪为 max(local_median × 3, 0.5s)
        且不超过下一个词的开始时间

第二遍 — 解决重叠：
  如果词 i 的结束时间 > 词 i+1 的开始时间：
    词 i 的结束时间 = 词 i+1 的开始时间
```

这样做的好处：
- 使用**局部中位数**而非全局固定阈值，适应语速变化（快语速时窗口窄，慢语速时窗口宽）
- 5 倍局部中位数的阈值保证了正常语速变化不会被误裁
- 第二遍保证时间轴单调递增，满足 SRT 格式要求

### 2.5 切分

时间戳后处理完成后，单词列表需要经过基于 LLM 的切分，将单词整理为字幕段。完整的切分算法描述、保护机制和参数详情请参见 [segmentation_pipeline_CN.md](segmentation_pipeline_CN.md)。

用法参数（`-seg_backend`、`-seg_model`、`-gpu-layers`）已在 README 中说明。

### 2.6 词级数据缓存

[`save_words_cache()` / `load_words_cache()`](../tools/cache.py)：

```python
def save_words_cache(all_words, cache_path):
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(all_words, f, ensure_ascii=False, indent=2)
```

ASR + 对齐对同一音频文件的输出是确定性的，因此通过缓存可以避免重复计算。`-no-cache` 参数控制是否跳过缓存：
- 首次运行：提取的词数据保存到 `cache/<stem>_words.json`
- 再次运行（缓存命中）：跳过 ASR 和对齐，直接加载缓存进入 LLM 切分

这样在调试切分参数时无需重复运行 ASR，每次可节省数分钟。

### 2.7 导出

[`export_srt()` / `export_txt()`](../tools/export.py) 和 [`format_srt_time()`](../tools/format.py)：

- **`export_srt()`**：遍历 segments 列表，用 `format_srt_time()` 将浮点秒数转换为 `HH:MM:SS,mmm` 格式
- **`export_txt()`**：`pure_text=False` 时输出完整 SRT 格式（含时间码），`pure_text=True` 时仅输出每段文本
- **`format_srt_time()`**：将秒数的小数部分转换为毫秒并用 `round()` 处理浮点误差

---

## 三、翻译流水线

### 3.1 翻译配置

[`translate_config.json`](../translate_config.json) 中控制翻译行为的字段：

| 字段 | 类型 | 默认值 | 说明 |
| ---- | ---- | ------ | ----------- |
| `source_lang` | 字符串 | `"English"` | 源语言名称（目前只支持英语；用于 LLM 提示词）。优先级比 CLI 参数 `-src_lang` 低。 |
| `target_lang` | 字符串 | `"Chinese"` | 目标语言名称（用于 LLM 提示词）。优先级比 CLI 参数 `-tgt_lang` 低。 |
| `target_lang_code` | 字符串 | `"CN"` | 目标语言代码（用于输出文件名后缀，如 `_CN.srt`）。优先级比 CLI 参数 `-tgt_lang_code` 低。 |
| `add_punctuation` | 布尔 | `false` | 是否在翻译结果中添加句末标点。`false` 时由后处理剥离标点。 |
| `allow_flexible_word_order` | 布尔 | `false` | 是否允许 LLM 跨行调整语序。**仅 `flexible` 模式下生效**，`accurate` 模式下不传递此参数（逐行翻译锁定）。强烈建议 **仅在使用在线 API 后端时启用**——本地模型使用可能导致行号偏移不稳定。 |
| `allow_simplify_wording` | 布尔 | `false` | 是否允许 LLM 精简口语化表达。**仅 `flexible` 模式下生效**，`accurate` 模式下不传递此参数。强烈建议 **仅在使用在线 API 后端时启用**——本地模型对该指令的遵循能力有限，效果不佳。 |
| `number_mode` | 字符串 | `"auto"` | 数字格式控制。可选值：`"auto"`（不做额外处理）、`"src_lang"`（强制保留原文数字格式）、`"digits"`（所有数字使用阿拉伯数字）、`"tgt_lang"`（数字转换为目标语言本地格式）。所有模式下均可用，本地模型也能生效，但受限于性能不一定会完全遵循。 |
| `space_between_cjk_and_latin` | 布尔 / 字符串 | `"auto"` | CJK-拉丁字符间距控制。`true`：插入空格；`false`：删除空格（覆盖 LLM 间距习惯）；`"auto"`：保留 LLM 原始输出。仅影响输出排版。 |
| `glossary` | 数组 | `["EXAMPLE1", ...]` | 术语表——这些词在翻译中**不得**被翻译，需保留原文。所有模式下均可用，本地模型也能生效，但受限于性能不一定会完全遵循。 |
| `custom_system_prompt` | 字符串或 null | `null` | 自定义系统提示词覆盖。非空时将完全替代自动构建的系统提示词。 |
| `cache_prompt` | 布尔 | `false` | 是否启用 LLM KV-cache 复用。在线 API 后端可通过此选项降低延迟；本地模型长任务中可能导致指令衰减，建议保持 `false`。不挑 mode。 |
| `drift_threshold` | 数字或布尔 | `false` | 内容漂移检测阈值。`false` 时完全跳过检测（零额外计算）；设为数字（如 `0.16`）时启用检测，当相邻两行翻译的字符集相似度与原文相似度之差超过该阈值时，判定为内容漂移并触发单行重译。数值越小越敏感。 |

API 凭证字段（`openai_api_key`、`anthropic_api_key`、`api_base_url`）见 [1.5 共用配置](#15-共用配置)。

### 3.2 输入解析

[`parse_srt()` / `rebuild_srt()`](../scripts/translate.py)

**解析** `parse_srt()`：逐行扫描，匹配序号 → 时间行（`HH:MM:SS,mmm --> HH:MM:SS,mmm`）→ 文本块，构建 `{start, end, text}` 字典列表。

### 3.3 滑动窗口翻译引擎

[`translate()`](../scripts/translate.py) → [`_translate_impl()`](../scripts/translate.py) 采用滑动窗口处理源文本。翻译区域使用 `tr_size` 行，前后的上下文行作为只读参考传入。

#### 3.3.1 翻译模式

[`translate()`](../scripts/translate.py) 支持两种滑动窗口模式，通过 `-mode` CLI 参数控制：

| 参数 | `accurate`（默认） | `flexible` |
| ------ | ------------------- | ---------- |
| `tr_size`（翻译窗口） | 2 行 | 4 行 |
| `ctx_size`（上下文窗口） | 4 行 | 8 行 |
| `first_step`（首次步进） | 1 行 | 2 行 |
| `later_step`（后续步进） | 2 行 | 4 行 |
| `show_timecodes`（显示时间码） | 否 | 是 |

- **`accurate`** — 较小的窗口实现严格的逐行翻译，时间轴错位风险最低。**推荐本地 Phi-4 模型使用**——本地模型推理速度较慢，小窗口可保持稳定的延迟和较低的资源占用。
- **`flexible`** — 较大的窗口配合时间码上下文，让 LLM 有更多空间产生自然措辞，包括跨行语序调整。**更适合在线 API 后端**——高速 API 可充分利用大窗口的上下文优势。

#### 3.3.2 系统提示词构建

根据配置动态组装系统提示词，由多个可选的指令块组成：

```
"Translate each line of {src_lang} text below to {tgt_lang}."
+ [add_punctuation? "添加适当标点"]
+ [flexible_word_order? "允许跨行调整语序" / "每行独立翻译"]
+ [simplify_wording? "精简口语化表达"]
+ [glossary? "不得翻译以下术语：term1, term2"]
+ [number_mode? "数字格式规则"]
+ 输出格式要求：输出行以 `[N]` 格式编号，如 `[1] Translation of source line 1.`
```

#### 3.3.3 窗口用户提示词构建

```python
def _build_window_user_prompt(ctx_lines, tr_start_idx, tr_end_idx, show_timecodes=False):
```

上下文行以 `(BEFORE)` / `(AFTER)` 标记为只读参考。翻译区域的行从 **1** 开始编号（`1`、`2` 等），提示 LLM 以 `[1]`、`[2]` 格式输出。

模型特定模板包裹（Phi-4 / Qwen / Ministral）由 LLM 调用层本身处理——见 [1.4 提示词模板格式化](#14-提示词模板格式化)。

#### 3.3.4 LLM 调用

每个滑动窗口通过 `llm_call.chat()` 发起一次 LLM 调用：

```python
response = llm_call.chat(system, user,
                          max_tokens=get_min_tokens(
                              getattr(llm_call, 'model', ''),
                              len(user)),
                          temperature=0.1, cache_prompt=CACHE_PROMPT)
if response:
    response = strip_reasoning(response)
```

**参数说明：**

- **`temperature=0.1`** — 低温度使翻译输出更稳定、更一致。切分引擎也使用类似温度，但翻译引擎刻意保持低值以防止自由发挥导致行号偏移。
- **`max_tokens=get_min_tokens(model, len(user))`** — 动态预测长度。`get_min_tokens()` 根据模型和输入长度计算合适的输出 token 上限，避免因 token 预算不足导致截断，同时防止过长输出浪费。
- **`cache_prompt=CACHE_PROMPT`** — 是否启用 LLM KV-cache 复用（由 `translate_config.json` 的 `cache_prompt` 字段控制）。在线后端降低延迟；本地模型长任务中可能导致指令衰减，建议保持 `false`。
- **`strip_reasoning(response)`** — 剥离思考链前言。处理三种 XML 思考标签（`<think>`、`<reason>`、`<reasoning>`）以及首个 `[N]` 结构行之前的散文体推理前言。注意 DeepSeek 的默认 thinking 模式是通过 API 参数（`body["thinking"] = {"type": "disabled"}`）在请求层禁用的，不经过此函数。目前 `_REASONING_MODELS` 中包含 `"qwen3.5-9b"`，其思考输出才由 `strip_reasoning()` 处理。

**空响应重试：** 若 LLM 返回 `None`（网络超时/服务端错误）或空字符串，以指数退避策略重试（5s → 15s → 30s），每个滑动窗口最多重试 3 次。三次均失败后保留原文。

#### 3.3.5 响应解析

[`translate()`](../scripts/translate.py) → `_translate_impl()` 中，LLM 返回的文本按行解析，预期格式为 `[N] text` 或 `[N-M] merged text`：

```python
m = re.match(
    r"\[(\d+)(?:\s*-\s*(\d+))?\]\s*"
    r"(?:\[([\d:,]+\s*-->\s*[\d:,]+)\]\s*)?"
    r"(.+)",
    resp_line,
)
if m:
    n = int(m.group(1)) - 1       # 0-based 索引
    m_n = int(m.group(2)) - 1 if m.group(2) else n  # 合并终点
    llm_tc = m.group(3)           # 时间码（可选）
    text = m.group(4).strip()
```

当 LLM 合并相邻行时（如 `[2-3]`），SRT 重建过程中会自动合并时间码。

#### 3.3.6 CJK 窗口级自动重试

**CJK 自动重试（第一层——窗口级）：** 部分模型偶尔会回显原文而非翻译。每个窗口的 LLM 响应解析后，引擎检测是否有**任一**输出行包含 CJK 字符。如果全部为英文，则以强化版 system prompt、temperature=0、cache_prompt=False 进行重试（每个窗口最多 2 次）。这可以捕获整窗翻译失败的情况。

### 3.4 后处理流水线

[`post_process()`](../scripts/translate.py) 及其依赖的各函数

每条翻译结果经过以下后处理步骤：

```
原始 LLM 输出
  │
  ├─ 1. [_strip_template_tokens()](../scripts/translate.py) — 移除 <|end|> 等模板标记
  ├─ 2. [_strip_end_punct()](../scripts/translate.py)       — 如 add_punct=False，移除句末标点
  ├─ 3. [_normalize_commas()](../scripts/translate.py)      — CJK 语言将 , 替换为 ，
  ├─ 4. [_normalize_quotes()](../scripts/translate.py)      — CJK 内容中的引号替换为""
  └─ 5. [_add_cjk_latin_spacing()](../scripts/translate.py) — 插入/删除/保留 CJK 与拉丁字符间的空格
```

**标点剥离** [`_strip_end_punct()`](../scripts/translate.py)：当 `add_punct=False` 时，用 `rstrip("。？！.!?")` 移除句末标点。这样做的好处是不将标点约束强加给 LLM，而是通过后处理实现——LLM 翻译质量不受影响，用户得到无标点输出。

**引号智能替换** [`_normalize_quotes()`](../scripts/translate.py)：根据引号内内容是否包含 CJK 字符决定使用全角还是半角引号——同一段文本内英文术语保持半角引号，中文部分使用全角。

### 3.5 溢出合并保护

当在线模型（如 DeepSeek）在 `accurate` 模式下翻译时，LLM 可能会将超出翻译窗口的内容合并进来。例如 `tr_size=2` 时，LLM 可能输出 `[2-3]`，表示它把第 2 行和第 3 行的内容一起翻译了（而第 3 行属于下一批）。

通过两种互补机制来处理：

**1. Clamp 修复（所有后端）：**
不去丢弃 `m_n >= actual_tr` 的输出，而是将合并范围限制在 `min(m_n, actual_tr - 1)`，确保当前窗口的翻译被存储。合并的内容被保留，溢出的内容会在其自己的窗口到来时被重新翻译。

**2. 溢出重拆分（所有后端）：**
当 `m_n >= actual_tr` 时，记录 `(merged_idx, overflowed_idx)`。所有批次完成后，`_fix_overflow()` 将两个源行及其当前译文重新发送给 LLM 进行干净的拆分。LLM 移除重复/重叠内容后返回两条独立的翻译。确保内容不重复也不丢失。

```
[Post-process] Fixing 14 overflow merge(s)...
  [Fix] Lines 8 & 9: re-split OK
  [Fix] Lines 10 & 11: re-split OK
  ...
```

---

### 3.6 内容漂移检测

当 LLM 在 `accurate` 模式下将相邻行的内容混入同一条输出时，产生"内容漂移"——翻译结果的行号正确，但内容被复制或混杂了相邻行的信息。该机制通过 Jaccard 字符集相似度检测并修复此类问题。

#### 相似度计算

两条字符串之间的相似度采用 **Jaccard 字符集相似度**（[`_char_sim()`](../scripts/translate.py)）：

```
_char_sim(a, b) = len(set(a) ∩ set(b)) / len(set(a) ∪ set(b))
```

即**交并比**：两字符串各自去重后取字符集合，交集大小除以并集大小。只关心字符的"有哪些"，不关心顺序。

例如：

- `"你好世界"`（字符集 `{你, 好, 世, 界}`）与 `"你好"`（`{你, 好}`）的相似度 = 2/4 = 0.5
- `"filter"`（`{f, i, l, t, e, r}`）与 `"滤波器"`（`{滤, 波, 器}`）的相似度 = 0/9 ≈ 0.0

#### 漂移检测逻辑

[`_detect_drift_pairs()`](../scripts/translate.py) 逐对检查相邻行：

1. 对第 `i` 行和第 `i+1` 行，计算翻译结果的相似度 `trans_sim = _char_sim(translation[i], translation[i+1])`
2. 计算原文的相似度 `src_sim = _char_sim(source[i], source[i+1])`
3. 计算差值 `gap = trans_sim - src_sim`
4. 若 `gap > threshold`，判定为漂移，两行均标记

**原理：** 如果原文两行说的不是同一件事（`src_sim` 低），但翻译结果却非常相似（`trans_sim` 高），说明 LLM 可能将某行的内容复制到了另一行。

#### 修复流程

被标记的行会逐条独立发送给 LLM，使用严格的一次一行提示词：

```
CRITICAL: Translate ONLY the single line provided below.
Do NOT combine it with any other text.
Output format: [1] <translation>
```

若脱离上下文后翻译结果与原来不同，则替换；否则保留原值。

#### 配置

由 `translate_config.json` 中的 `drift_threshold` 字段控制：


| 值 | 行为 |
|:---|:-----|
| `false` | 完全跳过检测（默认），不计算任何相似度 |
| `0.16` | 启用检测，`gap > 0.16` 时判定为漂移 |

### 3.7 CJK 逐行后处理

[`_translate_impl()`](../scripts/translate.py) 在完成溢出合并修复和内容漂移检测之后，执行第三道后处理——CJK 逐行检测与修复。该步骤捕获那些通过了窗口级 CJK 检测（[3.3.6 CJK 窗口级自动重试](#336-cjk-窗口级自动重试)）但本身仍是英文的单行。

**触发条件：** 窗口级 CJK 检测（[3.3.6 CJK 窗口级自动重试](#336-cjk-窗口级自动重试)）检查的是整个窗口是否有**任一**行包含 CJK 字符。当窗口内有混入 CJK 的行时，整窗通过检测，但那些原文被保留或 LLM 回显了英文的行不会在窗口级被发现。

**检测流程：** 扫描所有结果行，同时满足以下条件时标记：
1. 翻译结果不为 `None`
2. 结果中不包含 CJK 字符（正则 `[一-鿿㐀-䶿　-〿＀-￯]`）
3. 是原文保留行（`results[i] == text_lines[i]`），或长度 ≥ 3 个词（排除短噪声行）

**修复流程：** 每条被标记的行独立发送给 LLM，使用无上下文的严格单行提示词：

```
CRITICAL: Translate ONLY the single line below.
Output format: [1] <translation>.
The translation MUST contain Chinese characters.
```

重试参数为 `temperature=0, cache_prompt=False`，与原行上下文无关。若新译文包含 CJK 字符则替换原值；否则保留。

```
  [CJK Fix] 3 line(s) lack CJK, re-translating individually...
    [CJK Fix] Line 5: fixed
      Old: This is the original English subtitle text.
      New: 这是原始英文字幕文本。
    [CJK Fix] Line 12: new output also no CJK, keeping original
  [CJK Fix] Done — 3 line(s) processed
```


### 3.8 SRT 重建与文件输出

```python
def rebuild_srt(blocks, translated_texts):
    out = []
    i = 0
    seq = 1
    while i < len(blocks):
        # 跳过被合并行吸收的块（None 翻译）
        if i < len(translated_texts) and translated_texts[i] is None:
            i += 1
            continue
        text = translated_texts[i] if i < len(translated_texts) else blocks[i]["text"]
        start = blocks[i]["start"]
        # 向前扫描：将后续 None 块的时间码合并到当前块
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

当 LLM 合并相邻行时（输出 `[2-3]` 表示合并了第 2、3 行），`translated_texts[2]`（0-based 索引 1）被设为 `None`，`rebuild_srt()` 自动将该行的时间码吸收到前一个有效块中，保证时间轴不丢失。


### 3.9 集成到 main.py

[`main.py`](../main.py) 的 CLI `__main__` 块中翻译操作通过 [`translate_file()`](../scripts/translate.py) 一行完成：

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

[`translate_file()`](../scripts/translate.py) 将 `read_input()` → `translate()` → `write_output()` 封装为一次调用，`main.py` 无需重复文件 I/O 逻辑。翻译作为进程内函数调用（非子进程），与切分流水线共享同一 `LLMCall` 层。
