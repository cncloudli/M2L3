# 转录与翻译技术实现

本文档介绍项目中语音转录（ASR → 对齐 → 后处理）和翻译（本地/在线 LLM）的技术实现细节。README 中已有的用法说明不再重复。

---

## 一、转录流水线

### 1.1 整体流程

```
输入音频/视频
  │
  ├─ FFmpeg 音频解码（[check_ffmpeg()](../tools/env_check.py)）
  ├─ WhisperX 批量转录（faster-whisper-large-v3 + Silero VAD）
  ├─ Wav2Vec2 音素级强制对齐
  ├─ 时间戳后处理（[fix_word_timestamps](../tools/extract.py)）
  ├─ LLM 分割（10 阶段，参见 [segmentation_pipeline.md](segmentation_pipeline.md)）
  └─ 导出 SRT + TXT
```

### 1.2 WhisperX 转录

[`video_to_srt()`](../main.py) 中的模型加载部分 — 调用 `whisperx.load_model()` 加载 faster-whisper-large-v3：

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

- **Silero VAD** 代替 Whisper 内置的 VAD。Silero 是一个轻量级（~5 MB）的 PyTorch 模型，在音频上滑动窗口检测语音活动。WhisperX 将 VAD 输出的有声片段切分成块，每个块独立送入 Whisper 解码，比直接处理整段音频更精确。
- **`word_timestamps=False`** Whisper 本身可以输出词级时间戳，但精度不足（通常偏差数百毫秒）。项目选择关掉内置时间戳，改用 Wav2Vec2 做专门的强制对齐以获得音素级精度。
- **`condition_on_previous_text=False`**：每个解码块独立处理，避免前面片段的错误向后传播。

### 1.3 Wav2Vec2 音素级强制对齐

[`video_to_srt()`](../main.py) 中的对齐部分：

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

### 1.4 时间戳后处理：[`fix_word_timestamps()`](../tools/extract.py)

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

### 1.5 词级数据缓存

[`save_words_cache()` / `load_words_cache()`](../tools/cache.py)：

```python
def save_words_cache(all_words, cache_path):
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(all_words, f, ensure_ascii=False, indent=2)
```

ASR + 对齐是整个流程中最耗时的部分（尤其是 Wav2Vec2 对齐）。`-no-cache` 参数控制是否跳过缓存：
- 首次运行：提取的词数据保存到 `cache/<stem>_words.json`
- 再次运行（缓存命中）：跳过 ASR 和对齐，直接加载缓存进入 LLM 分割

这样在调试分割参数时无需重复运行 ASR，每次可节省数分钟。

### 1.6 导出

[`export_srt()` / `export_txt()`](../tools/export.py) 和 [`format_srt_time()`](../tools/format.py)：

- **`export_srt()`**：遍历 segments 列表，用 `format_srt_time()` 将浮点秒数转换为 `HH:MM:SS,mmm` 格式
- **`export_txt()`**：`pure_text=False` 时输出完整 SRT 格式（含时间码），`pure_text=True` 时仅输出每段文本
- **`format_srt_time()`**：将秒数的小数部分转换为毫秒并用 `round()` 处理浮点误差

---

## 二、翻译流水线

### 2.1 整体架构

[translate.py](../translate.py) 采用**后端抽象+工厂模式**，支持 7 种翻译后端：

```
translate.py
  ├─ [TranslatorLocal](../translate.py)      # 本地 llama-server 子进程
  ├─ [TranslatorOpenAI](../translate.py)     # OpenAI 兼容 API（DeepSeek, OpenAI, Qwen, Gemini, Ollama）
  └─ [TranslatorAnthropic](../translate.py)  # Anthropic Messages API
```

[`create_translator()`](../translate.py) 工厂函数根据 `-backend` 参数选择后端：

```python
def create_translator(backend, model=None, gpu_layers=None):
    if backend == "local":
        return TranslatorLocal(model=m, gpu_layers=gpu_layers)
    if backend == "anthropic":
        return TranslatorAnthropic(model=m, api_key=api_key, ...)
    # 其他后端 → TranslatorOpenAI
    return TranslatorOpenAI(backend=backend, model=m, ...)
```

### 2.2 本地翻译：[`TranslatorLocal`](../translate.py)

#### 2.2.1 llama-server 子进程管理

`TranslatorLocal` 管理一个 `llama-server.exe` 子进程的生命周期：

**启动** — [`TranslatorLocal.start()`](../translate.py)：
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

**推理** — [`TranslatorLocal.translate_batch()`](../translate.py)：
```python
r = requests.post(
    f"{self.server_url}/completion",
    json={
        "prompt": prompt,           # 已格式化的完整提示词
        "n_predict": max(200, int(len(user) * 1.5)),  # 动态预测长度
        "temperature": 0.1,         # 低温度 → 确定性的翻译输出
        "cache_prompt": CACHE_PROMPT,  # KV-cache 复用（可选）
    },
    timeout=120,                    # 单次推理超时
)
```

**关闭** — [`TranslatorLocal.stop()`](../translate.py)：三级关闭策略：
1. `proc.terminate()` + 5s 等待
2. 如失败 → `proc.kill()` + 3s 等待
3. 如仍存活 → `taskkill /F /PID` 强制终止

#### 2.2.2 GPU 层数自动检测：[`auto_gpu_layers()`](../translate.py)

```python
def auto_gpu_layers(model="phi4"):
    available = total_vram - 1.5             # 预留 1.5 GB 给系统/ASR 模型
    per_layer = model_size / n_layers * 1.1  # 每层大小 × 1.1 安全系数
    return max(0, min(n_layers, int(available / per_layer)))
```

根据 GPU 显存总量、模型文件大小、层数估算可容纳的层数，预留 1.5 GB 给其他进程。

### 2.3 在线翻译后端

#### 2.3.1 OpenAI 兼容后端：[`TranslatorOpenAI`](../translate.py)

统一使用 `/chat/completions` 端点，构造标准的 `system` + `user` 消息格式：

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

**兼容性处理：** DeepSeek V4 默认启用思考（thinking）模式，会消耗 token 预算导致 `content` 为空。因此自动添加 `body["thinking"] = {"type": "disabled"}` 来禁用。

支持的后端共用一个客户端实现，仅在 `base_url` 和默认模型上不同：

| 后端 | base_url | 默认模型 |
|---------|-----------|---------------|
| deepseek | `api.deepseek.com` | deepseek-v4-flash |
| openai | `api.openai.com` | gpt-5.6-terra |
| qwen | `dashscope-intl.aliyuncs.com/.../v1` | qwen3.5-plus |
| gemini | `generativelanguage.googleapis.com/.../openai` | gemini-3.5-flash |
| ollama | `localhost:11434/v1` | llama4 |

#### 2.3.2 Anthropic 后端：[`TranslatorAnthropic`](../translate.py)

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
    headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
)
```

**特殊处理：** DeepSeek 支持两种 API 格式，通过 `api_base_url` 是否包含 `"anthropic"` 自动判断使用 OpenAI 格式还是 Anthropic Messages 格式（见 [`create_translator()`](../translate.py) 中的 DeepSeek 分支）。

### 2.4 批处理与提示词设计

#### 2.4.1 分批翻译

[`translate_texts()`](../translate.py) 中，为避免上下文过长，源文本按 `BATCH_SIZE`（默认 8 行）分批发送。每批独立构造提示词。

#### 2.4.2 系统提示词构建：[`_build_system_prompt()`](../translate.py)

根据配置动态组装系统提示词，由多个可选的指令块组成：

```
"Translate each line of {src_lang} text below to {tgt_lang}."
+ [add_punctuation? "添加适当标点"]
+ [flexible_word_order? "允许跨行调整语序" / "每行独立翻译"]
+ [simplify_wording? "精简口语化表达"]
+ [glossary? "不得翻译以下术语：term1, term2"]
+ [number_mode? "数字格式规则"]
+ 输出格式要求："每行以 'N: 翻译结果' 格式输出"
```

**用户提示词** — [`_build_user_prompt()`](../translate.py)：
```python
def _build_user_prompt(lines):
    return "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))
```

**模板格式化** — [`_fmt_prompt()`](../translate.py)：
```python
def _fmt_prompt(model, system, user):
    return (
        f"<|system|>\n{system}\n<|end|>\n"
        f"<|user|>\n{user}\n<|end|>\n"
        f"<|assistant|>\n"
    )
```
这是 Phi-4 的聊天模板格式。对于在线 API 后端，system/user 通过 API 原生字段传递，不经过此函数。

#### 2.4.3 响应解析

[`translate_texts()`](../translate.py) 中，LLM 返回的文本按行解析，预期格式为 `N: translated text`：

```python
m = re.match(r"(?:Line\s*)?(\d+)\s*[:：]\s*(.+)", resp_line, re.IGNORECASE)
if m:
    num = int(m.group(1)) - 1  # → 0-based index
    if 0 <= num < len(batch):
        translated[num] = m.group(2).strip()
```

**兜底策略：** 如果 LLM 忽略编号格式（例如某些模型不遵守指令），则按文档顺序逐行对应。

**空响应重试：** 使用指数退避（5s → 15s）重试空响应，应对临时性 API 限流。

### 2.5 后处理流水线

[`post_process()`](../translate.py) 及其依赖的各函数

每条翻译结果经过以下后处理步骤：

```
原始 LLM 输出
  │
  ├─ 1. [_strip_template_tokens()](../translate.py) — 移除 <|end|> 等模板标记
  ├─ 2. [_strip_end_punct()](../translate.py)       — 如 add_punct=False，移除句末标点
  ├─ 3. [_normalize_commas()](../translate.py)      — CJK 语言将 , 替换为 ，
  ├─ 4. [_normalize_quotes()](../translate.py)      — CJK 内容中的引号替换为“”
  └─ 5. [_add_cjk_latin_spacing()](../translate.py) — CJK 与拉丁字符间插入空格
```

**标点剥离** [`_strip_end_punct()`](../translate.py)：当 `add_punct=False` 时，用 `rstrip("。？！.!?")` 移除句末标点。这样做的好处是不将标点约束强加给 LLM，而是通过后处理实现——LLM 翻译质量不受影响，用户得到无标点输出。

**引号智能替换** [`_normalize_quotes()`](../translate.py)：根据引号内内容是否包含 CJK 字符决定使用全角还是半角引号——同一段文本中英文术语保持半角引号，中文部分使用全角。

### 2.6 配置系统

[`_load_env()` / `_load_config()`](../translate.py)

采用三层配置优先级：

1. **代码内默认值**（最低优先级）→ `_load_config()` 中的 `defaults` 字典
2. **`translate_config.json`**（覆盖默认值）→ 用户编辑的 JSON 文件
3. **`.env` 文件 + 环境变量**（仅 API 密钥）→ 不在 JSON 中设置时回退到 `.env`

[`_load_env()`](../translate.py) 使用 `os.environ.setdefault()`，因此 JSON 中设置的密钥优先级更高，但留空的情况下 `.env` 自动生效。

### 2.7 SRT 解析与重建

[`parse_srt()` / `rebuild_srt()`](../translate.py)

**解析** `parse_srt()`：逐行扫描，匹配序号 → 时间行（`HH:MM:SS,mmm --> HH:MM:SS,mmm`）→ 文本块，构建 `{start, end, text}` 字典列表。

**重建** `rebuild_srt()`：将翻译后的文本列表按索引填入原始 SRT 块的时间码结构：

```python
for i, blk in enumerate(blocks):
    text = translated_texts[i] if i < len(translated_texts) else blk["text"]
    out.append(str(i + 1))
    out.append(f"{blk['start']} --> {blk['end']}")
    out.append(text)
    out.append("")
```

这样翻译输出的 SRT 完整保留了原始时间码、序号和空行分隔。

### 2.8 集成：转写 → 翻译

[`main.py`](../main.py) 的 CLI `__main__` 块中 `-translate` 的实现：

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

翻译作为子进程运行，与主转录进程隔离。这样即使翻译过程中出错（如 API 超时），转录生成的 SRT 文件仍然完整可用。

---
