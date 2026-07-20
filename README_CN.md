# 基于多层LLM的字幕切分工具
**(Multi-Layer LLM-Based Subtitle Segmentation / M2L3)**

集成英语转录 + 字幕切分 + 翻译的一站式字幕生成工具，所有处理流程支持**完全离线运行**

| 功能 | 方案 | 使用场景 |
|------|------|----------|
| **语音识别 (ASR)** | WhisperX (faster-whisper-large-v3) + Silero VAD | 转录 (scripts/transcribe.py) |
| **对齐 (Alignment)** | Wav2Vec2 音素级强制对齐 | 转录 (scripts/transcribe.py) |
| **切分 (Segment)** | 通过 llama-server 调用 本地 LLM 或使用在线 API | 转录 (scripts/transcribe.py) |
| **翻译 (Translate)** | 通过 llama-server 调用 本地 LLM 或使用在线 API | 翻译 (scripts/translate.py) |


---

## 文件参考

### 根目录

| 文件 | 用途                                                                                  |
|------|-------------------------------------------------------------------------------------|
| [main.py](main.py) | **编排入口。** 在一个命令中调用 `scripts/transcribe.py` 转录和/或 `scripts/translate.py` 翻译。         |
| [batch.py](batch.py) | **批处理。** 遍历输入文件夹中的所有文件，为每个文件启动一个独立的 `main.py` 子进程（每个进程拥有独立的 CUDA 上下文）。              |
| [translate_config.json](translate_config.json) | **翻译配置。** 语言、标点、术语表、API 密钥——由用户编辑，`scripts/translate.py` 启动时读取。**注：API 密钥也适用于字幕切分** |
| [README_CN.md](README_CN.md) | 本文件——中文版使用指南、安装说明及文件参考。                                                             |
| [README.md](README.md) | 英文版使用指南及文档。                                                                         |

### `scripts/` 目录

| 脚本 | 用途 |
|------|------|
| [scripts/transcribe.py](scripts/transcribe.py) | **转录流水线。** WhisperX ASR → Wav2Vec2 对齐 → LLM 切分 → SRT + TXT 导出。可独立运行。 |
| [scripts/translate.py](scripts/translate.py) | **翻译。** 使用本地或基于 API 的 LLM 翻译现有 SRT（保留时间码）或 TXT 文件。可独立运行。 |

### `tools/` 目录

| 模块 | 用途                                                                                                                                      |
|------|-----------------------------------------------------------------------------------------------------------------------------------------|
| [tools/\_\_init\_\_.py](tools/__init__.py) | 包标记——仅含模块文档字符串。                                                                                                                         |
| [tools/config.py](tools/config.py) | **系统配置。** 设置 `HTTP_PROXY` / `NO_PROXY`（绕过 PyTorch/HuggingFace CDN 的代理）和 `CUDA_VISIBLE_DEVICES`。在导入时执行，早于任何模型下载。                         |
| [tools/env_check.py](tools/env_check.py) | **环境验证。** `check_ffmpeg()` — 验证 FFmpeg 是否在 PATH 中。`check_cuda()` — GPU 检测 + 实际内核启动测试，返回 `True`/`False`。                                 |
| [tools/extract.py](tools/extract.py) | **从 ASR 输出中提取词级数据。** `extract_words_from_result()` — 从 WhisperX 结果字典中提取 `(start, end, text)` 词元组。`fix_word_timestamps()` — 修正膨胀时间戳的后处理。 |
| [tools/cache.py](tools/cache.py) | **词级缓存。** `save_words_cache()` / `load_words_cache()` — 将提取的词数据持久化/恢复为 JSON，用于重新切分调试。                                                   |
| [tools/format.py](tools/format.py) | **SRT 时间格式化。** `format_srt_time()` — 将浮点秒数转换为 `HH:MM:SS,mmm` 的 SRT 时间格式。                                                                |
| [tools/export.py](tools/export.py) | **字幕导出。** `export_srt()` — 写入带序号和时间戳的 SRT。`export_txt()` — 写入纯文本或 SRT 格式文本。`export_word_level()` — 调试用的词级 SRT/TXT（_当前已注释掉_）。            |
| [tools/download_models.py](tools/download_models.py) | **模型预下载器。** 一次性脚本，将 Silero VAD、Wav2Vec2 和 NLTK 数据下载到本地缓存，使流水线完全离线运行。                                                                    |
| [tools/segment.py](tools/segment.py) | **LLM 切分引擎。** `segment_words()` 函数 — 运行完整 10 阶段切分算法（参见 [docs/segmentation_pipeline_CN.md](docs/segmentation_pipeline_CN.md) 获取完整算法流程图）。 |
| [tools/non_split_bigrams.py](tools/non_split_bigrams.py) | **短语保护。** `NON_SPLIT_BIGRAMS` 集合（约 5700 条）包含固定表达、短语动词和搭配，这些内容不得在字幕段间切分。`would_break_phrase()` 查找函数被所有切分阶段使用。                            |
| [tools/llama/](tools/llama/) | **llama.cpp 二进制文件。** 预编译的 `llama-server.exe` + CUDA 12 DLL，用于本地 LLM 推理。已从 git 中排除——参见 [安装 → llama.cpp 二进制文件](#llamacpp-二进制文件)。          |

### `docs/` 目录

| 文件 | 用途                                                          |
|------|-------------------------------------------------------------|
| [docs/segmentation_pipeline.md](docs/segmentation_pipeline.md) | **10 阶段 LLM 切分算法文档（英文）。** 完整的算法流程图及各阶段的详细说明、保护机制和 LLM 调用总结。 |
| [docs/segmentation_pipeline_CN.md](docs/segmentation_pipeline_CN.md) | **10 阶段 LLM 切分算法文档（中文）。** 同上，中文版。                           |
| [docs/transcription_n_translation.md](docs/transcription_n_translation.md) | **转录与翻译技术实现（英文）。** ASR→对齐→切分流水线及翻译工厂后端的设计思路、参数选择和代码介绍。      |
| [docs/transcription_n_translation_CN.md](docs/transcription_n_translation_CN.md) | **转录与翻译技术实现（中文）。** 同上，中文版。                                  |
| [docs/models_n_performance.md](docs/models_n_performance.md) | **模型与性能（英文）。** 硬件需求，LLM 选型和推理基准测试。            |
| [docs/models_n_performance_CN.md](docs/models_n_performance_CN.md) | **模型与性能（中文）。** 同上，中文版。                                      |

---

## 安装

### 系统要求

| 类别 | 示例 | 安装方式 |
|----------|----------|---------------|
| **系统工具** | NVIDIA 驱动（≥ CUDA 12.8）、FFmpeg、Python venv | 手动（一次性） |
| **llama.cpp 二进制** | `llama-server.exe` | 下载 zip → 解压到 `tools/llama/`（参见下方变体说明） |
| **预下载模型** | faster-whisper-large-v3、Phi-4 GGUF | 手动下载到 `models/` |
| **自动下载模型** | Silero VAD、Wav2Vec2、NLTK | `python tools/download_models.py` |

> **GPU vs CPU**：整个流水线（ASR + 对齐 + LLM）**自动检测 CUDA**，若无 GPU 则**回退至 CPU**。无需特殊标志——只需选择下方对应的 llama.cpp 变体。

### NVIDIA 驱动与 CUDA

项目依赖 PyTorch CUDA 12.8（参见 `requirements.txt` 中的 `torch==2.8.0+cu128`），**必须配合兼容的 NVIDIA 驱动**才能启用 GPU 加速。

| 需求 | 说明 |
|------|------|
| **NVIDIA GPU** | 架构 ≥ Maxwell（GTX 900 系列或更新），最低 4 GB 显存（推荐 8 GB+） |
| **NVIDIA 驱动** | 版本 ≥ **R570**（CUDA 12.8 的最终驱动支持发布于 2026 年 7 月；请更新至最新 Game Ready / Studio Driver） |
| **CUDA Toolkit** | **不需要安装。** PyTorch 自带 CUDA 运行时，`requirements.txt` 中的 `torch==2.8.0+cu128` 版本已经包含所需的 `cudart` 和 `cublas`。只需确保驱动足够新即可。 |
| **无 GPU / CPU 模式** | 程序自动检测 CUDA，不可用时自动回退到 CPU（速度较慢）。无需任何额外配置。 |

验证方式：运行 `nvidia-smi`，查看顶部显示的 **CUDA Version** 是否为 12.8 或更高。如果版本过低或没有输出，请从 [NVIDIA 驱动下载](https://www.nvidia.com/Download/index.aspx) 更新驱动。

### FFmpeg

Whisper 提取音频所必需。

```powershell
winget install FFmpeg
# 或: choco install ffmpeg
# 或从 https://ffmpeg.org/download.html 下载并添加至 PATH
# 国内镜像: https://mirrors.tuna.tsinghua.edu.cn/ffmpeg/
```

验证：`ffmpeg -version`

### Python 虚拟环境

```powershell
# 创建 .venv（Python 3.10）
python -m venv .venv
# 激活
.venv\Scripts\activate
# 安装依赖
pip install -r requirements.txt
```

### llama.cpp 二进制文件

[llama.cpp](https://github.com/ggerganov/llama.cpp) 提供本地 LLM 推理引擎。根据你的硬件选择对应变体：

| 你的环境 | 下载 zip |
|-----------|--------------|
| **NVIDIA GPU（CUDA 12.x）** | `llama-bNNNN-bin-win-cuda12.4-x64.zip`（例如 [b9888](https://github.com/ggml-org/llama.cpp/releases/tag/b9888)） |
| **仅 CPU / 无 GPU** | `llama-bNNNN-bin-win-cpu-x64.zip`（同一发布页面） |

1. 访问 [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases) 页面
   - 国内加速代理：`https://ghproxy.com/https://github.com/ggml-org/llama.cpp/releases`
2. 下载对应环境的 zip 文件
3. 将**所有文件**解压至 `tools/llama/`


> **说明**：发布页上还有一个 **"CUDA 12.4 DLLs"** 独立包（`cudart-llama-bin-win-cuda-12.4-x64.zip`，约 750 MB）。这是可选的补充包——它提供 NVIDIA 官方的 cuBLAS 库，在部分 GPU 上比 `ggml-cuda.dll` 内置实现有轻微性能提升。标准 CUDA 版不依赖它也能正常工作。如需使用，将其中 3 个 DLL 解压到同一 `tools/llama/` 目录即可。


### 模型

将以下文件放入 `models/` 目录：

```
models/
├── faster-whisper-large-v3/     # ~3 GB，来自 HuggingFace
└── *.gguf                       # LLM 模型（Phi-4 等）——下载方式见 [docs/models_n_performance_CN.md](docs/models_n_performance_CN.md)
```

**faster-whisper-large-v3：**

```powershell
pip install huggingface-hub

# 方式 A — 默认下载
hf download guillaumek64/faster-whisper-large-v3 --local-dir models/faster-whisper-large-v3

# 方式 B — 国内镜像 (hf-mirror.com)，速度更快
# Windows PowerShell:
$env:HF_ENDPOINT = "https://hf-mirror.com"
hf download guillaumek64/faster-whisper-large-v3 --local-dir models/faster-whisper-large-v3

# 或使用镜像直链手动下载：
# https://hf-mirror.com/guillaumek64/faster-whisper-large-v3/tree/main
```

### 自动下载（其他小型模型）

```powershell
python tools/download_models.py
```
> 若下载缓慢，可在运行前设置镜像：PowerShell 中执行 `$env:HF_ENDPOINT = "https://hf-mirror.com"`

下载到 `models/hub/` ：

| 模型 | 大小 | 使用者 | 描述 |
|-------|------|---------|-------------|
| **Silero VAD** | ~35 MB | whisperx（VAD） | 语音活动检测——在音频中定位语音片段 |
| **Wav2Vec2（base 960h）** | ~360 MB | WhisperX 对齐 | 音素级词时间戳对齐 |
| **NLTK punkt_tab** | ~64 MB | WhisperX 对齐（内部） | 句子边界检测，用于音素级强制对齐 |

---

## 快速上手

```powershell
# 1. 转写单个视频/音频（可接受 mp4/mkv/avi/mov/wav/mp3/m4a/flac）
python scripts/transcribe.py -i input/input.mp4

# 2. 翻译现有字幕（自动识别 SRT 或 TXT）
python scripts/translate.py -i output/input.srt

# 3. 转写 + 翻译一条命令
python main.py -i input/input.mp4 -translate true

# 4. 转写 + 翻译使用不同后端
python main.py -i input/input.mp4 -translate true -seg_backend local -transl_backend deepseek

# 5. 批处理文件夹中所有视频
python batch.py

# 6. 批处理 + 翻译
python batch.py -translate true
```

---

## 使用说明

### 1. LLM 后端与共享配置

**转录切分**（`scripts/transcribe.py`）和**翻译引擎**（`scripts/translate.py`）共用同一套 LLM 调用层。支持的后端：

| 后端 | API 格式 | 默认模型 | 认证方式 |
|------|----------|---------|---------|
| `local` | llama-server 子进程 | `phi4` | — |
| `deepseek` | 自动检测：OpenAI `/chat/completions` **或** Anthropic Messages | `deepseek-v4-flash` | `openai_api_key` |
| `openai` | OpenAI 兼容 | `gpt-5.6-terra` | `openai_api_key` |
| `qwen` | OpenAI 兼容 | `qwen3.7-max` | `openai_api_key` |
| `gemini` | OpenAI 兼容 | `gemini-3.5-flash` | `openai_api_key` |
| `anthropic` | Anthropic Messages | `claude-opus-4-8` | `anthropic_api_key` |

**API 密钥和自定义 base URL** 在 [translate_config.json](translate_config.json) 中配置：

```json
{
    "source_lang": "English",
    "target_lang": "Chinese",
    "target_lang_code": "CN",
    "add_punctuation": false,
    "allow_flexible_word_order": false,
    "allow_simplify_wording": false,
    "number_mode": "auto",
    "space_between_cjk_and_latin": "auto",
    "glossary": ["EXAMPLE1", "EXAMPLE2", "VRAM", "CUDA"],
    "custom_system_prompt": null,
    "cache_prompt": false,
    "drift_threshold": false,
    "openai_api_key": "",
    "anthropic_api_key": "",
    "api_base_url": ""
}
```

API 密钥字段（`openai_api_key`、`anthropic_api_key`、`api_base_url`）是**共享的**——同时作用于切分后端和翻译后端。翻译专用字段（`target_lang`、`target_lang_code` 等）详见 [transcription_n_translation_CN.md](docs/transcription_n_translation_CN.md) - "3.1 翻译配置"。

或使用 `.env` 文件设置 API 密钥（路径：项目根目录 `.env`）：**仅当 `translate_config.json` 中的密钥字段为空时生效**

```ini
# ── LLM 后端 API 密钥（切分与翻译共用） ──

OPENAI_API_KEY=sk-your-key-here

# Anthropic 专用（仅后端为 anthropic 时需要）
ANTHROPIC_API_KEY=sk-ant-your-key-here

```

### 2. `scripts/transcribe.py` — 转录流水线

```powershell
python scripts/transcribe.py -i <输入文件> [选项]
```

通过 WhisperX + Wav2Vec2 将音视频转写为词级对齐的时间戳，然后通过 LLM 将词切分为字幕块。输出 SRT + TXT。
重新运行时加载词级 JSON 缓存（`cache/<stem>_words.json`），跳过转录 + 对齐的步骤。更多细节请参考 [transcription_n_translation_CN.md](docs/transcription_n_translation_CN.md) - "二、转录流水线"。

#### 参数

| 参数 | 默认值 | 描述                                                  |
|------|--------|-----------------------------------------------------|
| `-i, --input <path>` | `input/input.mp4` | 输入视频或音频文件（.mp4/.mkv/.avi/.mov/.wav/.mp3/.m4a/.flac） |
| `-o, --output <path>` | `output/<stem>.srt` | 输出 SRT 文件路径                                         |
| `-seg_backend <name>` | `local` | 切分后端。可选项见 [1. LLM 后端与共享配置](#1-llm-后端与共享配置)。         |
| `-seg_model <name>` | 各后端默认 | 切分用模型（如 `phi4`、`gpt-5.6-terra`；默认取决于后端）             |
| `-gpu-layers <N>` | 自动检测 | 本地 LLM 使用的 GPU 层数（0 = 仅 CPU）                        |
| `-no-cache` | — | 禁用词级 `.json` 缓存（默认启用缓存） |

#### 流水线

```
输入视频
  │
  ├─ 1. WhisperX ASR（faster-whisper-large-v3 + Silero VAD）
  ├─ 2. Wav2Vec2 音素级强制对齐
  ├─ 3. LLM 标点与切分（通过 -seg_backend / -seg_model 指定后端与模型）
  │      阶段 1-4：  LLM 填充缺失标点（分块 + 上下文）
  │      阶段 5：    根据 LLM 推断的句子边界构建片段
  │      阶段 6：    基于逗号的超长段强制切分
  │      阶段 7：    连词切分（LLM 分类器处理歧义 "and"）
  │      阶段 8：    超长句重标点（递归 LLM 切分）
  │      阶段 9：    连词片段合并（LLM 分类器）
  │      阶段 10：   应急字符数切分
  └─ 4. 导出 SRT + TXT
```

#### 输出文件

| 文件 | 描述 |
|------|------|
| `output/<stem>.srt` | 带时间戳的字幕文件 |
| `output/<stem>.txt` | 纯文本（含时间戳标记） |
| `cache/<stem>_words.json` | 缓存的 ASR 词数据（重新运行时复用） |

#### 示例

```powershell
# 默认
python scripts/transcribe.py -i input/input.mp4

# 自定义输出路径
python scripts/transcribe.py -i input/input.mp4 -o D:/output/lecture.srt

# 仅 CPU 切分（无 GPU offload）
python scripts/transcribe.py -i input/input.mp4 -gpu-layers 0

# 使用在线 LLM 切分
python scripts/transcribe.py -i input/input.mp4 -seg_backend openai -seg_model gpt-5.6-terra

# 不使用缓存
python scripts/transcribe.py -i input/input.mp4 -no-cache
```

---

### 3. `scripts/translate.py` — 翻译流水线

使用**本地 LLM**（通过 llama-server 调用）或**5 种在线 API 后端**翻译 SRT（保留时间码）或纯文本 TXT 文件。
自动检测输入格式——SRT 文件保留时间码，TXT 文件按纯文本行处理。

支持两种翻译模式：
- **`accurate`**（默认）：**推荐本地模型使用**。
- **`flexible`**：**推荐在线 API 后端使用**。`allow_flexible_word_order` 和 `allow_simplify_wording` 配置项仅在此模式下生效。

两种翻译模式的细节详见 [transcription_n_translation_CN.md](docs/transcription_n_translation_CN.md) - "3.3.1 翻译模式"。

```powershell
python scripts/translate.py -i <input> [选项]
```

#### 参数

| 参数 | 默认值 | 描述                                                                               |
|------|--------|----------------------------------------------------------------------------------|
| `-i, --input <path>` | **（必需）** | 输入 `.srt` 或 `.txt` 文件（按内容自动检测格式）                                                 |
| `-o, --output <path>` | 自动 | 输出文件路径                                                                           |
| `-transl_backend <name>` | `local` | 翻译后端。可选项同 scripts/transcribe.py。                                                 |
| `-transl_model <name>` | 各后端默认 | 所选后端对应的模型名称，可选项同 scripts/transcribe.py。                                          |
| `-src_lang <name>` | `English` | 源语言名称（目前只支持英语）。覆盖 translate_config.json 中的 `source_lang`。                                 |
| `-tgt_lang <name>` | `Chinese` | 目标语言名称。覆盖 translate_config.json 中的 `target_lang`。省略时自动补全 `-tgt_lang_code`。       |
| `-tgt_lang_code <code>` | `CN` | 用于文件名后缀的语言代码。覆盖 translate_config.json 中的 `target_lang_code`。省略时自动补全 `-tgt_lang`。 |
| `-mode <name>` | `accurate` | 翻译模式：`accurate`（每次 2 行）+ `flexible`（每次 4 行，含时间码）                                 |
| `-gpu-layers <N>` | 自动检测 | 本地 LLM 使用的 GPU 层数（0 = 仅 CPU），同 scripts/transcribe.py                                                    |


#### 示例

```powershell
# 本地 Phi-4 — 将 TXT 翻译为中文（默认）
python scripts/translate.py -i output/input.txt
# → output/input_CN.txt

# 本地 Phi-4 — 翻译 SRT（保留时间码）
python scripts/translate.py -i output/input.srt
# → output/input_CN.srt + output/input_CN.txt

# 在线后端
python scripts/translate.py -i output/input.srt -transl_backend deepseek
python scripts/translate.py -i output/input.srt -transl_backend openai -transl_model gpt-5.6-terra
python scripts/translate.py -i output/input.srt -transl_backend anthropic -transl_model claude-opus-4-8

# 使用 DeepSeek（flexible 模式）翻译为日语
python scripts/translate.py -i output/input.txt -transl_backend deepseek -tgt_lang Japanese -tgt_lang_code JP -mode flexible

# 自定义输出路径
python scripts/translate.py -i output/input.txt -o D:/output/lecture_JP.srt

# 指定源语言
python scripts/translate.py -i output/input.txt -src_lang English -tgt_lang Chinese
```

---

### 4. `main.py` — 编排入口（转写 + 翻译）

```powershell
python main.py -i <输入文件> [选项]
```

`main.py` 是编排入口，直接导入调用 `scripts.transcribe.transcribe_file()` 和/或 `scripts.translate.translate_file()`。
它不重新实现流水线——调用的是与独立脚本相同的函数。

| `-transcribe` | `-translate` | 行为 |
| ------------ | ----------- | ---- |
| `true`（默认） | `false`（默认） | 仅转写（ASR + 切分 → SRT/TXT） |
| `true` | `true` | 先转写，再翻译生成的 SRT |
| `false` | `true` | 纯翻译（输入需为 `.srt` 或 `.txt`） |
| `false` | `false` | 错误：至少需启用一项 |

#### 参数

| 参数 | 默认值 | 描述                                             |
|------|--------|------------------------------------------------|
| `-i, --input <path>` | `input/input.mp4` | 输入文件（转写：音视频；纯翻译：`.srt`/`.txt`）                 |
| `-o, --output <path>` | `output/<stem>.srt` | 同 scripts/transcribe.py                        |
| `-transcribe` | `true` | 启用转写（`true`/`false`）                           |
| `-translate` | `false` | 启用翻译（`true`/`false`）                           |
| `-seg_backend <name>` | `local` | 同 scripts/transcribe.py                        |
| `-seg_model <name>` | 各后端默认 | 同 scripts/transcribe.py                        |
| `-transl_backend <name>` | `local` | 同 scripts/translate.py                         |
| `-transl_model <name>` | 各后端默认 | 同 scripts/translate.py                         |
| `-src_lang <name>` | `English` | 同 scripts/translate.py                         |
| `-tgt_lang <name>` | `Chinese` | 同 scripts/translate.py                         |
| `-tgt_lang_code <code>` | `CN` | 同 scripts/translate.py                         |
| `-mode <name>` | `accurate` | 同 scripts/translate.py                         |
| `-gpu-layers <N>` | 自动检测 | 同 scripts/transcribe.py 和 scripts/translate.py |
| `-no-cache` | — | 同 scripts/transcribe.py                        |

> **注意：** `-mode` 要求 `-translate true`。无翻译时传入 `-mode` 会报错退出。

#### 示例

```powershell
# 仅转写（默认）
python main.py -i input/input.mp4

# 转写 + 翻译为中文（本地 Phi-4）
python main.py -i input/input.mp4 -translate true

# 转写 + 在线 API 翻译（flexible 模式）
python main.py -i input/input.mp4 -translate true -transl_backend deepseek -mode flexible

# 纯翻译（现有字幕）
python main.py -transcribe false -translate true -i output/lecture.srt

# 切分和翻译使用不同模型
python main.py -i input/input.mp4 -translate true -seg_backend local -seg_model phi4 -transl_backend openai -transl_model gpt-5.6-terra
```

---

### 5. `batch.py` — 批处理

处理文件夹中的所有文件，每个文件在独立的子进程中运行（每个文件拥有独立的 CUDA 上下文——文件间无 VRAM 泄漏）。

通过 `-ext` 参数支持视频（.mp4、.mkv、.avi、.mov）和音频（.wav、.mp3、.m4a、.flac）输入。

```powershell
python batch.py [选项]
```

#### 参数

| 参数 | 默认值 | 描述                             |
|------|--------|--------------------------------|
| `-i, --input <dir>` | `input/` | 输入文件夹                          |
| `-o, --output <dir>` | `output/` | 输出文件夹                          |
| `-ext <ext>` | `.mp4` / `.srt` / `.txt` | 要搜索的文件扩展名（如 `.mp3`、`.wav`、`.mkv`）。transcribe 模式默认 `.mp4`，translate-only 模式默认 `.srt` 和 `.txt`。 |
| `-transcribe` | `true` | 同 main.py                      |
| `-translate` | `false` | 同 main.py                      |
| `-seg_backend <name>` | `local` | 同 scripts/transcribe.py        |
| `-seg_model <name>` | 各后端默认 | 同 scripts/transcribe.py        |
| `-transl_backend <name>` | `local` | 同 scripts/translate.py         |
| `-transl_model <name>` | 各后端默认 | 同 scripts/translate.py         |
| `-src_lang <name>` | `English` | 同 scripts/translate.py         |
| `-tgt_lang <name>` | `Chinese` | 同 scripts/translate.py         |
| `-tgt_lang_code <code>` | `CN` | 同 scripts/translate.py         |
| `-mode <name>` | `accurate` | 同 scripts/translate.py         |
| `-gpu-layers <N>` | 自动检测 | 同 scripts/transcribe.py 和 scripts/translate.py      |
| `-no-cache` | — | 同 scripts/transcribe.py        |

#### 示例

```powershell
# 处理 input/ 中所有 MP4 文件
python batch.py

# 自定义文件夹
python batch.py -i D:/videos -o D:/output

# 处理 MKV 文件
python batch.py -ext .mkv

# 处理音频文件（MP3、WAV）
python batch.py -ext .mp3
python batch.py -ext .wav

# 批处理 + 翻译所有文件（本地 Phi-4）
python batch.py -translate true

# 批处理 + 在线 API 翻译（flexible 模式）
python batch.py -translate true -transl_backend deepseek -transl_model deepseek-v4-flash -mode flexible

# 批量使用在线 API 翻译
python batch.py -transcribe false -translate true -i D:/subtitles/ -o D:/subtitles_translated/ -transl_backend deepseek

# 在线切分 + 本地翻译（不建议）
python batch.py -seg_backend openai -seg_model gpt-5.6-terra -translate true
```

## 支持的输入格式

`.mp4`、`.mkv`、`.avi`、`.mov`、`.wav`、`.mp3`、`.m4a`、`.flac`

---

## 目录结构

```
M2L3/
├── main.py                 # 编排入口——一条命令完成转写 + 翻译
├── batch.py                # 批处理入口
├── translate_config.json   # 翻译用户配置
├── README.md               # 英文版文档
├── README_CN.md            # 中文版文档（本文件）
├── requirements.txt
├── scripts/
│   ├── transcribe.py       # 转录流水线（ASR → 对齐 → 切分 → 导出）
│   └── translate.py        # 翻译流水线（6 种后端，accurate/flexible 模式）
├── models/
│   ├── hub/                        # 本地模型缓存
│   │   ├── checkpoints/            # Wav2Vec2 对齐模型（~360 MB）
│   │   ├── nltk_data/              # NLTK 分词数据（~64 MB）
│   │   └── snakers4_silero-vad_master/  # Silero VAD 语音活动检测（~35 MB）
│   ├── faster-whisper-large-v3/    # ASR 模型（~3 GB）
│   └── *.gguf           # 用于切分与翻译的 LLM
├── docs/
│   ├── segmentation_pipeline.md           # 10-phase LLM segmentation algorithm (English)
│   ├── segmentation_pipeline_CN.md        # 10 阶段 LLM 切分算法文档（中文）
│   ├── transcription_n_translation.md     # Transcription & translation technical implementation (English)
│   ├── transcription_n_translation_CN.md  # 转录与翻译技术实现文档（中文）
│   ├── models_n_performance.md            # Models & performance (English)
│   └── models_n_performance_CN.md         # 模型与性能（中文）
├── tools/
│   ├── llama/
│   │   └── llama-server.exe        # llama.cpp 推理服务器（CUDA）
│   ├── __init__.py                 # 包标记
│   ├── config.py                   # 代理与 GPU 环境配置
│   ├── env_check.py                # FFmpeg 与 CUDA 验证
│   ├── extract.py                  # 从 ASR 输出中提取词级数据
│   ├── cache.py                    # 词级 JSON 缓存
│   ├── format.py                   # SRT 时间格式化
│   ├── export.py                   # SRT / TXT 导出
│   ├── download_models.py          # 自动下载脚本
│   ├── segment.py                  # LLM 切分引擎（10 阶段）
│   ├── non_split_bigrams.py        # 短语保护二元组
│   └── _patch_transformers.py      # Transformers 离线补丁（内部）
├── input/                  # 默认输入文件夹
├── output/                 # 默认输出文件夹
└── cache/                  # 词级 ASR 缓存
```

---

## 许可证

本项目采用 [MIT 许可证](LICENSE)。


