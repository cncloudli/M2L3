# 基于多层LLM的字幕分割工具 
**(Multi-Layer LLM-Based Subtitle Segmentation / M2L3)**

基于本地 GPU 加速的字幕生成工具，集成 WhisperX + Wav2Vec2 对齐 + LLM 增强分割与翻译。

| 功能 | 方案 | 使用场景 |
|---------|---------------------------------------------------|-------------|
| **语音识别 (ASR)** | WhisperX (faster-whisper-large-v3) + Silero VAD | 始终使用 |
| **对齐 (Alignment)** | Wav2Vec2 音素级强制对齐 | 始终使用 |
| **分割 (Segment)** | 通过 llama-server 调用 Phi-4（10 阶段 LLM 精炼） | 所有内容类型 |
| **翻译 (Translate)** | 通过 llama-server 调用 Phi-4 或在线 LLM（需 API） | 转写后 EN → 任意语言 |

所有处理流程**完全离线运行**——无需上传数据至外部服务器。(在线 LLM 翻译是**可选的**)

---

## 文件参考

### 根目录

| 文件 | 用途 |
|------|------|
| [main.py](main.py) | **单文件流水线入口。** 将视频/音频文件转写为带时间戳的词级对齐结果 → LLM 分割 → SRT + TXT 导出。 |
| [batch_pipeline.py](batch_pipeline.py) | **批处理。** 遍历输入文件夹中的所有文件，为每个文件启动一个独立的 `main.py` 子进程（每个进程拥有独立的 CUDA 上下文）。 |
| [translate.py](translate.py) | **翻译。** 使用本地或基于 API 的 LLM 翻译现有 SRT（保留时间码）或 TXT 文件。支持 7 种后端（local、OpenAI、DeepSeek、Qwen、Gemini、Ollama、Anthropic）。 |
| [translate_config.json](translate_config.json) | **翻译配置。** 语言、标点、术语表、API 密钥——由用户编辑，`translate.py` 启动时读取。 |
| [README_CN.md](README_CN.md) | 本文件——中文版使用指南、安装说明及文件参考。 |
| [README.md](README.md) | 英文版使用指南及文档。 |

### `tools/` 目录

| 模块 | 用途 |
|------|------|
| [tools/\_\_init\_\_.py](tools/__init__.py) | 包标记——仅含模块文档字符串。 |
| [tools/config.py](tools/config.py) | **系统配置。** 设置 `HTTP_PROXY` / `NO_PROXY`（绕过 PyTorch/HuggingFace CDN 的代理）和 `CUDA_VISIBLE_DEVICES`。在导入时执行，早于任何模型下载。 |
| [tools/env_check.py](tools/env_check.py) | **环境验证。** `check_ffmpeg()` — 验证 FFmpeg 是否在 PATH 中。`check_cuda()` — GPU 检测 + 实际内核启动测试，返回 `True`/`False`。 |
| [tools/extract.py](tools/extract.py) | **从 ASR 输出中提取词级数据。** `extract_words_from_result()` — 从 WhisperX 结果字典中提取 `(start, end, text)` 词元组。`fix_word_timestamps()` — 修正膨胀时间戳的后处理。 |
| [tools/cache.py](tools/cache.py) | **词级缓存。** `save_words_cache()` / `load_words_cache()` — 将提取的词数据持久化/恢复为 JSON，用于重新分割调试。 |
| [tools/format.py](tools/format.py) | **SRT 时间格式化。** `format_srt_time()` — 将浮点秒数转换为 `HH:MM:SS,mmm` 的 SRT 时间格式。 |
| [tools/export.py](tools/export.py) | **字幕导出。** `export_srt()` — 写入带序号和时间戳的 SRT。`export_txt()` — 写入纯文本或 SRT 格式文本。`export_word_level()` — 调试用的词级 SRT/TXT（_当前已注释掉_）。 |
| [tools/download_models.py](tools/download_models.py) | **模型预下载器。** 一次性脚本，将 Silero VAD、Wav2Vec2 和 NLTK 数据下载到本地缓存，使流水线完全离线运行。 |
| [tools/llm_pipeline.py](tools/llm_pipeline.py) | **LLM 分割引擎。** `LLMPipeline` 类 — 管理 llama-server 子进程生命周期，运行 10 阶段分割算法（参见 [docs/segmentation_pipeline_CN.md](docs/segmentation_pipeline_CN.md) 获取完整算法流程图）。`segment_with_llm()` 便捷函数用于一次性 CLI 使用。 |
| [tools/non_split_bigrams.py](tools/non_split_bigrams.py) | **短语保护。** `NON_SPLIT_BIGRAMS` 集合（约 5700 条）包含固定表达、短语动词和搭配，这些内容不得在字幕段间分割。`would_break_phrase()` 查找函数被所有分割阶段使用。 |
| [tools/llama/](tools/llama/) | **llama.cpp 二进制文件。** 预编译的 `llama-server.exe` + CUDA 12 DLL，用于本地 LLM 推理。已从 git 中排除——参见[安装 → llama.cpp 二进制文件](#llamacpp-二进制文件)。 |

### `docs/` 目录

| 文件 | 用途                                                          |
|------|-------------------------------------------------------------|
| [docs/segmentation_pipeline.md](docs/segmentation_pipeline.md) | **10 阶段 LLM 分割算法文档（英文）。** 完整的算法流程图及各阶段的详细说明、保护机制和 LLM 调用总结。 |
| [docs/segmentation_pipeline_CN.md](docs/segmentation_pipeline_CN.md) | **10 阶段 LLM 分割算法文档（中文）。** 同上，中文版。                           |
| [docs/transcription_n_translation.md](docs/transcription_n_translation.md) | **转录与翻译技术实现（英文）。** ASR→对齐→分割流水线及翻译工厂后端的设计思路、参数选择和代码介绍。      |
| [docs/transcription_n_translation_CN.md](docs/transcription_n_translation_CN.md) | **转录与翻译技术实现（中文）。** 同上，中文版。                                  |
| [docs/models_n_hardware_reqs.md](docs/models_n_hardware_reqs.md) | **模型与硬件需求说明。**（TODO — 将介绍所使用的 LLM 模型、显存/内存/硬盘需求及性能参考。）      |
| [docs/models_n_hardware_reqs_CN.md](docs/models_n_hardware_reqs_CN.md) | **模型与硬件需求说明。**（TODO — 同上，中文版。）                              |

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
# 若需加速，可使用国内 PyPI 镜像：
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
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
└── phi-4-Q4_K_M.gguf            # ~8.5 GB，Phi-4（分割和翻译均需要）
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

**Phi-4（14B，GGUF）：**

单个文件下载——任选一种方式：

```powershell
# 方式 A — 浏览器 / curl（直链，~8.5 GB，需科学上网）
curl -L -o models\phi-4-Q4_K_M.gguf ^
  https://huggingface.co/unsloth/phi-4-GGUF/resolve/main/phi-4-Q4_K_M.gguf

# 方式 A（国内镜像）— 替换为 hf-mirror.com
curl -L -o models\phi-4-Q4_K_M.gguf ^
  https://hf-mirror.com/unsloth/phi-4-GGUF/resolve/main/phi-4-Q4_K_M.gguf

# 方式 B — huggingface_hub（支持断点续传，无需重命名）
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/phi-4-GGUF', 'phi-4-Q4_K_M.gguf', local_dir='models')
"

# 方式 B（国内镜像）— 设置 HF_ENDPOINT
$env:HF_ENDPOINT = "https://hf-mirror.com"
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/phi-4-GGUF', 'phi-4-Q4_K_M.gguf', local_dir='models')
"
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
# 转写单个视频
python main.py -i input/input.mp4

# 转写 + 翻译为中文
python main.py -i input/input.mp4 -translate

# 处理文件夹中所有视频
python batch_pipeline.py

# 批处理 + 翻译
python batch_pipeline.py -translate

# 翻译现有的 TXT 文件
python translate.py -i output/input.txt
```

---

## 使用说明

### 1. `main.py` — 单文件转写

```powershell
python main.py -i <input> [options]
```

#### 参数

| 参数 | 默认值 | 描述 |
|----------|---------|-------------|
| `-i, --input <path>` | `input/input.mp4` | 输入视频或音频文件（.mp4/.mkv/.avi/.mov/.wav/.mp3/.m4a/.flac） |
| `-o, --output <path>` | `output/<stem>.srt` | 输出 SRT 文件路径 |
| `-gpu-layers <N>` | 自动检测 | LLM 使用的 GPU 层数（0 = 仅 CPU） |
| `-no-cache` | — | 跳过词级 `.json` 缓存 |
| `-translate` | — | 转写后自动翻译输出 |
| `-backend <name>` | `local` | 翻译后端（`local`、`deepseek`、`openai`、`qwen`、`gemini`、`ollama`、`anthropic`） |
| `-model <name>` | 各后端默认值 | 翻译模型（默认为后端对应的默认模型；仅在与 `-translate` 同时使用时有效） |

#### 流水线

```
输入视频
  │
  ├─ 1. WhisperX ASR（faster-whisper-large-v3 + Silero VAD）
  ├─ 2. Wav2Vec2 音素级强制对齐
  ├─ 3. LLM 标点与分割（通过 llama-server 调用 Phi-4）
  │      阶段 1-4：  LLM 填充缺失标点（分块 + 上下文）
  │      阶段 5：    根据 LLM 推断的句子边界构建片段
  │      阶段 6：    基于逗号的超长段强制分割
  │      阶段 7：    连词分割（LLM 分类器处理歧义 "and"）
  │      阶段 8：    超长句重标点（递归 LLM 分割）
  │      阶段 9：    连词片段合并（LLM 分类器）
  │      阶段 10：    应急字符数分割
  └─ 4. 导出 SRT + TXT
```

#### 输出文件

| 文件 | 描述 |
|------|-------------|
| `output/<stem>.srt` | 带时间戳的字幕文件 |
| `output/<stem>.txt` | 纯文本（含时间戳标记） |
| `output/<stem>_wl.srt` | 词级字幕（调试用——当前已注释掉） |
| `output/<stem>_wl.txt` | 词级文本（调试用——当前已注释掉） |
| `cache/<stem>_words.json` | 缓存的 ASR 词数据（重新运行时复用） |

#### 示例

```powershell
# 默认
python main.py -i input/input.mp4

# 路径中若包括空格或特殊字符请加引号
python main.py -i input/"input.mp4" 

# 自定义输出路径
python main.py -i input/input.mp4 -o D:/output/lecture.srt

# 转写 + 翻译为中文
python main.py -i input/input.mp4 -translate

# 不使用缓存
python main.py -i input/input.mp4 -no-cache
```

### 2. `batch_pipeline.py` — 批处理

处理文件夹中的所有文件，每个文件在独立的子进程中运行（每个文件拥有独立的 CUDA 上下文——文件间无 VRAM 泄漏）。

通过 `-ext` 参数支持视频（.mp4、.mkv、.avi、.mov）和音频（.wav、.mp3、.m4a、.flac）输入。

```powershell
python batch_pipeline.py [options]
```

#### 参数

| 参数 | 默认值 | 描述 |
|----------|---------|-------------|
| `-i, --input <dir>` | `input/` | 输入文件夹 |
| `-o, --output <dir>` | `output/` | 输出文件夹 |
| `-gpu-layers <N>` | 自动检测 | LLM 使用的 GPU 层数 |
| `-no-cache` | — | 跳过词级缓存 |
| `-ext <ext>` | `.mp4` | 要搜索的文件扩展名（例如 `.mp3`、`.wav`、`.mkv`） |
| `-dry-run` | — | 仅列出文件，不处理 |
| `-translate` | — | 每个文件转写后自动翻译 |
| `-backend <name>` | `local` | 翻译后端（`local`、`deepseek`、`openai`、`qwen`、`gemini`、`ollama`、`anthropic`） |
| `-model <name>` | 各后端默认值 | 翻译模型（仅在与 `-translate` 同时使用时有效） |

#### 示例

```powershell
# 处理 input/ 中所有 MP4 文件
python batch_pipeline.py

# 自定义文件夹
python batch_pipeline.py -i D:/videos -o D:/output

# 处理 MKV 文件
python batch_pipeline.py -ext .mkv

# 处理音频文件（MP3、WAV）
python batch_pipeline.py -ext .mp3
python batch_pipeline.py -ext .wav

# 试运行，查看将被处理的文件列表
python batch_pipeline.py -dry-run

# 批处理 + 翻译所有文件
python batch_pipeline.py -translate
```

### 3. `translate.py` — 语言翻译

使用**本地 LLM**（通过 llama-server 调用 Phi-4）或**6 种在线 API 后端**翻译 SRT（保留时间码）或纯文本 TXT 文件。
自动检测输入格式——SRT 文件保留时间码，TXT 文件按纯文本行处理。

支持两种翻译模式：
- **`accurate`**（默认）：小型滑动窗口（每次 2 行，上下文 4 行），逐行独立翻译，极低的时间轴错位风险。**推荐本地 Phi-4 模型使用**——本地模型推理速度较慢，小窗口可保持稳定的延迟和较低的资源占用。
- **`flexible`**：大型滑动窗口（每次 4 行，上下文 8 行），含时间码提示，允许 LLM 跨行调整语序以获得更自然的措辞。**更适合在线 API 后端**——高速 API 可充分利用大窗口的上下文优势。

```powershell
python translate.py -i <input> [options]
```

#### 参数

| 参数 | 默认值 | 描述 |
|----------|---------|-------------|
| `-i, --input <path>` | **（必需）** | 输入 `.srt` 或 `.txt` 文件（按内容自动检测格式） |
| `-o, --output <path>` | 自动 | 输出文件路径 |
| `-tgt-lang <name>` | `Chinese` | 目标语言名称 |
| `-tgt-lang-code <code>` | `CN` | 用于文件名后缀的语言代码 |
| `-src-lang <name>` | `English` | 源语言名称 |
| `-backend <name>` | `local` | 翻译后端：`local`、`deepseek`、`openai`、`qwen`、`gemini`、`ollama`、`anthropic` |
| `-model <name>` | 各后端默认值 | 所选后端对应的模型名称（参见下方表格） |
| `-gpu-layers <N>` | 自动检测 | GPU 层数（仅本地后端） |
| `-mode <name>` | `accurate` | 翻译模式：`accurate`（精准，每次 2 行）+ `flexible`（灵活，每次 4 行，含时间码） |
| `-mode <name>` | `accurate` | 翻译模式：`accurate`（精准，每次 2 行）+ `flexible`（灵活，每次 4 行，含时间码） |

#### 翻译后端

| 后端 | API 格式 | 默认模型 | 认证方式 |
|---------|-----------|---------------|------|
| `local` | llama-server 子进程 | `phi-4` | — |
| `deepseek` | 自动检测：OpenAI `/chat/completions` **或** Anthropic Messages（见脚注） | `deepseek-v4-flash` | `openai_api_key` |
| `openai` | OpenAI 兼容 | `gpt-5.6-terra` | `openai_api_key` |
| `qwen` | OpenAI 兼容 | `qwen3.5-plus` | `openai_api_key` |
| `gemini` | OpenAI 兼容 | `gemini-3.5-flash` | `openai_api_key` |
| `ollama` | OpenAI 兼容 | `llama4` | — |
| `anthropic` | Anthropic Messages | `claude-opus-4-8` | `anthropic_api_key` |

> **OpenAI 兼容后端**（openai、qwen、gemini、ollama）共用同一个 `OPENAI_API_KEY` 配置字段。Ollama 忽略该密钥（本地运行）。
> **DeepSeek** 支持两种 API 格式，通过 `api_base_url` 自动检测：
> - `"https://api.deepseek.com"`（默认）→ OpenAI `/chat/completions`
> - `"https://api.deepseek.com/anthropic"` → Anthropic Messages API（更稳定）
> 每个请求通过检查 URL 是否包含 `anthropic` 来决定使用哪种格式。使用 `openai_api_key` 进行身份验证。
> 每个后端的默认模型反映截至 2026 年 7 月的最新版本。

#### 用户配置

编辑项目根目录下的 [translate_config.json](translate_config.json)，或使用 `.env` 文件设置 API 密钥（见下方说明）：

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

#### 配置字段

| 字段 | 类型 | 默认值 | 描述                                                                                                                                                                               |
|-------|------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `target_lang` | string | `"Chinese"` | 目标语言名称（用于提示词）                                                                                                                                                                    |
| `target_lang_code` | string | `"CN"` | 追加到输出文件名的语言代码                                                                                                                                                                    |
| `source_lang` | string | `"English"` | 源语言名称（用于提示词）                                                                                                                                                                     |
| `add_punctuation` | bool | `true` | 在翻译中添加句末标点（。？！）。当为 `false` 时，通过后处理去除句末标点——不将此约束强加给 LLM。                                                                                                                          |
| `allow_flexible_word_order` | bool | `false` | 允许在相邻行间重新分配内容以获得更自然的措辞。**⚠ 仅 `flexible` 模式下可用**（`accurate` 模式下不会传递此参数）。**强烈建议仅在 API 后端中启用**——本地模型下使用可能导致不稳定的行号偏移。                                                                     |
| `allow_simplify_wording` | bool | `false` | 允许精简每行内的口语化/冗长表达（填充词、冗余、啰嗦）。**⚠ 仅 `flexible` 模式下可用**（`accurate` 模式下不会传递此参数）。**强烈建议仅在 API 后端中启用**——本地模型效果不佳。                                          |
| `number_mode` | string | `"auto"` | 数字处理模式：`"auto"`（LLM 自行决定）、`"src_lang"`（保留源格式）、`"digits"`（统一阿拉伯数字）、`"tgt_lang"`（目标语言数字形式）。**建议在在线 API 中使用**——本地模型也能正常生效，但效果可能不稳定。在两种模式下均可使用。     |
| `space_between_cjk_and_latin` | bool | `false` | 在中日韩字符与拉丁字符之间插入空格                                                                                                                                                              |
| `glossary` | string[] | `[]` | 保持不翻译的术语（原样保留）。**建议在在线 API 中使用**——本地模型也能正常生效，但效果可能不稳定。在两种模式下均可使用。                                                                                 |
| `custom_system_prompt` | string\|null | `none` | 覆盖整个 LLM 系统提示词（禁用所有其他提示标志）                                                                                                                                                     |
| `cache_prompt` | bool | `false` | 启用跨批次的 KV-cache 复用（仅 `local` 后端）。降低延迟，但可能在长翻译任务中导致指令衰减。**强烈建议仅在 API 后端中启用**，本地模型应设为 `false`。在两种模式下均可使用。                                                     |
| `openai_api_key` | string | `""` | OpenAI 兼容后端的 API 密钥。**建议：** 改为在 `.env` 中设置 `OPENAI_API_KEY`                                                                                                                      |
| `anthropic_api_key` | string | `""` | Anthropic 后端的 API 密钥。**建议：** 改为在 `.env` 中设置 `ANTHROPIC_API_KEY`                                                                                                                  |
| `api_base_url` | string | `""` | 覆盖所选 API 后端的 base URL。例如 DeepSeek：`"https://api.deepseek.com"`（OpenAI 格式）或 `"https://api.deepseek.com/anthropic"`（Anthropic Messages 格式）。代码通过检查 URL 是否包含 `anthropic` 自动检测使用哪种格式。 |

> **通过 `.env` 设置 API 密钥**（推荐）：在项目根目录创建 `.env` 文件（已在 `.gitignore` 中，因此不会提交至 git）：
> ```
> OPENAI_API_KEY=sk-...
> ANTHROPIC_API_KEY=sk-ant-...
> ```
> 在 `translate_config.json` 中设置的密钥仍然优先，但留空并使用 `.env` 可将密钥隔离在版本控制之外。

#### 示例

```powershell
# 本地 Phi-4 — 将 TXT 翻译为中文（默认）
python translate.py -i output/input.txt
python translate.py -i output/"input.txt"
# → output/input_CN.txt

# 本地 Phi-4 — 翻译 SRT（保留时间码）
python translate.py -i output/input.srt
# → output/input_CN.srt + output/input_CN.txt

# 在线后端
python translate.py -i output/input.srt -backend openai -model gpt-5.6-terra
python translate.py -i output/input.srt -backend deepseek -model deepseek-v4-flash
python translate.py -i output/input.srt -backend gemini -model gemini-3.5-flash
python translate.py -i output/input.srt -backend qwen -model qwen3.5-plus
python translate.py -i output/input.srt -backend ollama -model llama4
python translate.py -i output/input.srt -backend anthropic -model claude-opus-4-8

# 使用 DeepSeek 翻译为日语
python translate.py -i output/input.txt -backend deepseek -tgt-lang Japanese -tgt-lang-code JP

# 自定义输出路径
python translate.py -i output/input.txt -o D:/output/lecture_JP.srt

# 指定源语言
python translate.py -i output/input.txt -src-lang English -tgt-lang Chinese
```

### 集成：转写 → 翻译

在任何 `main.py` 或 `batch_pipeline.py` 命令中添加 `-translate` 参数，即可在转写后自动执行翻译。

```powershell
# 单文件 — 本地翻译（默认）
python main.py -i input/input.mp4 -translate

# 批处理 — 本地翻译
python batch_pipeline.py -translate

# 单文件 — 在线后端翻译
python main.py -i input/input.mp4 -translate -backend openai -model gpt-5.6-terra

# 批处理 — 在线后端翻译
python batch_pipeline.py -translate -backend anthropic -model claude-opus-4-8
```

上述命令将在生成的 `.srt` 文件上以子进程方式运行 `translate.py`（全程保留时间码），生成 `output/<stem>_CN.srt` 和 `output/<stem>_CN.txt`。

---

## 支持的输入格式

`.mp4`、`.mkv`、`.avi`、`.mov`、`.wav`、`.mp3`、`.m4a`、`.flac`

---

## 目录结构

```
M2L3/
├── main.py                 # 单文件转写入口
├── batch_pipeline.py       # 批处理入口
├── translate.py            # 翻译入口
├── translate_config.json   # 翻译用户配置
├── README.md               # 英文版文档
├── README_CN.md            # 中文版文档（本文件）
├── requirements.txt
├── models/
│   ├── hub/                        # 本地模型缓存
│   │   ├── checkpoints/            # Wav2Vec2 对齐模型（~360 MB）
│   │   ├── nltk_data/              # NLTK 分词数据（~64 MB）
│   │   └── snakers4_silero-vad_master/  # Silero VAD 语音活动检测（~35 MB）
│   ├── faster-whisper-large-v3/    # ASR 模型（~3 GB）
│   └── phi-4-Q4_K_M.gguf           # 用于分割与翻译的 LLM（~8.5 GB，Phi-4 14B）
├── docs/
│   ├── segmentation_pipeline.md           # 10-phase LLM segmentation algorithm (English)
│   ├── segmentation_pipeline_CN.md        # 10 阶段 LLM 分割算法文档（中文）
│   ├── transcription_n_translation.md     # Transcription & translation technical implementation (English)
│   ├── transcription_n_translation_CN.md  # 转录与翻译技术实现文档（中文）
│   ├── models_n_hardware_reqs.md          # (TODO) Models & hardware requirements
│   └── models_n_hardware_reqs_CN.md       # (TODO) 模型与硬件需求说明
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
│   ├── llm_pipeline.py             # LLM 分割引擎（10 阶段）
│   └── non_split_bigrams.py        # 短语保护二元组
├── input/                  # 默认输入文件夹
├── output/                 # 默认输出文件夹
└── cache/                  # 词级 ASR 缓存
```

---

## 国内镜像资源汇总

| 资源 | 官方地址 | 国内镜像地址 |
|------|---------|-------------|
| **HuggingFace 模型** | `huggingface.co` | `hf-mirror.com`（设置 `$env:HF_ENDPOINT = "https://hf-mirror.com"`） |
| **GitHub Releases** | `github.com` | `ghproxy.com/https://github.com/...`（GitHub 加速代理） |
| **PyPI 包** | `pypi.org` | `pypi.tuna.tsinghua.edu.cn/simple`（清华 TUNA 镜像） |
| **FFmpeg** | `ffmpeg.org` | `mirrors.tuna.tsinghua.edu.cn/ffmpeg/` |
