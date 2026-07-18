# 模型与性能

本文档从三个方面介绍 M2L3 流水线：**硬件需求**（转录、切分、翻译的运行条件）；**LLM 选型**（各 LLM 模型的对比与选择建议）；以及 **LLM 推理基准测试**（所有测试模型的性能数据）。

---

## 第一部分 — 硬件需求

### 1.1 WhisperX 转录（本地 ASR + 对齐）

如果你只在本机运行转录，切分和翻译使用在线 API，那么以下模型是唯一需要本地算力的：

| 模型 | 磁盘空间 | GPU 要求 | CPU 回退 |
|------|---------|----------|---------|
| **faster-whisper-large-v3** | ~3 GB | NVIDIA GPU，推荐 4 GB+ 显存 | 可用 CPU（速度明显变慢） |
| **Silero VAD** | ~35 MB | 可选（不需要 GPU） | CPU 足够 |
| **Wav2Vec2 (base 960h)** | ~360 MB | 可选（GPU 加速效果明显） | CPU 足够 |

> **建议：** 推荐配备专用 NVIDIA GPU（4 GB+ 显存）以获得可接受的转录速度。整个转录流水线可以纯 CPU 运行，但 ASR 速度会显著变慢（通常比 GPU 慢 3–5 倍）。

> **纯 API 场景：** 如果你的切分**和**翻译都使用在线 API，则除了转录所需的硬件外，不需要额外的 LLM 本地算力。

### 1.2 本地 LLM 推理

对于在本机运行 LLM 切分和/或翻译（而非使用在线 API）的用户，默认模型为 **Phi-4（14B，Q4_K_M GGUF ~8.5 GB）**。推荐硬件配置如下：

| 场景 | 系统内存 | GPU 要求 | 说明 |
|------|---------|---------|------|
| **仅 CPU**（无 GPU） | 32 GB+ | 不需要 | Phi-4 14B（Q4_K_M ~8.5 GB）完全载入系统内存。推理速度较慢，但对批处理来说够用。 |
| **CPU + GPU**（混合） | 16 GB+ | NVIDIA GPU，6 GB+ 显存 | 通过 `-gpu-layers N` 将部分层卸载到 GPU。`auto_gpu_layers()` 函数通过 `nvidia-smi` 查询实际空闲显存，预留 1.2 GB 作为 CUDA 上下文 + KV 缓存的 buffer，然后在剩余显存中尽可能多地载入 LLM 层。|


### 1.3 测试平台

本文档中的所有基准数据均在以下硬件上采集：

| 组件 | 规格 |
|------|------|
| **CPU** | AMD Ryzen AI H350 |
| **内存** | 32 GB LPDDR5 |
| **GPU** | NVIDIA RTX 5060 笔记本电脑（8 GB 显存） |
| **操作系统** | Windows 11 |

性能数据来自两轮独立测试（同一硬件）。**同轮测试结果可直接对比**，跨轮对比存在约 ±10% 的系统误差（系统负载差异）。

---

## 第二部分 — LLM 选型

本节介绍流水线中两个 LLM 驱动任务可用的模型：

- **切分** — 10 阶段 LLM 流水线，将原始转录拆分为字幕段（见 [segmentation_pipeline_CN.md](segmentation_pipeline_CN.md)）
- **翻译** — 滑动窗口翻译引擎，翻译 SRT/TXT 文件的同时保留时间码（见 [transcription_n_translation_CN.md](transcription_n_translation_CN.md) - "翻译流水线"）

> 三个较小的非 LLM 模型——**faster-whisper-large-v3**（ASR）、**Silero VAD**（语音活动检测）和 **Wav2Vec2**（音素对齐）——在 [transcription_n_translation_CN.md](transcription_n_translation_CN.md) - "转录流水线" 中介绍。

### 2.1 本地模型

所有本地模型均通过 llama.cpp（llama-server.exe）加载，使用 GGUF 量化格式（量化参数见大小列）。

| 模型 | 大小（GGUF） | 内存/显存 | 模型源和下载链接 | 说明                                           | 
|------|------------|-----------|-----------------|----------------------------------------------|
| **Phi-4 14B**（默认） | 8.5 GB（Q4_K_M） | 8–16 GB | **模型源：** [unsloth/phi-4-GGUF](https://huggingface.co/unsloth/phi-4-GGUF)<br>**国内镜像：** `hf-mirror.com/unsloth/phi-4-GGUF`<br>**下载：** `phi-4-Q4_K_M.gguf` | 综合推理能力强，兼顾字幕切分和翻译；适用于只用一种 LLM 解决所有问题的情况      |  
| **Qwen3.5-9B** | 8.9 GB（Q8_0） | 12–16 GB | **模型源：** [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)<br>**国内镜像：** `hf-mirror.com/unsloth/Qwen3.5-9B-GGUF`<br>**下载：** `Qwen3.5-9B-Q8_0.gguf` | 翻译质量中等，但切分效果最差（不推荐用于切分）                      | 
| **Ministral-3-8B-Instruct 8B** | 8.5 GB（Q8_0） | 12–16 GB | **模型源：** [unsloth/Ministral-3-8B-Instruct-2512-GGUF](https://huggingface.co/unsloth/Ministral-3-8B-Instruct-2512-GGUF)<br>**国内镜像：** `hf-mirror.com/unsloth/Ministral-3-8B-Instruct-2512-GGUF`<br>**下载：** `Ministral-3-8B-Instruct-2512-Q8_0.gguf` | 模型体积最小，翻译速度一般且翻译质量不如 Phi-4 和 Qwen3.5（不推荐用于翻译） |  
| **Ministral-3-14B-Instruct 14B** | 9.0 GB（Q5_K_M） | 12–16 GB | **模型源：** [mistralai/Ministral-3-14B-Instruct-2512-GGUF](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512-GGUF)<br>**国内镜像：** `hf-mirror.com/mistralai/Ministral-3-14B-Instruct-2512-GGUF`<br>**下载：** `Ministral-3-14B-Instruct-2512-Q5_K_M.gguf` | 翻译质量和 8B 版本几乎一致，推理太慢              | 

> **说明：** 我们还测试了 **DeepSeek-R1-Distill-Llama-8B（Q8_0）** 和 **DeepSeek-Coder-V2-Lite-Instruct（16B，Q4_K_M）**，效果均不理想——R1 蒸馏变体在指令跟随上表现不佳，Coder 变体在翻译中存在格式漂移和内容重复问题。如想尝试其他模型，建议选用**更偏向自然语言处理的 Instruct 模型**，而非面向代码或推理蒸馏的变体。 

完整的模型名列表（用于在使用本地后端的情况下指定 \`-seg_model\` 或 \`-transl_model\`）见 [transcription_n_translation_CN.md](transcription_n_translation_CN.md) - "1.2 本地后端"。

### 2.2 API 模型

所有 API 后端需要互联网连接和 API 密钥（通过 `translate_config.json` 或 `.env` 配置）。完整的后端列表及 base_url 见 [transcription_n_translation_CN.md](transcription_n_translation_CN.md) - "1.3 在线后端"。

### 2.3 何时选择哪个方案

| 场景 | 推荐方案                                                                    | 
|------|-------------------------------------------------------------------------|
| **完全离线，敏感内容** | 本地：Phi-4 | 
| **追求速度与质量，有 API 预算** | API：DeepSeek (v4-flash) | 
| **追求最佳质量，不敏感成本** | API：Anthropic (claude-opus-4-8) | 
| **混合模式——本地 ASR，API LLM** | 本地 Phi-4 执行字幕切分，DeepSeek API 执行翻译（推荐） | 
| **低配置硬件（4 GB 或更少显存）** | 全部使用 API | 

---

## 第三部分 — LLM 推理基准测试

> **测试视频源**: [Learning Dune 3 — Mod Matrix Explained](https://youtu.be/0WD4KMbOWGA?si=UKXTDKPL6kNrq1jD)（12分09秒）  
> **参考字幕**: 208 段（人工校对）

### 3.1 转录切分性能

切分流水线依次执行 ASR 语音识别 → 时间轴对齐 → LLM 精炼切分，将原始转录文本拆分为字幕段。

**破坏率公式**: `破坏性切分数 / 总切分数`  
*破坏性切分* 指从非语法边界处切断句子，产生主谓结构不完整的碎片或不自然的短语中断。在自然连词或从句边界处切分不计为错误。

| 模型 | 用时 | 切分数 | 破坏性切分¹ |
|------| :----: | :----: | :----------: |
| Phi-4 14B（Q4_K_M） | 202.0s | 199 | 3（1.5%） |
| Ministral-3-8B-Instruct 8B（Q8_0） | 195.6s | 200 | 3（1.5%） |
| Ministral-3-14B-Instruct 14B（Q5_K_M） | 254.5s | 206 | 3（1.5%） |
| Qwen3.5-9B（Q8_0） | 346.1s | 211 | 14（6.6%） |
| DeepSeek API（deepseek-v4-flash） | 84.9s | 205 | 4（2.0%） |

> **说明**：
> - 用时为**纯 LLM 切分时间**（在已转录并缓存的词上运行），所有模型共享同一批缓存的 ASR 结果，ASR 与对齐时间不计入模型间对比。

### 3.2 翻译性能

将参考字幕通过 LLM 进行翻译，支持两种模式：
- **Accurate（精确模式）**：2 行滑动窗口（单段翻译最准确）
- **Flexible（灵活模式）**：4 行滑动窗口（上下文更连贯，可能合并段落）

| 模型 | Accurate 用时 |         Accurate 效果         | Flexible 用时 |          Flexible 效果          |
|------|:------------:|:---------------------------:|:------------:|:-----------------------------:|
| Phi-4 (14B, Q4_K_M) | 627.3s |    B — 少量术语不一致，少量术语错误翻译     | 675.3s | B — 质量同 accurate 模式, 部分段落自然合并 |
| Qwen3.5-9B (Q8_0) | 542.2s |  B — 质量和 Phi-4 相近，但推理速度较快   | 561.5s |  B — 表达比 accurate 更自然，推理速度较快  |
| Ministral-3-8B-Instruct (8B, Q8_0) | 580.3s | C — 少量术语不一致，少量术语错误翻译，语言不够自然 | 528.7s |   C — 术语一致, 少量单词未翻译，语言不够自然    |
| Ministral-3-14B-Instruct (14B, Q5_K_M) | 657.4s |  C — 和 8B 版内容基本一致，但推理速度略慢   | 1099.1s |     C — 和 8B 版内容基本一致，推理太慢     |
| DeepSeek API (deepseek-v4-flash) | 207.7s |   A — 表达更自然，个别术语错误，无其他问题    | 112.8s |     A — 表达更自然，基本无误, 合并较多      |

> **说明：** 用时为翻译 208 段参考字幕的实际耗时。质量评级综合考虑术语一致性、上下文理解准确度和中文自然度；本地模型在中文自然度上普遍不如在线 API。评级：A=优秀，B=良好，C=一般，D=较差。
