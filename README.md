# Video2Text

把视频变成文字稿和摘要的本地命令行工具。纯本地运行，不联网，中文友好，支持 GPU 加速和批量处理。

## 它能做什么

输入一段视频，输出两个文件：

- **文字稿** (`*_transcript.txt`)：完整的语音转写
- **摘要** (`summary/*_summary.txt`)：大语言模型生成的结构化要点（简体中文）

支持单文件或整个目录批量处理。已自动跳过处理过的视频，支持 `--force` 强制重跑。

## 工作原理

```
视频 → ffmpeg 提取音频 → Whisper 转写 → Qwen 生成摘要
                              ↑
                        Silero VAD 过滤无人声
```

详细的技术原理、历史脉络和参数选择故事见 [docs/explainer.md](docs/explainer.md)。简洁版设计文档见 [docs/design.md](docs/design.md)。

## 环境要求

- Python 3.12+
- ffmpeg（系统命令）
- whisper.cpp（建议带 CUDA 编译）
- 约 4 GB 磁盘存放模型，8 GB 显存可享受 GPU 加速

## 安装

### 1. Python 依赖

```bash
uv sync
```

### 2. 编译 whisper.cpp（带 CUDA）

```bash
git clone https://github.com/ggerganov/whisper.cpp /tmp/whisper.cpp
cd /tmp/whisper.cpp
cmake -B build -DGGML_CUDA=on
cmake --build build -j --config Release
```

如果机器没有 NVIDIA GPU，去掉 `-DGGML_CUDA=on` 即可（速度会慢 5–10 倍）。

### 3. 下载模型

把以下三个模型放到 `models/` 目录：

| 模型 | 大小 | 来源 |
|------|------|------|
| `ggml-medium-q8_0.bin` | 786 MB | [HuggingFace ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp) |
| `ggml-silero-v5.1.2.bin` | 865 KB | [HuggingFace ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp) |
| `qwen3.5-4b-instruct-Q4_K_M.gguf` | 2.6 GB | [ModelScope Qwen](https://modelscope.cn/models/Qwen) |

国内用户访问 HuggingFace 不畅时，可换用 [ModelScope](https://modelscope.cn/) 镜像。

## 使用

```bash
# 单个视频
python main.py video.mp4

# 整个目录
python main.py videos/

# 自定义输出目录 + 详细日志
python main.py videos/ -o my_output -v

# 强制重新处理已存在的输出
python main.py video.mp4 --force

# 关闭 VAD（音频质量好时可加快速度）
python main.py video.mp4 --no-vad
```

### 全部参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input` | *(必填)* | 视频文件或目录 |
| `-o/--output` | `./output/` | 输出目录 |
| `-m/--model` | `models/qwen3.5-4b-instruct-Q4_K_M.gguf` | LLM 模型路径 |
| `--whisper-model` | `models/ggml-medium-q8_0.bin` | Whisper 模型路径 |
| `--whisper-cli` | `/tmp/whisper.cpp/build/bin/whisper-cli` | whisper-cli 二进制路径 |
| `--vad-model` | `models/ggml-silero-v5.1.2.bin` | VAD 模型路径 |
| `--no-vad` | false | 禁用 VAD 过滤 |
| `--no-speech-thold` | `0.6` | 无人声阈值 (0.0–1.0) |
| `-f/--force` | false | 强制重新处理 |
| `-v/--verbose` | false | 详细日志 |

## 输出结构

```
output/
├── video1_transcript.txt        # 文字稿
├── video2_transcript.txt
├── summary/
│   ├── video1_summary.txt       # 摘要
│   └── video2_summary.txt
└── video3_error.log             # 出错时的错误日志
```

## 项目结构

```
video2text/
├── main.py              # CLI 入口、流程编排
├── audio.py             # ffmpeg 音频提取
├── transcriber.py       # whisper.cpp 语音转写
├── summarizer.py        # Qwen 摘要 + OpenCC 简繁转换
├── docs/
│   ├── design.md        # 简洁设计文档
│   └── explainer.md     # 通俗读本（含技术原理与历史）
├── models/              # 模型文件（git 忽略）
└── output/              # 默认输出目录（git 忽略）
```

## 性能参考

测试机器：RTX 3070 Ti (8 GB) + Ubuntu 22.04

| 视频时长 | 处理耗时 | 备注 |
|----------|----------|------|
| 5 分钟 | ~30 秒 | VAD + medium + Qwen 4B |
| 30 分钟 | ~3 分钟 | 同上 |
| 批量 84 个视频 | 约 2 小时 | 总成功率 97.6% |

## 已知限制

- 仅支持中文语音转写
- 无人声 / 纯音乐视频会被 VAD 过滤
- 转写超过 10000 字符自动截断
- 需本地模型文件，约 3.4 GB

## 致谢

站在巨人肩上：

- [FFmpeg](https://ffmpeg.org/) — 音视频处理瑞士军刀
- [OpenAI Whisper](https://github.com/openai/whisper) — 多语言语音识别
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) / [llama.cpp](https://github.com/ggerganov/llama.cpp) — Georgi Gerganov 的本地 AI 推理生态
- [Qwen](https://github.com/QwenLM/Qwen) — 阿里开源的中文友好大模型
- [Silero VAD](https://github.com/snakers4/silero-vad) — 轻量人声检测
- [OpenCC](https://github.com/BYVoid/OpenCC) — 中文简繁转换

## 许可证

MIT
