# unsloth-buddy

<p align="center"><img src="images/unsloth_gaslamp.png" width="75%" alt="unsloth-buddy" /></p>

<p align="center">
  <a href="https://github.com/TYH-labs/unsloth-buddy"><img src="https://img.shields.io/github/stars/TYH-labs/unsloth-buddy?style=flat&logo=github&color=181717&logoColor=white" alt="GitHub" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License" /></a>
  <a href="#快速开始"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="https://gaslamp.dev/unsloth"><img src="https://img.shields.io/badge/%F0%9F%94%A5%20Gaslamp-Compatible-ff6b00?logoColor=white" alt="Gaslamp Compatible" /></a>
  <a href="#openclaw"><img src="https://img.shields.io/badge/%F0%9F%A6%9E%20OpenClaw-Compatible-ff4444" alt="OpenClaw Compatible" /></a>
  <a href="#快速开始"><img src="https://img.shields.io/badge/%F0%9F%A4%96%20Agent-Claude%20Code%20%2F%20Codex%20%2F%20Gemini-8b5cf6" alt="Agent Compatible" /></a>
  <a href="https://discord.gg/mZe4mbCQ6a"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="Discord" /></a>
  <a href="#本地部署"><img src="https://img.shields.io/badge/后端-Unsloth%20%7C%20MLX%20%7C%20llama.cpp-00b4d8" alt="后端：Unsloth / MLX / llama.cpp" /></a>
</p>

<p align="center"><code>/unsloth-buddy 我有 500 份医患问诊记录，想训练一个能自动摘要的模型，我只有一台 MacBook Air</code></p>

<p align="center">
  <a href="#快速开始"><img src="https://img.shields.io/badge/立即体验-1 分钟-black?style=for-the-badge" alt="立即体验" /></a>
  <a href="demos/"><img src="https://img.shields.io/badge/演示案例-示例-6e40c9?style=for-the-badge" alt="演示案例" /></a>
  <a href="SKILL.md"><img src="https://img.shields.io/badge/10%2B%20功能-功能详情-0969da?style=for-the-badge" alt="10+ 功能" /></a>
</p>

<p align="center">
  <a href="https://www.bilibili.com/video/BV1VWAFzmECy/"><img src="https://img.shields.io/badge/▶%20演示-Bilibili-00A1D6?style=for-the-badge&logo=bilibili&logoColor=white" alt="Bilibili 演示" /></a>
  <a href="https://youtu.be/wG28uxDGjHE"><img src="https://img.shields.io/badge/▶%20Demo-YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube Demo" /></a>
</p>

<p align="center">
  <a href="README.md">English</a> | 简体中文 | <a href="README_zh-Hant.md">繁體中文</a>
</p>

---

## 这是什么？

一个像同事一样交流的微调 Agent。描述你的需求，它会提出合适的问题、找到或整理你的数据、选择合适的技术和模型、在你的硬件上训练、验证效果，并打包部署。

支持通过 [Unsloth](https://github.com/unslothai/unsloth) 在 NVIDIA GPU 上运行，也支持通过 [mlx-tune](https://github.com/ml-explore/mlx-lm) 在 Apple Silicon 上原生运行。隶属于 [Gaslamp](https://gaslamp.dev/) AI 开发平台 — [文档](https://gaslamp.dev/unsloth)。

---

## 一句话，一个微调模型。

```
你：在我的客户支持 FAQ 上微调一个小模型，我有一个 CSV 文件。

[第 0 阶段] 创建项目：customer_faq_sft_2026_03_17/
[第 1 阶段] 需求访谈...
           方法：SFT   模型：Qwen2.5-0.5B   部署：Ollama
[第 2 阶段] 数据策略...
           从 faq.csv 加载 1,200 行 → 重新格式化为对话消息
           保存到 data/train.jsonl（已验证：messages 列 ✓）
[第 3 阶段] 环境：Apple M4 24GB，mlx-tune 0.4.3，Python 3.12
           准备就绪，可以开始训练
[第 4 阶段] 训练... 200 步
           最终损失：1.42 → 已保存到 outputs/adapters/
[第 5 阶段] 评估（基础模型 vs 微调模型，贪心解码）：
           问：如何重置我的密码？
           [基础]  我可以帮您。是哪种密码？
           [微调]  前往登录页面 → 点击"忘记密码" → 查看邮件。
[第 6 阶段] 导出 → outputs/model-q4_k_m.gguf
           运行：ollama create my-faq-bot -f Modelfile && ollama run my-faq-bot
```

一次对话，七个阶段，最终得到一个可部署的模型 — 以及一个可分享的演示页面。

---

## 快速开始

该技能包含子技能和工具脚本 — 请安装完整仓库，而非单个文件。

**Claude Code** *(推荐)*
```
/plugin marketplace add TYH-labs/unsloth-buddy
/plugin install unsloth-buddy@TYH-labs/unsloth-buddy
```
然后描述你想微调什么。技能会自动激活。

**Gemini CLI**
```bash
gemini extensions install https://github.com/TYH-labs/unsloth-buddy --consent
```

**任何支持 [Agent Skills](https://agentskills.io/) 标准的 Agent**
```bash
git clone https://github.com/TYH-labs/unsloth-buddy.git .agents/skills/unsloth-buddy
```

---

## 有何不同？

大多数工具假设你已经知道该怎么做，而这个工具不会。

| 你的顾虑 | 实际发生的事 |
|---|---|
| **"我不知道从哪里开始"** | 2 个问题的访谈锁定任务、受众和数据，然后推荐合适的模型、硬件和方法 |
| **"我没有数据，或者格式不对"** | 专门的数据阶段负责获取、生成或重新格式化数据，精确匹配训练器所需的 schema |
| **"SFT？DPO？GRPO？选哪个？"** | 将你的目标映射到正确的技术，并用通俗语言解释原因 |
| **"选哪个模型？能装进我的 GPU 吗？"** | 检测你的硬件，映射到可用的模型大小，必要时估算云端成本 |
| **"Unsloth 在我的机器上安装不了"** | 两阶段环境检测捕获不匹配问题，并打印适合你环境的精确安装命令 |
| **"我训练好了，但它有效吗？"** | 将微调适配器与基础模型并排运行，让你看到差距，而不只是一个损失数值 |
| **"怎么部署？"** | 你指定目标（Ollama、vLLM、HF Hub）— 它运行转换命令 |
| **"之后怎么复现，或者交给别人？"** | 每个项目都会生成一份 `gaslamp.md` 路书：记录每个已确定的决策及其原因，并附带 📖 学习模块解释底层 ML 概念，任何 Agent 或人员都可以端到端复现整个项目 |

---

## 工作原理

七个阶段，每个阶段都限定在一个独立的带日期项目目录中，不会影响你的仓库根目录。

| 阶段 | 发生的事 | 产出文件 |
|---|---|---|
| **0. 初始化** | 创建 `{name}_{date}/` 标准目录结构 | `gaslamp.md`、`progress_log.md` |
| **1. 访谈** | 2 个问题的访谈 — 任务 + 数据；捕获领域/受众 | `project_brief.md` |
| **2. 数据** | 获取、验证并格式化为训练器 schema | `data_strategy.md` |
| **3. 环境** | 硬件扫描 → Python 环境检查 → 阻塞直到就绪 | `detect_env_result.json` |
| **4. 训练** | 生成并运行 `train.py`，流式输出到日志 | `outputs/adapters/` |
| **5. 评估** | 批量测试、交互式 REPL、基础模型对比微调模型 | `logs/eval.log` |
| **5.5. 演示** | 生成可分享的静态 HTML 页面 — 基础模型 vs 微调模型并排展示 | `demos/<name>/index.html` |
| **6. 导出** | GGUF、合并 16-bit 或 Hub 推送 | `outputs/` |
| **6.5. 本地部署** | 可选：量化 → 基准测试 → 启动服务 + Gaslamp Chat WebUI（需要 llama.cpp）| `outputs/*.gguf` |

```
customer_faq_sft_2026_03_17/
├── train.py              eval.py
├── data/                 outputs/adapters/
├── logs/
├── gaslamp.md            ← 可复现路书
├── project_brief.md      data_strategy.md
├── memory.md             progress_log.md
```

---

## 硬件支持

| 硬件 | 后端 | 能跑什么 |
|---|---|---|
| NVIDIA T4 (16 GB) | `unsloth` | 7B QLoRA，小规模 GRPO |
| NVIDIA A100 (80 GB) | `unsloth` | 70B QLoRA，14B LoRA 16-bit |
| Apple M1 / M2 / M3 / M4 | `mlx-tune` / `mlx-vlm` / `trl` | SFT/DPO：10 GB 跑 7B，24 GB 跑 13B；通过 `mlx-vlm` 支持视觉 SFT；GRPO：1–7B（TRL + PyTorch MPS）|
| Google Colab (T4/L4/A100) | `unsloth` 通过 `colab-mcp` | 免费云端 GPU，可选接入 |

Unsloth 相比标准 HuggingFace 训练速度快约 2 倍，VRAM 使用量减少高达 80%，且使用精确梯度。

**支持的训练方法：** SFT、DPO、GRPO、ORPO、KTO、SimPO、视觉 SFT（Qwen2.5-VL、Llama 3.2 Vision、Gemma 3、Gemma 4）

---

## 实时训练看板

每次本地训练都会自动在 **http://localhost:8080/** 打开实时看板：

- **任务感知面板** — 传入 `task_type="sft"|"dpo"|"grpo"|"vision"` 自动启用对应图表
- **SSE 流式传输** — 通过 `EventSource` 即时推送更新，无轮询延迟
- **EMA 平滑损失** — 清晰的趋势线覆盖嘈杂的原始损失，附带运行均值
- **动态阶段徽章** — 空闲 → 训练中 → 已完成 / 错误，含色彩标识的任务类型徽章
- **ETA、已用时间与轮次** — 剩余时间估算及当前 epoch 进度
- **GPU 内存分解** — 基线（模型加载）vs LoRA 训练开销 vs 总量，以仪表条形式展示；同时支持 NVIDIA（CUDA）和 Apple Silicon（MPS，使用 `driver_allocated_memory` / `recommended_max_memory`）
- **GRPO 面板** — 奖励 ± 标准差置信带 + KL 散度图表
- **DPO 面板** — 选中 vs 拒绝奖励 + KL 散度图表
- **梯度范数与 tokens/sec** — 实时统计行，有数据时自动显示
- **训练完成摘要横幅** — 训练结束时展示最终内存与运行时间统计
- **终端 UI (Plotext)** — `scripts/terminal_dashboard.py` 支持 `--once` 一次性快照；DPO/GRPO 自动升级为 2×2 布局
- **演示服务器** — `python scripts/demo_server.py --task grpo --hardware mps|nvidia` 提供丰富的模拟数据，无需 GPU 即可预览所有面板

同时支持 NVIDIA（通过 `GaslampDashboardCallback(task_type=...)`）和 Apple Silicon（通过 `MlxGaslampDashboard(task_type=...)`）。

---

## 演示生成器

评估完成后，Agent 可以生成一个**静态 HTML 演示页面**，并排展示基础模型与微调模型的输出 — 在任意浏览器中打开即可，无需服务器。非常适合与团队成员、利益相关方分享结果，或放入作品集。

演示生成器是 [Gaslamp](https://gaslamp.dev/) 平台展示工具的一部分。我们为 unsloth-buddy 做了简化，内置两套主题并支持基于领域的自动配色：

| 主题 | 适用场景 | 风格 |
|---|---|---|
| **crisp-light** | 商务、医疗、教育、通用 | 简洁、极简、浅色背景 |
| **dark-signal** | 代码、数学、安全、DevOps | 大胆、高对比度、等宽输出 |

强调色根据模型领域自动选取（如医疗用青色、教育用琥珀色、代码用电光青）— 也可自行指定。

**查看在线示例：** [`demos/qwen2.5-0.5b-chip2-sft/index.html`](demos/qwen2.5-0.5b-chip2-sft/index.html) — 下载后在任意浏览器中打开。

---

## 本地部署

GGUF 导出完成后，如果系统中检测到 llama.cpp（在第 3 阶段检测），Agent 会提供一键本地部署：

```bash
python scripts/llamacpp.py deploy \
    --model outputs/model-f16.gguf --quant q4_k_m --bench --serve
```

将运行完整流水线：量化 → 基准测试 → 启动 OpenAI 兼容服务器 → 在浏览器中打开 Gaslamp Chat WebUI（`http://localhost:8081/`）。也可单独使用各子命令：

```bash
python scripts/llamacpp.py install              # 安装 llama.cpp（brew / cmake）
python scripts/llamacpp.py quantize --input model.gguf --types q4_k_m q8_0
python scripts/llamacpp.py bench --models model-q4_k_m.gguf
python scripts/llamacpp.py serve --model model-q4_k_m.gguf --port 8081
python scripts/llamacpp.py chat --model model-q4_k_m.gguf
```

需要安装 [llama.cpp](https://github.com/ggml-org/llama.cpp) — 可通过 `llamacpp.py install` 自动安装。

---

## Google Colab 云端训练

Apple Silicon 用户如需更大的模型或 CUDA 专属功能，可将训练卸载到免费 Colab GPU：

1. 在 Claude Code 中安装 `colab-mcp`：
   ```bash
   uv python install 3.13
   claude mcp add colab-mcp -- uvx --from git+https://github.com/googlecolab/colab-mcp --python 3.13 colab-mcp
   ```
2. 打开 Colab 笔记本，连接到 T4/L4 GPU 运行时
3. Agent 自动连接、安装 Unsloth，在后台线程中开始训练，并每 30 秒轮询指标
4. 训练完成后从 Colab 文件浏览器下载适配器

本地 mlx-tune 仍是默认选项 — Colab 为需要更多算力时的可选方案。

---

## Gaslamp

`unsloth-buddy` 可独立使用，也可作为 [Gaslamp](https://gaslamp.dev/) 大型项目的一部分运行 — Gaslamp 是一个协调从研究到训练再到部署整个 ML 生命周期的 Agentic 平台。通过 Gaslamp 调用时，项目目录和状态在各个技能之间共享，结果会自动传递到下一阶段。

每个项目还会生成一份 **`gaslamp.md` 路书** — 记录所有已确定的决策及其原因，并附带 📖 底层 ML 概念学习模块。任何 Agent 或人员都可以将此文件交给新会话，端到端复现整个项目，或用于理解每个决策背后的原因。

[gaslamp.dev/unsloth](https://gaslamp.dev/unsloth) — [gaslamp.dev](https://gaslamp.dev/)

---

## OpenClaw

**unsloth-buddy 是一个 [OpenClaw](https://github.com/openclaw/openclaw) 兼容技能。** 将仓库 URL 分享给 OpenClaw，描述你想微调的内容 — 它会读取 `AGENTS.md`，理解流程，并自动运行一切。

```
1. 将 https://github.com/TYH-labs/unsloth-buddy 分享给 OpenClaw
2. OpenClaw 读取 AGENTS.md → 理解 7 阶段微调生命周期
3. 说："在我的客户支持数据上微调一个模型"
4. 完成 — OpenClaw 运行访谈、格式化数据、训练、评估并导出
```

对于 Claude Code、Gemini CLI、Codex 或任何 ACP 兼容的 Agent：将 `AGENTS.md` 作为上下文提供，Agent 将自动引导相同的工作流程。

---

## 更新日志

- **2026-04-12** — 新增 **llama.cpp 本地部署**（第 6.5 阶段）：GGUF 导出完成后，若检测到 llama.cpp，Agent 将提供一键流水线 — 量化 → 基准测试 → 启动服务 + 打开 Gaslamp Chat WebUI（`templates/chat_ui.html`）。`scripts/llamacpp.py` 提供 7 个子命令（`install`、`quantize`、`bench`、`ppl`、`serve`、`chat`、`deploy`），在 Apple Silicon（Metal）和 NVIDIA 上自动启用 GPU 卸载。`scripts/detect_system.py` 现在也会检测 llama.cpp 二进制文件，若未安装则输出提示命令。
- **2026-04-10** — 新增 Apple Silicon 原生**视觉 SFT 支持**：集成 [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) 以支持在 M 系列芯片上进行多模态微调（如 Gemma 4 Vision、Qwen2.5-VL）。新增 `scripts/unsloth_mlx_vision_example.py` 训练模板和 `mlx_eval_vision_template.py` 视觉对比评估脚本。演示生成器现已支持宽屏 VLM 布局（`vlm-crisp`、`vlm-dark`）及 PNG 资源本地打包，确保多模态演示看板的离线可用性。
- **2026-04-09** — 演示生成器改进：自动将概念/电影类关键词（如 "matrix" → nvidia，"star wars" → spacex）解析并映射至最匹配的品牌；区分浅层与深层 DESIGN.md 覆盖 —— 深层覆盖（如全黑或局部大面积结构性分割布局）将跳过 CSS 注入点，从头开始写入演示文件。将 `scripts/search_design.py` 补充至技能脚本汇总，即使环境内无 `npx` 也可直接获取品牌设计模板。
- **2026-04-04** — 新增演示生成器（第 5.5 阶段）：评估完成后生成静态 HTML 演示页面，并排展示基础模型与微调模型输出。内置两套主题（crisp-light、dark-signal），支持基于领域的自动强调色。无需服务器 — 在任意浏览器中直接打开。属于 [Gaslamp](https://gaslamp.dev/) 展示工具的简化版本。访谈从 5 点合同简化为 2 个问题（任务 + 数据），同时捕获用户领域/受众以用于演示主题选择。
- **2026-03-22** — 新增 `gaslamp.md` 可复现路书：每个项目现在都会记录所有已确定的决策及其原因，并附带 📖 ML 概念学习模块（方法、模型、数据、超参数、评估、导出），任何 Agent 或人员均可端到端复现项目并理解每个决策背后的原因。模板文件位于 `templates/gaslamp_template.md`，由 `init_project.py` 自动生成。
- **2026-03-21** — 增强训练看板：任务感知面板（SFT/DPO/GRPO/Vision）、GPU 内存分解（基线 vs LoRA vs 总量）、GRPO 奖励 ± 标准差及 KL 散度图表、DPO 选中/拒绝奖励及 KL 图表、轮次追踪、训练完成摘要横幅、终端 DPO/GRPO 2×2 布局，以及新增 `scripts/demo_server.py` 无需 GPU 即可预览所有面板的模拟服务器。
- **2026-03-19** — 新增终端训练看板（`scripts/terminal_dashboard.py`）：在终端中实时显示 `plotext` 损失和学习率图表，支持 `--once` 模式供 Claude Code 一次性检查训练进度。
- **2026-03-18** — 新增通过 [colab-mcp](https://github.com/googlecolab/colab-mcp) 的 Google Colab 云端训练支持：可在 Claude Code 中直接使用免费 T4/L4/A100 GPU，支持后台线程训练、实时轮询进度及适配器下载流程。

---

## License

参见 `LICENSE.txt`。Unsloth 使用 MIT 许可证，mlx-tune 使用 MIT 许可证。
