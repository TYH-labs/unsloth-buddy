# Gaslamp 展廊 — 微调示例

一系列真实的 `gaslamp.md` 路书——每一份都是完整、可复现的微调运行记录。

每个示例只有一个文件。将它交给运行 [unsloth-buddy](https://gaslamp.dev/unsloth) 的 Agent，整个项目即可端到端复现：数据下载、环境配置、训练和评估。

```
/unsloth-buddy reproduce using demos/qwen2.5-0.5b-chip2-sft/gaslamp.md
```

---

## 展廊

| 示例 | 方法 | 模型 | 数据集 | 硬件 | 产出 |
|------|------|------|--------|------|------|
| [qwen2.5-0.5b-chip2-sft](./qwen2.5-0.5b-chip2-sft/gaslamp.md) | SFT | Qwen2.5-0.5B-Instruct | OIG unified_chip2（20 万对话） | Apple Silicon | 轻量指令跟随模型，M 系芯片约 3 分钟完成训练 |

---

## 什么是 gaslamp.md？

`gaslamp.md` 是一份**可复现路书**——不是日志，不是 README，不是 Notebook。它记录了微调过程中每一个已做出并保留的决策，包括：

- 每项选择背后的理由
- 📖 学习模块：解释底层 ML 概念及其与替代方案的权衡
- 精确的数据格式、解析逻辑和划分参数
- 完整的 LoRA 配置和超参数，每个非默认值均附一行说明
- 文件清单中的"来源"列，明确哪些文件需要复制、哪些需要重新生成、哪些需要从头编写
- 导出章节中的完整"加载+生成"代码片段，可端到端验证模型是否正常工作

**设计验证标准：** 一个从未见过原始项目的全新 Agent，仅凭 `gaslamp.md` 和已安装的技能，无需访问原始会话或项目文件，即可完整复现整个训练流程——从原始数据集下载到评估。本展廊中的每个示例均经过此验证。

---

## 如何复现示例

1. 确认已安装 unsloth-buddy：
   ```
   /install-plugin https://github.com/TYH-labs/unsloth-buddy
   ```

2. 指向路书文件：
   ```
   /unsloth-buddy reproduce using demos/qwen2.5-0.5b-chip2-sft/gaslamp.md
   ```

Agent 将读取路书、检测你的硬件，并在新的带日期目录中重建整个项目。

---

## 如何贡献示例

贡献内容只需一个 `gaslamp.md` 文件。

1. 使用 unsloth-buddy 完成一次微调（任何方法：SFT、DPO、GRPO、视觉）
2. 验证仅凭路书即可干净地复现
3. 移除所有个人或机器相关路径（使用 `.venv/` 而非 `/Users/yourname/...`）
4. 提交 PR，添加 `demos/<描述性名称>/gaslamp.md`

建议使用 `{模型}-{数据集}-{方法}` 的命名模式，例如：
- `llama3-openhermes-dpo`
- `phi3-mini-gsm8k-grpo`
- `qwen2.5-vl-chartqa-vision-sft`

每个示例应覆盖不同的模型系列、数据集或训练方法，使展廊保持多样性和实用性。
