# unsloth-buddy

<p align="center"><img src="images/unsloth_gaslamp.png" width="75%" alt="unsloth-buddy" /></p>

<p align="center">
  <a href="https://github.com/TYH-labs/unsloth-buddy"><img src="https://img.shields.io/github/stars/TYH-labs/unsloth-buddy?style=flat&logo=github&color=181717&logoColor=white" alt="GitHub" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License" /></a>
  <a href="#快速開始"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="https://gaslamp.dev/unsloth"><img src="https://img.shields.io/badge/%F0%9F%94%A5%20Gaslamp-Compatible-ff6b00?logoColor=white" alt="Gaslamp Compatible" /></a>
  <a href="#openclaw"><img src="https://img.shields.io/badge/%F0%9F%A6%9E%20OpenClaw-Compatible-ff4444" alt="OpenClaw Compatible" /></a>
  <a href="#快速開始"><img src="https://img.shields.io/badge/%F0%9F%A4%96%20Agent-Claude%20Code%20%2F%20Codex%20%2F%20Gemini-8b5cf6" alt="Agent Compatible" /></a>
  <a href="https://discord.gg/mZe4mbCQ6a"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="Discord" /></a>
</p>

<p align="center"><code>/unsloth-buddy 我有 500 份醫病問診紀錄，想訓練一個能自動摘要的模型，我只有一台 MacBook Air</code></p>

<p align="center">
  <a href="#快速開始"><img src="https://img.shields.io/badge/立即體驗-1 分鐘-black?style=for-the-badge" alt="立即體驗" /></a>
  <a href="demos/"><img src="https://img.shields.io/badge/示範案例-範例-6e40c9?style=for-the-badge" alt="示範案例" /></a>
  <a href="SKILL.md"><img src="https://img.shields.io/badge/10%2B%20功能-功能詳情-0969da?style=for-the-badge" alt="10+ 功能" /></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh-Hans.md">简体中文</a> | 繁體中文
</p>

---

## 這是什麼？

一個像同事一樣對話的微調 Agent。描述你的需求，它會提出合適的問題、找到或整理你的資料、選擇合適的技術和模型、在你的硬體上訓練、驗證效果，並打包部署。

支援透過 [Unsloth](https://github.com/unslothai/unsloth) 在 NVIDIA GPU 上執行，也支援透過 [mlx-tune](https://github.com/ml-explore/mlx-lm) 在 Apple Silicon 上原生執行。隸屬於 [Gaslamp](https://gaslamp.dev/) AI 開發平台 — [文件](https://gaslamp.dev/unsloth)。

---

## 一句話，一個微調模型。

```
你：在我的客戶支援 FAQ 上微調一個小模型，我有一個 CSV 檔案。

[第 0 階段] 建立專案：customer_faq_sft_2026_03_17/
[第 1 階段] 需求訪談...
           方法：SFT   模型：Qwen2.5-0.5B   部署：Ollama
[第 2 階段] 資料策略...
           從 faq.csv 載入 1,200 列 → 重新格式化為對話訊息
           儲存至 data/train.jsonl（已驗證：messages 欄位 ✓）
[第 3 階段] 環境：Apple M4 24GB，mlx-tune 0.4.3，Python 3.12
           準備就緒，可以開始訓練
[第 4 階段] 訓練... 200 步
           最終損失：1.42 → 已儲存至 outputs/adapters/
[第 5 階段] 評估（基礎模型 vs 微調模型，貪婪解碼）：
           問：如何重置我的密碼？
           [基礎]  我可以協助您。是哪種密碼？
           [微調]  前往登入頁面 → 點擊「忘記密碼」→ 查看電子郵件。
[第 6 階段] 匯出 → outputs/model-q4_k_m.gguf
           執行：ollama create my-faq-bot -f Modelfile && ollama run my-faq-bot
```

一次對話，七個階段，最終得到一個可部署的模型。

---

## 快速開始

該技能包含子技能和工具腳本 — 請安裝完整儲存庫，而非單一檔案。

**Claude Code** *(推薦)*
```
/plugin marketplace add TYH-labs/unsloth-buddy
/plugin install unsloth-buddy@TYH-labs/unsloth-buddy
```
然後描述你想微調什麼。技能會自動啟動。

**Gemini CLI**
```bash
gemini extensions install https://github.com/TYH-labs/unsloth-buddy --consent
```

**任何支援 [Agent Skills](https://agentskills.io/) 標準的 Agent**
```bash
git clone https://github.com/TYH-labs/unsloth-buddy.git .agents/skills/unsloth-buddy
```

---

## 有何不同？

大多數工具假設你已經知道該怎麼做，而這個工具不會。

| 你的顧慮 | 實際發生的事 |
|---|---|
| **「我不知道從哪裡開始」** | 5 點訪談鎖定方法、模型、資料、硬體和部署目標，然後再撰寫任何程式碼 |
| **「我沒有資料，或者格式不對」** | 專門的資料階段負責取得、生成或重新格式化資料，精確符合訓練器所需的 schema |
| **「SFT？DPO？GRPO？選哪個？」** | 將你的目標對應到正確的技術，並用淺顯易懂的語言解釋原因 |
| **「選哪個模型？能裝進我的 GPU 嗎？」** | 偵測你的硬體，對應到可用的模型大小，必要時估算雲端成本 |
| **「Unsloth 在我的機器上安裝不了」** | 兩階段環境偵測捕獲不相符問題，並印出適合你環境的精確安裝指令 |
| **「我訓練好了，但它有效嗎？」** | 將微調適配器與基礎模型並排執行，讓你看到差異，而不只是一個損失數值 |
| **「怎麼部署？」** | 你指定目標（Ollama、vLLM、HF Hub）— 它執行轉換指令 |

---

## 運作原理

七個階段，每個階段都限定在一個獨立的含日期專案目錄中，不會影響你的儲存庫根目錄。

| 階段 | 發生的事 | 產出檔案 |
|---|---|---|
| **0. 初始化** | 建立 `{name}_{date}/` 標準目錄結構 | `progress_log.md` |
| **1. 訪談** | 5 點 Unsloth 合約 — 方法、模型、資料、硬體、部署 | `project_brief.md` |
| **2. 資料** | 取得、驗證並格式化為訓練器 schema | `data_strategy.md` |
| **3. 環境** | 硬體掃描 → Python 環境檢查 → 阻塞直到就緒 | `detect_env_result.json` |
| **4. 訓練** | 生成並執行 `train.py`，串流輸出至日誌 | `outputs/adapters/` |
| **5. 評估** | 批量測試、互動式 REPL、基礎模型對比微調模型 | `logs/eval.log` |
| **6. 匯出** | GGUF、合併 16-bit 或 Hub 推送 | `outputs/` |

```
customer_faq_sft_2026_03_17/
├── train.py              eval.py
├── data/                 outputs/adapters/
├── logs/
├── project_brief.md      data_strategy.md
├── memory.md             progress_log.md
```

---

## 硬體支援

| 硬體 | 後端 | 能跑什麼 |
|---|---|---|
| NVIDIA T4 (16 GB) | `unsloth` | 7B QLoRA，小規模 GRPO |
| NVIDIA A100 (80 GB) | `unsloth` | 70B QLoRA，14B LoRA 16-bit |
| Apple M1 / M2 / M3 / M4 | `mlx-tune` | 10 GB 統一記憶體跑 7B，24 GB 跑 13B |
| Google Colab (T4/L4/A100) | `unsloth` 透過 `colab-mcp` | 免費雲端 GPU，可選接入 |

Unsloth 相比標準 HuggingFace 訓練速度快約 2 倍，VRAM 使用量減少高達 80%，且使用精確梯度。

**支援的訓練方法：** SFT、DPO、GRPO、ORPO、KTO、SimPO、視覺 SFT（Qwen2.5-VL、Llama 3.2 Vision、Gemma 3）

---

## 即時訓練儀表板

每次本地訓練都會自動在 **http://localhost:8080/** 開啟即時儀表板：

- **SSE 串流** — 透過 `EventSource` 即時推送更新，無輪詢延遲
- **EMA 平滑損失** — 清晰的趨勢線覆蓋雜訊原始損失，附帶滾動均值
- **動態階段徽章** — 閒置 → 訓練中 → 已完成 / 錯誤
- **ETA 與已用時間** — 根據步驟進度估算剩餘時間
- **梯度範數** — 有資料時自動顯示
- **評估指標** — 帶動畫空狀態的評估損失/準確率
- **峰值 VRAM** — 追蹤 GPU（CUDA）和 Apple MPS 記憶體用量

同時支援 NVIDIA（透過 `GaslampDashboardCallback`）和 Apple Silicon（透過 `MlxGaslampDashboard` 標準輸出攔截器）。

---

## Google Colab 雲端訓練

Apple Silicon 使用者如需更大的模型或 CUDA 專屬功能，可將訓練卸載到免費 Colab GPU：

1. 在 Claude Code 中安裝 `colab-mcp`：
   ```bash
   uv python install 3.13
   claude mcp add colab-mcp -- uvx --from git+https://github.com/googlecolab/colab-mcp --python 3.13 colab-mcp
   ```
2. 開啟 Colab 筆記本，連接到 T4/L4 GPU 執行階段
3. Agent 自動連接、安裝 Unsloth，在背景執行緒中開始訓練，並每 30 秒輪詢指標
4. 訓練完成後從 Colab 檔案瀏覽器下載適配器

本地 mlx-tune 仍是預設選項 — Colab 為需要更多算力時的選擇。

---

## Gaslamp

`unsloth-buddy` 可獨立使用，也可作為 [Gaslamp](https://gaslamp.dev/) 大型專案的一部分執行 — Gaslamp 是一個協調從研究到訓練再到部署整個 ML 生命週期的 Agentic 平台。透過 Gaslamp 呼叫時，專案目錄和狀態在各個技能之間共享，結果會自動傳遞到下一階段。

[gaslamp.dev/unsloth](https://gaslamp.dev/unsloth) — [gaslamp.dev](https://gaslamp.dev/)

---

## OpenClaw

**unsloth-buddy 是一個 [OpenClaw](https://github.com/openclaw/openclaw) 相容技能。** 將儲存庫 URL 分享給 OpenClaw，描述你想微調的內容 — 它會讀取 `AGENTS.md`，理解流程，並自動執行一切。

```
1. 將 https://github.com/TYH-labs/unsloth-buddy 分享給 OpenClaw
2. OpenClaw 讀取 AGENTS.md → 理解 7 階段微調生命週期
3. 說：「在我的客戶支援資料上微調一個模型」
4. 完成 — OpenClaw 執行訪談、格式化資料、訓練、評估並匯出
```

對於 Claude Code、Gemini CLI、Codex 或任何 ACP 相容的 Agent：將 `AGENTS.md` 作為上下文提供，Agent 將自動引導相同的工作流程。

---

## 更新日誌

- **2026-03-18** — 新增透過 [colab-mcp](https://github.com/googlecolab/colab-mcp) 的 Google Colab 雲端訓練支援：可在 Claude Code 中直接使用免費 T4/L4/A100 GPU，支援背景執行緒訓練、即時輪詢進度及適配器下載流程。

---

## License

參見 `LICENSE.txt`。Unsloth 使用 MIT 授權條款，mlx-tune 使用 MIT 授權條款。
