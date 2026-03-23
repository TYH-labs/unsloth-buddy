# Gaslamp 展廊 — 微調範例

一系列真實的 `gaslamp.md` 路書——每一份都是完整、可重現的微調執行記錄。

每個範例只有一個檔案。將它交給執行 [unsloth-buddy](https://gaslamp.dev/unsloth) 的 Agent，整個專案即可端到端重現：資料下載、環境設定、訓練與評估。

```
/unsloth-buddy reproduce using demos/qwen2.5-0.5b-chip2-sft/gaslamp.md
```

---

## 展廊

| 範例 | 方法 | 模型 | 資料集 | 硬體 | 產出 |
|------|------|------|--------|------|------|
| [qwen2.5-0.5b-chip2-sft](./qwen2.5-0.5b-chip2-sft/gaslamp.md) | SFT | Qwen2.5-0.5B-Instruct | OIG unified_chip2（20 萬對話） | Apple Silicon | 輕量指令遵循模型，M 系晶片約 3 分鐘完成訓練 |

---

## 什麼是 gaslamp.md？

`gaslamp.md` 是一份**可重現路書**——不是日誌，不是 README，不是 Notebook。它記錄了微調過程中每一個已做出並保留的決策，包含：

- 每項選擇背後的理由
- 📖 學習模組：解說底層 ML 概念及其與替代方案的取捨
- 精確的資料格式、解析邏輯與劃分參數
- 完整的 LoRA 設定和超參數，每個非預設值均附一行說明
- 檔案清單中的「來源」欄，明確標示哪些檔案需要複製、哪些需要重新產生、哪些需要從零撰寫
- 匯出章節中的完整「載入＋生成」程式碼片段，可端到端驗證模型是否正常運作

**設計驗證標準：** 一個從未見過原始專案的全新 Agent，僅憑 `gaslamp.md` 和已安裝的技能，無需存取原始工作階段或專案檔案，即可完整重現整個訓練流程——從原始資料集下載到評估。本展廊中的每個範例均經過此驗證。

---

## 如何重現範例

1. 確認已安裝 unsloth-buddy：
   ```
   /install-plugin https://github.com/TYH-labs/unsloth-buddy
   ```

2. 指向路書檔案：
   ```
   /unsloth-buddy reproduce using demos/qwen2.5-0.5b-chip2-sft/gaslamp.md
   ```

Agent 將讀取路書、偵測你的硬體，並在新的含日期目錄中重建整個專案。

---

## 如何貢獻範例

貢獻內容只需一個 `gaslamp.md` 檔案。

1. 使用 unsloth-buddy 完成一次微調（任何方法：SFT、DPO、GRPO、視覺）
2. 驗證僅憑路書即可乾淨地重現
3. 移除所有個人或機器相關路徑（使用 `.venv/` 而非 `/Users/yourname/...`）
4. 提交 PR，新增 `demos/<描述性名稱>/gaslamp.md`

建議使用 `{模型}-{資料集}-{方法}` 的命名模式，例如：
- `llama3-openhermes-dpo`
- `phi3-mini-gsm8k-grpo`
- `qwen2.5-vl-chartqa-vision-sft`

每個範例應涵蓋不同的模型系列、資料集或訓練方法，使展廊保持多樣性與實用性。
