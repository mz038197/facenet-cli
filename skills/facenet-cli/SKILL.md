---
name: "cli-anything-facenet-pytorch"
description: "使用內建 facenet-pytorch backend 進行 embeddings 建庫與拍照辨識的 CLI"
---

# cli-anything-facenet-pytorch

## 何時使用

- 需要以命令列方式進行拍照/資料夾 embedding 輸出與比對
- 需要 agent 可解析的 JSON 輸出

## 主要命令

- `embedding capture [--output-csv <path>]`：模式1，拍照後輸出 embedding
- `embedding help`：列出所有 embedding 子命令與範例
- `embedding find [--database-csv <path>]`：模式2，拍照比對後輸出姓名
- `embedding images --input-folder <path> --output-csv <path>`：模式3，批次讀取資料夾影像輸出 `filename + embedding` CSV
- `embedding compare --image1 <path> --image2 <path> --threshold <float>`：模式4，輸出兩張圖的距離與是否同人

## Agent 指引

- 盡量加上 `--json` 讓輸出可程式化解析
- 拍照相關命令都支援 `--dry-run` 做流程驗證
- 若拍照流程失敗，先檢查 facenet/openCV 依賴與攝影機權限
