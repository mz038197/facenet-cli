---
name: facenet-cli
description: >-
  Operates the facenet-cli Click tool (fnet) for face embeddings and 1:1 or 1:N matching via facenet-pytorch,
  with optional JSON output for automation. Use when the user works in the facenet-cli repo, runs fnet / embedding
  commands, needs camera capture or CSV face databases, or mentions facenet-pytorch CLI harness.
---

# facenet-cli（`fnet`）

本專案為 **facenet-pytorch** 的命令列封裝，主指令為 **`fnet`**。亦可以模組執行：`uv run python -m cli_anything.facenet`（需 Python **≥3.13**，見 `pyproject.toml`）。儲存庫：<https://github.com/mz038197/facenet-cli>

## 安裝

建議使用 **[uv](https://docs.astral.sh/uv/)**。從 GitHub 直接安裝到目前環境：

```bash
uv pip install git+https://github.com/mz038197/facenet-cli.git
```

全域安裝 `fnet` 指令（可選）：

```bash
uv tool install git+https://github.com/mz038197/facenet-cli.git
```

本機 clone 後開發：`uv venv --python 3.13` → `uv sync`（見專案 `README.md`）。

## 何時套用此技能

- 使用者要在終端機做**人臉 embedding、建庫、拍照比對、兩張圖比對**
- 使用者提到 **`fnet`**、`facenet-cli`、`embedding capture/find/images/compare`
- 需要 **可機讀輸出**（JSON）給腳本或 agent 解析
- 在 **本倉庫**內除錯 CLI、測試或撰寫文件

與 **`samples/`** 目錄的差異：`samples/*.py` 為**獨立範例腳本**（直接呼叫 `facenet_pytorch`），**不是** `fnet` 子命令；若使用者要「用專案內建 CLI」，請引導 `fnet`，不要混用範例檔名當成指令。

## 全域選項

| 選項 | 說明 |
|------|------|
| `--json` | 將結果以 JSON 印出（**建議 agent 永遠加上**） |
| `--workspace <路徑>` | 工作目錄，預設 `.`；實際行為見 `cli_anything/facenet/utils/` |

## 子命令群組：`embedding`

| 子命令 | 用途 | 常用參數 |
|--------|------|----------|
| `help` | 列出子命令與範例字串 | — |
| `capture` | 從攝影機擷取一次 embedding，可寫 CSV | `--output-csv`、`--dry-run` |
| `find` | 擷取後與資料庫 CSV 比對，輸出姓名／距離 | `--database-csv`（預設 `face_embeddings_database.csv`）、`--threshold`（預設 `1.0`）、`--dry-run` |
| `images` | 批次資料夾影像 → 輸出 **檔名 + embedding** 的 CSV | `--input-folder`、`--output-csv`（必填）、`--dry-run` |
| `compare` | 兩張圖與閾值，輸出距離與是否同人 | `--image1`、`--image2`、`--threshold`（必填）、`--dry-run` |

含攝影機或讀圖的流程皆可加 **`--dry-run`**：不開相機、不讀實檔，只驗證參數與流程。

## 指令範例（複製即用）

列出說明（JSON）：

```bash
fnet --json embedding help
```

拍照 embedding、可選 CSV：

```bash
fnet --json --workspace . embedding capture
fnet --json --workspace . embedding capture --output-csv single_embedding.csv
```

拍照與資料庫比對：

```bash
fnet --json --workspace . embedding find --database-csv face_embeddings_database.csv --threshold 1.0
```

資料夾批次：

```bash
fnet --json --workspace . embedding images --input-folder face_images --output-csv batch_result.csv
```

兩張圖比對：

```bash
fnet --json embedding compare --image1 ./a.jpg --image2 ./b.jpg --threshold 1.2
```

Dry-run（不碰相機／實圖）：

```bash
fnet --json embedding capture --dry-run
fnet --json embedding find --dry-run
fnet --json embedding images --input-folder face_images --output-csv out.csv --dry-run
fnet --json embedding compare --image1 a.jpg --image2 b.jpg --threshold 1.2 --dry-run
```

## JSON 輸出與解析

- 加上 `--json` 時，stdout 為 **單一 JSON 物件**（`ensure_ascii=False`，可能含中文）。
- `embedding help` 固定含 `"ok": true`、`"group": "embedding"`、`"commands": [...]`。
- 其餘子命令回傳結構由 **`cli_anything/facenet/core/recognition.py`** 與 backend 決定；常見欄位包含 `ok`、錯誤時的訊息、成功時的 `payload`（例如距離、姓名、`is_same_person` 等）。
- Agent 應 **只解析 stdout**；若需除錯再請使用者提供 stderr 或終端機完整輸出。

## 測試（給 agent／維護者）

```bash
uv run pytest -v --tb=no cli_anything/facenet/tests/
```

若需強制走「已安裝的 `fnet` 可執行檔」路徑（而非 `python -m`）：

- **cmd**：`set CLI_ANYTHING_FORCE_INSTALLED=1`
- **PowerShell**：`$env:CLI_ANYTHING_FORCE_INSTALLED=1`

然後執行 `uv run pytest -v -s --tb=no cli_anything/facenet/tests/test_full_e2e.py`。

## 疑難排解指引

| 狀況 | 建議 |
|------|------|
| 找不到 `fnet` | `uv pip install git+https://github.com/mz038197/facenet-cli.git` 或 `uv tool install git+https://github.com/mz038197/facenet-cli.git`；本機開發用 `uv pip install -e .`；或改用 `uv run python -m cli_anything.facenet` |
| 拍照失敗 | 檢查攝影機權限、裝置索引、OpenCV／驅動；先用 `embedding capture --dry-run` |
| 依賴／建置錯誤（如 Pillow） | 專案要求 Python ≥3.13；鎖檔為 `uv.lock`，請用相容版本的 Pillow／wheel，必要時調整 `pyproject.toml` 後 `uv lock` |
| 比對找不到人名 | 確認 `--database-csv` 路徑與內容格式與後端預期一致 |

## 程式碼導覽（必要時再讀檔）

- 入口：`cli_anything/facenet/cli.py`（Click 定義）
- 流程：`cli_anything/facenet/core/recognition.py`
- 執行後端：`cli_anything/facenet/utils/backend.py`、`embedded_backend.py`
