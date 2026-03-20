# facenet-cli

以 [facenet-pytorch](https://github.com/timesler/facenet-pytorch) 為後端的 **Click** CLI，封裝常見人臉 embedding／比對流程，並可選 `--json` 輸出供自動化或 agent 使用。

## 專案架構

```
facenet-cli/
├── pyproject.toml          # 套件與依賴（與 setup.py 並列，建議以本檔為準）
├── setup.py                # setuptools 後備設定
├── cli_anything/
│   └── facenet/
│       ├── __init__.py
│       ├── __main__.py     # python -m cli_anything.facenet
│       ├── cli.py          # Click 群組：embedding（capture / find / images / compare）
│       ├── core/
│       │   └── recognition.py   # 業務流程（dry-run 與實際呼叫）
│       ├── utils/
│       │   ├── backend.py       # 呼叫內嵌腳本／執行流程
│       │   └── embedded_backend.py
│       └── tests/               # pytest
├── samples/                # 獨立範例腳本（直接 facenet_pytorch，非 CLI 套件）
│   ├── generate_embeddings.py
│   └── face_recognition.py
└── skills/
    └── SKILL.md            # Agent 技能說明（可選）
```

安裝後主程式入口為 **`fnet`**（定義於 `pyproject.toml` / `setup.py` 的 `console_scripts`）。亦可不經安裝，在專案根目錄以模組方式執行：

```bash
py -3.13 -m cli_anything.facenet --help
```

## 功能概覽

| 模式 | 說明 |
|------|------|
| 1 | 拍照取得 embedding，可選寫入 CSV |
| 2 | 拍照後與資料庫 CSV 比對，輸出人名與距離 |
| 3 | 批次讀取資料夾影像，輸出檔名 + embedding 至 CSV |
| 4 | 兩張影像與閾值比對，輸出距離與是否同人 |

## 依賴

- Python **3.13+**（`requires-python` 為 `>=3.13`）
- 安裝時一併安裝：`facenet-pytorch`、`opencv-python`、`pillow`、`numpy`、`matplotlib`、`click`、`prompt-toolkit`

## 安裝

在專案根目錄（`facenet-cli`）執行：

```bash
py -3.13 -m pip install -e .
```

使用 `uv`（推薦）：

```bash
uv venv .venv --python 3.13
uv pip install --python .venv\Scripts\python.exe -e .
```

以 **uv tool** 安裝成可全域呼叫的指令（套件名稱為 `facenet-cli`）：

```bash
uv tool install --from . facenet-cli
```

不使用 venv、直接裝到目前 Python：

```bash
uv pip install --system -e .
```

> 需 Python **3.13 或以上**，請以符合版本的解譯器執行 `pip` / `uv`。

## 指令說明與範例

全域或 venv 啟用後，主指令為 **`fnet`**。`--workspace` 預設為目前目錄，指向 facenet 相關工作目錄（依 `utils` 內邏輯而定）；在專案根目錄開發時常用 `--workspace .`。

先列出 `embedding` 子命令說明：

```bash
fnet --json embedding help
```

### 1) 拍照取得 embedding（可選 CSV）

```bash
fnet --json --workspace . embedding capture
fnet --json --workspace . embedding capture --output-csv single_embedding.csv
```

輸出示例（節錄）：

```json
{
  "ok": true,
  "payload": {
    "dimension": 512,
    "output_csv": "C:/.../single_embedding.csv"
  }
}
```

### 2) 拍照辨識：與資料庫 CSV 比對

```bash
fnet --json --workspace . embedding find --database-csv face_embeddings_database.csv
```

輸出示例（節錄）：

```json
{
  "ok": true,
  "payload": {
    "name": "ruby_lin",
    "distance": 0.73,
    "threshold": 1.0,
    "is_same_person": true
  }
}
```

### 3) 資料夾批次：檔名 + embedding → CSV

```bash
fnet --json --workspace . embedding images --input-folder face_images --output-csv batch_result.csv
```

### 4) 兩張圖比對

```bash
fnet --json embedding compare --image1 ./img1.jpg --image2 ./img2.jpg --threshold 1.2
```

輸出示例（節錄）：

```json
{
  "ok": true,
  "payload": {
    "distance": 0.84,
    "threshold": 1.2,
    "is_same_person": true
  }
}
```

## Dry-run

不開攝影機、不讀實際影像，僅驗證參數與流程：

```bash
fnet --json embedding capture --dry-run
fnet --json embedding find --dry-run
fnet --json embedding images --input-folder face_images --output-csv batch.csv --dry-run
fnet --json embedding compare --image1 a.jpg --image2 b.jpg --threshold 1.2 --dry-run
```

## 測試

需已安裝 `pytest`（若未列入專案依賴，請自行 `pip install pytest`）：

```bash
pytest -v --tb=no cli_anything/facenet/tests/
```

或使用 `uv`：

```bash
uv run pytest -v --tb=no cli_anything/facenet/tests/
```

強制走「已安裝的 CLI」路徑的端對端測試：

```bash
set CLI_ANYTHING_FORCE_INSTALLED=1
pytest -v -s --tb=no cli_anything/facenet/tests/test_full_e2e.py
```

（PowerShell 請改為：`$env:CLI_ANYTHING_FORCE_INSTALLED=1`）

## samples 目錄

`samples/` 內為獨立腳本（批次產生 embedding CSV、即時攝影機辨識等），直接依賴 `facenet_pytorch`，與 `fnet` CLI 套件分開維護；執行前請自行調整路徑與參數。
