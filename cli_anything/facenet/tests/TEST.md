# TEST PLAN

## Test Inventory Plan

- `test_core.py`: 3 unit tests planned
- `test_full_e2e.py`: 4 E2E/subprocess tests planned

## Unit Test Plan

- `recognition.py` (core module name)
  - `capture_embedding_once(..., dry_run=True)`
  - `recognize_match_name(..., dry_run=True)`
  - `recognize_folder_images_to_csv(..., dry_run=True)`
  - `compare_images(..., dry_run=True)`

## E2E Test Plan

- 以 subprocess 呼叫已安裝 CLI command
- 驗證 `--help`
- 驗證三種調用命令之 `--dry-run` 鏈路

## Realistic Workflow Scenarios

- **Workflow name**: Four-case embedding flow
  - **Simulates**: 拍照擷取 embedding、拍照比對、資料夾批次 embedding、雙圖比對
  - **Operations chained**:
    - `embedding capture --dry-run`
    - `embedding find --dry-run`
    - `embedding images --input-folder ... --output-csv ... --dry-run`
    - `embedding compare --image1 ... --image2 ... --threshold ... --dry-run`
  - **Verified**: 指令返回 JSON 結構且 `ok=true`

---

# TEST RESULTS

```text
============================= test session starts =============================
platform win32 -- Python 3.13.0, pytest-9.0.2, pluggy-1.6.0 -- C:\Python\Python313\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\mz038\Desktop\testtest\facenet-pytorch\facenet-cli
configfile: pyproject.toml
plugins: anyio-4.12.1, langsmith-0.6.6, typeguard-4.5.1
collecting ... collected 7 items

cli_anything/facenet/tests/test_core.py::test_capture_embedding_dry_run_payload PASSED [ 14%]
cli_anything/facenet/tests/test_core.py::test_recognize_match_name_dry_run_payload PASSED [ 28%]
cli_anything/facenet/tests/test_core.py::test_recognize_folder_to_csv_dry_run_payload PASSED [ 36%]
cli_anything/facenet/tests/test_core.py::test_compare_images_dry_run_payload PASSED [ 45%]
cli_anything/facenet/tests/test_full_e2e.py::TestCLISubprocess::test_help PASSED [ 54%]
cli_anything/facenet/tests/test_full_e2e.py::TestCLISubprocess::test_recognition_help_json PASSED [ 63%]
cli_anything/facenet/tests/test_full_e2e.py::TestCLISubprocess::test_capture_embedding_dry_run PASSED [ 72%]
cli_anything/facenet/tests/test_full_e2e.py::TestCLISubprocess::test_match_name_dry_run PASSED [ 81%]
cli_anything/facenet/tests/test_full_e2e.py::TestCLISubprocess::test_folder_to_csv_dry_run PASSED [ 90%]
cli_anything/facenet/tests/test_full_e2e.py::TestCLISubprocess::test_compare_images_dry_run PASSED [100%]

============================== 7 passed in 0.57s ==============================
```

## Summary Statistics

- Total tests: 7
- Pass rate: 100%
- Execution time: 0.57s

## Coverage Notes

- 已覆蓋三種調用命令在 `dry-run` 模式下的 core 與 subprocess 鏈路。
- 尚未在自動化測試中直接啟動真實攝影機辨識（避免無攝影機環境失敗）。
