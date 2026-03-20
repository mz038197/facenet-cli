import json
import os
import shutil
import subprocess
import sys


def _resolve_cli(name):
    """預設用當前 Python 的 -m 執行 CLI，避免 PATH 上指向過期/錯誤路徑的 console script。

    若要強制測試已安裝的指令檔，請設定環境變數：CLI_ANYTHING_FORCE_INSTALLED=1
    """
    force = os.environ.get("CLI_ANYTHING_FORCE_INSTALLED", "").strip() == "1"
    if force:
        path = shutil.which(name)
        if not path:
            raise RuntimeError(
                f"{name} 未在 PATH 找到。請先安裝：pip install -e .（於 facenet-cli 目錄）"
            )
        print(f"[_resolve_cli] Using installed command: {path}")
        return [path]
    module = "cli_anything.facenet.cli"
    print(f"[_resolve_cli] Using: {sys.executable} -m {module}")
    return [sys.executable, "-m", module]


class TestCLISubprocess:
    CLI_BASE = _resolve_cli("fnet")

    def _run(self, args, check=True):
        return subprocess.run(
            self.CLI_BASE + args,
            capture_output=True,
            text=True,
            check=check,
        )

    def test_help(self):
        r = self._run(["--help"])
        assert r.returncode == 0
        assert "embedding" in r.stdout

    def test_embedding_help_json(self):
        r = self._run(["--json", "embedding", "help"])
        assert r.returncode == 0
        payload = json.loads(r.stdout)
        assert payload["ok"] is True
        assert payload["group"] == "embedding"
        assert any(item["command"] == "compare" for item in payload["commands"])

    def test_capture_embedding_dry_run(self):
        r = self._run(["--json", "--workspace", "..", "embedding", "capture", "--dry-run"])
        assert r.returncode == 0
        payload = json.loads(r.stdout)
        assert payload["ok"] is True

    def test_match_name_dry_run(self):
        r = self._run(["--json", "--workspace", "..", "embedding", "find", "--dry-run"])
        assert r.returncode == 0
        payload = json.loads(r.stdout)
        assert payload["ok"] is True

    def test_folder_to_csv_dry_run(self):
        r = self._run(
            [
                "--json",
                "--workspace",
                "..",
                "embedding",
                "images",
                "--input-folder",
                "face_images",
                "--output-csv",
                "batch_result.csv",
                "--dry-run",
            ]
        )
        assert r.returncode == 0
        payload = json.loads(r.stdout)
        assert payload["ok"] is True

    def test_compare_images_dry_run(self):
        r = self._run(
            [
                "--json",
                "--workspace",
                "..",
                "embedding",
                "compare",
                "--image1",
                "a.jpg",
                "--image2",
                "b.jpg",
                "--threshold",
                "1.2",
                "--dry-run",
            ]
        )
        assert r.returncode == 0
        payload = json.loads(r.stdout)
        assert payload["ok"] is True
