import io
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from cli_anything.facenet.utils.embedded_backend import (
    capture_embedding,
    compare_two_images,
    generate_embeddings,
    match_once,
    recognize_folder_to_csv,
    run_realtime_recognition,
)

def run_generate_embeddings(workspace: str) -> dict[str, Any]:
    out = io.StringIO()
    err = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            payload = generate_embeddings(workspace=workspace)
        return {
            "ok": True,
            "returncode": 0,
            "stdout": out.getvalue(),
            "stderr": err.getvalue(),
            "script": "embedded.generate_embeddings",
            "payload": payload,
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": 1,
            "stdout": out.getvalue(),
            "stderr": f"{err.getvalue()}\n{exc}".strip(),
            "script": "embedded.generate_embeddings",
        }


def run_face_recognition(workspace: str) -> dict[str, Any]:
    out = io.StringIO()
    err = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            payload = run_realtime_recognition(workspace=workspace)
        return {
            "ok": True,
            "returncode": 0,
            "stdout": out.getvalue(),
            "stderr": err.getvalue(),
            "script": "embedded.face_recognition",
            "payload": payload,
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": 1,
            "stdout": out.getvalue(),
            "stderr": f"{err.getvalue()}\n{exc}".strip(),
            "script": "embedded.face_recognition",
        }


def run_capture_embedding(workspace: str, output_csv: str | None = None) -> dict[str, Any]:
    out = io.StringIO()
    err = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            payload = capture_embedding(workspace=workspace, output_csv=output_csv)
        return {
            "ok": True,
            "returncode": 0,
            "stdout": out.getvalue(),
            "stderr": err.getvalue(),
            "script": "embedded.capture_embedding",
            "payload": payload,
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": 1,
            "stdout": out.getvalue(),
            "stderr": f"{err.getvalue()}\n{exc}".strip(),
            "script": "embedded.capture_embedding",
        }


def run_match_once(workspace: str, database_csv: str, threshold: float) -> dict[str, Any]:
    out = io.StringIO()
    err = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            payload = match_once(workspace=workspace, database_csv=database_csv, threshold=threshold)
        return {
            "ok": True,
            "returncode": 0,
            "stdout": out.getvalue(),
            "stderr": err.getvalue(),
            "script": "embedded.match_once",
            "payload": payload,
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": 1,
            "stdout": out.getvalue(),
            "stderr": f"{err.getvalue()}\n{exc}".strip(),
            "script": "embedded.match_once",
        }


def run_recognize_folder_to_csv(
    workspace: str,
    input_folder: str,
    output_csv: str,
    database_csv: str,
    threshold: float,
) -> dict[str, Any]:
    out = io.StringIO()
    err = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            payload = recognize_folder_to_csv(
                workspace=workspace,
                input_folder=input_folder,
                output_csv=output_csv,
                database_csv=database_csv,
                threshold=threshold,
            )
        return {
            "ok": True,
            "returncode": 0,
            "stdout": out.getvalue(),
            "stderr": err.getvalue(),
            "script": "embedded.recognize_folder_to_csv",
            "payload": payload,
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": 1,
            "stdout": out.getvalue(),
            "stderr": f"{err.getvalue()}\n{exc}".strip(),
            "script": "embedded.recognize_folder_to_csv",
        }


def run_compare_two_images(image_path_1: str, image_path_2: str, threshold: float) -> dict[str, Any]:
    out = io.StringIO()
    err = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            payload = compare_two_images(image_path_1=image_path_1, image_path_2=image_path_2, threshold=threshold)
        return {
            "ok": True,
            "returncode": 0,
            "stdout": out.getvalue(),
            "stderr": err.getvalue(),
            "script": "embedded.compare_two_images",
            "payload": payload,
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": 1,
            "stdout": out.getvalue(),
            "stderr": f"{err.getvalue()}\n{exc}".strip(),
            "script": "embedded.compare_two_images",
        }
