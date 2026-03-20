from cli_anything.facenet.utils.backend import (
    run_compare_two_images,
    run_capture_embedding,
    run_recognize_folder_to_csv,
    run_match_once,
)


def capture_embedding_once(workspace: str, output_csv: str | None = None, dry_run: bool = False) -> dict:
    if dry_run:
        return {
            "ok": True,
            "returncode": 0,
            "stdout": "dry-run: capture embedding flow prepared",
            "stderr": "",
            "script": "embedded.capture_embedding",
        }
    return run_capture_embedding(workspace=workspace, output_csv=output_csv)


def recognize_match_name(workspace: str, database_csv: str, threshold: float, dry_run: bool = False) -> dict:
    if dry_run:
        return {
            "ok": True,
            "returncode": 0,
            "stdout": "dry-run: match once flow prepared",
            "stderr": "",
            "script": "embedded.match_once",
            "payload": {"name": "Stranger"},
        }
    return run_match_once(workspace=workspace, database_csv=database_csv, threshold=threshold)


def recognize_folder_images_to_csv(
    workspace: str,
    input_folder: str,
    output_csv: str,
    dry_run: bool = False,
) -> dict:
    if dry_run:
        return {
            "ok": True,
            "returncode": 0,
            "stdout": "dry-run: recognize folder to csv flow prepared",
            "stderr": "",
            "script": "embedded.recognize_folder_to_csv",
            "payload": {"input_folder": input_folder, "output_csv": output_csv},
        }
    return run_recognize_folder_to_csv(
        workspace=workspace,
        input_folder=input_folder,
        output_csv=output_csv,
        database_csv="face_embeddings_database.csv",
        threshold=1.0,
    )


def compare_images(image_path_1: str, image_path_2: str, threshold: float, dry_run: bool = False) -> dict:
    if dry_run:
        return {
            "ok": True,
            "returncode": 0,
            "stdout": "dry-run: compare two images flow prepared",
            "stderr": "",
            "script": "embedded.compare_two_images",
            "payload": {
                "image1": image_path_1,
                "image2": image_path_2,
                "threshold": threshold,
                "distance": 0.0,
                "is_same_person": True,
            },
        }
    return run_compare_two_images(image_path_1=image_path_1, image_path_2=image_path_2, threshold=threshold)
