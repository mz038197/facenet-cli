from cli_anything.facenet.core.recognition import (
    compare_images,
    capture_embedding_once,
    recognize_folder_images_to_csv,
    recognize_match_name,
)


def test_capture_embedding_dry_run_payload():
    result = capture_embedding_once(workspace=".", output_csv="one.csv", dry_run=True)
    assert result["ok"] is True
    assert result["script"] == "embedded.capture_embedding"


def test_recognize_match_name_dry_run_payload():
    result = recognize_match_name(workspace=".", database_csv="db.csv", threshold=1.0, dry_run=True)
    assert result["ok"] is True
    assert result["payload"]["name"] == "Stranger"


def test_recognize_folder_to_csv_dry_run_payload():
    result = recognize_folder_images_to_csv(
        workspace=".",
        input_folder="face_images",
        output_csv="folder_result.csv",
        dry_run=True,
    )
    assert result["ok"] is True
    assert result["payload"]["input_folder"] == "face_images"


def test_compare_images_dry_run_payload():
    result = compare_images(
        image_path_1="a.jpg",
        image_path_2="b.jpg",
        threshold=1.2,
        dry_run=True,
    )
    assert result["ok"] is True
    assert "distance" in result["payload"]
    assert "is_same_person" in result["payload"]
