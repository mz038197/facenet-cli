import csv
import os
import time
from datetime import datetime
from pathlib import Path


def _supported_image(name: str) -> bool:
    return name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))


def _load_backend_deps():
    try:
        from facenet_pytorch import InceptionResnetV1, MTCNN
        from PIL import Image
        import cv2
        import numpy as np
        import torch
    except Exception as exc:
        raise RuntimeError(
            "缺少 backend 依賴。請安裝: pip install 'cli-anything-facenet-pytorch[backend]'"
        ) from exc
    return InceptionResnetV1, MTCNN, Image, cv2, np, torch


def _capture_one_frame(cv2):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("無法開啟攝影機")
    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("無法讀取攝影機畫面")
        return frame
    finally:
        cap.release()


def _load_database_embeddings(db_path: str, np):
    if not os.path.exists(db_path):
        raise RuntimeError(f"找不到 embeddings 資料庫: {db_path}")
    db: list[tuple[str, object]] = []
    with open(db_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            db.append((os.path.splitext(row[0])[0], np.array([float(x) for x in row[1:]])))
    if not db:
        raise RuntimeError("embeddings 資料庫是空的")
    return db


def _create_command_run_dir(workspace: str, command_name: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    run_dir = os.path.join(os.path.abspath(workspace), f"{command_name}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def generate_embeddings(workspace: str, image_folder: str = "face_images", output_csv: str = "face_embeddings_database.csv") -> dict:
    """Embedded backend implementation for database generation."""
    InceptionResnetV1, MTCNN, Image, _cv2, _np, torch = _load_backend_deps()

    abs_workspace = os.path.abspath(workspace)
    image_dir = os.path.join(abs_workspace, image_folder)
    csv_path = os.path.join(abs_workspace, output_csv)

    if not os.path.isdir(image_dir):
        raise RuntimeError(f"找不到影像資料夾: {image_dir}")

    image_files = sorted([x for x in os.listdir(image_dir) if _supported_image(x)])
    if not image_files:
        raise RuntimeError("影像資料夾內沒有支援的圖片格式")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval()
    if device == "cuda":
        resnet = resnet.cuda()

    rows: list[list[float | str]] = []
    ok_count = 0
    for fname in image_files:
        path = os.path.join(image_dir, fname)
        try:
            img = Image.open(path).convert("RGB")
            crop = mtcnn(img)
            if crop is None:
                continue
            crop = crop.unsqueeze(0)
            if device == "cuda":
                crop = crop.cuda()
            with torch.no_grad():
                emb = resnet(crop).detach().cpu().numpy()[0]
            rows.append([fname, *emb.tolist()])
            ok_count += 1
        except Exception:
            continue

    os.makedirs(os.path.dirname(csv_path) or abs_workspace, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", *[f"embedding_{i}" for i in range(512)]])
        writer.writerows(rows)

    return {
        "ok": True,
        "workspace": abs_workspace,
        "image_folder": image_folder,
        "output_csv": csv_path,
        "total_images": len(image_files),
        "embedded_images": ok_count,
    }


def _extract_single_embedding_from_frame(frame, MTCNN, InceptionResnetV1, Image, torch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval()
    if device == "cuda":
        resnet = resnet.cuda()
    rgb = frame[:, :, ::-1]
    pil_img = Image.fromarray(rgb)
    aligned = mtcnn(pil_img)
    if aligned is None:
        raise RuntimeError("未偵測到可用人臉")
    aligned = aligned.unsqueeze(0)
    if device == "cuda":
        aligned = aligned.cuda()
    with torch.no_grad():
        emb = resnet(aligned).detach().cpu().numpy()[0]
    return emb, device


def _extract_single_embedding_from_pil_image(img, MTCNN, InceptionResnetV1, torch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn_detect = MTCNN(keep_all=True, device=device)
    mtcnn = MTCNN(keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval()
    if device == "cuda":
        resnet = resnet.cuda()
    boxes, _ = mtcnn_detect.detect(img)
    if boxes is None or len(boxes) == 0:
        raise RuntimeError("未偵測到可用人臉")
    first_box = [int(v) for v in boxes[0]]
    aligned = mtcnn(img)
    if aligned is None:
        raise RuntimeError("未偵測到可用人臉")
    aligned = aligned.unsqueeze(0)
    if device == "cuda":
        aligned = aligned.cuda()
    with torch.no_grad():
        emb = resnet(aligned).detach().cpu().numpy()[0]
    return emb, device, first_box


def capture_embedding(workspace: str, output_csv: str | None = None) -> dict:
    InceptionResnetV1, MTCNN, Image, cv2, _np, torch = _load_backend_deps()
    from PIL import ImageDraw

    frame = _capture_one_frame(cv2)
    rgb = frame[:, :, ::-1]
    pil_img = Image.fromarray(rgb)
    embedding, device, box = _extract_single_embedding_from_pil_image(pil_img, MTCNN, InceptionResnetV1, torch)
    run_dir = _create_command_run_dir(workspace, "capture-embedding")
    capture_name = f"capture_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jpg"
    original_path = os.path.join(run_dir, f"original_{capture_name}")
    boxed_path = os.path.join(run_dir, f"boxed_{capture_name}")
    pil_img.save(original_path)
    draw_img = pil_img.copy()
    draw = ImageDraw.Draw(draw_img)
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    draw_img.save(boxed_path)

    result = {
        "ok": True,
        "workspace": os.path.abspath(workspace),
        "run_dir": os.path.abspath(run_dir),
        "device": device,
        "dimension": int(embedding.shape[0]),
        "box_points": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "boxed_images_dir": os.path.abspath(run_dir),
        "original_image_path": os.path.abspath(original_path),
        "boxed_image_path": os.path.abspath(boxed_path),
    }
    if output_csv:
        out_name = os.path.basename(output_csv)
        out_path = os.path.join(run_dir, out_name)
        write_header = not os.path.exists(out_path)
        with open(out_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["filename", *[f"embedding_{i}" for i in range(int(embedding.shape[0]))]])
            writer.writerow([capture_name, *embedding.tolist()])
        result["output_csv"] = os.path.abspath(out_path)
    return result


def match_once(workspace: str, database_csv: str = "face_embeddings_database.csv", threshold: float = 1.0) -> dict:
    InceptionResnetV1, MTCNN, Image, cv2, np, torch = _load_backend_deps()
    abs_workspace = os.path.abspath(workspace)
    db_path = database_csv if os.path.isabs(database_csv) else os.path.join(abs_workspace, database_csv)
    db = _load_database_embeddings(db_path, np)

    frame = _capture_one_frame(cv2)
    emb, device = _extract_single_embedding_from_frame(frame, MTCNN, InceptionResnetV1, Image, torch)
    best_name = "Stranger"
    best_dist = float("inf")
    for name, ref in db:
        dist = float(np.linalg.norm(emb - ref))
        if dist < best_dist:
            best_dist = dist
            best_name = name if dist <= threshold else "Stranger"

    return {
        "ok": True,
        "workspace": abs_workspace,
        "database_csv": os.path.abspath(db_path),
        "device": device,
        "name": best_name,
        "distance": best_dist,
        "threshold": threshold,
    }


def recognize_folder_to_csv(
    workspace: str,
    input_folder: str,
    output_csv: str,
    database_csv: str = "face_embeddings_database.csv",
    threshold: float = 1.0,
) -> dict:
    InceptionResnetV1, MTCNN, Image, _cv2, np, torch = _load_backend_deps()
    from PIL import ImageDraw

    abs_workspace = os.path.abspath(workspace)
    folder_path = input_folder if os.path.isabs(input_folder) else os.path.join(abs_workspace, input_folder)
    if not os.path.isdir(folder_path):
        raise RuntimeError(f"找不到輸入資料夾: {folder_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval()
    if device == "cuda":
        resnet = resnet.cuda()

    files = sorted([f for f in os.listdir(folder_path) if _supported_image(f)])
    run_dir = _create_command_run_dir(workspace, "images")
    out_name = os.path.basename(output_csv)
    out_path = os.path.join(run_dir, out_name)
    boxed_dir = os.path.join(run_dir, "boxed_images")
    os.makedirs(boxed_dir, exist_ok=True)

    recognized = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", *[f"embedding_{i}" for i in range(512)]])
        for fname in files:
            img_path = os.path.join(folder_path, fname)
            try:
                img = Image.open(img_path).convert("RGB")
                boxes, _ = mtcnn.detect(img, landmarks=False)
                aligned = mtcnn(img)
                if aligned is None or boxes is None or len(boxes) == 0:
                    continue
                aligned = aligned.unsqueeze(0)
                if device == "cuda":
                    aligned = aligned.cuda()
                with torch.no_grad():
                    emb = resnet(aligned).detach().cpu().numpy()[0]
                writer.writerow([fname, *emb.tolist()])
                draw_img = img.copy()
                draw = ImageDraw.Draw(draw_img)
                x1, y1, x2, y2 = [int(v) for v in boxes[0]]
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw_img.save(os.path.join(boxed_dir, fname))
                recognized += 1
            except Exception:
                continue

    return {
        "ok": True,
        "workspace": abs_workspace,
        "run_dir": os.path.abspath(run_dir),
        "input_folder": os.path.abspath(folder_path),
        "output_csv": os.path.abspath(out_path),
        "total_images": len(files),
        "recognized_images": recognized,
        "format": "filename + embedding_0..embedding_511",
        "boxed_images_dir": os.path.abspath(boxed_dir),
    }


def compare_two_images(image_path_1: str, image_path_2: str, threshold: float) -> dict:
    InceptionResnetV1, MTCNN, Image, _cv2, np, torch = _load_backend_deps()
    p1 = os.path.abspath(image_path_1)
    p2 = os.path.abspath(image_path_2)
    if not os.path.exists(p1):
        raise RuntimeError(f"找不到影像1: {p1}")
    if not os.path.exists(p2):
        raise RuntimeError(f"找不到影像2: {p2}")
    img1 = Image.open(p1).convert("RGB")
    img2 = Image.open(p2).convert("RGB")
    emb1, device = _extract_single_embedding_from_pil_image(img1, MTCNN, InceptionResnetV1, torch)
    emb2, _ = _extract_single_embedding_from_pil_image(img2, MTCNN, InceptionResnetV1, torch)
    distance = float(np.linalg.norm(emb1 - emb2))
    return {
        "ok": True,
        "image1": p1,
        "image2": p2,
        "device": device,
        "threshold": threshold,
        "distance": distance,
        "is_same_person": distance < threshold,
    }


def run_realtime_recognition(workspace: str, database_csv: str = "face_embeddings_database.csv", threshold: float = 1.0) -> dict:
    """Embedded backend for live recognition via webcam."""
    InceptionResnetV1, MTCNN, Image, cv2, np, torch = _load_backend_deps()

    abs_workspace = os.path.abspath(workspace)
    csv_path = os.path.join(abs_workspace, database_csv)
    if not os.path.exists(csv_path):
        raise RuntimeError(f"找不到 embeddings 資料庫: {csv_path}")

    db: list[tuple[str, object]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            db.append((os.path.splitext(row[0])[0], np.array([float(x) for x in row[1:]])))

    if not db:
        raise RuntimeError("embeddings 資料庫是空的")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn_detect = MTCNN(keep_all=True, device=device, min_face_size=40)
    mtcnn_embed = MTCNN(keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval()
    if device == "cuda":
        resnet = resnet.cuda()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("無法開啟攝影機")

    window = "Face Recognition (cli-anything)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    start = time.time()
    frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            boxes, _ = mtcnn_detect.detect(pil_img)

            if boxes is not None:
                aligned = mtcnn_embed(pil_img)
                name = "unknown"
                dist = float("inf")
                if aligned is not None:
                    aligned = aligned.unsqueeze(0)
                    if device == "cuda":
                        aligned = aligned.cuda()
                    with torch.no_grad():
                        emb = resnet(aligned).detach().cpu().numpy()[0]
                    for n, ref in db:
                        d = float(np.linalg.norm(emb - ref))
                        if d < dist:
                            dist = d
                            name = n if d <= threshold else "Stranger"

                x1, y1, x2, y2 = [int(v) for v in boxes[0]]
                color = (0, 255, 0) if name != "Stranger" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} ({dist:.3f})", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return {"ok": True, "frames": frames, "duration_sec": round(time.time() - start, 3)}
