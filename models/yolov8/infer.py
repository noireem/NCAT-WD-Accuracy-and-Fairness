import json
from pathlib import Path
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TARGET_SUBFOLDERS = {
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
}
MAX_IMAGES = 0  # 0 = all images
RESUME_LOG = "infer_processed.txt"
GUN_CLASS_ID = 0
GUN_OUTPUT = "gun_detections.jsonl"


def detect_objects(
    image_path: str,
    weights_path: str,
    conf: float = 0.7,
    target_class_id: int = GUN_CLASS_ID,
    output_path: str = GUN_OUTPUT,
):
    model = YOLO(weights_path)

    path = Path(image_path)
    sources = []
    processed_set = set()
    log_path = Path(RESUME_LOG)
    if log_path.exists():
        processed_set = {p.strip() for p in log_path.read_text().splitlines() if p.strip()}

    if path.is_dir():
        for sub in TARGET_SUBFOLDERS:
            sub_dir = path / sub
            if not sub_dir.exists():
                continue
            sources.extend(
                str(f)
                for f in sorted(sub_dir.rglob("*"))
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            )
    else:
        sources = [str(path)]

    if MAX_IMAGES > 0:
        sources = sources[:MAX_IMAGES]

    detections = []
    failed = 0
    processed = 0
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()
    for source in sources:
        if source in processed_set:
            continue
        try:
            results = model(source, conf=conf, stream=True, batch=1, imgsz=640)
            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls)
                    if class_id != target_class_id:
                        continue
                    detection = {
                        "source": source,
                        "class_id": class_id,
                        "confidence": float(box.conf),
                        "bbox_xyxy": box.xyxy.tolist()[0],
                    }
                    detections.append(detection)
                    with output_file.open("a", encoding="utf-8") as out_f:
                        out_f.write(json.dumps(detection) + "\n")
            processed += 1
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"{source}\n")
        except Exception as exc:
            failed += 1
            print(f"Skipped {source} due to error: {exc}")

    print(f"Processed: {processed}, Failed: {failed}")
    print(f"Gun detections: {len(detections)}")
    print(f"Saved gun detections to: {output_file}")

    return detections


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    image_path = repo_root / "data" / "raw" / "ucf_crime" / "archive (4)" / "Train"

    detections = detect_objects(
        image_path=str(image_path),
        weights_path="runs/detect/train13/weights/best.pt",
        conf=0.5,
    )

    for d in detections[:20]:
        print(d)
    if len(detections) > 20:
        print(f"Showing first 20 of {len(detections)} gun detections.")