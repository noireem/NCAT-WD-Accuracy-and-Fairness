"""Run YOLOv8 inference on intervention-dataset images ingested from Azure Blob."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_prep import (  # noqa: E402
    DEFAULT_DEST,
    collect_images,
    download_blobs,
    resolve_image_root,
)

RESUME_LOG = "infer_processed.txt"
GUN_CLASS_ID = 0
GUN_OUTPUT = "gun_detections.jsonl"
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "weights" / "yolo_v8_baseline.pt"


def detect_objects(
    image_path: str,
    weights_path: str,
    conf: float = 0.7,
    target_class_id: int = GUN_CLASS_ID,
    output_path: str = GUN_OUTPUT,
    max_images: int = 0,
    resume_log: str = RESUME_LOG,
):
    model = YOLO(weights_path)

    path = Path(image_path)
    processed_set = set()
    log_path = Path(resume_log)
    if log_path.exists():
        processed_set = {p.strip() for p in log_path.read_text().splitlines() if p.strip()}

    if path.is_dir():
        sources = [str(p) for p in collect_images(path)]
    else:
        sources = [str(path)]

    if max_images > 0:
        sources = sources[:max_images]

    detections = []
    failed = 0
    processed = 0
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    print(f"Images queued: {len(sources)}")
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
            print(f"Inferred {Path(source).name}")
        except Exception as exc:
            failed += 1
            print(f"Skipped {source} due to error: {exc}")

    print(f"Processed: {processed}, Failed: {failed}")
    print(f"Gun detections: {len(detections)}")
    print(f"Saved gun detections to: {output_file}")
    return detections


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest intervention-dataset images from Azure, then run YOLOv8."
    )
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_DEST),
        help="Local cache for Azure images (default: data/intervention).",
    )
    parser.add_argument("--prefix", default=None, help="Optional blob prefix.")
    parser.add_argument("--max-blobs", type=int, default=None, help="Limit Azure files to consider.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit inference images. 0 = all.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use images already on disk; do not contact Azure.",
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Path to YOLOv8 weights.",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--output", default=GUN_OUTPUT, help="JSONL path for gun detections.")
    args = parser.parse_args()

    dest = Path(args.dest)
    if not args.skip_download:
        print(f"Ingesting intervention-dataset images from Azure into {dest} ...")
        downloaded = download_blobs(
            prefix=args.prefix,
            max_blobs=args.max_blobs,
            dest_dir=dest,
            images_only=True,
        )
        print(f"New images downloaded: {downloaded}")

    image_root = resolve_image_root(dest)
    print(f"Running inference on: {image_root}")
    detections = detect_objects(
        image_path=str(image_root),
        weights_path=args.weights,
        conf=args.conf,
        output_path=args.output,
        max_images=args.max_images,
    )

    for detection in detections[:20]:
        print(detection)
    if len(detections) > 20:
        print(f"Showing first 20 of {len(detections)} gun detections.")


if __name__ == "__main__":
    main()
