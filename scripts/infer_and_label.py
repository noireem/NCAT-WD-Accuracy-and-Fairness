"""Run resumable inference with the distilled YOLO model.

Defaults target the roboflow_distill run and process every second frame from
all train-set classes except NormalVideos.
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.shared_paths import REPO_ROOT, require_weights_path

# ── Config ──────────────────────────────────────────────────────────────
WEIGHTS = "runs/detect/roboflow_distill/weights/best.pt"
CONF = 0.25
IMGSZ = 640
GUN_CLASS_ID = 0
DETECTIONS_FILE = "gun_detections_distill.jsonl"
PROCESSED_LOG = "infer_processed_distill.txt"
LABEL_DIR = "labeledImages_distill"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXCLUDED_SUBFOLDERS = {"NormalVideos"}
FRAME_STEP = 2


def collect_images(root: Path, target_subfolders: list[str], frame_step: int, show_progress: bool = False):
    sources = []
    for sub in target_subfolders:
        sub_dir = root / sub
        if not sub_dir.exists():
            continue
        all_files = [
            f.resolve() for f in sorted(sub_dir.rglob("*"))
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ]
        sources.extend(all_files[::frame_step])
        if show_progress:
            print(f"  collected {len(all_files[::frame_step])} frames from {sub}", flush=True)
    return sources


def find_target_subfolders(root: Path, excluded: set[str]) -> list[str]:
    return sorted(
        child.name for child in root.iterdir()
        if child.is_dir() and child.name not in excluded
    )


def load_already_processed(log_path: Path) -> set:
    if not log_path.exists():
        return set()
    return {p.strip() for p in log_path.read_text().splitlines() if p.strip()}


def draw_and_save(image_path: Path, dets: list, label_dir: Path):
    """Draw bounding boxes and save the labeled image."""
    with Image.open(image_path).convert("RGB") as img:
        draw = ImageDraw.Draw(img)
        for d in dets:
            x1, y1, x2, y2 = d["bbox_xyxy"]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            label = f"gun {d['confidence']:.2f}"
            draw.text((max(0, int(x1)), max(0, int(y1) - 14)), label, fill="red")
        img.save(label_dir / image_path.name)


def main():
    parser = argparse.ArgumentParser(description="Run resumable YOLO inference over training image folders.")
    parser.add_argument("--weights", default=WEIGHTS, help="Path to model weights.")
    parser.add_argument("--conf", type=float, default=CONF, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="Inference image size.")
    parser.add_argument("--gun-class-id", type=int, default=GUN_CLASS_ID, help="Target class ID to keep.")
    parser.add_argument("--detections-file", default=DETECTIONS_FILE, help="JSONL file for detections.")
    parser.add_argument("--processed-log", default=PROCESSED_LOG, help="Resume log of processed images.")
    parser.add_argument("--label-dir", default=LABEL_DIR, help="Directory for labeled output images.")
    parser.add_argument(
        "--exclude-class",
        action="append",
        default=[],
        help="Class folder to exclude. Repeatable.",
    )
    parser.add_argument("--frame-step", type=int, default=FRAME_STEP, help="Process every Nth frame.")
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Print periodic progress updates in the terminal.",
    )
    args = parser.parse_args()

    repo_root = REPO_ROOT
    image_root = repo_root / "data" / "raw" / "ucf_crime" / "archive (4)" / "Train"
    det_path = repo_root / args.detections_file
    log_path = repo_root / args.processed_log
    label_dir = repo_root / args.label_dir
    label_dir.mkdir(parents=True, exist_ok=True)

    excluded = EXCLUDED_SUBFOLDERS | set(args.exclude_class)
    target_subfolders = find_target_subfolders(image_root, excluded)
    weights_path = require_weights_path(args.weights, repo_root=repo_root)
    if args.frame_step < 1:
        raise ValueError("--frame-step must be >= 1")

    print("Loading model …", flush=True)
    print(f"Weights: {weights_path}", flush=True)
    model = YOLO(str(weights_path))

    print("Collecting images …", flush=True)
    print(f"Target folders: {', '.join(target_subfolders)}", flush=True)
    print(f"Excluded folders: {', '.join(sorted(excluded))}", flush=True)
    print(f"Frame step: {args.frame_step}", flush=True)
    all_sources = collect_images(image_root, target_subfolders, args.frame_step, args.show_progress)
    print(f"Total images found: {len(all_sources)}", flush=True)

    already_done = load_already_processed(log_path)
    sources = [s for s in all_sources if str(s) not in already_done]
    print(f"Already processed: {len(already_done)}", flush=True)
    print(f"Remaining: {len(sources)}", flush=True)

    if not sources:
        print("Nothing left to process!", flush=True)
        return

    total_det = 0
    labeled = 0
    processed = 0

    for src in sources:
        try:
            results = model(str(src), conf=args.conf, stream=True, batch=1,
                            imgsz=args.imgsz, verbose=False)
            gun_dets = []
            for r in results:
                for box in r.boxes:
                    cid = int(box.cls)
                    if cid != args.gun_class_id:
                        continue
                    det = {
                        "source": str(src),
                        "class_id": cid,
                        "confidence": float(box.conf),
                        "bbox_xyxy": box.xyxy.tolist()[0],
                    }
                    gun_dets.append(det)

            # Append detections to JSONL
            if gun_dets:
                with det_path.open("a", encoding="utf-8") as f:
                    for d in gun_dets:
                        f.write(json.dumps(d) + "\n")
                draw_and_save(src, gun_dets, label_dir)
                total_det += len(gun_dets)
                labeled += 1

            # Mark as processed (crash-safe: always append)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(str(src) + "\n")

            processed += 1
            progress_interval = 25 if args.show_progress else 200
            if processed % progress_interval == 0:
                done = len(already_done) + processed
                print(f"  [{done}/{len(all_sources)}] "
                      f"detections: {total_det}, labeled: {labeled}",
                      flush=True)

        except Exception as exc:
            print(f"  SKIP {src.name}: {exc}", file=sys.stderr, flush=True)

    done = len(already_done) + processed
    print(f"\nFinished! Processed {done}/{len(all_sources)} total images.", flush=True)
    print(f"Gun detections: {total_det}", flush=True)
    print(f"Labeled images saved: {labeled} → {label_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
