"""Download shared model weights from the Google Drive folder into repo-standard locations."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.shared_paths import DATASET_DRIVE_URL, REPO_ROOT


DEFAULT_OUTPUT = REPO_ROOT / "downloads" / "shared_assets"
DESTINATIONS = {
    "yolov8n.pt": REPO_ROOT / "models" / "yolov8" / "yolov8n.pt",
    "yolo26n.pt": REPO_ROOT / "yolo26n.pt",
    "train13_best.pt": REPO_ROOT / "runs" / "detect" / "train13" / "weights" / "best.pt",
    "roboflow_distill_best.pt": REPO_ROOT / "runs" / "detect" / "roboflow_distill" / "weights" / "best.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download shared weight files from Google Drive.")
    parser.add_argument("--url", default=DATASET_DRIVE_URL, help="Google Drive folder URL containing the shared assets.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Temporary directory for downloaded shared assets.",
    )
    parser.add_argument(
        "--keep-downloads",
        action="store_true",
        help="Keep the raw downloaded files after copying them into repo locations.",
    )
    return parser.parse_args()


def copy_matching_weights(download_root: Path) -> list[Path]:
    copied = []
    for file_path in download_root.rglob("*.pt"):
        name = file_path.name
        destination = DESTINATIONS.get(name)
        if destination is None and name == "best.pt":
            parts = {part.lower() for part in file_path.parts}
            if "roboflow_distill" in parts or "distill" in parts:
                destination = REPO_ROOT / "runs" / "detect" / "roboflow_distill" / "weights" / "best.pt"
            elif "train13" in parts:
                destination = REPO_ROOT / "runs" / "detect" / "train13" / "weights" / "best.pt"
        if destination is None:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, destination)
        copied.append(destination)
    return copied


def main() -> None:
    args = parse_args()
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit("gdown is required. Run 'pip install -r requirements.txt' first.") from exc

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = gdown.download_folder(
        url=args.url,
        output=str(output_dir),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
    )
    if not downloaded:
        raise SystemExit("No shared assets were downloaded. Check the Google Drive sharing settings.")

    copied = copy_matching_weights(output_dir)
    if not copied:
        raise SystemExit(
            "No expected .pt files were found. Ensure the Drive folder contains yolov8n.pt, yolo26n.pt, train13_best.pt, roboflow_distill_best.pt, or best.pt inside train13/ or roboflow_distill/."
        )

    for path in copied:
        print(f"Placed {path}")

    if not args.keep_downloads:
        shutil.rmtree(output_dir)


if __name__ == "__main__":
    main()