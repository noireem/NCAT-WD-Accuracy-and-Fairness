"""Download UCF-Crime images from Azure Blob Storage.

Run:
    python src/data_prep.py --max-blobs 20
"""

from __future__ import annotations

import os
from argparse import ArgumentParser
from pathlib import Path

from azure.storage.blob import ContainerClient
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "data" / "raw" / "ucf_crime"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
UCF_CLASS_FOLDERS = {
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

load_dotenv(REPO_ROOT / ".env")


def get_container() -> ContainerClient:
    sas_url = os.getenv("UCF_CRIME_SAS_URL")
    if not sas_url:
        raise RuntimeError(
            "Missing UCF_CRIME_SAS_URL in .env. Add the Azure container SAS URL."
        )
    return ContainerClient.from_container_url(sas_url)


def is_image_blob(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTS


def download_blobs(
    prefix: str | None,
    max_blobs: int | None,
    dest_dir: Path,
    images_only: bool = True,
) -> int:
    """Download blobs into dest_dir, preserving container folder structure."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        container = get_container()
    except Exception as exc:
        print(f"Failed to connect to Azure: {exc}")
        return 0

    print("Connected to Azure Blob. Scanning for files...")
    downloaded = 0
    skipped = 0
    scanned = 0

    for blob in container.list_blobs(name_starts_with=prefix):
        scanned += 1
        if images_only and not is_image_blob(blob.name):
            continue
        if max_blobs is not None and downloaded >= max_blobs:
            print(f"Reached limit of {max_blobs} new files.")
            break

        target_path = dest_dir / blob.name
        if target_path.exists() and target_path.stat().st_size > 0:
            skipped += 1
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            stream = container.download_blob(blob)
            target_path.write_bytes(stream.readall())
            downloaded += 1
            if downloaded % 50 == 0:
                print(f"Downloaded {downloaded} new files...")
        except Exception as exc:
            print(f"Failed to download {blob.name}: {exc}")

    print(
        f"Scan complete. scanned={scanned} downloaded={downloaded} skipped_existing={skipped}"
    )
    return downloaded


def resolve_image_root(dest_dir: Path) -> Path:
    """Pick the folder that actually holds classed UCF frames."""
    dest_dir = Path(dest_dir)
    candidates = [
        dest_dir / "Train",
        dest_dir / "archive (4)" / "Train",
        dest_dir,
    ]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        class_dirs = [p for p in candidate.iterdir() if p.is_dir() and p.name in UCF_CLASS_FOLDERS]
        if class_dirs:
            return candidate
    return dest_dir


def collect_images(
    root: Path,
    target_subfolders: set[str] | None = None,
    excluded_subfolders: set[str] | None = None,
) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []

    excluded = excluded_subfolders or set()
    sources: list[Path] = []

    if root.is_file() and root.suffix.lower() in IMAGE_EXTS:
        return [root.resolve()]

    subfolders = target_subfolders or UCF_CLASS_FOLDERS
    found_class_dir = False
    for sub in sorted(subfolders):
        if sub in excluded:
            continue
        sub_dir = root / sub
        if not sub_dir.is_dir():
            continue
        found_class_dir = True
        sources.extend(
            f.resolve()
            for f in sorted(sub_dir.rglob("*"))
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )

    if not found_class_dir:
        sources.extend(
            f.resolve()
            for f in sorted(root.rglob("*"))
            if f.is_file()
            and f.suffix.lower() in IMAGE_EXTS
            and not any(part in excluded for part in f.parts)
        )
    return sources


def main() -> None:
    parser = ArgumentParser(description="Download UCF-Crime images from Azure Blob Storage")
    parser.add_argument("--prefix", default=None, help="Filter by folder (e.g. 'Robbery/' or 'Train/')")
    parser.add_argument("--max-blobs", type=int, default=None, help="Limit number of new downloads")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Local destination folder")
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Download every blob, including videos. Default is images only.",
    )
    args = parser.parse_args()

    print(f"Saving to: {args.dest}")
    count = download_blobs(
        prefix=args.prefix,
        max_blobs=args.max_blobs,
        dest_dir=args.dest,
        images_only=not args.all_files,
    )
    print(f"Download complete. New files: {count}")


if __name__ == "__main__":
    main()
