"""Download intervention-dataset images from Azure Blob Storage.

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
DEFAULT_DEST = REPO_ROOT / "data" / "intervention"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_EXTS = {".txt", ".json", ".yaml", ".yml", ".xml"}
SAS_ENV_VARS = (
    "INTERVENTION_DATASET_SAS_URL",
    "AZURE_BLOB_SAS_URL",
)

load_dotenv(REPO_ROOT / ".env")


def get_sas_url() -> str:
    for name in SAS_ENV_VARS:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    raise RuntimeError(
        "Missing Azure SAS URL. Set INTERVENTION_DATASET_SAS_URL in .env "
        "to the interventiondataset container SAS URL."
    )


def get_container() -> ContainerClient:
    return ContainerClient.from_container_url(
        get_sas_url(),
        connection_timeout=20,
        retry_total=2,
    )


def is_image_blob(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTS


def is_dataset_blob(name: str) -> bool:
    suffix = Path(name).suffix.lower()
    return suffix in IMAGE_EXTS or suffix in LABEL_EXTS


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

    print("Connected to Azure Blob (intervention dataset). Scanning for files...")
    downloaded = 0
    skipped = 0
    scanned = 0
    matched = 0

    for blob in container.list_blobs(name_starts_with=prefix):
        scanned += 1
        keep = is_image_blob(blob.name) if images_only else is_dataset_blob(blob.name)
        if not keep:
            continue

        matched += 1
        if max_blobs is not None and matched > max_blobs:
            print(f"Reached limit of {max_blobs} matching files.")
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
            print(f"Downloaded {blob.name}")
        except Exception as exc:
            print(f"Failed to download {blob.name}: {exc}")

    print(
        f"Scan complete. scanned={scanned} matched={matched} "
        f"downloaded={downloaded} skipped_existing={skipped}"
    )
    return downloaded


def resolve_image_root(dest_dir: Path) -> Path:
    """Pick the folder that actually holds downloaded images."""
    dest_dir = Path(dest_dir)
    candidates = [
        dest_dir / "images" / "train",
        dest_dir / "images" / "val",
        dest_dir / "images",
        dest_dir / "train",
        dest_dir / "Train",
        dest_dir,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(
            f.is_file() and f.suffix.lower() in IMAGE_EXTS for f in candidate.rglob("*")
        ):
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
    if root.is_file() and root.suffix.lower() in IMAGE_EXTS:
        return [root.resolve()]

    if target_subfolders:
        sources: list[Path] = []
        found_class_dir = False
        for sub in sorted(target_subfolders):
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
        if found_class_dir:
            return sources

    return [
        f.resolve()
        for f in sorted(root.rglob("*"))
        if f.is_file()
        and f.suffix.lower() in IMAGE_EXTS
        and not any(part in excluded for part in f.parts)
    ]


def main() -> None:
    parser = ArgumentParser(
        description="Download intervention-dataset images from Azure Blob Storage"
    )
    parser.add_argument("--prefix", default=None, help="Filter by folder prefix")
    parser.add_argument("--max-blobs", type=int, default=None, help="Limit matching files")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Local destination folder")
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Also download label/metadata files. Default is images only.",
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
