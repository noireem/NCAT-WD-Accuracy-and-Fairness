#Connecting Azure Blob to Codebase. When you run "python src/download_data.py --max-blobs #", downloaded frames go to data/raw/ucf-crime...
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from azure.storage.blob import ContainerClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def download_blobs(prefix: str | None, max_blobs: int | None, dest_dir: Path) -> int:
    sas_url = os.getenv("UCF_CRIME_SAS_URL")
    if not sas_url:
        print("Error: Missing UCF_CRIME_SAS_URL in .env file.")
        print("Make sure you created the .env file and added the URL!")
        return 0

    # Ensure the destination directory exists
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        container = ContainerClient.from_container_url(sas_url)
    except Exception as e:
        print(f"Failed to connect to Azure: {e}")
        return 0

    print(f"Connected to Azure. Scanning for files...")
    
    downloaded = 0
    # List blobs (files) in the container
    # If prefix is None, it lists everything. If "Robbery/", it filters.
    blob_iterator = container.list_blobs(name_starts_with=prefix)

    for blob in blob_iterator:
        if max_blobs is not None and downloaded >= max_blobs:
            print(f"Reached limit of {max_blobs} files.")
            break

        # Define local path (e.g., data/raw/ucf_crime/Abuse/Abuse001.mp4)
        target_path = dest_dir / blob.name
        
        # Skip if already downloaded
        if target_path.exists():
            print(f"Skipping {blob.name} (Already exists)")
            continue

        # Create subfolders (e.g., 'Abuse', 'Arrest')
        target_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading: {blob.name} ...")
        
        try:
            stream = container.download_blob(blob)
            target_path.write_bytes(stream.readall())
            downloaded += 1
        except Exception as e:
            print(f"Failed to download {blob.name}: {e}")

    return downloaded

def main() -> None:
    # Default path: points to data/raw/ucf_crime relative to this script
    # This assumes script is in src/ and data is in data/
    default_dest = Path(__file__).resolve().parents[1] / "data" / "raw" / "ucf_crime"

    parser = ArgumentParser(description="Download UCF-Crime dataset from Azure Blob Storage")
    parser.add_argument("--prefix", default=None, help="Filter by folder (e.g., 'Robbery/', 'Arrest/')")
    parser.add_argument("--max-blobs", type=int, default=None, help="Limit number of downloads (for testing)")
    parser.add_argument("--dest", type=Path, default=default_dest, help="Local destination folder")

    args = parser.parse_args()

    print(f"Saving to: {args.dest}")
    count = download_blobs(prefix=args.prefix, max_blobs=args.max_blobs, dest_dir=args.dest)
    print(f"Download complete. Total files: {count}")

# THIS IS THE PART THAT PRESSES THE 'GO' BUTTON
if __name__ == "__main__":
    main()