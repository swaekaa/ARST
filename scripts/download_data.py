"""
Downloads and extracts the CMI - Detect Behavior with Sensor Data dataset from Kaggle.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def main():
    print("Initializing Kaggle API...")
    api = KaggleApi()

    # Authenticate (uses ~/.kaggle/kaggle.json)
    try:
        api.authenticate()
    except Exception as e:
        print(f"Error authenticating with Kaggle API: {e}")
        print("Please ensure your kaggle.json is correctly placed in ~/.kaggle/kaggle.json")
        sys.exit(1)

    competition_name = "cmi-fitness-behavior-dataset"  # Trying the most likely slug

    # Actually, the competition is likely "cmi-detect-behavior-with-sensor-data" based on the name.
    # Let's search for it to be sure.
    print("Searching for CMI competitions...")
    comps = api.competitions_list(search="cmi")
    target_comp = None
    for c in comps:
        if (
            "sensor" in c.title.lower()
            or "behavior" in c.title.lower()
            or "fitness" in c.title.lower()
        ):
            target_comp = c.ref
            print(f"Found match: {c.title} (slug: {c.ref})")
            break

    if not target_comp:
        # Fallback to the exact name in the user prompt
        target_comp = "cmi-detect-behavior-with-sensor-data"
        print(f"Could not find exact match in search. Defaulting to slug: {target_comp}")

    project_root = Path(__file__).resolve().parents[1]
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading dataset for {target_comp}...")
    print(f"Destination: {raw_data_dir}")
    print("This may take a while depending on your internet connection.")

    try:
        api.competition_download_files(target_comp, path=str(raw_data_dir), quiet=False)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Did you accept the competition rules on the Kaggle website?")
        sys.exit(1)

    zip_path = raw_data_dir / f"{target_comp}.zip"
    if zip_path.exists():
        print(f"\nExtracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_data_dir)
        print("Extraction complete. Cleaning up zip file...")
        zip_path.unlink()

    print("\nDataset download and extraction finished successfully!")
    print(f"Data is located at: {raw_data_dir}")


if __name__ == "__main__":
    main()
