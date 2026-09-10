"""Download Kaggle Customer Support on Twitter dataset."""
import shutil
from pathlib import Path

import kagglehub

from src.config import ROOT


def main():
    print("Downloading thoughtvector/customer-support-on-twitter from Kaggle...")
    path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
    src = Path(path) / "twcs" / "twcs.csv"
    dst = ROOT / "data" / "raw" / "twcs.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not dst.exists():
        print(f"Copying {src} -> {dst}")
        shutil.copy2(src, dst)
    else:
        print(f"Dataset already exists at {dst}")

    print("Done.")


if __name__ == "__main__":
    main()
