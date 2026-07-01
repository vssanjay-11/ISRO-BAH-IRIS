"""
download_datasets.py
IRIS-AI Dataset Downloader

Supports: FLIR ADAS, LLVIP, KAIST, TNO, SENSIAC
Reuses: config.DATASETS_DIR, data/image_folder.py::IMG_EXTENSIONS
Run:  python download_datasets.py
"""

from __future__ import annotations

import os
import sys
import zipfile
import tarfile
import shutil
import webbrowser
from pathlib import Path
from typing import Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config
# Reuse existing image-extension list from the repo
from data.image_folder import IMG_EXTENSIONS, is_image_file

# ---------------------------------------------------------------------------
# Dataset Registry
# ---------------------------------------------------------------------------
# Entries with download_url=None require manual download due to licensing.
# open_url is the page the user is directed to for manual download.
DATASETS: dict[str, dict] = {
    "LLVIP": {
        "description": "Low-Light Visible-Infrared Paired dataset (15k paired images)",
        "license":     "CC BY-NC 4.0",
        "auto":        False,
        "open_url":    "https://bupt-ai-cz.github.io/LLVIP/",
        "subfolder":   "LLVIP",
        "structure":   ["infrared/train", "infrared/test", "visible/train", "visible/test"],
    },
    "FLIR_ADAS": {
        "description": "FLIR Thermal Dataset for ADAS (26k thermal images + annotations)",
        "license":     "Non-commercial research only",
        "auto":        False,
        "open_url":    "https://www.flir.com/oem/adas/adas-dataset-form/",
        "subfolder":   "FLIR_ADAS",
        "structure":   ["train/thermal_8_bit", "val/thermal_8_bit"],
    },
    "KAIST": {
        "description": "KAIST Multispectral Pedestrian Dataset",
        "license":     "Research only",
        "auto":        False,
        "open_url":    "https://soonminhwang.github.io/rgbt-ped-detection/",
        "subfolder":   "KAIST",
        "structure":   ["images/set00", "images/set01"],
    },
    "TNO": {
        "description": "TNO Image Fusion Dataset (military IR + visible pairs)",
        "license":     "Non-commercial",
        "auto":        False,
        "open_url":    "https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029",
        "subfolder":   "TNO",
        "structure":   ["InfraRed", "Visible"],
    },
    "SENSIAC": {
        "description": "SENSIAC ATR Dataset (ground-vehicle thermal infrared)",
        "license":     "Requires registration",
        "auto":        False,
        "open_url":    "https://www.dsiac.org/resources/journals/dsiac/winter-2017-volume-4-number-1/atrevaluation-framework",
        "subfolder":   "SENSIAC",
        "structure":   ["images"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_images(folder: str) -> int:
    count = 0
    for root, _, files in os.walk(folder):
        count += sum(1 for f in files if is_image_file(f))  # reuse existing function
    return count


def _verify_structure(base: str, expected_subdirs: list) -> list:
    missing = [
        s for s in expected_subdirs
        if not os.path.isdir(os.path.join(base, s))
    ]
    return missing


def _extract_archive(archive_path: str, dest: str) -> bool:
    print(f"  Extracting {os.path.basename(archive_path)} → {dest} …")
    try:
        if archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as z:
                z.extractall(dest)
        elif archive_path.endswith((".tar.gz", ".tgz", ".tar")):
            with tarfile.open(archive_path) as t:
                t.extractall(dest)
        else:
            print(f"  Unknown archive format: {archive_path}")
            return False
        print(f"  Extracted OK.")
        return True
    except Exception as e:
        print(f"  Extraction failed: {e}")
        return False


def _scan_and_organize(dataset_name: str, source_folder: str) -> int:
    """
    Scan source_folder for IR images and move them into the canonical
    datasets/<DatasetName>/infrared/ structure.
    Returns count of organized images.
    """
    dest_ir  = os.path.join(config.DATASETS_DIR, dataset_name, "infrared")
    os.makedirs(dest_ir, exist_ok=True)

    moved = 0
    for root, _, files in os.walk(source_folder):
        for fname in files:
            if is_image_file(fname):   # reuse data/image_folder.py::is_image_file()
                src = os.path.join(root, fname)
                dst = os.path.join(dest_ir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    moved += 1
    return moved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_dataset(name: str) -> dict:
    """Return status dict for a dataset."""
    ds    = DATASETS[name]
    base  = os.path.join(config.DATASETS_DIR, ds["subfolder"])
    exists = os.path.isdir(base)
    images = _count_images(base) if exists else 0
    missing_dirs = _verify_structure(base, ds["structure"]) if exists else ds["structure"]

    return {
        "name":        name,
        "exists":      exists,
        "image_count": images,
        "missing_dirs": missing_dirs,
        "base_path":   base,
        "description": ds["description"],
        "license":     ds["license"],
        "open_url":    ds["open_url"],
    }


def download_dataset(name: str, auto_open_browser: bool = True) -> dict:
    """
    Download or guide the user to download a dataset.

    For datasets requiring manual download:
    - Opens the download page in the browser.
    - Creates the canonical folder structure.
    - Prints instructions for where to place files.

    Returns the status dict after processing.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

    ds   = DATASETS[name]
    base = os.path.join(config.DATASETS_DIR, ds["subfolder"])
    os.makedirs(base, exist_ok=True)

    # Create expected subfolder structure
    for sub in ds["structure"]:
        os.makedirs(os.path.join(base, sub), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Dataset: {name}")
    print(f"  {ds['description']}")
    print(f"  License: {ds['license']}")
    print(f"{'='*60}")

    if ds["auto"]:
        # Future: implement auto-download for openly licensed datasets
        print("  Auto-download enabled (not yet implemented).")
    else:
        print(f"\n  This dataset requires MANUAL download.")
        print(f"  Please visit:")
        print(f"    {ds['open_url']}")
        print(f"\n  After downloading, place the files in:")
        print(f"    {base}")
        print(f"\n  Expected structure:")
        for sub in ds["structure"]:
            print(f"    {os.path.join(base, sub)}/")

        if auto_open_browser:
            try:
                webbrowser.open(ds["open_url"])
                print("\n  Browser opened automatically.")
            except Exception:
                pass

    return check_dataset(name)


def check_all() -> dict:
    """Return status for all registered datasets."""
    return {name: check_dataset(name) for name in DATASETS}


def download_all(auto_open_browser: bool = True) -> dict:
    """
    Process all datasets — auto-download where possible,
    open browser for manual downloads.
    """
    print("\n" + "=" * 60)
    print("  IRIS-AI Dataset Downloader")
    print("  ISRO BAH 2026 PS-10")
    print("=" * 60)

    results = {}
    for name in DATASETS:
        try:
            results[name] = download_dataset(name, auto_open_browser=auto_open_browser)
        except Exception as e:
            print(f"  [{name}] Error: {e}")
            results[name] = {"name": name, "error": str(e)}

    print("\n" + "=" * 60)
    print("  Dataset Summary")
    print("=" * 60)
    for name, status in results.items():
        imgs   = status.get("image_count", 0)
        exists = "OK" if status.get("exists") else "NOT DOWNLOADED"
        print(f"  {name:<15} [{exists}]  {imgs} images found")
    print("=" * 60 + "\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IRIS-AI Dataset Downloader")
    parser.add_argument("--dataset", type=str, default="all",
                        help="Dataset name or 'all'. Options: " + ", ".join(DATASETS.keys()))
    parser.add_argument("--no-browser", action="store_true",
                        help="Suppress automatic browser opening")
    args = parser.parse_args()

    auto_open = not args.no_browser

    if args.dataset.lower() == "all":
        download_all(auto_open_browser=auto_open)
    else:
        if args.dataset not in DATASETS:
            print(f"Unknown dataset '{args.dataset}'. Options: {list(DATASETS.keys())}")
            sys.exit(1)
        download_dataset(args.dataset, auto_open_browser=auto_open)
