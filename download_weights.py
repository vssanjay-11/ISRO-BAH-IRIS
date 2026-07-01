"""
download_weights.py
IRIS-AI Automatic Weight Downloader

Checks, downloads, and verifies all required pretrained weights.
Reuses config.py for all paths and URLs.
Run: python download_weights.py
"""

from __future__ import annotations

import os
import sys
import hashlib
import urllib.request
from pathlib import Path
from typing import Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config

# ---------------------------------------------------------------------------
# Known SHA-256 checksums for integrity verification
# ---------------------------------------------------------------------------
CHECKSUMS: dict[str, str] = {
    "RealESRGAN_x4plus.pth": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
    "FastSAM-s.pt":           "",   # populated once stable release is pinned
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _verify(path: str, name: str) -> bool:
    expected = CHECKSUMS.get(name, "")
    if not expected:
        return True   # no checksum registered → skip verification
    actual = _sha256(path)
    if actual == expected:
        print(f"  [checksum] OK  {name}")
        return True
    print(f"  [checksum] FAIL {name}: got {actual[:16]}… expected {expected[:16]}…")
    return False


def _download(url: str, dest: str, label: str) -> bool:
    """Download url → dest, show progress, return True on success."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    print(f"  Downloading {label} …")
    print(f"    URL  : {url}")
    print(f"    Dest : {dest}")

    try:
        def _hook(blocks, block_size, total):
            if total > 0:
                pct = min(100, blocks * block_size * 100 // total)
                bar = "#" * (pct // 5)
                print(f"\r    [{bar:<20}] {pct:3d}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=_hook)
        print()   # newline after progress bar
        size_mb = os.path.getsize(dest) / 1024 / 1024
        print(f"  Downloaded {size_mb:.1f} MB → {dest}")
        return True
    except Exception as e:
        print(f"\n  Download failed: {e}")
        if os.path.isfile(dest):
            os.remove(dest)
        return False


def _is_valid(path: str) -> bool:
    """A weight file is considered valid if it exists and is > 1 KB."""
    return os.path.isfile(path) and os.path.getsize(path) > 1024


# ---------------------------------------------------------------------------
# Public download functions (called by ModelManager and CLI)
# ---------------------------------------------------------------------------

def download_realesrgan() -> str:
    """
    Ensure Real-ESRGAN weights exist. Download if missing.
    Returns the path to the weights file.
    """
    path = config.REALESRGAN_WEIGHTS_PATH
    name = os.path.basename(path)

    if _is_valid(path):
        print(f"[WeightDL] Real-ESRGAN already present: {path}")
        return path

    print(f"[WeightDL] Real-ESRGAN weights missing — downloading …")
    ok = _download(config.REALESRGAN_WEIGHTS_URL, path, name)
    if ok:
        _verify(path, name)
    return path if _is_valid(path) else None


def download_fastsam() -> str:
    """
    Ensure FastSAM weights exist. Download if missing.
    Returns the path to the weights file.
    """
    path = config.FASTSAM_WEIGHTS_PATH
    name = os.path.basename(path)

    if _is_valid(path):
        print(f"[WeightDL] FastSAM already present: {path}")
        return path

    print(f"[WeightDL] FastSAM weights missing — downloading …")
    ok = _download(config.FASTSAM_WEIGHTS_URL, path, name)
    if ok:
        _verify(path, name)
    return path if _is_valid(path) else None


def download_colorization_checkpoint() -> Optional[str]:
    """
    Ensure the IR-colorization GAN checkpoint exists.

    Strategy:
    1. Check if latest_net_G.pth exists and is > 1 KB.
    2. Try downloading via gdown (Google Drive).
    3. If gdown unavailable, open browser to the Drive folder.
    4. Return path if found, None otherwise.
    """
    # Accept either naming convention used by base_model.py
    candidates = [
        os.path.join(config.COLORIZATION_CHECKPOINT_DIR,
                     config.COLORIZATION_EXPERIMENT_NAME, "latest_net_G.pth"),
        os.path.join(config.COLORIZATION_CHECKPOINT_DIR,
                     config.COLORIZATION_EXPERIMENT_NAME, "latest_g.pth"),
    ]

    for p in candidates:
        if _is_valid(p):
            print(f"[WeightDL] Colorization checkpoint found: {p}")
            return p

    print("[WeightDL] Colorization checkpoint missing — attempting download …")
    dest_dir = os.path.join(
        config.COLORIZATION_CHECKPOINT_DIR, config.COLORIZATION_EXPERIMENT_NAME
    )
    os.makedirs(dest_dir, exist_ok=True)

    # Try gdown (Google Drive downloader)
    try:
        import gdown
        folder_url = f"https://drive.google.com/drive/folders/{config.COLORIZATION_GDRIVE_FILE_ID}"
        print(f"[WeightDL] Using gdown to download from: {folder_url}")
        gdown.download_folder(folder_url, output=dest_dir, quiet=False, use_cookies=False)

        for p in candidates:
            if _is_valid(p):
                print(f"[WeightDL] Checkpoint downloaded: {p}")
                return p
    except ImportError:
        print("[WeightDL] gdown not installed. Run: pip install gdown")
    except Exception as e:
        print(f"[WeightDL] gdown failed: {e}")

    # Fallback: open browser
    drive_url = f"https://drive.google.com/drive/folders/{config.COLORIZATION_GDRIVE_FILE_ID}"
    print(f"\n[WeightDL] MANUAL DOWNLOAD REQUIRED:")
    print(f"  1. Open: {drive_url}")
    print(f"  2. Download the .pth files")
    print(f"  3. Place them in: {dest_dir}")
    try:
        import webbrowser
        webbrowser.open(drive_url)
        print("[WeightDL] Browser opened automatically.")
    except Exception:
        pass

    return None


def download_all() -> dict:
    """
    Download all required weights.
    Returns status dict: {model_name: path_or_None}
    """
    print("\n" + "=" * 60)
    print("  IRIS-AI Weight Downloader")
    print("  ISRO BAH 2026 PS-10")
    print("=" * 60 + "\n")

    results = {}

    print("[1/3] Colorization Checkpoint (IR-colorization GAN)")
    results["colorization"] = download_colorization_checkpoint()
    print()

    print("[2/3] Real-ESRGAN Super Resolution")
    results["realesrgan"] = download_realesrgan()
    print()

    print("[3/3] FastSAM Segmentation")
    results["fastsam"] = download_fastsam()
    print()

    print("=" * 60)
    print("  Download Summary")
    print("=" * 60)
    for name, path in results.items():
        status = "OK" if path and _is_valid(path) else "MISSING"
        print(f"  {name:<20} [{status}]  {path or 'Not found'}")
    print("=" * 60 + "\n")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    download_all()
