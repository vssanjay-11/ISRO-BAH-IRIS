"""
startup.py — IRIS-AI Cloud Startup Script

Runs automatically before the Streamlit app on Hugging Face Spaces / Streamlit Cloud.
Downloads all required model weights if not already present.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def main():
    print("=" * 60)
    print("  IRIS-AI — Cloud Startup")
    print("=" * 60)

    # Download Real-ESRGAN and FastSAM automatically
    try:
        from download_weights import download_realesrgan, download_fastsam
        print("\n[Startup] Checking Real-ESRGAN weights...")
        download_realesrgan()
        print("\n[Startup] Checking FastSAM weights...")
        download_fastsam()
    except Exception as e:
        print(f"[Startup] Warning: Weight download issue: {e}")

    print("\n[Startup] Ready.\n")


if __name__ == "__main__":
    main()
