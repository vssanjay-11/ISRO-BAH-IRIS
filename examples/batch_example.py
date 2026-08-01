"""
Example script demonstrating batch processing of a folder containing infrared images.
"""

import os
import sys
import argparse

# Ensure the root of the project is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.model_manager import ModelManager
from backend.batch_processor import run_batch

def main():
    parser = argparse.ArgumentParser(description="IRIS-AI Batch Processing Example")
    parser.add_argument("--src", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--outdir", type=str, default="outputs/batch", help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.src) or not os.path.isdir(args.src):
        print(f"Error: Source directory '{args.src}' does not exist or is not a directory.")
        sys.exit(1)

    print("Initializing ModelManager...")
    mm = ModelManager.get_instance()
    mm.preload_all()

    print(f"Starting batch processing of directory: {args.src}")
    
    summary = run_batch(
        source=args.src,
        model_manager=mm,
        recursive=True,
        run_segmentation=True,
        run_caption=True,
        outdir=args.outdir
    )

    print("\nBatch Processing Summary:")
    print("-" * 30)
    print(f"Total processed: {summary.total}")
    print(f"Succeeded:       {summary.succeeded}")
    print(f"Failed:          {summary.failed}")
    print("-" * 30)

if __name__ == "__main__":
    main()
