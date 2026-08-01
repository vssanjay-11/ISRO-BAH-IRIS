"""
Example script demonstrating programmatic usage of the IRIS-AI inference pipeline on a single image.
"""

import os
import sys
import argparse
from PIL import Image

# Ensure the root of the project is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.model_manager import ModelManager

def main():
    parser = argparse.ArgumentParser(description="IRIS-AI Single Image Inference Example")
    parser.add_argument("--image", type=str, required=True, help="Path to input infrared image")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Input image file '{args.image}' not found.")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    print("Initializing ModelManager...")
    mm = ModelManager.get_instance()
    
    # Preload models
    print("Preloading models...")
    mm.preload_all()

    print(f"Running pipeline on image: {args.image}")
    try:
        img = Image.open(args.image)
        results = mm.run(
            image_input=img,
            run_superres=True,
            run_segmentation=True,
            run_caption=True
        )
        
        # Save output images
        colorized = results["colorized_pil"]
        colorized.save(os.path.join(args.outdir, "example_colorized.png"))
        print(f"Saved colorized image to {args.outdir}/example_colorized.png")

        if "overlay_pil" in results and results["overlay_pil"]:
            results["overlay_pil"].save(os.path.join(args.outdir, "example_overlay.png"))
            print(f"Saved segmentation overlay to {args.outdir}/example_overlay.png")

        if "interpretation" in results:
            print("\nGenerated Scene Interpretation:")
            print("-" * 50)
            print(results["interpretation"])
            print("-" * 50)

        print("\nProcessing timings:")
        for step, timing in results.get("timings", {}).items():
            print(f" - {step}: {timing:.4f}s")

    except Exception as e:
        print(f"An error occurred during inference: {e}")

if __name__ == "__main__":
    main()
