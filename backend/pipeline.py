"""
backend/pipeline.py
Master IRIS-AI Inference Pipeline

Orchestrates: Enhancement → Super Resolution → Colorization →
              Object Detection → Image Captioning → Save Outputs
"""

import os
import sys
import time
import uuid
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config
from backend.enhancer         import enhance_image
from backend.super_resolution import super_resolve
from backend.colorizer        import colorize
from backend.detector         import detect_objects
from backend.captioner        import generate_caption


def run_pipeline(
    image_input,
    run_enhancement: bool  = True,
    run_superres: bool     = True,
    run_colorize: bool     = True,
    run_detection: bool    = True,
    run_caption: bool      = True,
    session_id: str        = None,
    progress_callback      = None,   # callable(step: str, pct: float)
) -> dict:
    """
    Full IRIS-AI pipeline.

    Args:
        image_input       : file path, PIL.Image, or numpy array
        run_* flags       : toggle individual stages
        session_id        : unique session identifier (auto-generated if None)
        progress_callback : optional callable for Streamlit progress updates

    Returns:
        Comprehensive result dict with all images, detections, captions, timings.
    """
    session_id  = session_id or str(uuid.uuid4())[:8]
    session_dir = os.path.join(config.OUTPUTS_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    def _progress(step, pct):
        if progress_callback:
            progress_callback(step, pct)
        print(f"[Pipeline] [{pct:3.0f}%] {step}")

    results = {
        "session_id":   session_id,
        "session_dir":  session_dir,
        "timings":      {},
        "errors":       [],
    }

    pipeline_t0 = time.perf_counter()

    # ──────────────────────────────────────────────
    # Stage 0: Load original image
    # ──────────────────────────────────────────────
    _progress("Loading image", 0)
    try:
        if isinstance(image_input, Image.Image):
            original_pil = image_input.convert("RGB")
        elif isinstance(image_input, str):
            original_pil = Image.open(image_input).convert("RGB")
        else:
            import cv2
            import numpy as np
            if isinstance(image_input, bytes):
                from io import BytesIO
                original_pil = Image.open(BytesIO(image_input)).convert("RGB")
            else:
                # numpy
                if image_input.ndim == 2:
                    image_input = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
                original_pil = Image.fromarray(image_input)
        results["original_pil"] = original_pil
        _save(original_pil, session_dir, "00_original.jpg")
    except Exception as e:
        results["errors"].append(f"Load: {e}")
        return results

    current_pil = original_pil.copy()

    # ──────────────────────────────────────────────
    # Stage 1: Enhancement
    # ──────────────────────────────────────────────
    _progress("Enhancing image (CLAHE, Denoising, Gamma …)", 10)
    if run_enhancement:
        try:
            enh = enhance_image(current_pil)
            results["enhanced_pil"]   = enh["enhanced_pil"]
            results["enhancement_steps"] = enh["steps"]
            results["timings"]["enhancement_ms"] = enh["elapsed_ms"]
            current_pil = enh["enhanced_pil"]
            _save(current_pil, session_dir, "01_enhanced.jpg")
        except Exception as e:
            results["errors"].append(f"Enhancement: {e}")
            results["enhanced_pil"] = current_pil
    else:
        results["enhanced_pil"] = current_pil

    # ──────────────────────────────────────────────
    # Stage 2: Super Resolution
    # ──────────────────────────────────────────────
    _progress("Super Resolution (Real-ESRGAN x4)", 30)
    if run_superres:
        try:
            sr = super_resolve(current_pil)
            results["sr_pil"]           = sr["sr_pil"]
            results["sr_method"]        = sr["method"]
            results["timings"]["sr_ms"] = sr["elapsed_ms"]
            current_pil = sr["sr_pil"]
            _save(current_pil, session_dir, "02_super_res.jpg")
        except Exception as e:
            results["errors"].append(f"SuperRes: {e}")
            results["sr_pil"] = current_pil
    else:
        results["sr_pil"] = current_pil

    # ──────────────────────────────────────────────
    # Stage 3: Colorization
    # ──────────────────────────────────────────────
    _progress("Colorizing infrared image …", 55)
    if run_colorize:
        try:
            col = colorize(current_pil)
            results["colorized_pil"]           = col["colorized_pil"]
            results["timings"]["colorize_ms"]  = col["elapsed_ms"]
            current_pil = col["colorized_pil"]
            _save(current_pil, session_dir, "03_colorized.jpg")
        except Exception as e:
            results["errors"].append(f"Colorize: {e}")
            results["colorized_pil"] = current_pil
    else:
        results["colorized_pil"] = current_pil

    # ──────────────────────────────────────────────
    # Stage 4: Object Detection
    # ──────────────────────────────────────────────
    _progress("Detecting objects (YOLOv11) …", 70)
    if run_detection:
        try:
            det = detect_objects(current_pil)
            results["annotated_pil"]           = det["annotated_pil"]
            results["detections"]              = det["detections"]
            results["detection_summary"]       = det["summary"]
            results["total_objects"]           = det["total"]
            results["timings"]["detection_ms"] = det["elapsed_ms"]
            _save(det["annotated_pil"], session_dir, "04_detected.jpg")
        except Exception as e:
            results["errors"].append(f"Detection: {e}")
            results["detections"]        = []
            results["detection_summary"] = {}
            results["total_objects"]     = 0
            results["annotated_pil"]     = current_pil
    else:
        results["detections"]        = []
        results["detection_summary"] = {}
        results["total_objects"]     = 0
        results["annotated_pil"]     = current_pil

    # ──────────────────────────────────────────────
    # Stage 5: Image Captioning
    # ──────────────────────────────────────────────
    _progress("Generating scene description …", 85)
    if run_caption:
        try:
            cap = generate_caption(
                results.get("colorized_pil", current_pil),
                results.get("detections", []),
            )
            results["caption"]                 = cap["caption"]
            results["caption_method"]          = cap.get("method", "unknown")
            results["timings"]["caption_ms"]   = cap["elapsed_ms"]
        except Exception as e:
            results["errors"].append(f"Caption: {e}")
            results["caption"]        = "Caption generation unavailable."
            results["caption_method"] = "error"
    else:
        results["caption"]        = ""
        results["caption_method"] = "disabled"

    # ──────────────────────────────────────────────
    # Totals
    # ──────────────────────────────────────────────
    results["timings"]["total_ms"] = (time.perf_counter() - pipeline_t0) * 1000
    _progress("Pipeline complete ✓", 100)

    return results


def _save(pil_img: Image.Image, directory: str, filename: str):
    """Save a PIL image to disk, ignore errors."""
    try:
        path = os.path.join(directory, filename)
        pil_img.save(path, quality=95)
    except Exception as e:
        print(f"[Pipeline] Save warning: {e}")
