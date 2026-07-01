"""
backend/pipeline_v2.py
IRIS-AI Master Pipeline v2

Replaces pipeline.py with:
- ModelManager integration (no duplicate model loading)
- FastSAM segmentation (replaces YOLO detection)
- Scene interpretation via interpreter.py
- Smart timestamped file organisation
- Reuses util/util.py::save_image(), mkdir(), tensor2im()

Existing pipeline.py is retained for backward compatibility.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import datetime
from typing import Callable, Optional
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config

# Reuse from existing repository
from util.util import save_image, mkdir, mkdirs, tensor2im

# Reuse from IRIS-AI backend (no duplication)
from backend.enhancer         import enhance_image, load_image_as_numpy
from backend.super_resolution import super_resolve
from backend.colorizer        import colorize
from backend.segmenter        import segment_image
from backend.interpreter      import generate_interpretation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_pil(pil_img: Image.Image, path: str) -> None:
    """Save PIL image — wraps util/util.py::save_image() for numpy input."""
    try:
        import numpy as np
        import cv2
        arr = np.array(pil_img.convert("RGB"))
        # util.save_image expects HxWx3 uint8
        save_image(arr, path)     # reuse existing repo function
    except Exception as e:
        print(f"[Pipeline_v2] Save warning: {e}")


def _timestamped_session_dir(base_dir: str, filename_hint: str = "") -> str:
    """
    Create a timestamped output folder:
        outputs/YYYYMMDD_HHMMSS_<stem>/
    Uses util/util.py::mkdir() — no custom makedirs.
    """
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = os.path.splitext(os.path.basename(filename_hint))[0][:20] if filename_hint else uuid.uuid4().hex[:6]
    path = os.path.join(base_dir, f"{ts}_{stem}")
    mkdir(path)    # reuse existing repo function
    return path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline_v2(
    image_input,
    model_manager       = None,
    run_enhancement:  bool = True,
    run_superres:     bool = True,
    run_colorize:     bool = True,
    run_segmentation: bool = True,
    run_caption:      bool = True,
    session_id: str        = None,
    progress_callback: Optional[Callable] = None,
    source_filename: str   = "",
) -> dict:
    """
    Full IRIS-AI v2 pipeline.

    Differences from pipeline_v2:
    - Takes optional model_manager; falls back to standalone functions
    - Uses FastSAM (segmenter.py) instead of YOLOv11
    - Generates ISRO interpreter paragraph (interpreter.py)
    - Smart timestamped output folder (util/util.py::mkdir)
    - All images saved with canonical filenames:
        00_original.png / 01_enhanced.png / 02_super_res.png
        03_colorized.png / 04_segmented.png / 05_overlay.png
        06_caption.txt

    Args:
        image_input       : PIL.Image, file path, bytes, or numpy array
        model_manager     : ModelManager instance (or None for standalone)
        run_*             : Stage toggles
        session_id        : Optional override for session naming
        progress_callback : callable(step_name: str, pct: float)
        source_filename   : Original filename for session folder naming

    Returns:
        Comprehensive results dict
    """
    session_id  = session_id or uuid.uuid4().hex[:8]
    session_dir = _timestamped_session_dir(config.OUTPUTS_DIR, source_filename)

    def _progress(step: str, pct: float):
        if progress_callback:
            try:
                progress_callback(step, pct)
            except Exception:
                pass
        print(f"[Pipeline_v2] [{pct:3.0f}%] {step}")

    results = {
        "session_id":  session_id,
        "session_dir": session_dir,
        "timings":     {},
        "errors":      [],
    }

    t_total = time.perf_counter()

    # ── Stage 0: Load original image ─────────────────────────────────────
    _progress("Loading image …", 0)
    try:
        if isinstance(image_input, Image.Image):
            original_pil = image_input.convert("RGB")
        elif isinstance(image_input, str):
            original_pil = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            from io import BytesIO
            original_pil = Image.open(BytesIO(image_input)).convert("RGB")
        else:
            import numpy as np
            if image_input.ndim == 2:
                import cv2
                image_input = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            original_pil = Image.fromarray(image_input)

        results["original_pil"] = original_pil
        results["original_size"] = original_pil.size
        _save_pil(original_pil, os.path.join(session_dir, "00_original.png"))
    except Exception as e:
        results["errors"].append(f"Load: {e}")
        return results

    current_pil = original_pil.copy()

    # ── Stage 1: Enhancement ─────────────────────────────────────────────
    _progress("Enhancing image (CLAHE, Denoising, Gamma) …", 10)
    if run_enhancement:
        try:
            enh = enhance_image(current_pil)    # reuse existing enhancer.py
            results["enhanced_pil"]       = enh["enhanced_pil"]
            results["enhancement_steps"]  = enh["steps"]
            results["timings"]["enhancement_ms"] = enh["elapsed_ms"]
            current_pil = enh["enhanced_pil"]
            _save_pil(current_pil, os.path.join(session_dir, "01_enhanced.png"))
        except Exception as e:
            results["errors"].append(f"Enhancement: {e}")
            results["enhanced_pil"] = current_pil
    else:
        results["enhanced_pil"] = current_pil

    # ── Stage 2: Super Resolution ─────────────────────────────────────────
    _progress("Super Resolution (Real-ESRGAN x4) …", 25)
    if run_superres:
        try:
            sr = super_resolve(current_pil)    # reuse existing super_resolution.py
            results["sr_pil"]           = sr["sr_pil"]
            results["sr_method"]        = sr["method"]
            results["timings"]["sr_ms"] = sr["elapsed_ms"]
            current_pil = sr["sr_pil"]
            _save_pil(current_pil, os.path.join(session_dir, "02_super_res.png"))
        except Exception as e:
            results["errors"].append(f"SuperRes: {e}")
            results["sr_pil"] = current_pil
    else:
        results["sr_pil"] = current_pil

    # ── Stage 3: Colorization ─────────────────────────────────────────────
    _progress("Colorizing infrared image …", 45)
    if run_colorize:
        try:
            col = colorize(current_pil)    # reuse existing colorizer.py
            results["colorized_pil"]          = col["colorized_pil"]
            results["timings"]["colorize_ms"] = col["elapsed_ms"]
            current_pil = col["colorized_pil"]
            _save_pil(current_pil, os.path.join(session_dir, "03_colorized.png"))
        except Exception as e:
            results["errors"].append(f"Colorize: {e}")
            results["colorized_pil"] = current_pil
    else:
        results["colorized_pil"] = current_pil

    # ── Stage 4: Semantic Segmentation ────────────────────────────────────
    _progress("Semantic Segmentation (FastSAM) …", 65)
    if run_segmentation:
        try:
            seg = segment_image(current_pil, model_manager=model_manager)
            results["segmented_pil"]           = seg["segmented_pil"]
            results["overlay_pil"]             = seg["overlay_pil"]
            results["masks"]                   = seg["masks"]
            results["region_summary"]          = seg["region_summary"]
            results["interpretation_labels"]   = seg["interpretation"]
            results["total_regions"]           = seg["total_regions"]
            results["timings"]["segment_ms"]   = seg["elapsed_ms"]
            results["seg_method"]              = seg["method"]

            _save_pil(seg["segmented_pil"], os.path.join(session_dir, "04_segmented.png"))
            _save_pil(seg["overlay_pil"],   os.path.join(session_dir, "05_overlay.png"))
        except Exception as e:
            results["errors"].append(f"Segmentation: {e}")
            results["segmented_pil"]         = current_pil
            results["overlay_pil"]           = current_pil
            results["region_summary"]        = {}
            results["interpretation_labels"] = []
            results["total_regions"]         = 0
    else:
        results["segmented_pil"]         = current_pil
        results["overlay_pil"]           = current_pil
        results["region_summary"]        = {}
        results["interpretation_labels"] = []
        results["total_regions"]         = 0

    # ── Stage 5: Scene Interpretation ────────────────────────────────────
    _progress("Generating ISRO interpretation …", 82)
    if run_caption:
        try:
            interp = generate_interpretation(
                pil_image   = results.get("colorized_pil", current_pil),
                seg_results = {
                    "interpretation": results.get("interpretation_labels", []),
                    "region_summary": results.get("region_summary", {}),
                },
                use_blip    = True,
                model_manager = model_manager,
            )
            results["caption"]                   = interp["interpretation"]
            results["interpretation"]            = interp["interpretation"]
            results["timings"]["caption_ms"]     = interp["elapsed_ms"]
            results["caption_method"]            = "blip+isro" if interp["blip_used"] else "template"

            # Save caption text — util/util.py doesn't have a text saver, use plain open
            caption_path = os.path.join(session_dir, "06_caption.txt")
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(results["caption"])
        except Exception as e:
            results["errors"].append(f"Interpretation: {e}")
            results["caption"]        = "Scene interpretation unavailable."
            results["interpretation"] = results["caption"]
            results["caption_method"] = "error"
    else:
        results["caption"]        = ""
        results["interpretation"] = ""
        results["caption_method"] = "disabled"

    # ── Totals ────────────────────────────────────────────────────────────
    results["timings"]["total_ms"] = (time.perf_counter() - t_total) * 1000
    _progress("Pipeline complete.", 100)

    return results
