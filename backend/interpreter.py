"""
backend/interpreter.py
ISRO Analyst Scene Interpretation

Generates a professional ISRO-style paragraph from:
  - FastSAM segmentation results (from backend/segmenter.py)
  - BLIP caption (reused from backend/captioner.py)
  - Image metadata

Key design: WRAPS captioner.py — does NOT duplicate generate_caption().
"""

from __future__ import annotations

import os
import sys
import time
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config

# Reuse existing captioner — no duplication
from backend.captioner import generate_caption, _rule_based_caption


# ---------------------------------------------------------------------------
# ISRO Analyst Paragraph Templates
# ---------------------------------------------------------------------------

_OPENING_TEMPLATES = [
    "Spectral analysis of the acquired infrared imagery reveals",
    "Post-processing assessment of the thermal infrared acquisition indicates",
    "Interpretation of the infrared satellite imagery identifies",
    "Radiometric analysis of the enhanced infrared frame confirms",
]

_REGION_DESCRIPTORS = {
    "vegetation":  "distributed vegetation cover indicative of agricultural or forested zones",
    "water":       "water bodies exhibiting characteristic low-emissivity thermal signatures",
    "road":        "linear transportation infrastructure and road network patterns",
    "building":    "built-up structures with elevated thermal radiance profiles",
    "bare_land":   "exposed bare terrain with heterogeneous surface emissivity",
    "urban":       "dense urban agglomeration exhibiting high thermal density",
    "background":  "background terrain features",
}

_CLOSING_TEMPLATES = [
    (
        "The colorized output facilitates human interpretation by mapping thermal "
        "gradients to visually distinguishable spectral bands. This imagery is "
        "suitable for land-use classification, change detection, and mission "
        "planning workflows within the ISRO operational remote sensing framework."
    ),
    (
        "Real-ESRGAN super-resolution enhancement has improved spatial detail "
        "retention, enabling more precise delineation of terrain boundaries. "
        "The processed output aligns with ISRO PS-10 objectives for improved "
        "object interpretation in infrared remote sensing applications."
    ),
    (
        "The IR-colorization pipeline preserves the underlying thermal structure "
        "while providing enhanced visual contrast. These results support downstream "
        "geospatial analysis including habitat mapping, urban heat island detection, "
        "and disaster response operations."
    ),
]


def generate_interpretation(
    pil_image: Image.Image,
    seg_results: dict  = None,
    blip_caption: str  = None,
    use_blip: bool     = True,
    model_manager      = None,
) -> dict:
    """
    Generate a professional ISRO analyst interpretation paragraph.

    Strategy:
    1. If BLIP caption available (from captioner.py) — use as base.
    2. Augment with segmentation region labels from segmenter.py.
    3. Wrap with ISRO-style opening and closing templates.

    Args:
        pil_image:    Colorized or enhanced PIL image for BLIP (if used)
        seg_results:  Dict returned by backend/segmenter.py::segment_image()
        blip_caption: Pre-generated caption string (skip BLIP call if provided)
        use_blip:     Whether to call BLIP captioner (True by default)
        model_manager: Optional ModelManager for BLIP model access

    Returns:
        dict:
            interpretation  str — full ISRO analyst paragraph
            blip_used       bool
            elapsed_ms      float
            region_labels   list[str]
    """
    t0 = time.perf_counter()

    # ── Collect region labels from segmentation ───────────────────────────
    region_labels: list[str] = []
    region_summary: dict     = {}

    if seg_results:
        region_labels  = seg_results.get("interpretation", [])
        region_summary = seg_results.get("region_summary", {})

    # ── BLIP caption (reuse backend/captioner.py) ─────────────────────────
    blip_used = False
    if blip_caption:
        base_caption = blip_caption
        blip_used    = True
    elif use_blip:
        try:
            cap_result   = generate_caption(pil_image, detections=[])
            base_caption = cap_result.get("caption", "")
            blip_used    = cap_result.get("method", "rule-based") == "blip"
        except Exception:
            base_caption = ""
    else:
        base_caption = ""

    # ── Build ISRO paragraph ─────────────────────────────────────────────
    import random
    random.seed(42)   # deterministic for reproducibility

    opening = random.choice(_OPENING_TEMPLATES)
    closing = random.choice(_CLOSING_TEMPLATES)

    # Describe detected regions
    if region_labels:
        region_descs = [
            _REGION_DESCRIPTORS.get(lbl, lbl)
            for lbl in region_labels
            if lbl != "background"
        ]
        if region_descs:
            if len(region_descs) == 1:
                region_text = region_descs[0]
            elif len(region_descs) == 2:
                region_text = f"{region_descs[0]} and {region_descs[1]}"
            else:
                region_text = (
                    ", ".join(region_descs[:-1]) + ", and " + region_descs[-1]
                )
            semantic_sentence = (
                f"{opening} {region_text}. "
                f"Semantic segmentation delineates {len(region_summary)} distinct "
                f"region class{'es' if len(region_summary) > 1 else ''}, "
                f"providing improved terrain discrimination capability."
            )
        else:
            semantic_sentence = (
                f"{opening} heterogeneous terrain features with complex thermal signatures. "
                f"Segmentation analysis was performed on the colorized infrared output."
            )
    else:
        semantic_sentence = (
            f"{opening} a complex infrared scene with multiple terrain features. "
            f"Colorization enhances the visual separation of thermal signatures "
            f"for improved human interpretation."
        )

    # Combine BLIP base with ISRO framing
    if base_caption and len(base_caption) > 30:
        # Strip the RS prompt prefix if BLIP echoed it
        clean_base = base_caption.replace(
            "An aerial or satellite infrared image showing", ""
        ).strip().strip(".")

        middle_sentence = (
            f" The image depicts {clean_base}. " if clean_base else " "
        )
    else:
        middle_sentence = " "

    full_paragraph = semantic_sentence + middle_sentence + closing

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "interpretation": full_paragraph,
        "blip_used":      blip_used,
        "elapsed_ms":     elapsed_ms,
        "region_labels":  region_labels,
    }
