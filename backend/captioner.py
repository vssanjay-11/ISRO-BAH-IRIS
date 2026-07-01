"""
backend/captioner.py
Lightweight Image Captioning Module for IRIS-AI

Uses Salesforce BLIP (blip-image-captioning-base) via Hugging Face.
Generates a scene-level paragraph for remote sensing interpretation.
"""

import os
import sys
import time
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config

_processor = None
_model     = None


def _load_model():
    global _processor, _model
    if _model is not None:
        return

    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch

        print(f"[Captioner] Loading BLIP: {config.CAPTION_MODEL_NAME}")
        _processor = BlipProcessor.from_pretrained(config.CAPTION_MODEL_NAME)
        _model     = BlipForConditionalGeneration.from_pretrained(
            config.CAPTION_MODEL_NAME
        ).to(config.DEVICE)
        _model.eval()
        print("[Captioner] BLIP loaded successfully.")
    except ImportError as e:
        print(f"[Captioner] ImportError: {e}. Run: pip install transformers accelerate")
        _processor = None
        _model     = None


# Remote-sensing vocabulary hints injected as prompt prefix
_RS_PROMPT = (
    "An aerial or satellite infrared image showing"
)


def generate_caption(pil_image: Image.Image,
                     detections: list = None) -> dict:
    """
    Generate a descriptive caption for a remote sensing image.

    Args:
        pil_image  : PIL.Image (the colorized or enhanced image)
        detections : optional list of detection dicts (used to enrich caption)

    Returns:
        dict with:
            caption     : str – one paragraph description
            elapsed_ms  : float
    """
    _load_model()
    t0 = time.perf_counter()

    if _model is None:
        # Fallback: build a rule-based caption from detections
        caption = _rule_based_caption(pil_image, detections)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {"caption": caption, "elapsed_ms": elapsed_ms, "method": "rule-based"}

    import torch

    # Use conditional generation with RS-specific prompt
    inputs = _processor(
        images=pil_image.convert("RGB"),
        text=_RS_PROMPT,
        return_tensors="pt",
    ).to(config.DEVICE)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_length=config.CAPTION_MAX_LENGTH,
            min_length=config.CAPTION_MIN_LENGTH,
            num_beams=config.CAPTION_NUM_BEAMS,
            early_stopping=True,
        )

    raw_caption = _processor.decode(output_ids[0], skip_special_tokens=True)

    # Augment caption with detection summary
    caption = _augment_caption(raw_caption, detections)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "caption":    caption,
        "elapsed_ms": elapsed_ms,
        "method":     "blip",
    }


def _augment_caption(base_caption: str, detections: list) -> str:
    """Append detection-derived context to BLIP caption."""
    if not detections:
        return base_caption

    from collections import Counter
    counter = Counter(d["label"] for d in detections)
    det_str = ", ".join(f"{v} {k.lower()}{'s' if v > 1 else ''}" for k, v in counter.items())

    return (
        f"{base_caption}. "
        f"Automated analysis identified {len(detections)} object(s) in the scene, "
        f"including {det_str}. "
        f"This imagery is consistent with a remote sensing observation and may be "
        f"suitable for further geospatial analysis or change detection workflows."
    )


def _rule_based_caption(pil_image: Image.Image, detections: list) -> str:
    """Generate a descriptive caption purely from detections (no ML model)."""
    if not detections:
        return (
            "This infrared satellite or aerial image has been processed through the "
            "IRIS-AI pipeline. The scene appears to contain terrain features. "
            "Colorization reveals thermal gradients across the landscape. "
            "Further manual inspection is recommended for detailed interpretation."
        )

    from collections import Counter
    counter = Counter(d["label"] for d in detections)
    det_str = ", ".join(f"{v} {k.lower()}{'s' if v > 1 else ''}" for k, v in counter.items())

    return (
        f"This infrared satellite or aerial image, processed through the IRIS-AI "
        f"enhancement and colorization pipeline, reveals a complex scene containing "
        f"{det_str}. The thermal signature pattern suggests active land use with "
        f"multiple objects of interest. Colorized representation enhances visual "
        f"contrast and aids in distinguishing object classes. This data may be used "
        f"for mission planning, surveillance, or environmental monitoring purposes."
    )
