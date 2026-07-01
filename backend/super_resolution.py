"""
backend/super_resolution.py
Real-ESRGAN Super Resolution Module for IRIS-AI

Uses ONLY pretrained inference – no training.
Auto-downloads weights on first run.
"""

import os
import sys
import time
import urllib.request
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config

_upsampler = None


def _download_weights():
    """Download Real-ESRGAN x4plus weights if not present."""
    weights_path = config.REALESRGAN_WEIGHTS_PATH
    if os.path.isfile(weights_path) and os.path.getsize(weights_path) > 1024:
        return weights_path

    print(f"[SuperRes] Downloading Real-ESRGAN weights → {weights_path}")
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    try:
        urllib.request.urlretrieve(
            config.REALESRGAN_WEIGHTS_URL,
            weights_path,
            reporthook=lambda b, bs, t: None,
        )
        print("[SuperRes] Download complete.")
    except Exception as e:
        print(f"[SuperRes] Download failed: {e}")
        print("[SuperRes] Falling back to bicubic upsampling.")
        if os.path.isfile(weights_path):
            os.remove(weights_path)
        return None

    return weights_path


def _load_upsampler():
    global _upsampler

    if _upsampler is not None:
        return _upsampler

    weights_path = _download_weights()

    if weights_path is None or not os.path.isfile(weights_path):
        _upsampler = "bicubic"
        print("[SuperRes] Using bicubic fallback (Real-ESRGAN weights unavailable).")
        return _upsampler

    try:
        # Real-ESRGAN may not be pip-installed; try import gracefully
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23, num_grow_ch=32,
            scale=config.REALESRGAN_SCALE,
        )
        _upsampler = RealESRGANer(
            scale=config.REALESRGAN_SCALE,
            model_path=weights_path,
            model=model,
            tile=config.REALESRGAN_TILE,
            tile_pad=config.REALESRGAN_TILE_PAD,
            pre_pad=config.REALESRGAN_PRE_PAD,
            half=(config.DEVICE == "cuda"),
        )
        print("[SuperRes] Real-ESRGAN loaded successfully.")
    except ImportError as e:
        print(f"[SuperRes] ImportError ({e}). Install: pip install realesrgan basicsr")
        _upsampler = "bicubic"

    return _upsampler


def super_resolve(pil_image: Image.Image) -> dict:
    """
    Upscale a PIL image using Real-ESRGAN x4.

    Args:
        pil_image: Input PIL.Image (RGB).

    Returns:
        dict with:
            sr_pil      : super-resolved PIL.Image
            scale       : actual scale factor applied
            method      : 'realesrgan' | 'bicubic'
            elapsed_ms  : float
    """
    upsampler = _load_upsampler()
    t0 = time.perf_counter()

    if upsampler == "bicubic":
        w, h = pil_image.size
        sr_img = pil_image.resize(
            (w * config.REALESRGAN_SCALE, h * config.REALESRGAN_SCALE),
            Image.BICUBIC,
        )
        method = "bicubic"
    else:
        # Real-ESRGAN expects BGR numpy
        import cv2
        img_np = np.array(pil_image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        try:
            output, _ = upsampler.enhance(img_bgr, outscale=config.REALESRGAN_SCALE)
            img_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            sr_img = Image.fromarray(img_rgb)
            method = "realesrgan"
        except Exception as e:
            print(f"[SuperRes] Inference error: {e}. Falling back to bicubic.")
            w, h = pil_image.size
            sr_img = pil_image.resize(
                (w * config.REALESRGAN_SCALE, h * config.REALESRGAN_SCALE),
                Image.BICUBIC,
            )
            method = "bicubic"

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "sr_pil":     sr_img,
        "scale":      config.REALESRGAN_SCALE,
        "method":     method,
        "elapsed_ms": elapsed_ms,
    }
