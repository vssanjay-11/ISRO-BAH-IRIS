"""
backend/enhancer.py
Image Enhancement Module for IRIS-AI

Pipeline: Raw IR Image → CLAHE → Denoising → Histogram Equalization →
           Gamma Correction → Adaptive Contrast → Resize → Normalize
"""

import cv2
import numpy as np
from PIL import Image
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def load_image_as_numpy(image_input) -> np.ndarray:
    """
    Accept PIL Image, numpy array, or file path.
    Returns HxWxC uint8 numpy array (BGR for OpenCV ops).
    """
    if isinstance(image_input, np.ndarray):
        img = image_input.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return img
    elif isinstance(image_input, Image.Image):
        img = np.array(image_input.convert("RGB"))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_input}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")


def apply_clahe(img_bgr: np.ndarray,
                clip_limit: float = config.CLAHE_CLIP_LIMIT,
                tile_grid: tuple = config.CLAHE_TILE_GRID) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization on L-channel."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def apply_denoising(img_bgr: np.ndarray, h: float = config.DENOISE_H) -> np.ndarray:
    """Fast Non-Local Means denoising."""
    return cv2.fastNlMeansDenoisingColored(img_bgr, None, h, h, 7, 21)


def apply_histogram_equalization(img_bgr: np.ndarray) -> np.ndarray:
    """Per-channel histogram equalization for additional contrast."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    out = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)


def apply_gamma_correction(img_bgr: np.ndarray, gamma: float = config.GAMMA_VALUE) -> np.ndarray:
    """Gamma correction using lookup table."""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img_bgr, table)


def apply_adaptive_contrast(img_bgr: np.ndarray) -> np.ndarray:
    """Unsharp masking for adaptive local contrast enhancement."""
    blurred = cv2.GaussianBlur(img_bgr, (9, 9), 10.0)
    sharpened = cv2.addWeighted(img_bgr, 1.5, blurred, -0.5, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def auto_resize(img_bgr: np.ndarray,
                target_size: tuple = config.TARGET_SIZE) -> np.ndarray:
    """Resize image to target size with aspect-ratio-preserving padding."""
    h, w = img_bgr.shape[:2]
    target_w, target_h = target_size

    # Compute scale
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Pad to exact target
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def normalize_image(img_bgr: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 255] range (min-max stretching)."""
    img_float = img_bgr.astype(np.float32)
    min_val, max_val = img_float.min(), img_float.max()
    if max_val - min_val < 1e-6:
        return img_bgr
    normalized = (img_float - min_val) / (max_val - min_val) * 255.0
    return normalized.astype(np.uint8)


def enhance_image(image_input,
                  apply_clahe_flag: bool = True,
                  apply_denoise_flag: bool = True,
                  apply_histeq_flag: bool = False,
                  apply_gamma_flag: bool = True,
                  apply_contrast_flag: bool = True,
                  apply_resize_flag: bool = True,
                  apply_norm_flag: bool = True,
                  target_size: tuple = config.TARGET_SIZE) -> dict:
    """
    Full enhancement pipeline. Returns a dict with:
        - original_bgr   : original image as BGR numpy array
        - enhanced_bgr   : final enhanced BGR numpy array
        - enhanced_pil   : final enhanced PIL RGB image
        - steps          : dict of intermediate step images (PIL)
        - elapsed_ms     : time taken in milliseconds
    """
    t0 = time.perf_counter()
    img = load_image_as_numpy(image_input)

    # Store original
    original_bgr = img.copy()
    steps = {}

    # Step 1: CLAHE
    if apply_clahe_flag:
        img = apply_clahe(img)
    steps["clahe"] = _bgr_to_pil(img)

    # Step 2: Denoising
    if apply_denoise_flag:
        img = apply_denoising(img)
    steps["denoised"] = _bgr_to_pil(img)

    # Step 3: Histogram Equalization (optional – can over-saturate)
    if apply_histeq_flag:
        img = apply_histogram_equalization(img)
    steps["histeq"] = _bgr_to_pil(img)

    # Step 4: Gamma Correction
    if apply_gamma_flag:
        img = apply_gamma_correction(img)
    steps["gamma"] = _bgr_to_pil(img)

    # Step 5: Adaptive Contrast
    if apply_contrast_flag:
        img = apply_adaptive_contrast(img)
    steps["adaptive_contrast"] = _bgr_to_pil(img)

    # Step 6: Auto Resize
    if apply_resize_flag:
        img = auto_resize(img, target_size)
    steps["resized"] = _bgr_to_pil(img)

    # Step 7: Normalize
    if apply_norm_flag:
        img = normalize_image(img)
    steps["normalized"] = _bgr_to_pil(img)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "original_bgr":  original_bgr,
        "original_pil":  _bgr_to_pil(original_bgr),
        "enhanced_bgr":  img,
        "enhanced_pil":  _bgr_to_pil(img),
        "steps":         steps,
        "elapsed_ms":    elapsed_ms,
    }


def _bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """Convert BGR numpy to RGB PIL image."""
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
