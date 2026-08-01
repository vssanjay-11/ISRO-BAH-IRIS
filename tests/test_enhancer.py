import os
import numpy as np
import pytest
from PIL import Image

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.enhancer import (
    load_image_as_numpy,
    apply_gamma_correction,
    auto_resize,
    enhance_image
)

def test_load_image_as_numpy():
    # Test loading from raw numpy array
    dummy_gray = np.zeros((100, 100), dtype=np.uint8)
    img_bgr = load_image_as_numpy(dummy_gray)
    assert img_bgr.shape == (100, 100, 3)

    # Test loading from PIL Image
    dummy_pil = Image.new("RGB", (120, 80), color="red")
    img_bgr_pil = load_image_as_numpy(dummy_pil)
    assert img_bgr_pil.shape == (80, 120, 3)

def test_apply_gamma_correction():
    dummy_bgr = np.ones((50, 50, 3), dtype=np.uint8) * 128
    corrected = apply_gamma_correction(dummy_bgr, gamma=1.0)
    # With gamma 1.0, value should stay close to 128
    assert np.allclose(corrected, 128, atol=2)

def test_auto_resize():
    dummy_bgr = np.zeros((100, 200, 3), dtype=np.uint8)
    resized = auto_resize(dummy_bgr, target_size=(256, 256))
    assert resized.shape == (256, 256, 3)

def test_enhance_image():
    dummy_pil = Image.new("RGB", (128, 128), color="gray")
    res = enhance_image(dummy_pil, target_size=(256, 256))
    assert "enhanced_pil" in res
    assert "steps" in res
    assert res["enhanced_pil"].size == (256, 256)
