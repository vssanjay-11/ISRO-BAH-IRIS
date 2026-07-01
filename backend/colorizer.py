"""
backend/colorizer.py
Wrapper around the existing IR-colorization model for single-image inference.

DOES NOT modify any existing repository code.
Calls: options/test_options.py, models/__init__.py, models/COLOR_model.py
"""

import os
import sys
import time
import numpy as np
from PIL import Image
import torch

# ── Ensure repo root is on path ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config
from util.util import tensor2im

# Lazy imports – only resolved on first call
_model = None
_opt   = None


def _build_fake_opt():
    """
    Build an argparse Namespace that mirrors what TestOptions().parse() would
    produce, but without needing sys.argv / a dataroot directory.
    Uses argparse.Namespace so existing base_model.py picks it up transparently.
    """
    from argparse import Namespace
    return Namespace(
        isTrain=False,
        # Model
        model=config.COLORIZATION_MODEL,
        netG=config.COLORIZATION_NETG,
        norm=config.COLORIZATION_NORM,
        input_nc=config.COLORIZATION_INPUT_NC,
        output_nc=config.COLORIZATION_OUTPUT_NC,
        ngf=config.COLORIZATION_NGF,
        no_dropout=True,
        no_antialias=False,
        no_antialias_up=False,
        init_type="xavier",
        init_gain=0.02,
        stylegan2_G_num_downsampling=1,
        # Checkpoint
        checkpoints_dir=config.COLORIZATION_CHECKPOINT_DIR,
        name=config.COLORIZATION_EXPERIMENT_NAME,
        epoch=config.COLORIZATION_EPOCH,
        load_iter=0,
        # Runtime
        gpu_ids=[0] if config.DEVICE == "cuda" and torch.cuda.is_available() else [],
        verbose=False,
        # Dataset (unused at inference but required by BaseModel)
        preprocess="none",
        direction="AtoB",
        # Unused training params (required by COLORModel.__init__ signature)
        attn_layers="4, 7, 9",
        patch_nums=256,
        patch_size=64,
        loss_mode="cos",
        use_norm=False,
        learned_attn=False,
        augment=False,
        T=0.07,
        lambda_spatial=10.0,
        lambda_spatial_idt=0.0,
        lambda_perceptual=1.0,
        lambda_style=1.0,
        lambda_identity=0.0,
        lambda_gradient=0.0,
    )


def _load_model():
    global _model, _opt
    if _model is not None:
        return

    from models import create_model

    _opt = _build_fake_opt()
    _model = create_model(_opt)

    # Provide a minimal dummy batch so data_dependent_initialize works
    device = torch.device(f"cuda:{_opt.gpu_ids[0]}" if _opt.gpu_ids else "cpu")
    h = w = config.COLORIZATION_CROP_SIZE
    dummy = {
        "A": torch.zeros(1, config.COLORIZATION_INPUT_NC, h, w),
        "B": torch.zeros(1, config.COLORIZATION_OUTPUT_NC, h, w),
        "A_paths": ["dummy"],
        "B_paths": ["dummy"],
    }
    _model.data_dependent_initialize(dummy)
    _model.setup(_opt)
    _model.parallelize()
    _model.eval()
    print("[Colorizer] Model loaded successfully.")


def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image → (1, 3, H, W) float tensor in [-1, 1]."""
    import torchvision.transforms.functional as TF
    t = TF.to_tensor(pil_img.convert("RGB"))   # [0,1]
    t = t * 2.0 - 1.0                          # [-1,1]
    return t.unsqueeze(0)


def colorize(pil_image: Image.Image) -> dict:
    """
    Run the IR colorization model on a single PIL image.

    Args:
        pil_image: PIL.Image – should be the enhanced image (RGB, 256×256).

    Returns:
        dict with:
            colorized_pil  : PIL.Image – colorized output
            elapsed_ms     : float – inference time in ms
    """
    _load_model()

    # Resize to expected crop size
    h = w = config.COLORIZATION_CROP_SIZE
    img_resized = pil_image.resize((w, h), Image.LANCZOS)

    device = torch.device(f"cuda:{_opt.gpu_ids[0]}" if _opt.gpu_ids else "cpu")
    tensor_A = pil_to_tensor(img_resized).to(device)
    tensor_B = torch.zeros_like(tensor_A)

    data = {
        "A": tensor_A,
        "B": tensor_B,
        "A_paths": ["input"],
        "B_paths": ["input"],
    }

    t0 = time.perf_counter()
    with torch.no_grad():
        _model.set_input(data)
        _model.test()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    visuals = _model.get_current_visuals()

    # The COLOR model stores output in 'fake_B'
    out_key = "fake_B" if "fake_B" in visuals else list(visuals.keys())[-1]
    colorized_np = tensor2im(visuals[out_key])      # HxWx3 uint8
    colorized_pil = Image.fromarray(colorized_np)

    return {
        "colorized_pil": colorized_pil,
        "elapsed_ms":    elapsed_ms,
    }
