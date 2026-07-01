"""
utils/model_manager.py
IRIS-AI ModelManager — Thread-safe Singleton for all model lifecycle.

Design constraints (enforced by dependency map):
- All model loading calls download_weights.py (not inline URL fetches)
- Colorizer opt built using same pattern as backend/colorizer.py::_build_fake_opt()
- Device resolved once via utils/gpu_utils.py::detect_device()
- .run() delegates entirely to backend/pipeline_v2.py::run_pipeline_v2()
- No duplication of enhancer.py, super_resolution.py, colorizer.py logic
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Optional, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config


class ModelManager:
    """
    Singleton model registry for IRIS-AI.

    Usage:
        mm = ModelManager.get_instance()
        results = mm.run(pil_image, ...)
    """

    _instance: Optional["ModelManager"] = None
    _lock: threading.Lock = threading.Lock()

    # Shared model handles — class-level so they persist across get_instance() calls
    _colorizer_model   = None
    _colorizer_opt     = None
    _esrgan_upsampler  = None
    _fastsam_model     = None
    _blip_processor    = None
    _blip_model        = None

    _colorizer_loaded: bool = False
    _esrgan_loaded:    bool = False
    _fastsam_loaded:   bool = False
    _blip_loaded:      bool = False

    def __init__(self):
        """Private — use ModelManager.get_instance()."""
        from utils.gpu_utils import detect_device     # reuse gpu_utils.py
        self.device_info = detect_device()
        self.device      = self.device_info.device
        print(f"[ModelManager] Initialized | Device: {self.device_info.device_name}")

    @classmethod
    def get_instance(cls) -> "ModelManager":
        """Thread-safe singleton accessor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ─────────────────────────────────────────────────────────────────────
    # Model Loaders
    # ─────────────────────────────────────────────────────────────────────

    def load_colorizer(self) -> bool:
        """
        Load the existing IR-colorization GAN (once).
        Uses same Namespace pattern as backend/colorizer.py::_build_fake_opt()
        — no duplication: _build_fake_opt() is imported directly.
        """
        if self._colorizer_loaded:
            return self._colorizer_model is not None

        try:
            import torch
            from models import create_model           # existing repo factory
            from backend.colorizer import _build_fake_opt   # reuse existing opt builder

            opt = _build_fake_opt()
            model = create_model(opt)

            h = w = config.COLORIZATION_CROP_SIZE
            dummy = {
                "A": torch.zeros(1, 3, h, w),
                "B": torch.zeros(1, 3, h, w),
                "A_paths": ["dummy"],
                "B_paths": ["dummy"],
            }
            model.data_dependent_initialize(dummy)
            model.setup(opt)
            model.parallelize()
            model.eval()

            self._colorizer_model  = model
            self._colorizer_opt    = opt
            self._colorizer_loaded = True
            print("[ModelManager] Colorizer loaded.")
            return True

        except Exception as e:
            print(f"[ModelManager] Colorizer load failed: {e}")
            self._colorizer_loaded = True   # mark as attempted to avoid retries
            return False

    def load_esrgan(self) -> bool:
        """
        Load Real-ESRGAN (once).
        Download delegated to download_weights.py::download_realesrgan().
        """
        if self._esrgan_loaded:
            return self._esrgan_upsampler not in (None, "bicubic")

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            from download_weights import download_realesrgan   # centralised downloader

            weights = download_realesrgan()

            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23, num_grow_ch=32,
                scale=config.REALESRGAN_SCALE,
            )
            self._esrgan_upsampler = RealESRGANer(
                scale      = config.REALESRGAN_SCALE,
                model_path = weights,
                model      = model,
                tile       = config.REALESRGAN_TILE,
                tile_pad   = config.REALESRGAN_TILE_PAD,
                pre_pad    = config.REALESRGAN_PRE_PAD,
                half       = (self.device == "cuda"),
            )
            self._esrgan_loaded = True
            print("[ModelManager] Real-ESRGAN loaded.")
            return True

        except Exception as e:
            print(f"[ModelManager] ESRGAN failed ({e}). Bicubic fallback.")
            self._esrgan_upsampler = "bicubic"
            self._esrgan_loaded    = True
            return False

    def load_fastsam(self) -> bool:
        """
        Load FastSAM (once).
        Download delegated to download_weights.py::download_fastsam().
        """
        if self._fastsam_loaded:
            return self._fastsam_model is not None

        try:
            from ultralytics import FastSAM
            from download_weights import download_fastsam   # centralised downloader

            weights = download_fastsam() or config.FASTSAM_WEIGHTS_PATH
            self._fastsam_model  = FastSAM(weights)
            self._fastsam_loaded = True
            print("[ModelManager] FastSAM loaded.")
            return True

        except Exception as e:
            print(f"[ModelManager] FastSAM failed: {e}")
            self._fastsam_model  = None
            self._fastsam_loaded = True
            return False

    def load_blip(self) -> bool:
        """Load BLIP captioning (once)."""
        if self._blip_loaded:
            return self._blip_model is not None

        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration

            self._blip_processor = BlipProcessor.from_pretrained(config.CAPTION_MODEL_NAME)
            self._blip_model     = BlipForConditionalGeneration.from_pretrained(
                config.CAPTION_MODEL_NAME
            ).to(self.device)
            self._blip_model.eval()
            self._blip_loaded = True
            print("[ModelManager] BLIP loaded.")
            return True

        except Exception as e:
            print(f"[ModelManager] BLIP failed: {e}")
            self._blip_loaded = True
            return False

    def preload_all(self) -> Dict[str, bool]:
        """Load all models — call once at Streamlit app startup."""
        return {
            "colorizer": self.load_colorizer(),
            "esrgan":    self.load_esrgan(),
            "fastsam":   self.load_fastsam(),
            "blip":      self.load_blip(),
        }

    def get_status(self) -> Dict[str, str]:
        """Return human-readable load status for the Streamlit sidebar."""
        def _s(loaded: bool, handle) -> str:
            if not loaded:
                return "Not loaded"
            if handle is None or handle == "bicubic":
                return "Fallback"
            return "Ready"

        return {
            "Colorizer (UNet++ GAN)": _s(self._colorizer_loaded, self._colorizer_model),
            "Real-ESRGAN x4":         _s(self._esrgan_loaded,    self._esrgan_upsampler),
            "FastSAM Segmentation":   _s(self._fastsam_loaded,   self._fastsam_model),
            "BLIP Captioning":        _s(self._blip_loaded,      self._blip_model),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Unified Run Interface
    # ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        image_input,
        run_enhancement:  bool = True,
        run_superres:     bool = True,
        run_colorize:     bool = True,
        run_segmentation: bool = True,
        run_caption:      bool = True,
        session_id:       str  = None,
        progress_callback       = None,
        source_filename:  str  = "",
    ) -> dict:
        """
        Execute the full IRIS-AI pipeline.
        This is the ONLY method the Streamlit UI should call.
        Delegates to backend/pipeline_v2.py — no logic duplication.
        """
        from backend.pipeline_v2 import run_pipeline_v2

        return run_pipeline_v2(
            image_input       = image_input,
            model_manager     = self,
            run_enhancement   = run_enhancement,
            run_superres      = run_superres,
            run_colorize      = run_colorize,
            run_segmentation  = run_segmentation,
            run_caption       = run_caption,
            session_id        = session_id,
            progress_callback = progress_callback,
            source_filename   = source_filename,
        )
