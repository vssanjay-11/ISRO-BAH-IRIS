"""
configs/settings.py
IRIS-AI User Settings Manager

Persists user-configurable overrides as JSON.
Falls back to config.py defaults for any missing key.
"""

from __future__ import annotations

import os
import sys
import json
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config

# ---------------------------------------------------------------------------
# Default settings (mirrors config.py values; user overrides are layered on top)
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    # Processing
    "device":              config.DEVICE,
    "realesrgan_scale":    config.REALESRGAN_SCALE,
    "realesrgan_tile":     config.REALESRGAN_TILE,
    "fastsam_conf":        config.FASTSAM_CONF,
    "fastsam_img_size":    config.FASTSAM_IMG_SIZE,
    # Enhancement
    "clahe_clip_limit":    config.CLAHE_CLIP_LIMIT,
    "gamma_value":         config.GAMMA_VALUE,
    "denoise_h":           config.DENOISE_H,
    "target_size":         list(config.TARGET_SIZE),
    # Output
    "output_dir":          config.OUTPUTS_DIR,
    "reports_dir":         config.REPORTS_DIR,
    "downloads_dir":       config.DOWNLOADS_DIR,
    # Captioning
    "caption_max_length":  config.CAPTION_MAX_LENGTH,
    "caption_num_beams":   config.CAPTION_NUM_BEAMS,
    # UI
    "theme":               "dark",
    "show_intermediate":   True,
    "auto_report":         False,
}


class Settings:
    """
    Singleton user-settings store.

    Usage:
        s = Settings.get()
        scale = s["realesrgan_scale"]
        s.set("realesrgan_scale", 2)
        s.save()
    """

    _instance: "Settings | None" = None

    def __init__(self):
        self._path = config.SETTINGS_PATH
        self._data: dict = {}
        self._load()

    @classmethod
    def get(cls) -> "Settings":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    def _load(self):
        """Load from JSON file; fall back to defaults for missing keys."""
        stored = {}
        if os.path.isfile(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    stored = json.load(f)
            except (json.JSONDecodeError, OSError):
                stored = {}
        # Merge: defaults <- stored
        self._data = {**_DEFAULTS, **stored}

    def save(self):
        """Persist current settings to JSON."""
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def reset(self):
        """Reset all settings to defaults and save."""
        self._data = dict(_DEFAULTS)
        self.save()

    # ------------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data.get(key, _DEFAULTS.get(key))

    def __setitem__(self, key: str, value: Any):
        self._data[key] = value

    def set(self, key: str, value: Any):
        self[key] = value

    def get_all(self) -> dict:
        return dict(self._data)

    def as_dict(self) -> dict:
        return self.get_all()

    # ------------------------------------------------------------------
    # Convenience typed accessors
    # ------------------------------------------------------------------
    @property
    def device(self) -> str:
        return self["device"]

    @property
    def realesrgan_scale(self) -> int:
        return int(self["realesrgan_scale"])

    @property
    def fastsam_conf(self) -> float:
        return float(self["fastsam_conf"])

    @property
    def clahe_clip_limit(self) -> float:
        return float(self["clahe_clip_limit"])

    @property
    def gamma_value(self) -> float:
        return float(self["gamma_value"])

    @property
    def target_size(self) -> tuple:
        raw = self["target_size"]
        return tuple(raw) if isinstance(raw, (list, tuple)) else config.TARGET_SIZE

    @property
    def theme(self) -> str:
        return self["theme"]
