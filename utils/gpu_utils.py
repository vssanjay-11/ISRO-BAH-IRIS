"""
utils/gpu_utils.py
GPU/Device Detection and Memory Monitoring for IRIS-AI
"""

from __future__ import annotations

import os
import sys
import platform
from dataclasses import dataclass, field
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@dataclass
class DeviceInfo:
    """Structured device information returned by detect_device()."""
    device: str                    # 'cuda', 'mps', or 'cpu'
    device_name: str               # Human-readable device name
    cuda_available: bool = False
    mps_available: bool  = False
    gpu_count: int       = 0
    vram_total_mb: float = 0.0
    vram_free_mb: float  = 0.0
    vram_used_mb: float  = 0.0
    ram_total_mb: float  = 0.0
    ram_free_mb: float   = 0.0
    python_version: str  = ""
    os_name: str         = ""
    torch_version: str   = ""
    cuda_version: str    = ""


def detect_device() -> DeviceInfo:
    """
    Auto-detect the best available compute device.

    Priority: CUDA → Apple MPS → CPU

    Returns:
        DeviceInfo dataclass with full hardware details.
    """
    import torch

    info = DeviceInfo(
        device="cpu",
        device_name="CPU",
        python_version=platform.python_version(),
        os_name=f"{platform.system()} {platform.release()}",
        torch_version=torch.__version__,
    )

    # ── CUDA ──────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        info.cuda_available = True
        info.device         = "cuda"
        info.gpu_count      = torch.cuda.device_count()
        info.cuda_version   = torch.version.cuda or "N/A"

        try:
            idx = torch.cuda.current_device()
            info.device_name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            info.vram_total_mb = props.total_memory / 1024 ** 2
            mem = torch.cuda.mem_get_info(idx)
            info.vram_free_mb  = mem[0] / 1024 ** 2
            info.vram_used_mb  = info.vram_total_mb - info.vram_free_mb
        except Exception:
            info.device_name = f"CUDA GPU ({info.gpu_count} device(s))"

    # ── Apple MPS (M-series) ──────────────────────────────────────────────
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info.mps_available = True
        info.device        = "mps"
        info.device_name   = "Apple Silicon (MPS)"

    # ── CPU Fallback ──────────────────────────────────────────────────────
    else:
        try:
            import cpuinfo  # py-cpuinfo (optional)
            info.device_name = cpuinfo.get_cpu_info().get("brand_raw", "CPU")
        except ImportError:
            info.device_name = platform.processor() or "CPU"

    # ── RAM ───────────────────────────────────────────────────────────────
    try:
        import psutil
        vm = psutil.virtual_memory()
        info.ram_total_mb = vm.total / 1024 ** 2
        info.ram_free_mb  = vm.available / 1024 ** 2
    except ImportError:
        pass

    return info


def get_gpu_memory_usage() -> dict:
    """
    Return current VRAM usage dict. Returns empty dict on CPU/MPS.
    Keys: vram_used_mb, vram_free_mb, vram_total_mb, utilization_pct
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        idx = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(idx)
        used  = total - free
        return {
            "vram_used_mb":      used  / 1024 ** 2,
            "vram_free_mb":      free  / 1024 ** 2,
            "vram_total_mb":     total / 1024 ** 2,
            "utilization_pct":   used / total * 100 if total > 0 else 0,
        }
    except Exception:
        return {}


def device_badge_html(info: DeviceInfo) -> str:
    """Return an HTML badge string for the Streamlit UI."""
    icons = {"cuda": "⚡", "mps": "🍎", "cpu": "💻"}
    colors = {"cuda": "#06D6A0", "mps": "#FF9F43", "cpu": "#9BAAC8"}
    icon  = icons.get(info.device, "💻")
    color = colors.get(info.device, "#9BAAC8")
    return (
        f'<span style="background:rgba(6,214,160,0.15); border:1px solid {color}; '
        f'color:{color}; border-radius:20px; padding:3px 12px; font-size:0.75rem; '
        f'font-weight:700;">{icon} {info.device_name.upper()}</span>'
    )
