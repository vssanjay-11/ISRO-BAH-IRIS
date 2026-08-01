import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.gpu_utils import detect_device, get_gpu_memory_usage, DeviceInfo, device_badge_html

def test_detect_device():
    info = detect_device()
    assert isinstance(info, DeviceInfo)
    assert info.device in ["cuda", "mps", "cpu"]
    assert isinstance(info.device_name, str)
    assert info.torch_version != ""

def test_get_gpu_memory_usage():
    usage = get_gpu_memory_usage()
    assert isinstance(usage, dict)
    if usage:
        assert "vram_used_mb" in usage
        assert "vram_total_mb" in usage

def test_device_badge_html():
    info = detect_device()
    badge = device_badge_html(info)
    assert isinstance(badge, str)
    assert "<span" in badge
