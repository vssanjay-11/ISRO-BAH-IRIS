"""
backend/exporter.py
IRIS-AI ZIP Export Module

One-click export of complete analysis:
  original + enhanced + super-res + colorized + segmented + overlay + caption + PDF report

Reuses:
  - backend/report_generator.py::generate_pdf_report()
  - util/util.py::mkdir()
  - config.ZIP_OUTPUT_DIR, ZIP_COMPRESSION_LEVEL
"""

from __future__ import annotations

import os
import sys
import io
import zipfile
import datetime
from pathlib import Path
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config

# Reuse existing modules
from util.util import mkdir
from backend.report_generator import generate_pdf_report


# ---------------------------------------------------------------------------
# Image keys to export
# ---------------------------------------------------------------------------

_IMAGE_EXPORTS = [
    ("original_pil",   "00_original.png"),
    ("enhanced_pil",   "01_enhanced.png"),
    ("sr_pil",         "02_super_resolution.png"),
    ("colorized_pil",  "03_colorized.png"),
    ("segmented_pil",  "04_segmented.png"),
    ("overlay_pil",    "05_overlay.png"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_zip(
    pipeline_results: dict,
    output_path: str = None,
    include_pdf: bool = True,
    include_caption: bool = True,
    compression_level: int = config.ZIP_COMPRESSION_LEVEL,
) -> bytes:
    """
    Generate a ZIP archive of all IRIS-AI analysis outputs.

    Reuses:
    - generate_pdf_report() from backend/report_generator.py
    - mkdir() from util/util.py
    - config.ZIP_OUTPUT_DIR, config.ZIP_COMPRESSION_LEVEL

    Args:
        pipeline_results  : dict from backend/pipeline_v2.py::run_pipeline_v2()
        output_path       : optional save path; if None returns bytes only
        include_pdf       : whether to generate and include PDF report
        include_caption   : whether to include caption .txt
        compression_level : ZIP compression (0=store, 9=max)

    Returns:
        ZIP bytes
    """
    session_id = pipeline_results.get("session_id", "iris_session")
    ts         = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name   = f"IRIS_AI_{session_id}_{ts}.zip"

    if output_path is None:
        mkdir(config.ZIP_OUTPUT_DIR)
        output_path = os.path.join(config.ZIP_OUTPUT_DIR, zip_name)

    buf = io.BytesIO()
    compress = zipfile.ZIP_DEFLATED if compression_level > 0 else zipfile.ZIP_STORED

    with zipfile.ZipFile(buf, mode="w", compression=compress,
                         compresslevel=compression_level) as zf:

        # ── Images ──────────────────────────────────────────────────────
        for key, filename in _IMAGE_EXPORTS:
            pil = pipeline_results.get(key)
            if pil is None:
                continue
            try:
                img_buf = io.BytesIO()
                pil.convert("RGB").save(img_buf, format="PNG")
                zf.writestr(os.path.join("images", filename), img_buf.getvalue())
            except Exception as e:
                print(f"[Exporter] Could not export {filename}: {e}")

        # ── Session folder images (already saved to disk) ────────────────
        session_dir = pipeline_results.get("session_dir", "")
        if session_dir and os.path.isdir(session_dir):
            for fname in sorted(os.listdir(session_dir)):
                fpath = os.path.join(session_dir, fname)
                if os.path.isfile(fpath):
                    with open(fpath, "rb") as f:
                        zf.writestr(os.path.join("session", fname), f.read())

        # ── Caption text ─────────────────────────────────────────────────
        if include_caption:
            caption = pipeline_results.get("caption") or pipeline_results.get("interpretation", "")
            if caption:
                zf.writestr("caption.txt", caption.encode("utf-8"))

        # ── PDF report — reuse report_generator.py ────────────────────────
        if include_pdf:
            try:
                pdf_bytes = generate_pdf_report(pipeline_results, output_path=None)
                zf.writestr(f"IRIS_AI_Report_{session_id}.pdf", pdf_bytes)
            except Exception as e:
                print(f"[Exporter] PDF generation failed: {e}")

        # ── Manifest ──────────────────────────────────────────────────────
        timings   = pipeline_results.get("timings", {})
        total_ms  = timings.get("total_ms", 0)
        manifest  = [
            "IRIS-AI Analysis Export",
            f"Session ID : {session_id}",
            f"Timestamp  : {ts}",
            f"Total Time : {total_ms:.0f} ms ({total_ms/1000:.2f} s)",
            f"Device     : {config.DEVICE.upper()}",
            f"Version    : IRIS-AI v{config.APP_VERSION}",
            "",
            "Contents:",
        ]
        for key, filename in _IMAGE_EXPORTS:
            if pipeline_results.get(key) is not None:
                manifest.append(f"  images/{filename}")
        if include_caption and pipeline_results.get("caption"):
            manifest.append("  caption.txt")
        if include_pdf:
            manifest.append(f"  IRIS_AI_Report_{session_id}.pdf")

        zf.writestr("MANIFEST.txt", "\n".join(manifest).encode("utf-8"))

    zip_bytes = buf.getvalue()

    # Save to disk
    mkdir(os.path.dirname(output_path))
    with open(output_path, "wb") as f:
        f.write(zip_bytes)

    size_mb = len(zip_bytes) / 1024 / 1024
    print(f"[Exporter] ZIP saved ({size_mb:.1f} MB) → {output_path}")

    return zip_bytes
