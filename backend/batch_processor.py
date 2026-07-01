"""
backend/batch_processor.py
IRIS-AI Batch Processing Engine

Supports: single image, multiple images, folder, recursive folder.

Key reuse (from dependency map):
- data/image_folder.py::make_dataset()    — recursive image discovery
- data/image_folder.py::is_image_file()   — extension filtering
- data/image_folder.py::IMG_EXTENSIONS    — supported extensions list
- util/util.py::mkdirs()                  — directory creation
- backend/pipeline_v2.py::run_pipeline_v2 — per-image inference
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Union
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config

# Reuse from existing repository
from data.image_folder import make_dataset, is_image_file, IMG_EXTENSIONS
from util.util import mkdirs

# Reuse from IRIS-AI
from backend.pipeline_v2 import run_pipeline_v2


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    """Result container for a single image in a batch."""
    index:      int
    path:       str
    success:    bool
    results:    dict       = field(default_factory=dict)
    error:      str        = ""
    elapsed_ms: float      = 0.0


@dataclass
class BatchSummary:
    """Aggregated summary for an entire batch run."""
    total:        int
    succeeded:    int
    failed:       int
    elapsed_ms:   float
    results:      List[BatchResult] = field(default_factory=list)
    output_dirs:  List[str]         = field(default_factory=list)


# ---------------------------------------------------------------------------
# Image Discovery (wraps data/image_folder.py)
# ---------------------------------------------------------------------------

def discover_images(
    source: Union[str, List[str]],
    recursive: bool = True,
    max_images: int = config.BATCH_MAX_IMAGES,
) -> List[str]:
    """
    Discover all supported images from a path or list of paths.

    Reuses:
      - data/image_folder.py::make_dataset() for directory scanning
      - data/image_folder.py::is_image_file() for single-file check
      - data/image_folder.py::IMG_EXTENSIONS for the extension list

    Args:
        source     : file path, folder path, or list of file paths
        recursive  : scan subdirectories (default True)
        max_images : upper limit (config.BATCH_MAX_IMAGES)

    Returns:
        Sorted list of absolute image paths
    """
    paths: List[str] = []

    sources = [source] if isinstance(source, str) else source

    for s in sources:
        s = os.path.abspath(s)

        if os.path.isfile(s):
            if is_image_file(s):    # reuse from data/image_folder.py
                paths.append(s)
            else:
                print(f"[Batch] Skipping non-image file: {s}")

        elif os.path.isdir(s):
            # make_dataset already walks recursively
            discovered = make_dataset(s, max_dataset_size=max_images)   # reuse
            paths.extend(discovered)
        else:
            print(f"[Batch] Path not found: {s}")

    # Deduplicate and sort
    paths = sorted(set(paths))
    if len(paths) > max_images:
        print(f"[Batch] Clamping to {max_images} images (found {len(paths)})")
        paths = paths[:max_images]

    return paths


# ---------------------------------------------------------------------------
# Batch Runner
# ---------------------------------------------------------------------------

def run_batch(
    source: Union[str, List[str]],
    model_manager              = None,
    recursive:       bool      = config.BATCH_RECURSIVE,
    max_images:      int       = config.BATCH_MAX_IMAGES,
    run_enhancement: bool      = True,
    run_superres:    bool      = True,
    run_colorize:    bool      = True,
    run_segmentation: bool     = True,
    run_caption:     bool      = True,
    progress_callback: Optional[Callable] = None,
    stop_event                 = None,   # threading.Event for cancellation
) -> BatchSummary:
    """
    Process one or many images through the full IRIS-AI pipeline.

    Progress callback signature: (current_idx, total, image_path, stage, stage_pct)

    Args:
        source:           file path, folder, or list of paths
        model_manager:    ModelManager instance (preferred)
        recursive:        scan subdirectories
        max_images:       safety cap
        run_*:            stage toggles (passed through to pipeline_v2)
        progress_callback: UI progress hook
        stop_event:       threading.Event — set() to cancel mid-batch

    Returns:
        BatchSummary with per-image results
    """
    # Discover images using existing data/image_folder.py functions
    image_paths = discover_images(source, recursive=recursive, max_images=max_images)
    total = len(image_paths)

    if total == 0:
        print(f"[Batch] No images found in: {source}")
        return BatchSummary(
            total=0, succeeded=0, failed=0, elapsed_ms=0.0
        )

    print(f"[Batch] Processing {total} image(s) …")
    print(f"[Batch] Supported extensions: {', '.join(IMG_EXTENSIONS[:6])} …")

    batch_results: List[BatchResult] = []
    output_dirs:   List[str]         = []
    succeeded = 0
    failed    = 0
    t_batch   = time.perf_counter()

    for idx, img_path in enumerate(image_paths):
        # Cancellation support
        if stop_event is not None and stop_event.is_set():
            print(f"[Batch] Cancelled at image {idx+1}/{total}")
            break

        print(f"\n[Batch] [{idx+1}/{total}] {os.path.basename(img_path)}")
        t_img = time.perf_counter()

        # Per-image progress hook wrapping pipeline_v2 progress
        def _img_progress(step: str, pct: float, _idx=idx, _total=total, _path=img_path):
            if progress_callback:
                try:
                    progress_callback(_idx, _total, _path, step, pct)
                except Exception:
                    pass

        # Estimate time remaining
        elapsed_so_far = (time.perf_counter() - t_batch)
        avg_per_image  = elapsed_so_far / max(idx, 1)
        remaining      = avg_per_image * (total - idx)

        print(f"  Est. remaining: {remaining:.0f}s | "
              f"Completed: {idx} | Remaining: {total - idx}")

        try:
            result = run_pipeline_v2(
                image_input       = img_path,
                model_manager     = model_manager,
                run_enhancement   = run_enhancement,
                run_superres      = run_superres,
                run_colorize      = run_colorize,
                run_segmentation  = run_segmentation,
                run_caption       = run_caption,
                progress_callback = _img_progress,
                source_filename   = img_path,
            )
            elapsed_img = (time.perf_counter() - t_img) * 1000

            br = BatchResult(
                index      = idx,
                path       = img_path,
                success    = len(result.get("errors", [])) == 0,
                results    = result,
                elapsed_ms = elapsed_img,
            )
            succeeded += 1
            output_dirs.append(result.get("session_dir", ""))

        except Exception as e:
            elapsed_img = (time.perf_counter() - t_img) * 1000
            print(f"  [Batch] Error on {img_path}: {e}")
            br = BatchResult(
                index      = idx,
                path       = img_path,
                success    = False,
                error      = str(e),
                elapsed_ms = elapsed_img,
            )
            failed += 1

        batch_results.append(br)

    total_elapsed = (time.perf_counter() - t_batch) * 1000

    print(f"\n[Batch] Complete: {succeeded} OK, {failed} failed in {total_elapsed/1000:.1f}s")

    return BatchSummary(
        total       = total,
        succeeded   = succeeded,
        failed      = failed,
        elapsed_ms  = total_elapsed,
        results     = batch_results,
        output_dirs = [d for d in output_dirs if d],
    )
