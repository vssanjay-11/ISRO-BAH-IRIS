"""
config.py
IRIS-AI Central Configuration
Intelligent Remote-sensing Infrared Interpretation Suite
ISRO Bharatiya Antariksh Hackathon 2026 – PS-10
"""

import os
import torch

# -----------------------------------------------------------------------------
# Device (auto-detected by gpu_utils.py; mirrored here for non-MM use)
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Root Paths
# -----------------------------------------------------------------------------
ROOT_DIR      = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR   = os.path.join(ROOT_DIR, "weights")
OUTPUTS_DIR   = os.path.join(ROOT_DIR, "outputs")
REPORTS_DIR   = os.path.join(ROOT_DIR, "reports")
ASSETS_DIR    = os.path.join(ROOT_DIR, "assets")
CONFIGS_DIR   = os.path.join(ROOT_DIR, "configs")
FRONTEND_DIR  = os.path.join(ROOT_DIR, "frontend")
BACKEND_DIR   = os.path.join(ROOT_DIR, "backend")
DATASETS_DIR  = os.path.join(ROOT_DIR, "datasets")
DOCS_DIR      = os.path.join(ROOT_DIR, "docs")
DOWNLOADS_DIR = os.path.join(ROOT_DIR, "downloads")

# -----------------------------------------------------------------------------
# Colorization Model  (Existing Repository — NEVER MODIFIED)
# Integration: backend/colorizer.py builds argparse.Namespace from these values
#              and passes it to models.create_model() without touching test.py
# -----------------------------------------------------------------------------
COLORIZATION_CHECKPOINT_DIR  = os.path.join(ROOT_DIR, "checkpoints")
COLORIZATION_EXPERIMENT_NAME = "experiment_name"
COLORIZATION_EPOCH           = "latest"
COLORIZATION_MODEL           = "COLOR"
COLORIZATION_NETG            = "UNet_2Plus"
COLORIZATION_NORM            = "instance"
COLORIZATION_INPUT_NC        = 3
COLORIZATION_OUTPUT_NC       = 3
COLORIZATION_NGF             = 64
COLORIZATION_CROP_SIZE       = 256
COLORIZATION_LOAD_SIZE       = 256

# Google Drive file-id for the colorization checkpoint.
# download_weights.py uses gdown to fetch it automatically.
COLORIZATION_GDRIVE_FILE_ID  = "1vVRhSVFs6yt2R-17dTURf5NeNzAscvkk"   # folder ID
COLORIZATION_WEIGHTS_PATH    = os.path.join(
    COLORIZATION_CHECKPOINT_DIR, COLORIZATION_EXPERIMENT_NAME, "latest_net_G.pth"
)

# -----------------------------------------------------------------------------
# Real-ESRGAN Super Resolution
# Integration: backend/super_resolution.py + ModelManager.load_esrgan()
# -----------------------------------------------------------------------------
REALESRGAN_WEIGHTS_URL  = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
REALESRGAN_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4plus.pth")
REALESRGAN_SCALE        = 4
REALESRGAN_MODEL_NAME   = "RealESRGAN_x4plus"
REALESRGAN_TILE         = 0       # 0 = no tiling; set ~400 for low-VRAM GPUs
REALESRGAN_TILE_PAD     = 10
REALESRGAN_PRE_PAD      = 0

# -----------------------------------------------------------------------------
# FastSAM Semantic Segmentation  (replaces YOLOv11 bounding-box detection)
# Integration: backend/segmenter.py + ModelManager.load_fastsam()
# NOTE: FastSAM is part of `ultralytics` — already in requirements.txt
# -----------------------------------------------------------------------------
FASTSAM_WEIGHTS_URL  = "https://github.com/ultralytics/assets/releases/download/v8.3.0/FastSAM-s.pt"
FASTSAM_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "FastSAM-s.pt")
FASTSAM_MODEL_NAME   = "FastSAM-s.pt"
FASTSAM_CONF         = 0.4
FASTSAM_IOU          = 0.9
FASTSAM_IMG_SIZE     = 1024

# Semantic overlay colours for remote-sensing classes (RGB tuples)
FASTSAM_CLASS_COLORS = {
    "road":       (255, 200,   0),
    "building":   (200,  80,  80),
    "vegetation": ( 60, 179,  60),
    "water":      ( 30, 144, 255),
    "bare_land":  (210, 180, 140),
    "urban":      (128,   0, 128),
    "default":    ( 50,  50,  50),
}

# -----------------------------------------------------------------------------
# Image Enhancement
# Integration: backend/enhancer.py reads these defaults
# NOTE: do NOT redefine IMG_EXTENSIONS here;
#       reuse data/image_folder.py::IMG_EXTENSIONS instead.
# -----------------------------------------------------------------------------
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID  = (8, 8)
DENOISE_H        = 10
GAMMA_VALUE      = 1.2
TARGET_SIZE      = (256, 256)

# -----------------------------------------------------------------------------
# Image Captioning — BLIP
# Integration: backend/captioner.py + ModelManager.load_blip()
# -----------------------------------------------------------------------------
CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
CAPTION_MAX_LENGTH = 200
CAPTION_MIN_LENGTH = 30
CAPTION_NUM_BEAMS  = 5

# -----------------------------------------------------------------------------
# PDF Report
# Integration: backend/report_generator.py
# -----------------------------------------------------------------------------
REPORT_LOGO_PATH = os.path.join(ASSETS_DIR, "isro_logo.png")
REPORT_FONT      = "Helvetica"
REPORT_AUTHOR    = "IRIS-AI | ISRO BAH 2026 PS-10"

# -----------------------------------------------------------------------------
# ZIP Export
# Integration: backend/exporter.py
# -----------------------------------------------------------------------------
ZIP_OUTPUT_DIR        = DOWNLOADS_DIR
ZIP_INCLUDE_ORIGINALS = True
ZIP_COMPRESSION_LEVEL = 6

# -----------------------------------------------------------------------------
# Batch Processing
# Integration: backend/batch_processor.py reuses data/image_folder.py::make_dataset()
# -----------------------------------------------------------------------------
BATCH_RECURSIVE  = True
BATCH_MAX_IMAGES = 500

# -----------------------------------------------------------------------------
# Settings
# Integration: configs/settings.py persists user overrides as JSON
# -----------------------------------------------------------------------------
SETTINGS_PATH = os.path.join(CONFIGS_DIR, "user_settings.json")

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
APP_TITLE    = "IRIS-AI"
APP_SUBTITLE = "Intelligent Remote-sensing Infrared Interpretation Suite"
APP_ICON     = "satellite"
APP_VERSION  = "2.0.0"

# -----------------------------------------------------------------------------
# Auto-create all required directories on import
# Uses os.makedirs — same as util/util.py::mkdir() but done once at import time
# -----------------------------------------------------------------------------
for _d in [
    WEIGHTS_DIR, OUTPUTS_DIR, REPORTS_DIR, ASSETS_DIR,
    CONFIGS_DIR, FRONTEND_DIR, BACKEND_DIR,
    DATASETS_DIR, DOCS_DIR, DOWNLOADS_DIR,
    os.path.join(COLORIZATION_CHECKPOINT_DIR, COLORIZATION_EXPERIMENT_NAME),
]:
    os.makedirs(_d, exist_ok=True)
