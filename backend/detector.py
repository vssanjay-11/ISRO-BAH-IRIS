"""
backend/detector.py
YOLOv11 Object Detection Module for IRIS-AI

Uses Ultralytics pretrained weights only — no training.
Detects COCO-class objects relevant to remote-sensing interpretation.
"""

import os
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config

_model = None

# COCO class names relevant to remote sensing (highlight these)
REMOTE_SENSING_CLASSES = {
    "car", "truck", "bus", "motorcycle", "bicycle",
    "airplane", "boat", "ship",
    "train",
    "person",
    "building",          # not standard COCO but kept for future
}

# COCO class id → friendly label mapping
COCO_FRIENDLY = {
    0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle",
    4: "Airplane", 5: "Bus", 6: "Train", 7: "Truck", 8: "Boat",
    9: "Traffic Light", 10: "Fire Hydrant", 11: "Stop Sign",
    13: "Bench", 14: "Bird", 15: "Cat", 16: "Dog",
    17: "Horse", 18: "Sheep", 19: "Cow", 20: "Elephant",
    21: "Bear", 22: "Zebra", 23: "Giraffe",
    56: "Chair", 57: "Couch", 58: "Potted Plant",
    59: "Bed", 60: "Dining Table", 61: "Toilet",
    62: "TV", 63: "Laptop", 67: "Cell Phone",
    72: "Refrigerator", 73: "Book",
}


def _load_model():
    global _model
    if _model is not None:
        return
    try:
        from ultralytics import YOLO
        weights = config.YOLO_WEIGHTS_PATH
        # Ultralytics will auto-download if the file is just a model name
        if not os.path.isfile(weights):
            weights = config.YOLO_MODEL_NAME   # e.g. "yolo11n.pt" → auto-download
        _model = YOLO(weights)
        print(f"[Detector] YOLOv11 loaded: {weights}")
    except ImportError:
        print("[Detector] ultralytics not installed. Run: pip install ultralytics")
        _model = None


def detect_objects(pil_image: Image.Image) -> dict:
    """
    Run YOLOv11 on a PIL image.

    Returns:
        dict with:
            annotated_pil  : PIL.Image with bounding boxes drawn
            detections     : list of dicts {label, confidence, bbox [x1,y1,x2,y2]}
            summary        : dict {class: count}
            total          : int – total objects detected
            elapsed_ms     : float
    """
    _load_model()
    t0 = time.perf_counter()

    detections = []
    summary    = {}
    annotated  = pil_image.copy()

    if _model is None:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "annotated_pil": annotated,
            "detections":    detections,
            "summary":       summary,
            "total":         0,
            "elapsed_ms":    elapsed_ms,
            "error":         "ultralytics not installed",
        }

    # Run inference
    results = _model(
        pil_image,
        conf=config.YOLO_CONF_THRESHOLD,
        iou=config.YOLO_IOU_THRESHOLD,
        imgsz=config.YOLO_IMG_SIZE,
        device=config.YOLO_DEVICE,
        verbose=False,
        classes=config.YOLO_CLASSES_OF_INTEREST,
    )

    # Parse results
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf   = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            label  = COCO_FRIENDLY.get(cls_id, result.names.get(cls_id, f"Class_{cls_id}"))

            detections.append({
                "label":      label,
                "confidence": conf,
                "bbox":       [x1, y1, x2, y2],
                "class_id":   cls_id,
            })
            summary[label] = summary.get(label, 0) + 1

    # Draw annotations
    annotated = _draw_boxes(pil_image.copy(), detections)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "annotated_pil": annotated,
        "detections":    detections,
        "summary":       summary,
        "total":         len(detections),
        "elapsed_ms":    elapsed_ms,
    }


def _draw_boxes(img: Image.Image, detections: list) -> Image.Image:
    """Draw bounding boxes and labels on a PIL image."""
    draw = ImageDraw.Draw(img)

    # Color palette for detection boxes
    colors = [
        "#FF4444", "#FF8C00", "#FFD700", "#00C851", "#33B5E5",
        "#AA66CC", "#FF6680", "#00B8D4", "#64DD17", "#FF6D00",
    ]

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for i, det in enumerate(detections):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['label']} {det['confidence']:.0%}"

        # Box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Label background
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1 - th - 6, x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - th - 3), label, fill="white", font=font)

    return img
