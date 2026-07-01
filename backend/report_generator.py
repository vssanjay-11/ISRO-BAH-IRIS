"""
backend/report_generator.py
PDF Report Generator for IRIS-AI

Generates a professional ISRO-themed PDF report using ReportLab.
"""

import os
import sys
import io
import datetime
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config


def generate_pdf_report(pipeline_results: dict, output_path: str = None) -> bytes:
    """
    Generate a professional PDF report from pipeline results.

    Args:
        pipeline_results : dict returned by backend.pipeline.run_pipeline()
        output_path      : optional file path to save PDF; if None returns bytes only

    Returns:
        PDF content as bytes
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm, mm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
            Table, TableStyle, HRFlowable, PageBreak,
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        from reportlab.pdfgen import canvas as rl_canvas
    except ImportError:
        raise ImportError("Install reportlab: pip install reportlab")

    # ── Colour palette ──────────────────────────────────────────────────────
    ISRO_ORANGE  = colors.HexColor("#FF6B00")
    ISRO_DARK    = colors.HexColor("#0A1628")
    ISRO_BLUE    = colors.HexColor("#1A3A6B")
    ISRO_LIGHT   = colors.HexColor("#F0F4FF")
    TEXT_DARK    = colors.HexColor("#1C2340")
    ACCENT_TEAL  = colors.HexColor("#00B4D8")
    GREY         = colors.HexColor("#6B7280")

    # ── Setup ───────────────────────────────────────────────────────────────
    if output_path is None:
        output_path = os.path.join(
            config.REPORTS_DIR,
            f"IRIS_AI_Report_{pipeline_results.get('session_id', 'session')}.pdf",
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    def _style(name, **kwargs):
        return ParagraphStyle(name, **kwargs)

    title_style = _style(
        "IRISTitle",
        fontSize=22, textColor=ISRO_ORANGE, alignment=TA_CENTER,
        fontName="Helvetica-Bold", spaceAfter=4,
    )
    subtitle_style = _style(
        "IRISSubtitle",
        fontSize=11, textColor=ISRO_BLUE, alignment=TA_CENTER,
        fontName="Helvetica", spaceAfter=2,
    )
    section_style = _style(
        "IRISSection",
        fontSize=13, textColor=ISRO_BLUE, fontName="Helvetica-Bold",
        spaceBefore=12, spaceAfter=4,
    )
    body_style = _style(
        "IRISBody",
        fontSize=9, textColor=TEXT_DARK, fontName="Helvetica",
        leading=14, alignment=TA_JUSTIFY, spaceAfter=6,
    )
    meta_style = _style(
        "IRISMeta",
        fontSize=8, textColor=GREY, fontName="Helvetica",
        spaceAfter=2,
    )
    caption_style = _style(
        "IRISCaption",
        fontSize=8, textColor=GREY, fontName="Helvetica-Oblique",
        alignment=TA_CENTER, spaceAfter=4,
    )

    story = []

    # ── Cover Header ────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("🛰️ IRIS-AI", title_style))
    story.append(Paragraph("Intelligent Remote-sensing Infrared Interpretation Suite", subtitle_style))
    story.append(Paragraph("ISRO Bharatiya Antariksh Hackathon 2026 — Problem Statement PS-10", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=ISRO_ORANGE, spaceAfter=8))

    # ── Report Metadata ─────────────────────────────────────────────────────
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_id = pipeline_results.get("session_id", "N/A")
    total_ms   = pipeline_results.get("timings", {}).get("total_ms", 0)

    meta_data = [
        ["Report Generated", now],
        ["Session ID",        session_id],
        ["Total Inference Time", f"{total_ms:.0f} ms ({total_ms/1000:.2f} s)"],
        ["Device",            config.DEVICE.upper()],
        ["Model Version",     f"IRIS-AI v{config.APP_VERSION}"],
        ["Colorization Model", f"{config.COLORIZATION_MODEL} / {config.COLORIZATION_NETG}"],
        ["Super Resolution",  f"Real-ESRGAN x{config.REALESRGAN_SCALE}"],
        ["Detection Model",   config.YOLO_MODEL_NAME],
        ["Caption Model",     config.CAPTION_MODEL_NAME],
    ]

    meta_table = Table(meta_data, colWidths=[5.5 * cm, 12 * cm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), ISRO_LIGHT),
        ("TEXTCOLOR",  (0, 0), (0, -1), ISRO_BLUE),
        ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("GRID",       (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.4 * cm))

    # ── Images Section ──────────────────────────────────────────────────────
    story.append(Paragraph("1. Pipeline Image Results", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=ISRO_BLUE, spaceAfter=6))

    img_keys = [
        ("original_pil",   "Original Infrared Input"),
        ("enhanced_pil",   "Enhanced Image (CLAHE + Denoising + Gamma)"),
        ("sr_pil",         f"Super Resolution Output (Real-ESRGAN x{config.REALESRGAN_SCALE})"),
        ("colorized_pil",  "Colorized Output (IR-Colorization Model)"),
        ("annotated_pil",  "Object Detection (YOLOv11)"),
    ]

    # Display 2 images per row
    img_w = 8.5 * cm
    img_h = 6.5 * cm
    row = []
    for key, label in img_keys:
        pil = pipeline_results.get(key)
        if pil is None:
            continue
        rl_img = _pil_to_rl_image(pil, img_w, img_h)
        cell = [rl_img, Paragraph(label, caption_style)]
        row.append(cell)
        if len(row) == 2:
            t = Table([row], colWidths=[img_w + 0.5 * cm, img_w + 0.5 * cm])
            t.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.3 * cm))
            row = []

    if row:  # leftover single image
        t = Table([row], colWidths=[img_w + 0.5 * cm])
        t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        story.append(t)
        story.append(Spacer(1, 0.3 * cm))

    # ── Inference Timings ───────────────────────────────────────────────────
    story.append(Paragraph("2. Inference Time Breakdown", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=ISRO_BLUE, spaceAfter=6))

    timings = pipeline_results.get("timings", {})
    timing_data = [["Stage", "Time (ms)", "Time (s)"]]
    stage_names = {
        "enhancement_ms": "Image Enhancement",
        "sr_ms":          "Super Resolution (Real-ESRGAN)",
        "colorize_ms":    "IR Colorization",
        "detection_ms":   "Object Detection (YOLOv11)",
        "caption_ms":     "Scene Captioning (BLIP)",
        "total_ms":       "TOTAL",
    }
    for k, name in stage_names.items():
        if k in timings:
            ms = timings[k]
            timing_data.append([name, f"{ms:.0f}", f"{ms/1000:.3f}"])

    timing_table = Table(timing_data, colWidths=[9 * cm, 4 * cm, 4 * cm])
    timing_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), ISRO_BLUE),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ISRO_LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("FONTNAME",      (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND",    (0, -1), (-1, -1), ISRO_ORANGE),
        ("TEXTCOLOR",     (0, -1), (-1, -1), colors.white),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(timing_table)
    story.append(Spacer(1, 0.4 * cm))

    # ── Object Detection Results ─────────────────────────────────────────────
    story.append(Paragraph("3. Object Detection Results", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=ISRO_BLUE, spaceAfter=6))

    summary = pipeline_results.get("detection_summary", {})
    total_obj = pipeline_results.get("total_objects", 0)
    story.append(Paragraph(f"Total Objects Detected: <b>{total_obj}</b>", body_style))

    if summary:
        det_data = [["Object Class", "Count", "Confidence (avg)"]]
        detections = pipeline_results.get("detections", [])
        from collections import defaultdict
        conf_map = defaultdict(list)
        for d in detections:
            conf_map[d["label"]].append(d["confidence"])

        for label, count in sorted(summary.items(), key=lambda x: -x[1]):
            avg_conf = sum(conf_map[label]) / len(conf_map[label]) if conf_map[label] else 0
            det_data.append([label, str(count), f"{avg_conf:.1%}"])

        det_table = Table(det_data, colWidths=[8 * cm, 4 * cm, 5 * cm])
        det_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), ISRO_BLUE),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ISRO_LIGHT]),
            ("GRID",          (0, 0), (-1, -1), 0.3, colors.lightgrey),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(det_table)
    else:
        story.append(Paragraph("No objects detected or detection was disabled.", meta_style))

    story.append(Spacer(1, 0.4 * cm))

    # ── Scene Caption ────────────────────────────────────────────────────────
    story.append(Paragraph("4. Automated Scene Interpretation", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=ISRO_BLUE, spaceAfter=6))

    caption = pipeline_results.get("caption", "No caption generated.")
    cap_method = pipeline_results.get("caption_method", "")
    story.append(Paragraph(caption, body_style))
    story.append(Paragraph(f"(Generated by: {cap_method})", meta_style))
    story.append(Spacer(1, 0.4 * cm))

    # ── Errors ───────────────────────────────────────────────────────────────
    errors = pipeline_results.get("errors", [])
    if errors:
        story.append(Paragraph("5. Pipeline Warnings / Errors", section_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.red, spaceAfter=6))
        for err in errors:
            story.append(Paragraph(f"⚠ {err}", meta_style))
        story.append(Spacer(1, 0.2 * cm))

    # ── Footer ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=ISRO_ORANGE, spaceAfter=6))
    story.append(Paragraph(
        f"IRIS-AI v{config.APP_VERSION} | {config.REPORT_AUTHOR} | Generated: {now}",
        _style("Footer", fontSize=7, textColor=GREY, alignment=TA_CENTER, fontName="Helvetica"),
    ))

    # ── Build PDF ────────────────────────────────────────────────────────────
    doc.build(story)
    pdf_bytes = buf.getvalue()

    if output_path:
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"[Report] PDF saved → {output_path}")

    return pdf_bytes


def _pil_to_rl_image(pil_img: Image.Image, max_w, max_h):
    """Convert PIL image to ReportLab Image flowable."""
    from reportlab.platypus import Image as RLImage

    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=90)
    buf.seek(0)

    # Maintain aspect ratio
    w, h = pil_img.size
    aspect = h / w if w > 0 else 1
    if aspect > 1:
        rh = min(max_h, max_h)
        rw = rh / aspect
    else:
        rw = min(max_w, max_w)
        rh = rw * aspect

    return RLImage(buf, width=rw, height=rh)
