"""
frontend/app.py
IRIS-AI Premium Streamlit Dashboard v2
Intelligent Remote-sensing Infrared Interpretation Suite
ISRO Bharatiya Antariksh Hackathon 2026 – PS-10

Pages:
  1. Home        — Landing page with hero, animated feature cards
  2. Inference   — Single-image pipeline run
  3. Batch       — Multi-image / folder batch processing
  4. Comparison  — Interactive slider + tab comparison
  5. Analytics   — Metrics, timings, region stats, device info
  6. Downloads   — Per-image download + ZIP export
  7. Settings    — User-configurable parameters
  8. About       — Architecture, diagrams, team

Reuses:
  - utils/model_manager.py     (ModelManager singleton)
  - utils/gpu_utils.py         (detect_device, device_badge_html)
  - backend/report_generator.py (generate_pdf_report)
  - backend/exporter.py         (export_zip)
  - backend/batch_processor.py  (run_batch, discover_images)
  - configs/settings.py         (Settings)
  - config.py                   (all constants)
  - streamlit-image-comparison   (image_comparison slider)
"""

import os
import sys
import io
import time
import threading
import datetime
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config
from utils.gpu_utils   import detect_device, get_gpu_memory_usage, device_badge_html
from utils.model_manager import ModelManager
from configs.settings  import Settings
from backend.report_generator import generate_pdf_report
from backend.exporter  import export_zip

# ────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = f"IRIS-AI | {config.APP_SUBTITLE}",
    page_icon  = "🛰️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ────────────────────────────────────────────────────────────────────────────
# Global CSS
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

  :root {
    --bg:        #0A1628;
    --bg2:       #0D1F3C;
    --card:      rgba(26,58,107,0.35);
    --border:    rgba(0,180,216,0.25);
    --orange:    #FF6B00;
    --teal:      #06D6A0;
    --blue:      #00B4D8;
    --text:      #E8EDF8;
    --muted:     #9BAAC8;
    --radius:    16px;
  }

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: var(--bg) !important;
    color: var(--text) !important;
  }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem !important; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1F3C 0%, #0A1628 100%);
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] .stRadio label {
    color: var(--muted) !important;
    font-size: 0.88rem;
  }

  /* Cards */
  .iris-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    transition: transform 0.2s, border-color 0.2s;
  }
  .iris-card:hover {
    transform: translateY(-3px);
    border-color: var(--blue);
  }

  /* Gradient title */
  .iris-gradient {
    background: linear-gradient(135deg, #FF6B00 0%, #00B4D8 60%, #06D6A0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 800;
  }

  /* Section header */
  .section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--blue);
    border-left: 3px solid var(--orange);
    padding-left: 0.6rem;
    margin-bottom: 1rem;
  }

  /* Metric tile */
  .metric-tile {
    background: linear-gradient(135deg, rgba(26,58,107,0.5), rgba(10,22,40,0.7));
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
  }
  .metric-value { font-size: 2rem; font-weight: 700; color: var(--teal); }
  .metric-label { font-size: 0.78rem; color: var(--muted); margin-top: 4px; }

  /* Pipeline step badge */
  .step-badge {
    display: inline-block;
    background: rgba(255,107,0,0.15);
    border: 1px solid rgba(255,107,0,0.4);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    color: var(--orange);
    font-weight: 600;
    margin: 2px;
  }

  /* Feature card (landing) */
  .feature-card {
    background: linear-gradient(135deg, rgba(26,58,107,0.4), rgba(10,22,40,0.6));
    border: 1px solid rgba(0,180,216,0.2);
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    transition: all 0.25s ease;
    height: 100%;
  }
  .feature-card:hover {
    border-color: var(--orange);
    transform: translateY(-4px);
    box-shadow: 0 8px 32px rgba(255,107,0,0.15);
  }
  .feature-icon { font-size: 2.2rem; margin-bottom: 0.7rem; }
  .feature-title { font-weight: 700; font-size: 1rem; color: var(--text); }
  .feature-desc { font-size: 0.82rem; color: var(--muted); margin-top: 0.4rem; }

  /* Progress bar custom */
  .stProgress > div > div { background: linear-gradient(90deg, var(--orange), var(--teal)); border-radius: 10px; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #FF6B00, #FF8C42);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: opacity 0.2s, transform 0.15s;
  }
  .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

  /* Image caption */
  .img-label {
    text-align: center;
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 4px;
    font-style: italic;
  }

  /* Tag chip */
  .chip {
    display: inline-block;
    background: rgba(6,214,160,0.12);
    border: 1px solid rgba(6,214,160,0.35);
    color: var(--teal);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.74rem;
    font-weight: 600;
    margin: 2px;
  }

  /* Interpretation paragraph */
  .interp-box {
    background: rgba(6,214,160,0.06);
    border-left: 3px solid var(--teal);
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    line-height: 1.7;
    color: var(--text);
  }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Cached singletons
# ────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initialising IRIS-AI models …")
def _get_model_manager() -> ModelManager:
    mm = ModelManager.get_instance()
    return mm


@st.cache_resource
def _get_device_info():
    return detect_device()


@st.cache_resource
def _get_settings() -> Settings:
    return Settings.get()


# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────

def _render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 0.5rem;">
          <div style="font-size:2.8rem;">🛰️</div>
          <div class="iris-gradient" style="font-size:1.8rem; font-weight:800;">IRIS-AI</div>
          <div style="font-size:0.72rem; color:#9BAAC8; margin-top:2px;">ISRO BAH 2026 · PS-10</div>
          <div style="font-size:0.68rem; color:#FF6B00; margin-top:4px; font-weight:600;">
            v{ver}
          </div>
        </div>
        <hr style="border-color:rgba(0,180,216,0.2); margin:0.8rem 0;">
        """.format(ver=config.APP_VERSION), unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["🏠 Home", "🔬 Inference", "📦 Batch", "🔀 Comparison",
             "📊 Analytics", "💾 Downloads", "⚙️ Settings", "ℹ️ About"],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:rgba(0,180,216,0.15);'>", unsafe_allow_html=True)

        # Device status
        dev = _get_device_info()
        st.markdown(device_badge_html(dev), unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.72rem;color:#9BAAC8;margin-top:6px;'>"
                    f"{dev.device_name}</div>", unsafe_allow_html=True)

        # VRAM if GPU
        vram = get_gpu_memory_usage()
        if vram:
            used = vram.get("vram_used_mb", 0)
            total = vram.get("vram_total_mb", 1)
            pct   = vram.get("utilization_pct", 0)
            st.progress(int(pct), text=f"VRAM {used:.0f}/{total:.0f} MB")

        # Model status
        st.markdown("<div class='section-header' style='margin-top:1rem;'>Models</div>",
                    unsafe_allow_html=True)
        mm = _get_model_manager()
        for name, status in mm.get_status().items():
            icon = "✅" if "Ready" in status else ("⚠️" if "Fallback" in status else "⬜")
            st.markdown(
                f"<div style='font-size:0.73rem;color:#9BAAC8;margin:2px 0;'>"
                f"{icon} {name}</div>",
                unsafe_allow_html=True,
            )

    return page


# ────────────────────────────────────────────────────────────────────────────
# Page 1 — Home
# ────────────────────────────────────────────────────────────────────────────

def page_home():
    # Hero
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0 2rem;">
      <div style="font-size:4rem; margin-bottom:0.5rem;">🛰️</div>
      <h1 class="iris-gradient" style="font-size:3.5rem; margin:0; line-height:1.1;">IRIS-AI</h1>
      <p style="font-size:1.2rem; color:#9BAAC8; margin:0.5rem 0 0.2rem;">
        Intelligent Remote-sensing Infrared Interpretation Suite
      </p>
      <p style="font-size:0.95rem; color:#FF6B00; font-weight:600; letter-spacing:0.05em;">
        ISRO Bharatiya Antariksh Hackathon 2026 &nbsp;·&nbsp; Problem Statement PS-10
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,107,0,0.3);margin:0 0 2rem;'>",
                unsafe_allow_html=True)

    # Pipeline flow
    st.markdown("<div class='section-header'>Processing Pipeline</div>",
                unsafe_allow_html=True)
    steps = [
        ("📤", "Upload", "IR Image"),
        ("🔬", "Enhance", "CLAHE + Gamma"),
        ("⚡", "Super-Res", "Real-ESRGAN ×4"),
        ("🎨", "Colorize", "UNet++ GAN"),
        ("🧩", "Segment", "FastSAM"),
        ("📝", "Interpret", "ISRO Paragraph"),
        ("📄", "Report", "PDF + ZIP"),
    ]
    cols = st.columns(len(steps))
    for i, (icon, title, sub) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div style="text-align:center;">
              <div style="font-size:1.8rem;">{icon}</div>
              <div style="font-size:0.8rem;font-weight:700;color:#E8EDF8;">{title}</div>
              <div style="font-size:0.68rem;color:#9BAAC8;">{sub}</div>
              {'<div style="font-size:0.65rem;color:#FF6B00;font-weight:700;">↓</div>' if i < len(steps)-1 else ''}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards
    st.markdown("<div class='section-header'>Key Features</div>", unsafe_allow_html=True)
    features = [
        ("🔬", "Image Enhancement", "CLAHE, NL-Means denoising, gamma correction, adaptive contrast"),
        ("⚡", "Super Resolution", "Real-ESRGAN ×4 — spatial detail recovery for HR analysis"),
        ("🎨", "IR Colorization", "UNet++ GAN trained on IR-visible pairs — existing repo"),
        ("🧩", "Semantic Segmentation", "FastSAM — roads, buildings, water, vegetation, bare land"),
        ("📝", "ISRO Interpretation", "Analyst-grade paragraph with RS-domain language"),
        ("📊", "Analytics Dashboard", "Inference timings, region stats, device memory"),
        ("📦", "Batch Processing", "Single / folder / recursive with progress bar"),
        ("💾", "ZIP Export", "All outputs + PDF report in one download"),
    ]
    rows = [features[i:i+4] for i in range(0, len(features), 4)]
    for row in rows:
        cols = st.columns(4)
        for col, (icon, title, desc) in zip(cols, row):
            with col:
                st.markdown(f"""
                <div class="feature-card">
                  <div class="feature-icon">{icon}</div>
                  <div class="feature-title">{title}</div>
                  <div class="feature-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 2, 2])
    with c2:
        if st.button("🚀  Start Analysis →", use_container_width=True):
            st.session_state["_nav"] = "🔬 Inference"
            st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# Page 2 — Inference
# ────────────────────────────────────────────────────────────────────────────

def page_inference():
    st.markdown("<h2 class='iris-gradient'>🔬 Single Image Inference</h2>",
                unsafe_allow_html=True)

    col_upload, col_controls = st.columns([3, 2])

    with col_upload:
        uploaded = st.file_uploader(
            "Upload Infrared Image",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            help="Supports PNG, JPG, TIFF, BMP",
        )
        if uploaded:
            orig = Image.open(uploaded).convert("RGB")
            st.image(orig, caption=f"{uploaded.name} — {orig.size[0]}×{orig.size[1]} px",
                     use_container_width=True)

    with col_controls:
        st.markdown("<div class='section-header'>Pipeline Stages</div>",
                    unsafe_allow_html=True)
        run_enh = st.toggle("Image Enhancement",   value=True)
        run_sr  = st.toggle("Super Resolution",    value=True)
        run_col = st.toggle("IR Colorization",     value=True)
        run_seg = st.toggle("Semantic Segmentation", value=True)
        run_cap = st.toggle("Scene Interpretation", value=True)

        st.markdown("<div class='section-header'>Quick Settings</div>",
                    unsafe_allow_html=True)
        settings = _get_settings()
        sr_scale = st.select_slider("ESRGAN Scale", options=[2, 4], value=settings.realesrgan_scale)
        seg_conf = st.slider("FastSAM Confidence", 0.1, 0.9, settings.fastsam_conf, 0.05)

    if uploaded and st.button("▶  Run IRIS-AI Pipeline", use_container_width=True):
        mm = _get_model_manager()

        progress_bar  = st.progress(0, text="Initialising …")
        status_text   = st.empty()
        stage_display = st.empty()

        def _cb(step, pct):
            progress_bar.progress(int(pct), text=step)
            status_text.markdown(
                f"<span class='step-badge'>{step}</span>", unsafe_allow_html=True
            )

        with st.spinner("Running pipeline …"):
            results = mm.run(
                image_input       = orig,
                run_enhancement   = run_enh,
                run_superres      = run_sr,
                run_colorize      = run_col,
                run_segmentation  = run_seg,
                run_caption       = run_cap,
                progress_callback = _cb,
                source_filename   = uploaded.name,
            )

        progress_bar.progress(100, text="Complete ✓")
        st.session_state["last_results"] = results
        st.session_state["last_uploaded_name"] = uploaded.name

        # Quick preview
        st.markdown("<div class='section-header'>Results Preview</div>",
                    unsafe_allow_html=True)
        preview_keys = [
            ("original_pil",  "Original"),
            ("enhanced_pil",  "Enhanced"),
            ("colorized_pil", "Colorized"),
            ("overlay_pil",   "Segmented Overlay"),
        ]
        pcols = st.columns(4)
        for (key, label), pcol in zip(preview_keys, pcols):
            pil = results.get(key)
            if pil:
                with pcol:
                    st.image(pil, use_container_width=True)
                    st.markdown(f"<div class='img-label'>{label}</div>",
                                unsafe_allow_html=True)

        # Interpretation
        interp = results.get("interpretation") or results.get("caption", "")
        if interp:
            st.markdown("<div class='section-header'>ISRO Scene Interpretation</div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='interp-box'>{interp}</div>",
                        unsafe_allow_html=True)

        # Errors
        for err in results.get("errors", []):
            st.warning(f"⚠️ {err}")

        total_ms = results.get("timings", {}).get("total_ms", 0)
        st.success(f"Pipeline completed in **{total_ms/1000:.2f}s** · "
                   f"Session: `{results.get('session_id', '—')}`")


# ────────────────────────────────────────────────────────────────────────────
# Page 3 — Batch
# ────────────────────────────────────────────────────────────────────────────

def page_batch():
    st.markdown("<h2 class='iris-gradient'>📦 Batch Processing</h2>",
                unsafe_allow_html=True)

    from backend.batch_processor import run_batch, discover_images
    from data.image_folder import IMG_EXTENSIONS   # reuse existing extension list

    col_input, col_opts = st.columns([3, 2])

    with col_input:
        st.markdown("<div class='section-header'>Input Source</div>",
                    unsafe_allow_html=True)
        mode = st.radio("Mode", ["Upload Files", "Folder Path"], horizontal=True)

        if mode == "Upload Files":
            files = st.file_uploader(
                "Upload Images",
                type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
                accept_multiple_files=True,
            )
            source_paths = []
            if files:
                tmp_dir = os.path.join(config.OUTPUTS_DIR, "_batch_tmp")
                os.makedirs(tmp_dir, exist_ok=True)
                for f in files:
                    p = os.path.join(tmp_dir, f.name)
                    with open(p, "wb") as fp:
                        fp.write(f.read())
                    source_paths.append(p)
                st.info(f"{len(source_paths)} file(s) ready. "
                        f"Formats: {', '.join(IMG_EXTENSIONS[:5])} …")
        else:
            folder = st.text_input("Folder Path",
                                   placeholder="C:/path/to/ir_images")
            recursive = st.checkbox("Recursive (include sub-folders)", value=True)
            source_paths = [folder] if folder and os.path.isdir(folder) else []
            if folder and not source_paths:
                st.error("Folder not found.")
            if source_paths:
                imgs = discover_images(source_paths, recursive=recursive)
                st.info(f"Found **{len(imgs)}** image(s).")

    with col_opts:
        st.markdown("<div class='section-header'>Pipeline Stages</div>",
                    unsafe_allow_html=True)
        b_enh = st.toggle("Enhancement",   value=True)
        b_sr  = st.toggle("Super-Res",     value=True)
        b_col = st.toggle("Colorization",  value=True)
        b_seg = st.toggle("Segmentation",  value=True)
        b_cap = st.toggle("Interpretation",value=True)
        max_imgs = st.number_input("Max images", 1, 500, config.BATCH_MAX_IMAGES)

    if source_paths and st.button("▶  Start Batch", use_container_width=True):
        mm      = _get_model_manager()
        prog    = st.progress(0, text="Starting …")
        stat    = st.empty()
        stop_ev = threading.Event()

        def _cb(idx, total, path, step, pct):
            overall = int((idx + pct / 100) / total * 100)
            prog.progress(overall, text=f"[{idx+1}/{total}] {os.path.basename(path)} — {step}")
            stat.markdown(
                f"<div style='font-size:0.8rem;color:#9BAAC8;'>"
                f"Completed: {idx} &nbsp;|&nbsp; Remaining: {total-idx} &nbsp;|&nbsp; "
                f"Est: {((total-idx)*5):.0f}s</div>",
                unsafe_allow_html=True,
            )

        with st.spinner("Batch in progress …"):
            summary = run_batch(
                source           = source_paths,
                model_manager    = mm,
                max_images       = int(max_imgs),
                run_enhancement  = b_enh,
                run_superres     = b_sr,
                run_colorize     = b_col,
                run_segmentation = b_seg,
                run_caption      = b_cap,
                progress_callback= _cb,
                stop_event       = stop_ev,
            )

        prog.progress(100, text="Batch complete ✓")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total",     summary.total)
        c2.metric("Succeeded", summary.succeeded)
        c3.metric("Failed",    summary.failed)
        c4.metric("Time",      f"{summary.elapsed_ms/1000:.1f}s")

        if summary.failed > 0:
            with st.expander("Failed images"):
                for r in summary.results:
                    if not r.success:
                        st.error(f"{r.path}: {r.error}")

        st.session_state["batch_summary"] = summary


# ────────────────────────────────────────────────────────────────────────────
# Page 4 — Comparison
# ────────────────────────────────────────────────────────────────────────────

def page_comparison():
    st.markdown("<h2 class='iris-gradient'>🔀 Image Comparison</h2>",
                unsafe_allow_html=True)

    results = st.session_state.get("last_results")
    if not results:
        st.info("Run the pipeline on the **Inference** page first.")
        return

    try:
        from streamlit_image_comparison import image_comparison
        has_slider = True
    except ImportError:
        has_slider = False
        st.warning("Install `streamlit-image-comparison` for interactive sliders.")

    tab_slider, tab_grid, tab_steps = st.tabs(
        ["↔ Slider", "🖼️ Grid View", "🔬 Enhancement Steps"]
    )

    key_pairs = [
        ("original_pil",  "enhanced_pil",  "Original",  "Enhanced"),
        ("enhanced_pil",  "colorized_pil", "Enhanced",  "Colorized"),
        ("colorized_pil", "overlay_pil",   "Colorized", "Segmented"),
        ("original_pil",  "overlay_pil",   "Original",  "Final"),
    ]

    with tab_slider:
        pair_labels = [f"{a} ↔ {b}" for _, _, a, b in key_pairs]
        choice = st.selectbox("Select comparison", pair_labels)
        idx = pair_labels.index(choice)
        k1, k2, l1, l2 = key_pairs[idx]
        img1 = results.get(k1)
        img2 = results.get(k2)

        if img1 and img2:
            # Match sizes for slider
            if img1.size != img2.size:
                img2 = img2.resize(img1.size, Image.LANCZOS)
            if has_slider:
                image_comparison(
                    img1=img1, img2=img2,
                    label1=l1, label2=l2,
                    width=800,
                )
            else:
                cc1, cc2 = st.columns(2)
                cc1.image(img1, caption=l1, use_container_width=True)
                cc2.image(img2, caption=l2, use_container_width=True)

    with tab_grid:
        all_imgs = [
            ("original_pil",  "Original"),
            ("enhanced_pil",  "Enhanced"),
            ("sr_pil",        "Super Resolution"),
            ("colorized_pil", "Colorized"),
            ("segmented_pil", "Segmentation Map"),
            ("overlay_pil",   "Overlay"),
        ]
        valid = [(k, l) for k, l in all_imgs if results.get(k) is not None]
        cols = st.columns(3)
        for i, (key, label) in enumerate(valid):
            with cols[i % 3]:
                st.image(results[key], use_container_width=True)
                st.markdown(f"<div class='img-label'>{label}</div>",
                            unsafe_allow_html=True)

    with tab_steps:
        steps = results.get("enhancement_steps", {})
        if not steps:
            st.info("Enhancement steps not available. Enable Enhancement stage and re-run.")
        else:
            step_names = list(steps.keys())
            if step_names:
                sel = st.select_slider("Enhancement step", step_names)
                if sel in steps:
                    st.image(steps[sel], caption=f"After: {sel}",
                             use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# Page 5 — Analytics
# ────────────────────────────────────────────────────────────────────────────

def page_analytics():
    import pandas as pd

    st.markdown("<h2 class='iris-gradient'>📊 Analytics</h2>", unsafe_allow_html=True)

    results = st.session_state.get("last_results")
    dev     = _get_device_info()

    # Device panel
    st.markdown("<div class='section-header'>Compute Device</div>",
                unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    d1.markdown(f"""<div class='metric-tile'>
      <div class='metric-value'>{dev.device.upper()}</div>
      <div class='metric-label'>Compute Backend</div>
    </div>""", unsafe_allow_html=True)
    d2.markdown(f"""<div class='metric-tile'>
      <div class='metric-value'>{dev.torch_version}</div>
      <div class='metric-label'>PyTorch Version</div>
    </div>""", unsafe_allow_html=True)
    d3.markdown(f"""<div class='metric-tile'>
      <div class='metric-value'>{dev.gpu_count or '—'}</div>
      <div class='metric-label'>GPU Count</div>
    </div>""", unsafe_allow_html=True)
    d4.markdown(f"""<div class='metric-tile'>
      <div class='metric-value'>{dev.ram_total_mb:.0f} MB</div>
      <div class='metric-label'>System RAM</div>
    </div>""", unsafe_allow_html=True)

    if not results:
        st.info("Run the pipeline first to see inference analytics.")
        return

    st.markdown("<div class='section-header'>Pipeline Timings</div>",
                unsafe_allow_html=True)
    timings = results.get("timings", {})
    stage_names = {
        "enhancement_ms": "Enhancement",
        "sr_ms":          "Super Resolution",
        "colorize_ms":    "IR Colorization",
        "segment_ms":     "Segmentation",
        "caption_ms":     "Interpretation",
        "total_ms":       "TOTAL",
    }
    rows = []
    for k, label in stage_names.items():
        ms = timings.get(k)
        if ms is not None:
            rows.append({"Stage": label, "ms": f"{ms:.0f}", "s": f"{ms/1000:.3f}"})

    if rows:
        df = pd.DataFrame(rows)
        df.columns = ["Stage", "Time (ms)", "Time (s)"]
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Bar chart
        chart_data = pd.DataFrame(
            {k: [v] for k, v in timings.items() if k != "total_ms"},
        ).T.rename(columns={0: "ms"})
        chart_data.index = [stage_names.get(i, i) for i in chart_data.index]
        st.bar_chart(chart_data)

    # Segmentation regions
    region_summary = results.get("region_summary", {})
    if region_summary:
        st.markdown("<div class='section-header'>Segmentation Regions</div>",
                    unsafe_allow_html=True)
        c1, c2 = st.columns([2, 3])
        with c1:
            df_r = pd.DataFrame(
                list(region_summary.items()), columns=["Region", "Mask Count"]
            )
            st.dataframe(df_r, use_container_width=True, hide_index=True)
        with c2:
            labels = results.get("interpretation_labels", [])
            if labels:
                chips = " ".join([f"<span class='chip'>{l}</span>" for l in labels])
                st.markdown(chips, unsafe_allow_html=True)
            seg_img = results.get("segmented_pil")
            if seg_img:
                st.image(seg_img, caption="Segmentation Map", use_container_width=True)

    # Image metadata
    st.markdown("<div class='section-header'>Image Metadata</div>",
                unsafe_allow_html=True)
    orig = results.get("original_pil")
    sr   = results.get("sr_pil")
    if orig:
        meta = [
            ["Input Resolution",  f"{orig.size[0]} × {orig.size[1]} px"],
            ["SR Resolution",     f"{sr.size[0]} × {sr.size[1]} px" if sr else "—"],
            ["Session ID",        results.get("session_id", "—")],
            ["Session Folder",    results.get("session_dir", "—")],
            ["Model Version",     f"IRIS-AI v{config.APP_VERSION}"],
            ["Colorization",      f"{config.COLORIZATION_MODEL}/{config.COLORIZATION_NETG}"],
            ["Caption Method",    results.get("caption_method", "—")],
            ["Errors",            str(len(results.get("errors", [])))],
        ]
        df_meta = pd.DataFrame(meta, columns=["Property", "Value"])
        st.dataframe(df_meta, use_container_width=True, hide_index=True)


# ────────────────────────────────────────────────────────────────────────────
# Page 6 — Downloads
# ────────────────────────────────────────────────────────────────────────────

def page_downloads():
    st.markdown("<h2 class='iris-gradient'>💾 Downloads</h2>", unsafe_allow_html=True)

    results = st.session_state.get("last_results")
    if not results:
        st.info("Run the pipeline first.")
        return

    # Individual images
    st.markdown("<div class='section-header'>Individual Images</div>",
                unsafe_allow_html=True)
    img_exports = [
        ("original_pil",  "00_original.png"),
        ("enhanced_pil",  "01_enhanced.png"),
        ("sr_pil",        "02_super_resolution.png"),
        ("colorized_pil", "03_colorized.png"),
        ("segmented_pil", "04_segmented.png"),
        ("overlay_pil",   "05_overlay.png"),
    ]
    cols = st.columns(3)
    for i, (key, fname) in enumerate(img_exports):
        pil = results.get(key)
        if pil is None:
            continue
        with cols[i % 3]:
            st.image(pil, use_container_width=True)
            buf = io.BytesIO()
            pil.convert("RGB").save(buf, format="PNG")
            st.download_button(
                label=f"⬇ {fname}",
                data=buf.getvalue(),
                file_name=fname,
                mime="image/png",
                use_container_width=True,
                key=f"dl_{key}",
            )

    # Caption
    caption = results.get("interpretation") or results.get("caption", "")
    if caption:
        st.markdown("<div class='section-header'>Scene Interpretation Text</div>",
                    unsafe_allow_html=True)
        st.text_area("Interpretation", caption, height=120)
        st.download_button(
            "⬇ Download caption.txt",
            data=caption.encode("utf-8"),
            file_name="caption.txt",
            mime="text/plain",
        )

    # PDF
    st.markdown("<div class='section-header'>PDF Report</div>",
                unsafe_allow_html=True)
    if st.button("📄 Generate PDF Report"):
        with st.spinner("Generating PDF …"):
            pdf_bytes = generate_pdf_report(results, output_path=None)
        st.download_button(
            "⬇ Download PDF Report",
            data=pdf_bytes,
            file_name=f"IRIS_AI_Report_{results.get('session_id','session')}.pdf",
            mime="application/pdf",
        )

    # ZIP
    st.markdown("<div class='section-header'>Complete Analysis ZIP</div>",
                unsafe_allow_html=True)
    st.markdown("One-click download of all outputs + PDF report.")
    if st.button("📦 Generate & Download ZIP"):
        with st.spinner("Packaging ZIP …"):
            zip_bytes = export_zip(results, output_path=None)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "⬇ Download ZIP",
            data=zip_bytes,
            file_name=f"IRIS_AI_{results.get('session_id','session')}_{ts}.zip",
            mime="application/zip",
        )


# ────────────────────────────────────────────────────────────────────────────
# Page 7 — Settings
# ────────────────────────────────────────────────────────────────────────────

def page_settings():
    st.markdown("<h2 class='iris-gradient'>⚙️ Settings</h2>", unsafe_allow_html=True)

    s = _get_settings()

    st.markdown("<div class='section-header'>Processing</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        device = st.selectbox("Device", ["auto", "cuda", "cpu", "mps"],
                              index=["auto", "cuda", "cpu", "mps"].index(
                                  s["device"] if s["device"] in ["auto","cuda","cpu","mps"] else "auto"
                              ))
        sr_scale = st.select_slider("Real-ESRGAN Scale", [2, 4], value=s.realesrgan_scale)
        sr_tile  = st.number_input("ESRGAN Tile (0=off)", 0, 800, int(s["realesrgan_tile"]), 100)
    with c2:
        seg_conf = st.slider("FastSAM Confidence", 0.1, 0.9, float(s.fastsam_conf), 0.05)
        seg_size = st.select_slider("FastSAM Img Size", [512, 640, 1024], value=int(s["fastsam_img_size"]))

    st.markdown("<div class='section-header'>Enhancement</div>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        clahe  = st.slider("CLAHE Clip Limit", 1.0, 8.0, float(s.clahe_clip_limit), 0.5)
        gamma  = st.slider("Gamma Value", 0.5, 2.5, float(s.gamma_value), 0.1)
    with c4:
        denoise_h = st.slider("Denoising Strength", 1, 30, int(s["denoise_h"]))
        tsize_sel = st.selectbox("Target Size", ["256×256", "512×512"],
                                 index=0 if s.target_size == (256, 256) else 1)

    st.markdown("<div class='section-header'>Output</div>", unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        out_dir = st.text_input("Output Directory", s["output_dir"])
        rpt_dir = st.text_input("Reports Directory", s["reports_dir"])
    with c6:
        cap_len  = st.number_input("Caption Max Length", 50, 500, int(s["caption_max_length"]))
        auto_rpt = st.checkbox("Auto-generate report after inference", value=bool(s["auto_report"]))

    col_save, col_reset = st.columns(2)
    with col_save:
        if st.button("💾 Save Settings", use_container_width=True):
            tsize = (256, 256) if tsize_sel == "256×256" else (512, 512)
            s.set("device",            device)
            s.set("realesrgan_scale",  sr_scale)
            s.set("realesrgan_tile",   sr_tile)
            s.set("fastsam_conf",      seg_conf)
            s.set("fastsam_img_size",  seg_size)
            s.set("clahe_clip_limit",  clahe)
            s.set("gamma_value",       gamma)
            s.set("denoise_h",         denoise_h)
            s.set("target_size",       list(tsize))
            s.set("output_dir",        out_dir)
            s.set("reports_dir",       rpt_dir)
            s.set("caption_max_length", cap_len)
            s.set("auto_report",       auto_rpt)
            s.save()
            st.success("Settings saved.")

    with col_reset:
        if st.button("↺ Reset to Defaults", use_container_width=True):
            s.reset()
            st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# Page 8 — About
# ────────────────────────────────────────────────────────────────────────────

def page_about():
    st.markdown("<h2 class='iris-gradient'>ℹ️ About IRIS-AI</h2>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class="iris-card">
      <h3 style="color:#FF6B00;margin-top:0;">Project Overview</h3>
      <p style="color:#9BAAC8;line-height:1.7;">
        IRIS-AI is a production-ready AI application built on top of the
        <strong>IR-colorization</strong> repository for ISRO BAH 2026 Problem Statement PS-10.
        It wraps the existing UNet++ GAN colorization model with a full inference pipeline
        including super-resolution, FastSAM segmentation, ISRO-style scene interpretation,
        and professional PDF/ZIP reporting — all without modifying the original model code.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tech stack
    st.markdown("<div class='section-header'>Technology Stack</div>",
                unsafe_allow_html=True)
    tech = [
        ("🎨", "IR Colorization", "UNet++ GAN (existing repo)", "#FF6B00"),
        ("⚡", "Super Resolution", "Real-ESRGAN x4plus", "#00B4D8"),
        ("🧩", "Segmentation", "FastSAM (ultralytics)", "#06D6A0"),
        ("📝", "Captioning", "BLIP-base (Salesforce)", "#9BAAC8"),
        ("🔬", "Enhancement", "OpenCV CLAHE + NL-Means", "#FFD700"),
        ("📄", "Reports", "ReportLab PDF", "#FF6B00"),
        ("🖥️", "UI", "Streamlit", "#FF4B4B"),
        ("🐳", "Deploy", "Docker + NVIDIA GPU", "#2496ED"),
    ]
    cols = st.columns(4)
    for i, (icon, name, desc, color) in enumerate(tech):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="iris-card" style="border-color:{color}33; text-align:center;">
              <div style="font-size:1.8rem;">{icon}</div>
              <div style="font-weight:700;color:{color};font-size:0.9rem;">{name}</div>
              <div style="font-size:0.75rem;color:#9BAAC8;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        if (i + 1) % 4 == 0:
            st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture diagram (Mermaid via markdown)
    st.markdown("<div class='section-header'>Architecture Diagram</div>",
                unsafe_allow_html=True)
    st.markdown("""
```mermaid
flowchart LR
  A[IR Image] --> B[enhancer.py]
  B --> C[super_resolution.py]
  C --> D[colorizer.py]
  D --> E[segmenter.py]
  E --> F[interpreter.py]
  F --> G[report_generator.py]
  G --> H[exporter.py]

  D -.uses.-> D1[COLOR_model.py]
  D -.uses.-> D2[util/util.py]
  E -.uses.-> E1[FastSAM]
  H -.uses.-> H1[generate_pdf_report]

  subgraph UI[frontend/app.py]
    I[ModelManager.run]
  end
  A --> I --> B
```
""")

    # Future scope
    st.markdown("<div class='section-header'>Future Scope</div>",
                unsafe_allow_html=True)
    future = [
        "Multi-temporal change detection",
        "SAR + multispectral fusion",
        "Semantic segmentation fine-tuned on IR datasets",
        "GeoTIFF / KML / GIS export",
        "Real-time satellite stream processing",
        "ISRO cloud deployment",
        "Interactive web annotation tool",
    ]
    for f in future:
        st.markdown(f"<span class='chip'>+ {f}</span>", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Main router
# ────────────────────────────────────────────────────────────────────────────

def main():
    page = _render_sidebar()

    # Navigation override from Home button
    if "_nav" in st.session_state:
        page = st.session_state.pop("_nav")

    if page == "🏠 Home":
        page_home()
    elif page == "🔬 Inference":
        page_inference()
    elif page == "📦 Batch":
        page_batch()
    elif page == "🔀 Comparison":
        page_comparison()
    elif page == "📊 Analytics":
        page_analytics()
    elif page == "💾 Downloads":
        page_downloads()
    elif page == "⚙️ Settings":
        page_settings()
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
