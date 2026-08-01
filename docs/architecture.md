# System Architecture

This document describes the high-level architecture and pipeline workflows of **IRIS-AI**.

---

## 🏗️ High-Level System Architecture

IRIS-AI separates user interfaces from deep-learning backend workloads:

```mermaid
graph TD
    subgraph Frontend [Streamlit Dashboard]
        UI[Multi-page Dashboard]
    end

    subgraph Backend [Inference Engine]
        API[FastAPI Router]
        Pipeline[Orchestration Pipeline]
        Manager[Model Manager Singleton]
    end

    subgraph Models [Deep Learning Models]
        Enhance[Enhancement Mod]
        SR[Real-ESRGAN x4]
        Color[UNet++ GAN Colorizer]
        Seg[FastSAM Segmenter]
        BLIP[BLIP Captioner]
    end

    UI <-->|HTTP / REST| API
    API --> Pipeline
    Pipeline --> Manager
    Manager --> Enhance
    Manager --> SR
    Manager --> Color
    Manager --> Seg
    Manager --> BLIP
```

---

## 🔄 Inference Pipeline (Step-by-Step)

When an infrared image is submitted, it flows through the following sequential pipeline:

```mermaid
sequenceDiagram
    autonumber
    User->>Frontend: Upload Infrared Image
    Frontend->>Backend: Post Image Content
    Backend->>Enhancer: Apply Denoising & CLAHE
    Enhancer-->>Backend: Return Enhanced Grayscale
    Backend->>SuperResolution: Upscale image x4 (Real-ESRGAN)
    SuperResolution-->>Backend: Return High-Res Grayscale
    Backend->>Colorizer: Colorize Grayscale (UNet++ GAN)
    Colorizer-->>Backend: Return Colorized RGB
    Backend->>Segmenter: Detect Classes (FastSAM)
    Segmenter-->>Backend: Return Segmentation Mask & Overlay
    Backend->>Interpreter: Generate analyst notes (BLIP & template)
    Interpreter-->>Backend: Return Interpretation Paragraph
    Backend->>ReportGenerator: Compile PDF Report
    ReportGenerator-->>Backend: Return PDF path
    Backend-->>User: Return complete ZIP bundle & PDF report
```

---

## 💾 Model Lifecycle Management

Preloading heavy models in deep learning scripts often causes duplicate GPU initialization overhead and crashes due to Out Of Memory (OOM) errors. 

To solve this, IRIS-AI implements a strict **Model Manager Singleton Pattern**:
- Configures lazy-loading of checkpoints.
- Tracks active memory.
- Safely moves weights between CPU and GPU/VRAM space when they are inactive.

---

## 🏋️ Training Flow (UNet++ GAN)

Below is the workflow showing how training data propagates through the generator and patch discriminator to update weight parameters:

```mermaid
graph TD
    subgraph Data Input
        IR[Infrared Source Image]
        RGB[Aligned Real RGB Image]
    end

    subgraph Generator Network [UNet++ Generator]
        Enc[Dense Encoder] -->|Dense Skip Connections| Dec[Dense Decoder]
        Dec --> Fake[Colorized Output]
    end

    subgraph Discriminator Network [PatchGAN Discriminator]
        Fake --> DiscFake[Discriminator Decision on Fake]
        RGB --> DiscReal[Discriminator Decision on Real]
    end

    subgraph Optimization
        LossG[Generator Adversarial + L1 Loss] --> OptG[Optimizer Update Generator]
        LossD[Discriminator Loss] --> OptD[Optimizer Update Discriminator]
    end

    IR --> Enc
    DiscFake --> LossG
    DiscFake --> LossD
    DiscReal --> LossD
```

