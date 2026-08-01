# Model Description

This document details the deep-learning model architectures utilized by **IRIS-AI**.

---

## 🎨 1. IR Colorizer (UNet++ GAN)
- **Role**: Performs the core translation from grayscale infrared/thermal images to RGB colors.
- **Architecture**: A generative adversarial network (GAN) where:
  - **Generator**: Based on **UNet++**, featuring nested, dense skip pathways that reduce semantic gaps between feature maps of the encoder and decoder.
  - **Discriminator**: PatchGAN discriminator ensuring high-frequency texture coherence.
- **Inference Mode**: Framed as a black box generator wrapper within the pipeline.

---

## 🔍 2. Super Resolution (Real-ESRGAN ×4)
- **Role**: Upscales low-resolution satellite/aerial imagery by a factor of 4.
- **Architecture**: Uses an Enhanced Super-Resolution Generative Adversarial Network (ESRGAN) with realistic noise synthesis.
- **Weights**: `RealESRGAN_x4plus.pth`.

---

## 📊 3. Semantic Segmenter (FastSAM)
- **Role**: Computes semantic segmentation masks for ground classification (roads, buildings, water, vegetation).
- **Architecture**: Fast Segment Anything Model (FastSAM) which solves the SAM task by framing segment-anything as an instance segmentation task on YOLOv8.
- **Weights**: `FastSAM-s.pt`.

---

## 📝 4. Scene Interpreter (BLIP Captioner)
- **Role**: Generates contextual annotations and text descriptions.
- **Architecture**: Bootstrapping Language-Image Pre-training (BLIP) model for unified vision-language understanding.
