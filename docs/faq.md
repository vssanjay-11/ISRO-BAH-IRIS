# Frequently Asked Questions (FAQ)

Here are answers to frequently asked questions about the **IRIS-AI** project.

---

## 🛰️ Remote Sensing & Application

### Q: Why do we use UNet++ instead of standard UNet?
**A:** UNet++ introduces nested, dense skip pathways. These pathways bridge the semantic gap between the encoder and decoder feature maps before fusion, resulting in significantly more accurate fine-texture recovery (such as roads and narrow crop boundaries) during infrared colorization.

### Q: Can this project process standard GeoTIFF (.tif) images?
**A:** Yes, PIL and OpenCV support loading multi-spectral and high-dynamic-range TIFF images. However, before passing them to the colorization network, they are normalized and converted to standard 8-bit RGB/grayscale. Geographic metadata preservation is currently planned on our [Roadmap](../ROADMAP.md).

---

## 💻 Hardware & Performance

### Q: What is the minimum VRAM required to run the full pipeline?
**A:** 
- **CPU Mode**: Runs on any modern processor (requires 8GB RAM).
- **GPU Mode**: A minimum of **4GB VRAM** is recommended to run the full pipeline (including CLAHE, Super-Resolution, UNet++ GAN, FastSAM, and BLIP) without encountering Out of Memory (OOM) errors.

### Q: How can I reduce GPU VRAM usage?
**A:** You can enable tiling for the Real-ESRGAN super-resolution model by editing `config.py`:
```python
REALESRGAN_TILE = 400  # Set to a non-zero value like 400
```
This forces the model to process the image in smaller patches, dramatically reducing the maximum VRAM spike during upscaling.
