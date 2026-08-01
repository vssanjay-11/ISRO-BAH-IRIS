# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-08-01

### Added
- **Image Enhancement Pipeline**: CLAHE, NL-Means denoising, and Gamma correction modules.
- **Super Resolution Integration**: Real-ESRGAN ×4 module.
- **Semantic Segmentation**: FastSAM implementation for identifying roads, buildings, water, and vegetation.
- **ISRO Scene Interpretation**: BLIP captioning model integration coupled with predefined analyst reporting templates.
- **Output Exporters**: PDF Report Generator (ReportLab-based) and ZIP export utility.
- **Premium Frontend**: 8-page Streamlit dashboard interface.
- **Robust ModelManager**: Singleton model loader ensuring zero VRAM duplication and efficient model preloading.
- **Developer Tools**: Unit tests, examples, and comprehensive API documentation.
