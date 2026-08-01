# Roadmap

This roadmap outlines the planned development and milestones for **IRIS-AI**.

---

## 🚀 Near-Term Goals (Q3-Q4 2026)

### 1. Model Enhancements
- [ ] Add support for alternative super-resolution models (e.g., SwinIR, LPIPS).
- [ ] Support on-the-fly model switching (choosing different colorizer checkpoints via the UI).

### 2. UI & Frontend Optimization
- [ ] Implement live inference progress bars for individual pipeline stages.
- [ ] Support interactive interactive maps for georeferenced TIF/GeoTIFF satellite imagery.

### 3. API & Backend
- [ ] Add asynchronous batch processing queues (Celery/Redis or FastAPI BackgroundTasks).
- [ ] Implement Swagger API authentication.

---

## 🎯 Long-Term Goals (2027+)

### 1. Geospatial & GIS Integration
- [ ] Support coordinate system projection (EPSG codes) preservation in colorized outputs.
- [ ] Export segmentations as GeoJSON/Shapefiles.

### 2. Multi-spectral Support
- [ ] Expand the colorizer core to accept multi-spectral inputs beyond standard thermal/LWIR bands.
- [ ] Introduce custom colorization profiles matching specific spectral ranges.
