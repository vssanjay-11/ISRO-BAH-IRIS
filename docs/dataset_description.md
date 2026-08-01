# Dataset Description

This document details the datasets supported by **IRIS-AI** for training and validation purposes.

---

## 📋 Summary of Supported Datasets

| Dataset | Total Images | Image Type | Main Purpose | License |
|---|---|---|---|---|
| **LLVIP** | 15,488 | High-res aligned IR + RGB pairs | Visible-infrared translation | CC BY-NC 4.0 |
| **FLIR ADAS** | 26,442 | Thermal + Visible | Autonomous driving & object detection | Non-commercial |
| **KAIST** | 95,328 | Thermal + RGB video sequences | Pedestrian detection and translation | Research |
| **TNO** | 261 | Multi-spectral band (LWIR, Intensified) | Image fusion | Non-commercial |

---

## 📂 Data Ingestion and Downloading

Use the automated download utility to fetch dataset mirrors:
```bash
python download_datasets.py
```
This script handles structural folder preparation and validation checks for the listed datasets.
