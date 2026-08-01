# Developer Guide

Welcome to the **IRIS-AI** developer guide. This document outlines development workflows, code organization principles, and design guidelines.

---

## 🎨 Architectural Principles

We adhere strictly to the **black-box** inference principle:
1. **Unchanged Core**: The original colorizer GAN code in `models/` must not be modified unless resolving critical bugs.
2. **Modular Architecture**: All new capabilities (enhancements, segmentations, super-resolution) reside inside the `backend/` directory as decoupled modules.
3. **Singleton Lifecycle**: Models are managed by the `ModelManager` class inside `utils/model_manager.py` to prevent duplicate GPU memory consumption.

---

## 📂 Core Development Directory Map

- **`backend/`**: House all inference pipeline algorithms.
- **`utils/`**: Helper methods, GPU memory monitors, and singleton managers.
- **`frontend/`**: Streamlit UI components and multipage views.
- **`tests/`**: Unit tests validating pipeline modules.
- **`examples/`**: Minimal code scripts showing programmatic usage.

---

## 🧪 Testing Guidelines

We utilize `pytest` to run tests. To write new tests:
- Place them under `tests/` with the prefix `test_`.
- Run tests:
  ```bash
  pytest tests/
  ```

---

## 📜 Coding Conventions

- **PEP 8**: Follow PEP 8 style formatting.
- **Static Typing**: Annotate function signatures where applicable.
- **Docstrings**: Document classes and public APIs using standard formats (Google style preferred).
- **GPU Device Management**: Avoid hardcoding `cuda` or `cpu`. Always call the generic device mapper in `utils/gpu_utils.py`.
