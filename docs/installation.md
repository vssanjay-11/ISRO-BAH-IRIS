# Installation Guide

This guide details how to set up the development and execution environments for **IRIS-AI**.

---

## 💻 Prerequisites

- **Python**: Version 3.10 (tested and recommended).
- **CUDA Toolkit** (Optional, for GPU acceleration): Version 11.8 or 12.1.
- **Operating Systems**: Windows 10/11, Ubuntu 20.04+, macOS.

---

## 🛠️ Local Installation

### 1. Clone the Repository
```bash
git clone https://github.com/vssanjay-11/ISRO-BAH-IRIS.git
cd ISRO-BAH-IRIS
```

### 2. Set Up a Virtual Environment
We recommend using Python's native `venv`:

**Windows**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch with GPU Support (Recommended)
If you have an NVIDIA GPU, install PyTorch with CUDA support first:

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Dependencies
Install all remaining package requirements:
```bash
pip install -r requirements.txt
```

---

## 🌐 Model Weights Download

To run the complete pipeline, you must download pre-trained weights for the models. Run the automated utility script:

```bash
python download_weights.py
```

This script will automatically:
1. Fetch and install **Real-ESRGAN** weights.
2. Fetch and install **FastSAM** segmenter weights.
3. Open a Google Drive URL for downloading the main IR colorization model (`latest_net_G.pth`).

Ensure that the downloaded colorization model weights are saved in:
`checkpoints/experiment_name/latest_net_G.pth`

---

## 🐳 Docker Setup

Alternatively, you can containerize the application to ensure consistency across environments.

### Build the Docker Image
```bash
docker build -t iris-ai .
```

### Run Container (GPU Enabled)
```bash
docker run --gpus all -p 8501:8501 -p 8000:8000 iris-ai
```

### Run Container (CPU Mode)
```bash
docker run -p 8501:8501 -p 8000:8000 iris-ai
```
