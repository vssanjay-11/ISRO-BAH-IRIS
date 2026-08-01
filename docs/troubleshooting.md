# Troubleshooting & FAQ

This document lists common issues encountered during setup or runtime and how to resolve them.

---

## 🛑 Common Issues

### 1. CUDA GPU is Not Detected
- **Symptoms**: Console outputs: `[ModelManager] Initialized | Device: cpu` despite having an NVIDIA GPU.
- **Fix**: Check if CUDA is available in PyTorch:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
  If `False`, reinstall PyTorch with the correct CUDA version:
  ```bash
  pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
  ```

### 2. Out of Memory (OOM) Errors on GPU
- **Symptoms**: Script crashes with `RuntimeError: CUDA out of memory`.
- **Fix**: 
  - Ensure other applications utilizing CUDA are closed.
  - Set segmenter or super-resolution to run on CPU if VRAM is less than 4GB. The `ModelManager` will dynamically manage loading active pipelines.

### 3. Missing Checkpoint Files
- **Symptoms**: `FileNotFoundError` when launching backend/api.
- **Fix**: Run `python download_weights.py` and verify that the file `checkpoints/experiment_name/latest_net_G.pth` is present.
