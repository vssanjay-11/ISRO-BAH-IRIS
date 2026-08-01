# Training Guide

This guide describes how to train or fine-tune the UNet++ GAN colorization model on custom visible-infrared paired datasets.

---

## 📂 Dataset Preparation

To train the colorizer, you need aligned visible (RGB) and infrared (grayscale or 3-channel LWIR) image pairs.

1. Create a dataset directory structured as follows:
   ```
   datasets/custom_dataset/
   ├── trainA/      # Infrared training images
   ├── trainB/      # Aligned RGB training images
   ├── valA/        # Infrared validation images
   └── valB/        # Aligned RGB validation images
   ```
2. Place matching image pairs in the respective subfolders (e.g., `trainA/image_001.png` must correspond exactly to `trainB/image_001.png`).

---

## 🚀 Running Training

Execute the training script with your custom configurations:

```bash
python train.py --dataroot ./datasets/custom_dataset --name custom_ir_colorizer --model COLOR --netG UNet_2Plus
```

### Key Training Options

- `--name`: The name of the experiment. Checkpoints will be saved under `checkpoints/<name>/`.
- `--gpu_ids`: GPU IDs to use (e.g. `0` or `0,1` for multi-GPU training). Use `-1` for CPU training.
- `--batch_size`: Batch size per GPU (default: `1`).
- `--n_epochs`: Number of epochs at starting learning rate (default: `100`).
- `--n_epochs_decay`: Number of epochs to linearly decay learning rate to zero (default: `100`).

---

## 📈 Monitoring Progress

Training losses and generated image quality can be monitored in real-time:
1. **TensorBoard / Visdom**: Start visdom server before training:
   ```bash
   pip install visdom
   python -m visdom.server
   ```
2. **Saved Visuals**: Sample generated outputs are saved to `checkpoints/custom_ir_colorizer/web/index.html` every few hundred iterations.
