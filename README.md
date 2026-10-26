# 🐾 PyTorch Animal Image Classification

This repository contains a custom Convolutional Neural Network (CNN) built with PyTorch to classify animal images. The project features optimized GPU training pipeline, model persistence, and an interactive inference script with dynamic file path checks.

---

## 📌 Features

- **Custom CNN Architecture**: Designed from scratch using PyTorch (`Conv2d`, `ReLU`, `MaxPool2d`, `Linear`).
- **Optimized Data Pipeline**: Utilizes CUDA acceleration along with `DataLoader` optimizations (`num_workers=6`, `pin_memory=True`).
- **Dynamic Inference**: Interactive command-line test script with `os.path.exists` validation to classify any input image dynamically.
- **Model Persistence**: Trained weights saved cleanly as `which_animal.pth` (~334 KB).

---

## 🛠️ Tech Stack & Dependencies

- **Framework**: Python 3.x, PyTorch (`torch`, `torchvision`)
- **Image Processing**: Pillow (`PIL`)
- **Hardware Acceleration**: NVIDIA CUDA (RTX Series recommended)

---

## 📁 Repository Structure

```text
├── train.py           # Model training script with GPU support
├── test.py          # Dynamic inference script for predicting new images
├── which_animal.pth   # Pre-trained CNN weight checkpoint
├── .gitignore         # Excludes dataset and temporary cache files
└── README.md          # Project documentation
