# CSC871_FinalProject_SFSU_Fall2025
CSC871_FinalProject_SFSU_Fall2025


# Pneumonia Detection from Chest X-ray Images

**CSC 871 Deep Learning - Fall 2025**  
**San Francisco State University**

---

## Team Members

- **Fayeeza** - Data & Infrastructure Lead
- **Veronica** - Model Development Lead  
- **Ratchagan** - Experiments & Analysis Lead

---

## Project Description

Automated pneumonia detection from chest X-ray images using deep learning. We will train and compare multiple CNN architectures to classify X-rays as NORMAL or PNEUMONIA.

---

## Dataset

**Source:** Chest X-Ray Images (Pneumonia) - Kaggle  
**Total Images:** 5,856 chest X-ray images

**Data Split:**
- Training: 5,216 images
- Validation: 782 images
- Test: 624 images

**Classes:** NORMAL (0) and PNEUMONIA (1)

---

## Installation
```bash
# Install dependencies
pip install torch torchvision numpy matplotlib pillow

# Run data module to test
python data_module.py
```

---

## Quick Start
```python
from data_module import get_data_loaders

# Load data
train_loader, val_loader, test_loader = get_data_loaders(
    batch_size=32,
    image_size=224
)

# Use in training
for images, labels in train_loader:
    # images: [32, 3, 224, 224]
    # labels: 0=NORMAL, 1=PNEUMONIA
    pass
```

---

## Project Structure
```
CSC871_Final_Project/
├── chest_xray/          # Dataset
├── data_module.py       # Data loading
├── create_val_split.py  # Validation split script
├── requirements.txt     # Dependencies
└── README.md           # This file
```

---

## Models to Test

- Baseline CNN
- ResNet-18 / ResNet-50
- DenseNet-121
- EfficientNet-B0
- VGG-16

---

## Important Notes

- **Class Imbalance:** Dataset has 3:1 ratio (pneumonia:normal)
- **Validation Set:** Original had only 16 images, we created new set with 782 images
- **Image Size:** All images resized to 224×224
- **Augmentation:** Training images have flip, rotation, and color jitter


---
