# U-Net implementation for 2D Brain Tumor Segmentation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)

This project implements a U-Net architecture for 2D brain tumor segmentation, including a full training and evaluation pipeline built with PyTorch.

It features a custom curriculum-based data augmentation strategy implemented using Albumentations, where augmentation intensity gradually increases during training. The pipeline also includes Z-score normalization and optional CLAHE contrast enhancement as preprocessing steps.

Segmentation performance is evaluated using torchmetrics and custom loss functions, with visualization tools built using matplotlib. Experiment tracking is handled via Weights & Biases.

The dataset used for training and evaluation is available on Kaggle:  
https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation

---

## Key Features

* **Custom U-Net Architecture** – U-Net with Residual Blocks.
* **Custom Hybrid Loss Function** – Custom hybrid loss function combining BCE and Dice Loss.
* **Performance Metrics** – Dice and IoU scores computed using torchmetrics (`BinaryF1Score`, `BinaryJaccardIndex`) along with custom loss implementations.
* **Curriculum-based Data Augmentation** – Augmentation strength increases progressively during training using Albumentations.
* **Preprocessing Pipeline** – Resizing, Z-score normalization and optional CLAHE contrast enhancement.
* **Segmentation Visualization** – Visualization of predictions on validation samples after each epoch.
* **Experiment Tracking** – Integrated experiment tracking using Weights & Biases.

---

## Tech Stack

- **Core**: Python, PyTorch  
- **Computer Vision**: OpenCV, Albumentations  
- **Metrics**: torchmetrics  
- **Visualization**: matplotlib  
- **Experiment Tracking**: Weights & Biases  

---

## Project Structure

```text
UNet/
├── augmentations.py        # Curriculum-based online data augmentation pipeline (Albumentations)
├── clahe_preprocessor.py   # CLAHE contrast enhancement (optional preprocessing)
├── dataset.py              # Dataset class and data loading logic
├── logging_config.py       # Configuration for logging and experiment tracking
├── main.py                 # Main
├── metrics.py              # Metrics and loss functions (Dice, IoU, BCE-Dice)
├── normalizer.py           # Z-score normalization
├── train.py                # Training and validation loops
├── unet.py                 # U-Net model with residual blocks
├── vis_augmentation.py     # Visualization of data augmentation effects
├── vis_segmentation.py     # Visualization of segmentation predictions
├── wandb_logger.py         # Weights & Biases logging utilities
```

--- 

### Running the Project 
1. Open `U-Net.ipynb` in Google Colab and run all the cells. 

--- 

## License 
This project was made for educational purposes.