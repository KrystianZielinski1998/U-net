# U-Net for 2D Brain Tumor Segmentation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)
![Weights & Biases](https://img.shields.io/badge/Weights%20%26%20Biases-1.0-FFBE00?logo=weightsandbiases)

### Summary

This project implements a U-Net architecture with ResNet blocks for 2D brain tumor segmentation. It includes training and evaluation pipeline implemented with PyTorch.

In this project, a preprocessing pipeline includes resizing images to a fixed resolution and applying Z-score normalization. Additionally, an optional CLAHE contrast enhancement step was introduced. The impact of preprocesing images with CLAHE contrast enhancement on segmentation performance is evaluated. Results of this experiment are presented in section below.

For this project, a custom curriculum-based online data augmentation strategy was implemented, in which augmentation intensity gradually increases during training. The augmentation pipeline includes horizontal flipping, random rotations, zooming, translations, and shearing.

Segmentation performance is evaluated using Dice and IoU scores, with additional segmentation visualization tools. 

The experiments were conducted in Google Colab on NVIDIA L4 GPU and the results were tracked via Weights & Biases. 

The dataset used for training and evaluation is available on Kaggle:  
https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation

---

## Key Features

* **Custom U-Net Architecture** – U-Net with Residual Blocks.
* **Custom Hybrid Loss Function** – Custom hybrid loss function combining BCE and Dice Loss.
* **Curriculum-based Data Augmentation** – Augmentation strength increases progressively during training.
* **Preprocessing Pipeline** – Resizing, Z-score normalization and optional CLAHE contrast enhancement.
* **Performance Metrics** – Dice and IoU scores computed using torchmetrics (`BinaryF1Score`, `BinaryJaccardIndex`).
* **Segmentation Visualization** – Visualization of predictions on validation samples after each epoch.
* **Experiment Tracking** – Integrated experiment tracking using Weights & Biases.
* **CLAHE Impact Analysis** – Assessment of CLAHE contrast enhancement impact on segmentation performance.
---

## Tech Stack

- **Core**: Python, PyTorch  
- **Computer Vision**: OpenCV, Albumentations  
- **Metrics**: torchmetrics  
- **Visualization**: matplotlib  
- **Experiment Tracking**: Weights & Biases  

---

## Results

The impact of contrast enhancement using CLAHE on segmentation performance was evaluated by comparing it against the same pipeline without the CLAHE preprocessing step. The results are reported as the best Dice and IoU scores achieved on the validation set. The results are presented in the table below.

| Preprocessing | Dice Score | IoU Score |
|---------------|------------|-----------|
| CLAHE         | 0.83188    | 0.71216   |
| Baseline      | 0.82761    | 0.70591   |

All training metrics were tracked using Weights & Biases, which enabled visualization of model performance. An example plot of the validation Dice score over training epochs is shown below.

<p align="center">
  <img src="images/dice.png" width="700"/>
</p>
<p align="center"><em>Figure 1: Comparison of Dice scores on the validation set for the baseline and CLAHE preprocessing.</em></p>

Additionaly, during training segmentation results on selected validation images were logged. Below are example plots for the baseline model and the model with CLAHE image preprocessing.

<p align="center">
  <img src="images/base_vis.png" width="700"/>
</p>  
<p align="center"><em>Figure 2: Segmentation performance visualization on validation set samples for the baseline model</em></p>

<p align="center">  
  <img src="images/clahe_vis.png" width="700"/>
</p>
<p align="center"><em>Figure 3: Segmentation performance visualization on validation set samples for the model with CLAHE image preprocessing.</em></p>

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