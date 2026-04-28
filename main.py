import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from tqdm import tqdm
import argparse
from pathlib import Path
import logging
import wandb
import random
import numpy as np

from train import Trainer
from unet import UNetModel 

from clahe_preprocessor import CLAHEPreprocessor
from normalizer import ZScoreNormalizer
from augmentations import Augmenter, AugmentationScheduler
from dataset import DataModule

from vis_augmentation import VisAugmentation
from vis_segmentation import VisSegmentation
from wandb_logger import WandbLogger
from logging_config import setup_logging

def parse_args():
    """
    Parse command-line arguments for training and logging configuration.
    Returns a namespace with all arguments.
    """

    parser = argparse.ArgumentParser(
        description="Segmentation using U-Net"
    )

    # Dataset name
    parser.add_argument("-d", "--dataset_name", type=str, required=True, help="Name of the dataset")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and validation")
    parser.add_argument("--max_epochs", type=int, default=80, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--base_lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimal lr")
    parser.add_argument("--img_size", type=float, default=224, help="Image size")
    parser.add_argument("--bce_loss_weight", type=float, default=0.5, help="Weight of the BCELoss part in the total DiceBCELoss")

    # Online augmentation parameters
    parser.add_argument("--use_aug", action="store_true", help="Enable augmentation")
    parser.add_argument("--aug_start_epoch", type=int, default=10, help="Epoch at which data augmentation begins to be applied (linearly increasing intensity)")
    parser.add_argument("--aug_end_epoch", type=int, default=70, help="Epoch at which augmentation reaches full intensity (1.0)")

    # CLAHE preprocessing
    parser.add_argument("--use_clahe", action="store_true", help="Enable CLAHE preprocessing")
    parser.add_argument("--clahe_clip_limit", type=float, default=1.25, help="Controls how much contrast is enhanced: lower values limit contrast amplification "
        "and reduce noise, higher values increase contrast but may amplify noise/artifacts.")
    
    # Wandb config
    parser.add_argument("--run_name", type=str, help="Name of the run in wandb logging")

    # Visualization
    parser.add_argument("--vis_augmentation", type=bool, default=True, help="Create and save fig of augmentation preview")
    parser.add_argument("--vis_segmentation", type=bool, default=True, help="Create and save fig of segmentation preview during training")

    args = parser.parse_args()

    return args

def main():

    # Setup logging
    setup_logging()

    # Set global seed
    set_global_seed(seed=42)

    # Parse args
    args = parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get U-Net model
    model = UNetModel()
    
    # Get CLAHE preprocessor
    if args.use_clahe:
        clahe_preprocessor = CLAHEPreprocessor(clahe_clip_limit=args.clahe_clip_limit)
    else:
        clahe_preprocessor = None

    # Get normalizer
    normalizer = ZScoreNormalizer()

    # Get augmenter 
    augmenter = Augmenter()

    # Visualize augmentation preview
    if args.vis_augmentation:
        visualize_augmentation(clahe_preprocessor, augmenter)

    # Augmentation Scheduler
    augmentation_scheduler = AugmentationScheduler(
        start_epoch=args.aug_start_epoch,
        end_epoch=args.aug_end_epoch
    )
    
    # Datamodule
    data = DataModule(
        images_path="dataset/images",
        masks_path="dataset/masks",
        img_size=args.img_size,
        batch_size=args.batch_size,
        clahe_preprocessor=clahe_preprocessor,
        normalizer=normalizer,
        augmenter=augmenter,
        augmentation_scheduler=augmentation_scheduler,
        num_workers=2 
    ).setup()

    # Train and Val loaders
    train_loader, val_loader = data.get_loaders()

    # Wandb logger
    wandb_logger = WandbLogger(args)

    trainer = Trainer(
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            base_lr=args.base_lr,
            min_lr=args.min_lr,
            bce_loss_weight=args.bce_loss_weight,
            augmentation_scheduler=augmentation_scheduler,
            wandb_logger=wandb_logger
        )

    trainer()

def visualize_augmentation(clahe_preprocessor, augmenter):

    args = parse_args()

    visualizer = VisAugmentation(
        images_path="dataset/images",
        masks_path="dataset/masks",
        clahe_preprocessor=clahe_preprocessor,
        augmenter=augmenter
    )
    
    visualizer(
        num_samples=10,
        save_path="aug_intensity_0.png",
        intensity=0.0
    )

    visualizer(
        num_samples=10,
        save_path="aug_intensity_1.png",
        intensity=1.0
    )

def set_global_seed(seed: int):
    # Python RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
   main()
    
    
















           






















    



            
    
    
    