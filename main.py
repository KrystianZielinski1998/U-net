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

from train import Trainer
from unet import UNetModel 

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

    # Dataset parameters
    parser.add_argument("-d", "--dataset_name", type=str, required=True, help="Name of the dataset")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and validation")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--base_lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimal lr")
    parser.add_argument("--img_size", type=float, default=224, help="Image size")
    parser.add_argument("--aug_start_epoch", type=int, default=10, help="Epoch at which data augmentation begins to be applied (linearly increasing intensity)")
    parser.add_argument("--aug_end_epoch", type=int, default=90, help="Epoch at which augmentation reaches full intensity (1.0)")
    parser.add_argument("--vis_augmentation", type=bool, default=True, help="Create and save fig of augmentation preview")
    parser.add_argument("--vis_segmentation", type=bool, default=True, help="Create and save fig of segmentation preview during training")
    args = parser.parse_args()

    return args

def main():

    # Setup logging
    setup_logging()

    # Parse args
    args = parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get U-Net model
    model = UNetModel()
    
    # Get normalizer
    normalizer = ZScoreNormalizer()

    # Get augmenter 
    augmenter = Augmenter()

    # Visualize augmentation preview
    if args.vis_augmentation:
        visualize_augmentation(augmenter)

    # Augmentation Scheduler
    augmentation_scheduler = AugmentationScheduler(
        max_epochs=args.max_epochs,
        no_aug_epochs=args.no_aug_epochs
    )
    
    # Datamodule
    data = DataModule(
        images_path="dataset/images",
        masks_path="dataset/masks",
        img_size=args.img_size,
        batch_size=args.batch_size,
        normalizer=normalizer,
        augmenter=augmenter,
        augmentation_scheduler=augmentation_scheduler,
        num_workers=4   
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
            augmentation_scheduler=augmentation_scheduler,
            wandb_logger=wandb_logger
        )

    trainer()

def visualize_augmentation(augmenter):

    args = parse_args()

    visualizer = VisAugmentation(
        images_path="dataset/images",
        masks_path="dataset/masks",
        augmenter=augmenter,
        intensity=0.0
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


if __name__ == "__main__":
   main()
    
    
















           






















    



            
    
    
    