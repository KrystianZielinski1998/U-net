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

from dataset import SegmentationDataset, GetLoaders
from train import Trainer
from utils.logging_config import setup_logging
from unet import UNet 

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
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and validation")
    parser.add_argument("--max_epochs", type=int, default=120, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--base_lr", type=float, default=5e-4, help="Initial learning rate")

    args = parser.parse_args()

    return args

def main():

    # Setup logging
    setup_logging()

    # Parse args
    args = parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get dataset
    segmentation_dataset = SegmentationDataset(
        images_path="dataset/images",
        masks_path="dataset/masks"
    )
    # Get loaders
    get_loaders = GetLoaders(args.dataset_path, args.batch_size)
    train_loader, val_loader = get_loaders()

    # Get U-Net model
    model = UNet()
    
    trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            base_lr=args.base_lr
        )

    trainer()

if __name__ == "__main__":
    main()
    
















           






















    



            
    
    
    