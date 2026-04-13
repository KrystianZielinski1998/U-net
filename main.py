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
from logging_config import setup_logging
from unet import UNetModel 

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
    parser.add_argument("--max_epochs", type=int, default=120, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--base_lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="Minimal lr")

    args = parser.parse_args()

    return args

def main():

    # Setup logging
    setup_logging()

    # Parse args
    args = parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get loaders
    get_loaders = GetLoaders(
        images_path="dataset/images",
        masks_path="dataset/masks",
        batch_size=args.batch_size,
        val_split=0.1, 
        transform=None
    )

    train_loader, val_loader = get_loaders()

    # Get U-Net model
    model = UNetModel()
    
    trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            base_lr=args.base_lr,
            min_lr=args.min_lr
        )

    trainer()

if __name__ == "__main__":
    main()
    
















           






















    



            
    
    
    