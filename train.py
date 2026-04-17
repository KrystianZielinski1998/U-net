import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from dataclasses import dataclass
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score
from tabulate import tabulate
from segmentation_vis import SegmentationVis
from augmentations import AugmentationScheduler
from dataset import SegmentationDataset, GetLoaders
import os

# ----------------------------
# Metrics
# ----------------------------
@dataclass
class Metrics:
    loss: float = 0.0
    dice: float = 0.0
    iou: float = 0.0
    dice_torch: float = 0.0
    iou_torch: float = 0.0
    
# ----------------------------
# Early Stopping
# ----------------------------
class EarlyStopping:
    """ Early stopping to stop training when validation metric doesn't improve. Saves the best checkpoint. """

    def __init__(self, 
        patience: int, 
        min_delta: float,
        verbose: bool
    ):
        """   
        Args:
            patience (int): Number of epochs to wait after min has been hit before stopping.
            min_delta (float): Minimum change in monitored value to qualify as improvement.
            verbose (bool): Add logs.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_dice = 0.0
        self.early_stop = False

        self.logger = logging.getLogger(__name__)

        os.makedirs("checkpoints", exist_ok=True)
        
    def __call__(self, val_dice, model):
        if val_dice > self.best_dice - self.min_delta:
            if self.verbose:
                self.logger.info(f'Dice score on validation set increased from {self.best_dice: .4f} to {val_dice: .4f}. Saving model...')
            self.best_dice = val_dice
            self.counter = 0
            
            # Save the model
            torch.save(model.state_dict(), f"checkpoints/best_model.pth")

        else:
            self.counter += 1
            if self.verbose:
                self.logger.info(f'Early stopping counter: {self.counter} out of {self.patience}.')

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.logger.info('Early stopping triggered.')

# ----------------------------
# Dice
# ----------------------------
def dice_coeff(preds, masks, eps=1e-6):
    """
    Args:
        preds (): [B,H,W] 
        masks (): [B,H,W]
    """

    preds = preds.float()
    masks = masks.float()

    intersection = (preds * masks).sum(dim=(1,2))
    union = preds.sum(dim=(1,2)) + masks.sum(dim=(1,2))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()

# ----------------------------
# IoU
# ----------------------------
def iou_score(preds, masks, eps=1e-6):
    """
    Args:
        preds (): [B,H,W] 
        masks (): [B,H,W]
    """

    preds = preds.float()
    masks = masks.float()

    intersection = (preds * masks).sum(dim=(1,2))
    union = (preds + masks - preds * masks).sum(dim=(1,2))
    return ((intersection + eps) / (union + eps)).mean()

# ----------------------------
# Metrics accumulator
# ----------------------------
class MetricsAccumulator:
    def __init__(self, device):
        self.device = device
        self.reset()
        

    def reset(self):
        self.total_loss = 0.0
        self.total_samples = 0
        self.dice_total = 0.0
        self.iou_total = 0.0
        self.dice_torch_total = 0.0
        self.iou_torch_total = 0.0
        
        self.dice_torch = BinaryF1Score().to(self.device)
        self.iou_torch = BinaryJaccardIndex().to(self.device)
        

    def update(self, loss, preds, masks):
        """
        Args:
            preds (): [B,H,W] 
            masks (): [B,H,W]
        """
        B = masks.size(0)

        self.total_loss += loss.item() * B

        self.dice_total += dice_coeff(preds, masks).item() * B
        self.iou_total += iou_score(preds, masks).item() * B
        
        self.dice_torch.update(preds, masks)
        self.iou_torch.update(preds, masks)

        self.total_samples += B


    def compute(self):
        avg_loss = self.total_loss / self.total_samples

        avg_dice = self.dice_total / self.total_samples
        avg_iou = self.iou_total / self.total_samples

        avg_dice_torch = self.dice_torch.compute().item()
        avg_iou_torch = self.iou_torch.compute().item()

        return Metrics(avg_loss, avg_dice, avg_iou, avg_dice_torch, avg_iou_torch)

# ----------------------------
# Loss
# ----------------------------
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, masks):
        """
        Args:
            logits (): [B, 1, H, W] from the model
            masks (): [B, H, W] ground truth
        """
        # Add channel dimension to masks to match logits
        masks = masks.unsqueeze(1).float()  # [B,1,H,W]

        # Sigmoid for Dice
        probs = torch.sigmoid(logits)
        dice = dice_coeff(probs, masks)  

        # BCE loss
        bce = self.bce(logits, masks)

        return self.bce_weight * bce + (1 - self.bce_weight) * (1 - dice)

# ----------------------------
# Trainer
# ----------------------------
class Trainer:
    def __init__(self, model, device, max_epochs, patience, batch_size, base_lr, min_lr, no_aug_epochs):
        self.model = model.to(device)
        
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.no_aug_epochs = no_aug_epochs

        self.criterion = BCEDiceLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=base_lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_epochs, eta_min=self.min_lr)
        self.early_stopping = EarlyStopping(patience=self.patience, min_delta=0.0, verbose=True)
        self.augmentation_scheduler = AugmentationScheduler(self.max_epochs, self.no_aug_epochs)

        self.get_loaders = GetLoaders(
            images_path="dataset/images",
            masks_path="dataset/masks",
            batch_size=self.batch_size,
            val_split=0.1, 
            transform=self.augmentation_scheduler
        )

        self.train_loader, self.val_loader = self.get_loaders()

        self.vis = SegmentationVis(self.val_loader, self.device)

        self.logger = logging.getLogger(__name__)
        
        self.log_model_info()

    def log_model_info(self):
        self.logger.info(self.model)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info(f"Total params: {total_params:,}")
        self.logger.info(f"Trainable params: {trainable_params:,}")

    def train_one_epoch(self):
        self.model.train()
        metrics = MetricsAccumulator(self.device)

        for imgs, masks in tqdm(self.train_loader, total=len(self.train_loader), desc="Training"):
            imgs, masks = imgs.to(self.device), masks.to(self.device)

            # Forward
            logits = self.model(imgs)
            loss = self.criterion(logits, masks)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Predictions for metrics
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()  

            preds = preds.squeeze(1)
            masks = masks.squeeze(1)

            metrics.update(loss, preds, masks)

        return metrics.compute()

    def evaluate(self):
        self.model.eval()
        metrics = MetricsAccumulator(self.device)

        with torch.no_grad():
            for imgs, masks in tqdm(self.val_loader, total=len(self.val_loader), desc="Evaluating"):
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                logits = self.model(imgs)
                loss = self.criterion(logits, masks)

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()

                preds = preds.squeeze(1)
                masks = masks.squeeze(1)

                metrics.update(loss, preds, masks)

        return metrics.compute()

    def log_metrics(self, train_metrics, eval_metrics):
    
        table = [
            ["Loss", train_metrics.loss, eval_metrics.loss],
            ["Dice", train_metrics.dice, eval_metrics.dice],
            ["Dice (torch)", train_metrics.dice_torch, eval_metrics.dice_torch],
            ["IoU", train_metrics.iou, eval_metrics.iou],
            ["IoU (torch)", train_metrics.iou_torch, eval_metrics.iou_torch],
        ]

        self.logger.info("\n" + tabulate(
            table,
            headers=["Metric", "Train", "Val"],
            floatfmt=".4f",
            tablefmt="fancy_grid"  
        ))

    def __call__(self):
        for epoch in range(self.max_epochs):
            self.logger.info(f"_____________________________________________________________________________________")
            self.logger.info(f"Epoch: {epoch+1} / {self.max_epochs}")
            self.logger.info(f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            self.augmentation_scheduler.set_epoch(epoch)

            train_metrics = self.train_one_epoch()
            eval_metrics = self.evaluate()

            self.log_metrics(train_metrics, eval_metrics)
            
            if (epoch+1) % 1 == 0:
                self.vis(model=self.model, epoch=epoch+1, num_samples=6)

            self.scheduler.step()
            self.early_stopping(eval_metrics.dice_torch, self.model)
            if self.early_stopping.early_stop:
                break

            

            

           



        

           






















    



            
    
    
    