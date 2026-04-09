
import torch
import torch.nn as nn
import torch.optim as optim
import logging

from collections import defaultdict
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class Metrics:
    loss: float = 0.0
    dice: float = 0.0
    iou: float = 0.0

def iou_score(preds, masks, eps=1e-6):
    """
    preds: [B, H, W] 
    targets: [B, H, W] 
    """
    preds = preds.float()
    masks = masks.float()

    intersection = (preds * masks).sum(dim=(1, 2))
    union = (preds + masks - preds * masks).sum(dim=(1, 2))

    iou = (intersection + eps) / (union + eps)
    return iou.mean()
    
def dice_coeff(preds, masks, eps=1e-6):
    """
    preds: [B, H, W] 
    targets: [B, H, W] 
    """
    preds = preds.float()
    masks = masks.float()

    intersection = (preds * masks).sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + masks.sum(dim=(1, 2))

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


class MetricsAccumulator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.total_samples = 0   
        self.dice_total = 0.0
        self.iou_total = 0.0
        self.n_batches = 0

    def update(self, loss, preds, masks):
        """
        preds: [B, H, W]
        masks: [B, H, W]
        """
        B = masks.size(0)

        self.total_loss += loss.item() * B
        self.dice_total += dice_coeff(preds, masks).item() * B
        self.iou_total += iou_score(preds, masks).item() * B

        self.total_samples += B

    def compute(self):
        avg_loss = self.total_loss / self.total_samples
        avg_dice = self.dice_total / self.total_samples
        avg_iou = self.iou_total / self.total_samples

        return Metrics(avg_loss, avg_dice, avg_iou)

class EarlyStopping:
    """ Early stopping to stop training when validation accuracy doesn't improve. Saves the best checkpoint. """

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
        self.best_loss = float("inf")
        self.early_stop = False

        self.logger = logging.getLogger(__name__)

        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            if self.verbose:
                self.logger.info(f'Accuracy on validation set decreased from {self.best_loss: .4f} to {val_loss: .4f}. Saving model...')
            self.best_loss = val_loss
            self.counter = 0
            
            # Save the model
            torch.save(model.state_dict(), f"best_model.pth")
        else:
            self.counter += 1
            if self.verbose:
                self.logger.info(f'Early stopping counter: {self.counter} out of {self.patience}.')

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.logger.info('Early stopping triggered.')


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, masks):
        probs = torch.sigmoid(logits)
        dice = dice_coeff(probs, masks)

        masks = masks.unsqueeze(1).float()
        logits = logits.unsqueeze(1).float()
        bce = self.bce(logits, masks)
        return self.bce_weight * bce + (1 - self.bce_weight) * (1 - dice)

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        max_epochs,
        patience,
        base_lr,
        min_lr
    ):
        self.model = model.to(device)
        self.device = device

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.max_epochs = max_epochs
        self.patience = patience
        self.base_lr = base_lr
        self.min_lr = min_lr

        self.criterion = BCEDiceLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=base_lr)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_epochs,   
            eta_min=self.min_lr      
        )

        self.early_stopping = EarlyStopping(patience=self.patience, min_delta=0.0, verbose=True)

        self.logger = logging.getLogger(__name__) 

    def train_one_epoch(self):
        self.model.train()
        metrics = MetricsAccumulator()

        for imgs, masks in tqdm(self.train_loader, total=len(self.train_loader), desc="Training"):
            imgs, masks = imgs.to(self.device), masks.to(self.device)

            # Forward
            pred_masks = self.model(imgs)

            # Loss
            loss = self.criterion(pred_masks, masks)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Mask predictions
            probs = torch.sigmoid(pred_masks)
            preds = (probs > 0.5).long().squeeze(1)

            metrics.update(loss, preds, masks)

        return metrics.compute()

    def evaluate(self):
        self.model.eval()
        metrics = MetricsAccumulator()

        with torch.no_grad():
            for imgs, masks in tqdm(self.val_loader, total=len(self.val_loader), desc="Evaluating"):
                imgs, masks = imgs.to(self.device), masks.to(self.device)

                # Forward pass
                pred_masks = self.model(imgs)

                # Calculate loss
                loss = self.criterion(pred_masks, masks)

                # Mask predictions
                probs = torch.sigmoid(pred_masks)
                preds = (probs > 0.5).long().squeeze(1)

                metrics.update(loss, preds, masks)

            return metrics.compute()

    def __call__(self):
        for epoch in range(self.max_epochs):
            self.logger.info(f"________________________________________________________________________")
            self.logger.info(f"Epoch: {epoch+1} / {self.max_epochs}")
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Current LR: {current_lr:.6f}")  

            train_metrics = self.train_one_epoch()
            eval_metrics = self.evaluate()
            
            self.logger.info(f"Train Loss: {train_metrics.loss:.4f}           || Val Loss: {eval_metrics.loss:.4f}")
            self.logger.info(f"Train Dice coeff.: {train_metrics.dice:.4f}    || Val Dice coeff: {eval_metrics.dice:.4f}")
            self.logger.info(f"Train IoU: {train_metrics.iou:.4f}             || Val IoU: {eval_metrics.iou:.4f}")

            self.scheduler.step()

            self.early_stopping(eval_metrics.loss, self.model)
            if self.early_stopping.early_stop:
                break

        

           






















    



            
    
    
    