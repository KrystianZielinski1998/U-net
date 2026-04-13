import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from dataclasses import dataclass
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
from torchmetrics.segmentation import MeanIoU, DiceScore
from segmentation_vis import SegmentationVis

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
        self.best_loss = float("inf")
        self.early_stop = False

        self.logger = logging.getLogger(__name__)

        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            if self.verbose:
                self.logger.info(f'Loss on validation set decreased from {self.best_loss: .4f} to {val_loss: .4f}. Saving model...')
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

def dice_coeff(preds, masks, eps=1e-6):
    """
    preds: [B,H,W] or [B,1,H,W]
    masks: [B,H,W] or [B,1,H,W]
    """
    if preds.ndim == 4:
        preds = preds.squeeze(1)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    preds = preds.float()
    masks = masks.float()

    intersection = (preds * masks).sum(dim=(1,2))
    union = preds.sum(dim=(1,2)) + masks.sum(dim=(1,2))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


def iou_score(preds, masks, eps=1e-6):
    """
    preds: [B,H,W] or [B,1,H,W]
    masks: [B,H,W] or [B,1,H,W]
    """
    if preds.ndim == 4:
        preds = preds.squeeze(1)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

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
        
        self.dice_torch = DiceScore().to(self.device)
        self.iou_torch = MeanIoU(num_classes=2).to(self.device)
        

    def update(self, loss, preds, masks):
        """
        preds: [B,1,H,W]
        masks: [B,1,H,W]
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
        logits: [B, 1, H, W] from the model
        masks: [B, H, W] ground truth
        """
        # Add channel dimension to masks to match logits
        masks = masks.unsqueeze(1).float()  # [B,1,H,W]

        # Sigmoid for Dice
        probs = torch.sigmoid(logits)
        dice = dice_coeff(probs, masks)  # safe dice, handles [B,1,H,W]

        # BCE loss
        bce = self.bce(logits, masks)

        return self.bce_weight * bce + (1 - self.bce_weight) * (1 - dice)
# ----------------------------
# Trainer
# ----------------------------
class Trainer:
    def __init__(self, model, train_loader, val_loader, device,
                 max_epochs, patience, base_lr, min_lr):
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
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_epochs, eta_min=self.min_lr)
        self.early_stopping = EarlyStopping(patience=self.patience, min_delta=0.0, verbose=True)
        self.vis = SegmentationVis(self.model, self.val_loader, self.device)

        self.logger = logging.getLogger(__name__)

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
            preds = (probs > 0.5).long()  # keep shape [B,1,H,W]
            metrics.update(loss, preds, masks)

        return metrics.compute()

    def evaluate(self):
        self.model.eval()
        metrics = MetricsAccumulator()
        with torch.no_grad():
            for imgs, masks in tqdm(self.val_loader, total=len(self.val_loader), desc="Evaluating"):
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                logits = self.model(imgs)
                loss = self.criterion(logits, masks)

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                metrics.update(loss, preds, masks)

        return metrics.compute()

    def __call__(self):
        for epoch in range(self.max_epochs):
            self.logger.info(f"_____________________________________________________________________________________")
            self.logger.info(f"Epoch: {epoch+1} / {self.max_epochs}")
            self.logger.info(f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            train_metrics = self.train_one_epoch()
            eval_metrics = self.evaluate()

            self.logger.info(f"Train Loss: {train_metrics.loss:.4f} || Val Loss: {eval_metrics.loss:.4f}")

            self.logger.info(f"Train Dice: {train_metrics.dice:.4f} || Val Dice: {eval_metrics.dice:.4f}")
            self.logger.info(f"Train Torch Dice: {train_metrics.dice_torch:.4f} || Val Dice: {eval_metrics.dice_torch:.4f}")

            self.logger.info(f"Train IoU: {train_metrics.iou:.4f} || Val IoU: {eval_metrics.iou:.4f}")
            self.logger.info(f"Train Torch IoU: {train_metrics.dice_iou:.4f} || Val Torch IoU: {eval_metrics.dice_iou:.4f}")
            
            if epoch % 5 == 0:
                self.vis(model=self.model, num_samples=6)

            self.scheduler.step()
            self.early_stopping(eval_metrics.loss, self.model)
            if self.early_stopping.early_stop:
                break

            

           



        

           






















    



            
    
    
    