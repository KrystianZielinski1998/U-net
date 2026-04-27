import torch
import torch.nn as nn
from dataclasses import dataclass, field
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score

# ----------------------------
# Metrics
# ----------------------------
@dataclass
class Metrics:
    bcedice_loss: float = 0.0
    dice_loss: float = 0.0
    iou_loss: float = 0.0
    dice_metric: float = 0.0
    iou_metric: float = 0.0
    
@dataclass
class MetricsHistory:
    bcedice_loss: list[float] = field(default_factory=list)
    dice_loss: list[float] = field(default_factory=list)
    iou_loss: list[float] = field(default_factory=list)
    dice_metric: list[float] = field(default_factory=list)
    iou_metric: list[float] = field(default_factory=list)
    
# ----------------------------
# IoU Loss
# ----------------------------
def iou_loss(logits, targets, eps=1e-6):
    targets = targets.unsqueeze(1).float()
    probs = torch.sigmoid(logits)

    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)

    return 1 - iou.mean()


# ----------------------------
# Dice Loss
# ----------------------------
def dice_loss(logits, targets, eps=1e-6):
    targets = targets.unsqueeze(1).float()  # [B,1,H,W]
    probs = torch.sigmoid(logits)

    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)

    return 1 - dice.mean()


# ----------------------------
# BCEDiceLoss
# ----------------------------
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_loss_weight):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_loss_weight = bce_loss_weight

    def forward(self, logits, masks):

        masks = masks.unsqueeze(1).float()  # [B,1,H,W]

        bce_loss = self.bce(logits, masks)
        d_loss = dice_loss(logits, masks)

        loss = self.bce_weight * bce_loss + (1 - self.bce_weight) * d_loss

        return loss


# ----------------------------
# Metrics accumulator
# ----------------------------
class MetricsAccumulator:
    def __init__(self, device):
        self.device = device
        self.history_train = MetricsHistory()
        self.history_val = MetricsHistory()

    def reset(self):
        self.total_bcedice_loss = 0.0
        self.total_dice_loss = 0.0
        self.total_iou_loss = 0.0
        self.dice_metric_total = 0.0
        self.iou_metric_total = 0.0
        self.total_samples = 0

        self.dice_metric = BinaryF1Score().to(self.device)
        self.iou_metric = BinaryJaccardIndex().to(self.device)
        

    def update(self, bcedice_loss, dice_loss, iou_loss, preds, masks):
        """
        Args:
            preds (): [B,H,W] 
            masks (): [B,H,W]
        """
        B = masks.size(0)

        self.total_bcedice_loss += bcedice_loss.item() * B
        self.total_dice_loss += dice_loss.item() * B
        self.total_iou_loss += iou_loss.item() * B

        self.dice_metric.update(preds, masks)
        self.iou_metric.update(preds, masks)

        self.total_samples += B


    def compute(self):
        avg_bcedice_loss = self.total_bcedice_loss / self.total_samples
        avg_dice_loss = self.total_dice_loss / self.total_samples
        avg_iou_loss = self.total_iou_loss / self.total_samples

        avg_dice_metric = self.dice_metric.compute().item()
        avg_iou_metric = self.iou_metric.compute().item()

        return Metrics(avg_bcedice_loss, avg_dice_loss, avg_iou_loss, avg_dice_metric, avg_iou_metric)

    def store(self, metrics: Metrics, mode: bool):
        match mode:
            case "train":
                self.history_train.bcedice_loss.append(metrics.bcedice_loss)
                self.history_train.dice_loss.append(metrics.dice_loss)
                self.history_train.iou_loss.append(metrics.iou_loss)
                self.history_train.dice_metric.append(metrics.dice_metric)
                self.history_train.iou_metric.append(metrics.iou_metric)        
            
            case "val":
                self.history_val.bcedice_loss.append(metrics.bcedice_loss)
                self.history_val.dice_loss.append(metrics.dice_loss)
                self.history_val.iou_loss.append(metrics.iou_loss)
                self.history_val.dice_metric.append(metrics.dice_metric)
                self.history_val.iou_metric.append(metrics.iou_metric)      

            

            

           



        

           






















    



            
    
    
    