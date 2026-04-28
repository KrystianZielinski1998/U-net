import torch
import torch.nn as nn
from dataclasses import dataclass, field
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score


# ----------------------------
# Metrics containers
# ----------------------------
@dataclass
class Metrics:
    """
    Container for storing averaged metrics for a single epoch.

    Attributes:
        bcedice_loss (float): Combined BCE + Dice loss
        dice_loss (float): Dice loss
        iou_loss (float): IoU loss
        dice_metric (float): Dice score (F1)
        iou_metric (float): IoU score
    """
    bcedice_loss: float = 0.0
    dice_loss: float = 0.0
    iou_loss: float = 0.0
    dice_metric: float = 0.0
    iou_metric: float = 0.0
    

@dataclass
class MetricsHistory:
    """
    Stores metric values across all epochs.

    Each field contains a list of values (one per epoch).
    """
    bcedice_loss: list[float] = field(default_factory=list)
    dice_loss: list[float] = field(default_factory=list)
    iou_loss: list[float] = field(default_factory=list)
    dice_metric: list[float] = field(default_factory=list)
    iou_metric: list[float] = field(default_factory=list)
    

# ----------------------------
# IoU Loss
# ----------------------------
def iou_loss(logits, targets, eps=1e-6):
    """
    Computes Intersection over Union (IoU) loss.

    Args:
        logits (torch.Tensor): Model outputs [B,1,H,W]
        targets (torch.Tensor): Ground truth masks [B,H,W]
        eps (float): Small constant to avoid division by zero

    Returns:
        torch.Tensor: Scalar IoU loss
    """

    # Add channel dimension [B,1,H,W]
    targets = targets.unsqueeze(1).float()

    # Convert logits into probabilities
    probs = torch.sigmoid(logits)

    # Flatten tensors [B, H*W]
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    # Compute intersection and union
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1) - intersection

    # Compute IoU per sample
    iou = (intersection + eps) / (union + eps)

    # Return IoU loss
    return 1 - iou.mean()


# ----------------------------
# Dice Loss
# ----------------------------
def dice_loss(logits, targets, eps=1e-6):
    """
    Computes Dice loss.

    Args:
        logits (torch.Tensor): Model outputs [B,1,H,W]
        targets (torch.Tensor): Ground truth masks [B,H,W]
        eps (float): Small constant for numerical stability

    Returns:
        torch.Tensor: Scalar Dice loss
    """

    # Add channel dimension
    targets = targets.unsqueeze(1).float()

    # Convert logits → probabilities
    probs = torch.sigmoid(logits)

    # Flatten tensors
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    # Compute Dice components
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    # Dice coefficient
    dice = (2 * intersection + eps) / (union + eps)

    return 1 - dice.mean()


# ----------------------------
# BCE + Dice combined loss
# ----------------------------
class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice loss.

    Final loss:
        loss = w * BCE + (1 - w) * Dice
    """

    def __init__(self, bce_loss_weight):
        """
        Args:
            bce_loss_weight (float): Weight for BCE loss (0–1)
        """
        super().__init__()

        # BCE with logits (numerically stable)
        self.bce = nn.BCEWithLogitsLoss()

        # Weight controlling BCE vs Dice contribution
        self.bce_loss_weight = bce_loss_weight

    def forward(self, logits, masks):
        """
        Args:
            logits (torch.Tensor): Model outputs [B,1,H,W]
            masks (torch.Tensor): Ground truth masks [B,H,W]

        Returns:
            torch.Tensor: Combined loss
        """

        # Add channel dimension
        masks = masks.unsqueeze(1).float()

        # Compute individual losses
        bce_loss = self.bce(logits, masks)
        d_loss = dice_loss(logits, masks)

        # Weighted combination
        loss = self.bce_loss_weight * bce_loss + (1 - self.bce_loss_weight) * d_loss

        return loss


# ----------------------------
# Metrics accumulator
# ----------------------------
class MetricsAccumulator:
    """
    Accumulates losses and metrics over an epoch.

    Responsibilities:
    - Track running sums
    - Compute averages
    - Store history for train/validation
    """

    def __init__(self, device):
        """
        Args:
            device (torch.device): Device for metric computation
        """
        self.device = device

        # Store full training history
        self.history_train = MetricsHistory()
        self.history_val = MetricsHistory()

    def reset(self):
        """
        Resets accumulators at the start of each epoch.
        """

        # Running sums for losses
        self.total_bcedice_loss = 0.0
        self.total_dice_loss = 0.0
        self.total_iou_loss = 0.0

        # Running metric states
        self.dice_metric_total = 0.0
        self.iou_metric_total = 0.0

        # Number of processed samples
        self.total_samples = 0

        # TorchMetrics objects (stateful)
        self.dice_metric = BinaryF1Score().to(self.device)
        self.iou_metric = BinaryJaccardIndex().to(self.device)
        

    def update(self, bcedice_loss, dice_loss, iou_loss, preds, masks):
        """
        Updates accumulators with batch results.

        Args:
            bcedice_loss (torch.Tensor): Loss value
            dice_loss (torch.Tensor): Dice loss
            iou_loss (torch.Tensor): IoU loss
            preds (torch.Tensor): Predictions [B,H,W]
            masks (torch.Tensor): Ground truth [B,H,W]
        """

        # Batch size
        B = masks.size(0)

        # Accumulate weighted losses
        self.total_bcedice_loss += bcedice_loss.item() * B
        self.total_dice_loss += dice_loss.item() * B
        self.total_iou_loss += iou_loss.item() * B

        # Update metrics (internal state)
        self.dice_metric.update(preds, masks)
        self.iou_metric.update(preds, masks)

        # Update sample count
        self.total_samples += B


    def compute(self):
        """
        Computes final averaged metrics for the epoch.

        Returns:
            Metrics: Aggregated results
        """

        # Average losses
        avg_bcedice_loss = self.total_bcedice_loss / self.total_samples
        avg_dice_loss = self.total_dice_loss / self.total_samples
        avg_iou_loss = self.total_iou_loss / self.total_samples

        # Compute torchmetrics
        avg_dice_metric = self.dice_metric.compute().item()
        avg_iou_metric = self.iou_metric.compute().item()

        return Metrics(
            avg_bcedice_loss,
            avg_dice_loss,
            avg_iou_loss,
            avg_dice_metric,
            avg_iou_metric
        )


    def store(self, metrics: Metrics, mode: bool):
        """
        Stores metrics into history.

        Args:
            metrics (Metrics): Computed metrics
            mode (str): "train" or "val"

        Returns:
            None
        """

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

            

            

           



        

           






















    



            
    
    
    