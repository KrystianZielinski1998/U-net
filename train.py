import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os

from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from tabulate import tabulate

from metrics import dice_loss, iou_loss, BCEDiceLoss, MetricsAccumulator
from vis_segmentation import VisSegmentation

# ----------------------------
# Early Stopping
# ----------------------------
class EarlyStopping:
    """
    Early stopping mechanism.

    This class:

    - Stops training when validation metric stops improving.
    - Saves the best model checkpoint.
    """

    def __init__(self, 
        patience: int, 
        min_delta: float,
        verbose: bool
    ):
        """
        Args:
            patience (int): Number of epochs to wait without improvement before stopping
            min_delta (float): Minimum improvement required to reset patience
            verbose (bool): Whether to log progress

        Returns:
            None
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
        """
        Checks if validation Dice improved and decides whether to stop.

        Args:
            val_dice (float): Current validation Dice score
            model (torch.nn.Module): Model to save if improved

        Returns:
            None
        """

        # Check if Dice improved
        if val_dice > self.best_dice - self.min_delta:

            # Log improvement
            if self.verbose:
                self.logger.info(
                    f'Dice score improved from {self.best_dice:.4f} to {val_dice:.4f}. Saving model...'
                )

            # Update best score
            self.best_dice = val_dice

            # Reset patience counter
            self.counter = 0
            
            # Save best model
            torch.save(model.state_dict(), "checkpoints/best_model.pth")

        else:
            # increase counter if there is no improvement
            self.counter += 1

            if self.verbose:
                self.logger.info(
                    f'Early stopping counter: {self.counter}/{self.patience}'
                )

            # Trigger early stopping
            if self.counter >= self.patience:
                self.early_stop = True

                if self.verbose:
                    self.logger.info('Early stopping triggered.')

# ----------------------------
# Trainer
# ----------------------------
class Trainer:
    """
    Main training loop manager.

    This class:
    - Trains model and evaluates it each epoch
    - Tracks and logs metrics
    """

    def __init__(self, 
        model, 
        device, 
        train_loader,
        val_loader,
        max_epochs, 
        patience, 
        batch_size, 
        base_lr, 
        min_lr, 
        bce_loss_weight,
        augmentation_scheduler,
        wandb_logger
    ):
        """
        Args:
            model (nn.Module): Segmentation model
            device (torch.device): CPU or GPU
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            max_epochs (int): Number of training epochs
            patience (int): Early stopping patience
            batch_size (int): Batch size
            base_lr (float): Initial learning rate
            min_lr (float): Minimum learning rate for scheduler
            bce_loss_weight (float): Weight for BCE in combined loss
            augmentation_scheduler: Controls augmentation intensity
            wandb_logger: Logger for experiment tracking

        Returns:
            None
        """

        # Move model to device
        self.model = model.to(device)
        self.device = device

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training config
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.bce_loss_weight = bce_loss_weight

        # Loss function
        self.criterion = BCEDiceLoss(bce_loss_weight=self.bce_loss_weight)

        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=base_lr)

        # Learning scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_epochs, eta_min=self.min_lr)

        # Early stopping
        self.early_stopping = EarlyStopping(patience=self.patience, min_delta=0.0, verbose=True)

        # Augmentation scheduler function
        self.augmentation_scheduler = augmentation_scheduler
        
        # Metrics tracker
        self.metrics = MetricsAccumulator(self.device)

        # Visualization tool
        self.vis = VisSegmentation(self.val_loader, self.device)

        # Logs
        self.logger = logging.getLogger(__name__)
        self.log_model_info()

        # Wandb logger
        self.wandb_logger = wandb_logger

    def log_model_info(self):
        """
        Logs model architecture and parameter counts.
        """

        # Print model structure
        self.logger.info(self.model)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info(f"Total params: {total_params:,}")
        self.logger.info(f"Trainable params: {trainable_params:,}")

    def train_one_epoch(self):
        """
        Runs one full training epoch.

        Returns:
            Metrics: Aggregated training metrics
        """

        # Set model to training mode
        self.model.train()

        # Reset metrics
        self.metrics.reset()

        for imgs, masks in tqdm(self.train_loader, total=len(self.train_loader), desc="Training"):
            # Move data to device
            imgs, masks = imgs.to(self.device, non_blocking=True), masks.to(self.device, non_blocking=True)

            # Forward pass
            logits = self.model(imgs)

            # ------------------
            # Loss computation
            # ------------------
            bcedice_loss_train = self.criterion(logits, masks)
            dice_loss_train = dice_loss(logits, masks)
            iou_loss_train = iou_loss(logits, masks)

            # ------------------
            # Backward pass
            # ------------------
            self.optimizer.zero_grad()
            bcedice_loss_train.backward()
            self.optimizer.step()

            # ------------------
            # Predictions
            # ------------------
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            preds = preds.squeeze(1)
            masks = masks.squeeze(1)

            # Update training metrics
            self.metrics.update(
                bcedice_loss_train, 
                dice_loss_train, 
                iou_loss_train, 
                preds, 
                masks)

        return self.metrics.compute()

    def evaluate(self):
        """
        Runs validation loop.

        Returns:
            Metrics: Aggregated validation metrics
        """

        # Set model to evaluation mode
        self.model.eval()

        # Reset evaluation metrics
        self.metrics.reset()

        with torch.no_grad():
            for imgs, masks in tqdm(self.val_loader, total=len(self.val_loader), desc="Evaluating"):
                # Move data to device
                imgs, masks = imgs.to(self.device, non_blocking=True), masks.to(self.device, non_blocking=True)
   
                # Forward pass
                logits = self.model(imgs)

                # ------------------
                # Loss computation
                # ------------------
                bcedice_loss_val = self.criterion(logits, masks)
                dice_loss_val = dice_loss(logits, masks)
                iou_loss_val = iou_loss(logits, masks)

                # ------------------
                # Predictions
                # ------------------
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()

                preds = preds.squeeze(1)
                masks = masks.squeeze(1)

                # Update validation metrics
                self.metrics.update(
                    bcedice_loss_val, 
                    dice_loss_val, 
                    iou_loss_val, 
                    preds, 
                    masks)

        return self.metrics.compute()

    def log_metrics(self, train_metrics, eval_metrics):
        """
        Logs metrics in table format.

        Args:
            train_metrics (Metrics): Training metrics
            eval_metrics (Metrics): Validation metrics
        """

        table = [
            ["BCE + Dice Loss", train_metrics.bcedice_loss, eval_metrics.bcedice_loss],
            ["Dice Loss", train_metrics.dice_loss, eval_metrics.dice_loss],
            ["IoU Loss", train_metrics.iou_loss, eval_metrics.iou_loss],
            ["Dice Metric", train_metrics.dice_metric, eval_metrics.dice_metric],
            ["IoU Metric", train_metrics.iou_metric, eval_metrics.iou_metric],
        ]

        self.logger.info("\n" + tabulate(
            table,
            headers=["Metric", "Train", "Val"],
            floatfmt=".4f",
            tablefmt="fancy_grid"  
        ))

    def __call__(self):
        """
        Runs full training process.

        Returns:
            None
        """

        for epoch in range(1, self.max_epochs+1):

            # Logging
            self.logger.info(f"_____________________________________________________________________________________")
            self.logger.info(f"Epoch: {epoch+1} / {self.max_epochs}")
            self.logger.info(f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Update augmentation intensity
            self.augmentation_scheduler.set_epoch(epoch)

            # Get training and validation metrics
            train_metrics = self.train_one_epoch()
            val_metrics = self.evaluate()

            # Store metrics 
            self.metrics.store(train_metrics, mode="train")
            self.metrics.store(val_metrics, mode="val")

            # Log metrics
            self.log_metrics(train_metrics, val_metrics)
            
            # Segmentation visualization
            if (epoch) % 1 == 0:
                vis_fig = self.vis(model=self.model, epoch=epoch, num_samples=8)
                self.wandb_logger.log_fig(vis_fig, epoch)

            # Scheduler step
            self.scheduler.step()
            
            # Early stopping check
            self.early_stopping(val_metrics.dice_metric, self.model)
            if self.early_stopping.early_stop:
                break

        # Save best model to W&B
        self.wandb_logger.log_artifact("checkpoints/best_model.pth", "best-model")

        # Log full history
        self.wandb_logger.log_metrics(self.metrics.history_train.bcedice_loss, self.metrics.history_val.bcedice_loss, name="BCE + Dice Loss") 
        self.wandb_logger.log_metrics(self.metrics.history_train.dice_loss, self.metrics.history_val.dice_loss, name="Dice Loss") 
        self.wandb_logger.log_metrics(self.metrics.history_train.iou_loss, self.metrics.history_val.iou_loss, name="IoU Loss")
        self.wandb_logger.log_metrics(self.metrics.history_train.dice_metric, self.metrics.history_val.dice_metric, name="Dice Metric")
        self.wandb_logger.log_metrics(self.metrics.history_train.iou_metric, self.metrics.history_val.iou_metric, name="IoU Metric")

        # Finish experiment
        self.wandb_logger.finish()




            

            

           



        

           






















    



            
    
    
    