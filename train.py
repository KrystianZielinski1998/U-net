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
# Trainer
# ----------------------------
class Trainer:
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
        augmentation_scheduler,
        wandb_logger
    ):

        self.model = model.to(device)
        self.device = device

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.min_lr = min_lr

        self.criterion = BCEDiceLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=base_lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_epochs, eta_min=self.min_lr)
        self.early_stopping = EarlyStopping(patience=self.patience, min_delta=0.0, verbose=True)
        self.augmentation_scheduler = augmentation_scheduler
        
        self.metrics = MetricsAccumulator(self.device)

        self.vis = VisSegmentation(self.val_loader, self.device)

        self.logger = logging.getLogger(__name__)
        self.log_model_info()

        self.wandb_logger = wandb_logger

    def log_model_info(self):
        self.logger.info(self.model)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info(f"Total params: {total_params:,}")
        self.logger.info(f"Trainable params: {trainable_params:,}")

    def train_one_epoch(self):
        self.model.train()
        self.metrics.reset()

        for imgs, masks in tqdm(self.train_loader, total=len(self.train_loader), desc="Training"):
            imgs, masks = imgs.to(self.device), masks.to(self.device)

            # Forward
            logits = self.model(imgs)

            # Calculate losses
            bcedice_loss_train = self.criterion(logits, masks)
            dice_loss_train = dice_loss(logits, masks)
            iou_loss_train = iou_loss(logits, masks)

            # Backward
            self.optimizer.zero_grad()
            bcedice_loss_train.backward()
            self.optimizer.step()

            # Predictions for metrics
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()  

            preds = preds.squeeze(1)
            masks = masks.squeeze(1)

            self.metrics.update(bcedice_loss_train, dice_loss_train, iou_loss_train, preds, masks)

        return self.metrics.compute()

    def evaluate(self):
        self.model.eval()
        self.metrics.reset()

        with torch.no_grad():
            for imgs, masks in tqdm(self.val_loader, total=len(self.val_loader), desc="Evaluating"):
                imgs, masks = imgs.to(self.device), masks.to(self.device)

                # Forward
                logits = self.model(imgs)

                # Calculate losses
                bcedice_loss_val = self.criterion(logits, masks)
                dice_loss_val = dice_loss(logits, masks)
                iou_loss_val = iou_loss(logits, masks)

                # Predictions for metrics
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()

                preds = preds.squeeze(1)
                masks = masks.squeeze(1)

                self.metrics.update(bcedice_loss_val, dice_loss_val, iou_loss_val, preds, masks)

        return self.metrics.compute()

    def log_metrics(self, train_metrics, eval_metrics):
    
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
        for epoch in range(self.max_epochs):
            self.logger.info(f"_____________________________________________________________________________________")
            self.logger.info(f"Epoch: {epoch+1} / {self.max_epochs}")
            self.logger.info(f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            self.augmentation_scheduler.set_epoch(epoch)

            train_metrics = self.train_one_epoch()
            val_metrics = self.evaluate()

            self.metrics.store(train_metrics, mode="train")
            self.metrics.store(val_metrics, mode="val")

            self.log_metrics(train_metrics, val_metrics)
            
            if (epoch+1) % 1 == 0:
                vis_fig = self.vis(model=self.model, epoch=epoch+1, num_samples=8)
                self.wandb_logger.log_fig(vis_fig, epoch+1)

            self.scheduler.step()
            self.early_stopping(val_metrics.dice_metric, self.model)
            if self.early_stopping.early_stop:
                break

        self.wandb_logger.log_artifact("checkpoints/best_model.pth", "best-model")

        self.wandb_logger.log_line_plot(self.metrics.history_train.bcedice_loss, self.metrics.history_val.bcedice_loss, name="BCE + Dice Loss") 
        self.wandb_logger.log_line_plot(self.metrics.history_train.dice_loss, self.metrics.history_val.dice_loss, name="Dice Loss") 
        self.wandb_logger.log_line_plot(self.metrics.history_train.iou_loss, self.metrics.history_val.iou_loss, name="IoU Loss")
        self.wandb_logger.log_line_plot(self.metrics.history_train.dice_metric, self.metrics.history_val.dice_metric, name="Dice Metric")
        self.wandb_logger.log_line_plot(self.metrics.history_train.iou_metric, self.metrics.history_val.iou_metric, name="IoU Metric")

        self.wandb_logger.finish()

            

            

           



        

           






















    



            
    
    
    