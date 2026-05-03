import wandb


import wandb


class WandbLogger:
    """
    Class for logging training experiments to Weights & Biases.

    This class handles:
    - Initialization of a W&B run with configuration tracking
    - Logging figures (e.g. segmentation visualizations)
    - Logging model artifacts (saved checkpoints)
    - Logging training/validation metrics over epochs
    """

    def __init__(self, args):
        """
        Initializes a Weights & Biases run.

        Args:
            args (argparse.Namespace): Training configuration 

        Returns:
            None
        """

        # Initialize W&B experiment
        wandb.init(
            project="Brain Tumor Segmentation",   # project name in W&B
            group="group 1",                      # grouping runs together
            name=args.run_name,                   # unique run name

            # Save all hyperparameters for reproducibility
            config={
                # training hyperparameters
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "base_lr": args.base_lr,
                "min_lr": args.min_lr,
                "img_size": args.img_size,
                "bce_loss_weight": args.bce_loss_weight,
                "val_split": args.val_split,

                # augmentation schedule
                "aug_start_epoch": args.aug_start_epoch,
                "aug_end_epoch": args.aug_end_epoch,

                # preprocessing settings
                "use_clahe": args.use_clahe,
                "clahe_clip_limit": args.clahe_clip_limit,
            }
        )

        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    def log_fig(self, fig, epoch: int):
        """
        Logs a matplotlib figure to Weights & Biases.

        Args:
            fig (matplotlib.figure.Figure): Figure to log
            epoch (int): Current training epoch

        Returns:
            None
        """
        # Convert matplotlib figure to W&B image and log it
        wandb.log({"fig": wandb.Image(fig)}, step=epoch)

    def log_artifact(self, model_path: str, artifact_name: str):
        """
        Logs a model checkpoint as a W&B artifact.

        Args:
            model_path (str): Path to saved model file
            artifact_name (str): Name of the artifact in W&B

        Returns:
            None
        """
        
        # Create artifact container for model versioning
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model"
        )

        # Attach model file to artifact
        artifact.add_file(model_path)

        # Log artifact to W&B
        wandb.log_artifact(artifact)

    def log_metrics(self, train_history, val_history):
        """
        Logs training and validation metrics to Weights & Biases.
        One step = one epoch.
        Additionally stores metrics from the best epoch (based on val/Dice Metric).
        """

        num_epochs = len(train_history.bcedice_loss)

        best_dice = -1
        best_epoch = -1
        best_metrics = {}

        for i in range(num_epochs):
            epoch = i + 1

            log_dict = {
                "epoch": epoch,

                "train/BCE + Dice Loss": train_history.bcedice_loss[i],
                "val/BCE + Dice Loss": val_history.bcedice_loss[i],

                "train/Dice Loss": train_history.dice_loss[i],
                "val/Dice Loss": val_history.dice_loss[i],

                "train/IoU Loss": train_history.iou_loss[i],
                "val/IoU Loss": val_history.iou_loss[i],

                "train/Dice Metric": train_history.dice_metric[i],
                "val/Dice Metric": val_history.dice_metric[i],

                "train/IoU Metric": train_history.iou_metric[i],
                "val/IoU Metric": val_history.iou_metric[i],
            }

            wandb.log(log_dict)

            
            if val_history.dice_metric[i] > best_dice:
                best_dice = val_history.dice_metric[i]
                best_epoch = epoch

              
                best_metrics = {
                    "best/epoch": epoch,

                    "best/train_BCE+Dice": train_history.bcedice_loss[i],
                    "best/val_BCE+Dice": val_history.bcedice_loss[i],

                    "best/train_Dice_Loss": train_history.dice_loss[i],
                    "best/val_Dice_Loss": val_history.dice_loss[i],

                    "best/train_IoU_Loss": train_history.iou_loss[i],
                    "best/val_IoU_Loss": val_history.iou_loss[i],

                    "best/train_Dice": train_history.dice_metric[i],
                    "best/val_Dice": val_history.dice_metric[i],

                    "best/train_IoU": train_history.iou_metric[i],
                    "best/val_IoU": val_history.iou_metric[i],
                }


        wandb.run.summary.update(best_metrics)

    def finish(self):
        """
        Finalizes the W&B run.

        Returns:
            None
        """
        wandb.finish()



    
  
    
    
    