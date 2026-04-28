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

                # augmentation schedule
                "aug_start_epoch": args.aug_start_epoch,
                "aug_end_epoch": args.aug_end_epoch,

                # preprocessing settings
                "use_clahe": args.use_clahe,
                "clahe_clip_limit": args.clahe_clip_limit,
            }
        )

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

    def log_metrics(self, y1, y2, name):
        """
        Logs training and validation metrics across epochs.

        Args:
            y1 (list or array): Training metric values per epoch
            y2 (list or array): Validation metric values per epoch
            name (str): Metric name (e.g. "loss", "dice")

        Returns:
            None
        """

        # Define epoch as global step
        wandb.define_metric("epoch")

        # Link train metric to epoch
        wandb.define_metric(f"train/{name}", step_metric="epoch")

        # Link validation metric to epoch
        wandb.define_metric(f"val/{name}", step_metric="epoch")

        # Create epoch index list
        epochs = list(range(1, len(y1) + 1))

        # Log metrics for each epoch
        for epoch, train, val in zip(epochs, y1, y2):

            wandb.log({
                "epoch": epoch,
                f"train/{name}": train,
                f"val/{name}": val
            })

    def finish(self):
        """
        Finalizes the W&B run.

        Returns:
            None
        """
        wandb.finish()



    
  
    
    
    