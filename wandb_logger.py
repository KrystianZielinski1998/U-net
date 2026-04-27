import wandb 

class WandbLogger:
    def __init__(self, args):
        wandb.init(
            project="Brain Tumor Segmentation",
            group="group 1",
            name=args.run_name,
            config = {
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

                # preprocessing
                "use_clahe": args.use_clahe,
                "clahe_clip_limit": args.clahe_clip_limit,
            }
        )

    def log_fig(self, fig, epoch):
        """ Log figure. """

        wandb.log({"fig": wandb.Image(fig)}, step=epoch)

    def log_artifact(self, model_path, artifact_name):
        """ Log model artifact. """

        artifact = wandb.Artifact(
            name=artifact_name,
            type="model"
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

    def log_metrics(self, y1, y2, name):
  
      wandb.define_metric("epoch")
      wandb.define_metric(f"train/{name}", step_metric="epoch")
      wandb.define_metric(f"val/{name}", step_metric="epoch")
      
      epochs = list(range(1, len(y1) + 1))
      
      for epoch, train, val in zip(epochs, y1, y2):
       
          wandb.log({
              "epoch": epoch,         
              f"train/{name}": train,       
              f"val/{name}": val
          })

    def finish(self):
        wandb.finish()



    
  
    
    
    