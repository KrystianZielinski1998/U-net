import wandb 

class WandbLogger:
    def __init__(self):
        pass

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

    def log_line_plot(self, y1, y2, name="line_plot", x_axis_name="x_axis", y_axis_name="y_axis"):
        """ Log line plots with shared x axis and custom axis names. """

        assert len(y1) == len(y2), 

        x = list(range(1, len(y1) + 1))

        table = wandb.Table(columns=[x_axis_name, y_axis_name, "series"])

        for i, (a, b) in enumerate(zip(y1, y2)):
            table.add_data(x[i], a, "train")
            table.add_data(x[i], b, "val")

        wandb.log({
            name: wandb.plot.line(
                table,
                x_axis_name,
                y_axis_name,
                title=name
            )
        })



    
  
    
    
    