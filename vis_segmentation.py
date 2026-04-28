import os
import torch
import numpy as np
import matplotlib.pyplot as plt


class VisSegmentation:
    """
    Utility class for segmentation model evaluation.

    This class:
    - Iterates sequentially through validation dataset
    - Runs model inference on selected samples
    - Visualizes:
        1. Original image (with index)
        2. Ground truth mask vs predicted mask contours
        3. TP / FP / FN color-coded segmentation map
    - Saves visualization plot as an image each epoch

    """
    
    def __init__(self, val_loader, device, save_dir="vis"):
        """
        Initializes visualization pipeline.

        Args:
            val_loader (DataLoader): Validation data loader
            device (torch.device): Device for inference (CPU/GPU)
            save_dir (str): Directory where visualizations will be saved

        Returns:
            None
        """

        # Extract dataset from dataloader
        self.val_dataset = val_loader.dataset   

        # Device used for model inference
        self.device = device

        # Directory to save visualization images
        self.save_dir = save_dir

        # Create directory if it does not exist
        os.makedirs(save_dir, exist_ok=True)

        # Pointer for sequential sampling 
        self.val_index_ptr = 0

    def _get_sequential_samples(self, num_samples):
        """
        Retrieves sequential samples from dataset.

        Args:
            num_samples (int): Number of samples to retrieve

        Returns:
            list: List of tuples ((img, mask), idx)
        """

        samples = []
        
        # Total dataset size
        dataset_size = len(self.val_dataset)

        # Collect samples sequentially
        for _ in range(num_samples):
            
            # Wrap around using modulo
            idx = self.val_index_ptr % dataset_size
            
            # Get sample (image, mask)
            sample = self.val_dataset[idx]

            # Store sample and its index
            samples.append((sample, idx))
            
            # Move pointer forward
            self.val_index_ptr += 1

        return samples
        

    def __call__(self, model, epoch: int, num_samples: int=6):
        """
        Generates visualization for a given model and epoch.

        Args:
            model (torch.nn.Module): Trained segmentation model
            epoch (int): Current training epoch (used in filename)
            num_samples (int): Number of samples to visualize

        Returns:
            matplotlib.figure.Figure: Generated figure
        """

        # Move model to device (CPU/GPU)
        self.model = model.to(self.device)

        # Set the model to evaluation mode
        self.model.eval()

        # Get sequential samples
        samples = self._get_sequential_samples(num_samples)

        # Create subplot grid: 3 rows (image, contours, TP/FP/FN)
        fig, axes = plt.subplots(3, num_samples, figsize=(5 * num_samples, 12))

        # Iterate over samples
        for i, ((img, mask), idx) in enumerate(samples):
            
            # Prepare image, ground truth mask and prediction
            img, true_mask, pred_mask = self._prepare(img, mask)

            # Row 1: Image + IDX
            axes[0, i].imshow(img, cmap="gray")
            axes[0, i].set_title(f"Original {idx}")
            axes[0, i].axis("off")

            # Row 2: Contours overlay
            axes[1, i].imshow(img, cmap="gray")

            # Ground Truth contour (green)
            axes[1, i].contour(
                true_mask,
                levels=[0.5],
                colors="green",
                linewidths=2
            )

            # Prediction contour (red)
            axes[1, i].contour(
                pred_mask,
                levels=[0.5],
                colors="red",
                linewidths=2
            )

            axes[1, i].set_title("GT = green | Pred = red")
            axes[1, i].axis("off")

            # Row 3: TP / FP / FN map
            colored_mask = self._create_colored_mask(true_mask, pred_mask)

            axes[2, i].imshow(colored_mask)
            axes[2, i].set_title(f"TP=green/FP=red/FN=white")
            axes[2, i].axis("off")

        plt.tight_layout()

        save_path = os.path.join(self.save_dir, f"epoch_{epoch:03d}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _prepare(self, img, mask):
        """
        Runs model inference and prepares numpy arrays.

        Args:
            img (torch.Tensor): Input image tensor
            mask (torch.Tensor): Ground truth mask tensor

        Returns:
            tuple:
                np.ndarray: Image (H, W)
                np.ndarray: Ground truth mask (H, W)
                np.ndarray: Predicted binary mask (H, W)
        """

        # Disable gradient computation for inference
        with torch.no_grad():

            # Move image to device and add batch dimension
            img = img.to(self.device).unsqueeze(0)

            # Run model prediction
            pred = self.model(img)

        # Convert image to numpy (remove batch/channel dims)
        img = img.squeeze().cpu().numpy()

        # Convert ground truth mask to numpy
        true_mask = mask.cpu().numpy()
        
        # Convert prediction to binary mask using threshold 0.5
        pred_mask = (pred.squeeze().cpu().numpy() > 0.5).astype(float)

        return img, true_mask, pred_mask

    def _create_colored_mask(self, true_mask, pred_mask):
        """
        Creates RGB visualization of TP, FP, FN regions.

        Args:
            true_mask (np.ndarray): Ground truth binary mask
            pred_mask (np.ndarray): Predicted binary mask

        Returns:
            np.ndarray: RGB image where:
                - TP (true positive)  -> green
                - FP (false positive) -> red
                - FN (false negative) -> white
        """
        
        # True Positives: correctly predicted pixel as tumor
        tp = (true_mask == 1) & (pred_mask == 1)

        # False Positives: predicted pixel as tumor but actually background 
        fp = (true_mask == 0) & (pred_mask == 1)
        
         # False Negatives: predicted pixel as background but actually tumor
        fn = (true_mask == 1) & (pred_mask == 0)

        # Initialize empty RGB image
        colored = np.zeros((*true_mask.shape, 3))

        # Assign colors
        colored[tp] = [0, 1, 0]     # green
        colored[fp] = [1, 0, 0]     # red
        colored[fn] = [1, 1, 1]     # white

        return colored






    



            
    
    
    