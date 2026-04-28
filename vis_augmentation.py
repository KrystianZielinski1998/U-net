import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


class VisAugmentation:
    """
    Utility class for visualizing dataset with image preprocessing step and online data augmentation.

    This class:
    - Loads grayscale images and corresponding segmented tumor masks
    - Optionally applies CLAHE contrast enhancement if clahe_preprocessor is used
    - Optionally applies online augmentation modifications if augmenter is used
    - Generates a visualization grid showing each processing step
    - Saves the result as an image

    The layout dynamically adapts depending on whether CLAHE and/or augmentation are used.
    """

    def __init__(self, images_path, masks_path, clahe_preprocessor=None, augmenter=None):
        """
        Initializes visualization pipeline.

        Args:
            images_path (str or Path): Path to directory with input images (.png)
            masks_path (str or Path): Path to directory with corresponding masks (.png)
            clahe_preprocessor (callable, optional): Function for CLAHE preprocessing
            augmenter (callable, optional): Function for data augmentation
        """

        # Load and sort all image paths
        self.images_path = sorted(list(Path(images_path).glob("*.png")))
        
        # Load and sort all mask paths (must align with images)
        self.masks_path = sorted(list(Path(masks_path).glob("*.png")))
        
        # Optional CLAHE preprocessing function
        self.clahe_preprocessor = clahe_preprocessor
        
        # Optional augmentation function
        self.augmenter = augmenter

    def load_image(self, path):
        """
        Loads an image and converts it to grayscale numpy array.

        Args:
            path (Path): Path to image file

        Returns:
            np.ndarray: Grayscale image as numpy array
        """

        # Open image, convert to grayscale ("L"), convert to numpy array
        return np.array(Image.open(path).convert("L"))

    def load_mask(self, path):
        """
        Loads a mask and converts it to binary format.

        Args:
            path (Path): Path to mask file

        Returns:
            np.ndarray: Binary mask (0.0 or 1.0)
        """
        # Load mask as grayscale
        mask = np.array(Image.open(path).convert("L"))
        
        # Convert to binary mask (threshold at 127)
        return (mask > 127).astype(np.float32)

    def __call__(self, num_samples: int=4, save_path: str="aug_vis.png", intensity: float = 0.0):
        """
        Generates visualization of datatset with optional CLIP contrast enhancement and online augmentation.

        Args:
            num_samples (int): Number of random samples to visualize
            save_path (str): Output file path for saved visualization
            intensity (float): Augmentation intensity parameter

        Returns:
            None: Saves visualization to file
        """

        # Randomly select indices of images to visualize
        indices = np.random.choice(len(self.images_path), num_samples, replace=False)

        # Check if CLAHE preprocessing is enabled
        use_clahe = self.clahe_preprocessor is not None

        # Check if augmentation is enabled
        use_aug = self.augmenter is not None

        # Define visualization rows dynamically
        # Each tuple: (Title, key used in data_map)
        row_defs = [("Original", "orig_img")]

        # Add CLAHE row if enabled
        if use_clahe:
            row_defs.append(("CLAHE Enhanced", "clahe_img"))

        # Add augmented image row if enabled
        if use_aug:
            row_defs.append(("Augmented", "aug_img"))

        # Always include original mask
        row_defs.append(("Mask", "orig_mask"))

        # Add augmented mask row if enabled
        if use_aug:
            row_defs.append(("Aug Mask", "aug_mask"))

        # Number of rows in visualization
        n_rows = len(row_defs)

        # Create matplotlib figure and axes grid
        fig, axes = plt.subplots(n_rows, num_samples, figsize=(5 * num_samples, 2 * n_rows))

        # Iterate over selected samples
        for i, idx in enumerate(indices):
            
            # Load original image    
            orig_img = self.load_image(self.images_path[idx])

            # Load corresponding mask
            orig_mask = self.load_mask(self.masks_path[idx])

            # Apply CLAHE if enabled, otherwise keep original img
            clahe_img = self.clahe_preprocessor(orig_img) if use_clahe else orig_img

            # Apply augmentation if enabled
            if use_aug:
                aug_img, aug_mask = self.augmenter(
                    image=clahe_img, mask=orig_mask, intensity=intensity
                )
            else:
                aug_img, aug_mask = None, None

            # Map keys to actual data for flexible plotting
            data_map = {
                "orig_img": orig_img,
                "clahe_img": clahe_img,
                "aug_img": aug_img,
                "orig_mask": orig_mask,
                "aug_mask": aug_mask,
            }

            # Plot each row dynamically
            for row_idx, (title, key) in enumerate(row_defs):

                # Display image in grayscale
                axes[row_idx, i].imshow(data_map[key], cmap="gray")

                # Set subplot title
                axes[row_idx, i].set_title(title)

                # Hide axis ticks
                axes[row_idx, i].axis("off")

        # Adjust layout to avoid overlaps
        plt.tight_layout()

        # Save figure to file
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
 








    



            
    
    
    