import albumentations as A
import cv2
from torch.utils.data import Dataset, DataLoader, Subset
import torch

class ZScoreNormalizer:
    """
    Z-score normalization module for grayscale images.

    This class:
    - Computes dataset-wide mean and standard deviation
    - Applies normalization: (img - mean) / std

    Expected input:
    - torch.Tensor images in range [0, 255], shape [1, H, W]
    """

    def __init__(self):
        """
        Initializes normalization parameters.

        Returns:
            None
        """

        # Mean of dataset (computed in fit)
        self.mean = None

        # Standard deviation of dataset (computed in fit)
        self.std = None

        # Small constant to avoid division by zero
        self.eps = 1e-8

    def fit(self, dataset, clahe_preprocessor, batch_size=16):
        """
        Computes mean and standard deviation from dataset.

        Args:
            dataset (Dataset): Dataset returning (image, mask)
            clahe_preprocessor (callable or None): Optional CLAHE preprocessing
            batch_size (int): Batch size for statistics computation

        Returns:
            self: Fitted normalizer
        """

        # Create DataLoader for efficient batch processing
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Accumulators for mean/std computation
        total_sum = 0.0
        total_sum_sq = 0.0
        total_pixels = 0

        for imgs, _ in loader:

            # CLAHE preprocessing if enabled
            if clahe_preprocessor is not None:
                processed = []

                # Apply CLAHE per image 
                for img in imgs:
                    img = clahe_preprocessor(img)
                    processed.append(img)

                # Stack back into batch
                imgs = torch.stack(processed)

            # Normalize to [0, 1] range
            imgs = imgs / 255.0

            # Accumulate statistics
            total_sum += imgs.sum()              
            total_sum_sq += (imgs ** 2).sum()    
            total_pixels += imgs.numel()         


        # Compute mean 
        self.mean = total_sum / total_pixels

        # Compute std 
        self.std = (total_sum_sq / total_pixels - self.mean ** 2) ** 0.5

        return self

    def __call__(self, img):
        """
        Applies Z-score normalization to a single image.

        Args:
            img (torch.Tensor): Image [1, H, W] in range [0, 255]

        Returns:
            torch.Tensor: Normalized image
        """

        # Scale to [0, 1]
        img = img / 255.0

        # Apply normalization
        return (img - self.mean) / (self.std + self.eps)



    



            
    
    
    