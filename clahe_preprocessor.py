import cv2
import numpy as np
import torch


class CLAHEPreprocessor:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE) preprocessing module.

    Applies local contrast enhancement to grayscale images using OpenCV.
    Supports both torch.Tensor and numpy.ndarray inputs.

    Expected input shape: [1, H, W] (single-channel image).
    """

    def __init__(self, clahe_clip_limit=2.0, tile_grid_size=(4, 4)):
        """
        Initializes CLAHE operator.

        Args:
            clahe_clip_limit (float): Threshold for contrast limiting.
            tile_grid_size (tuple): Size of grid for histogram equalization.
        """
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=tile_grid_size
        )

    def __call__(self, img: torch.Tensor):
        """
        Applies CLAHE to an input image.

        Args:
            img (torch.Tensor or np.ndarray): Grayscale image of shape [1, H, W]

        Returns:
            torch.Tensor or np.ndarray: Contrast-enhanced image with same shape
        """

        is_tensor = isinstance(img, torch.Tensor)

        # Convert to numpy if needed
        if is_tensor:
            img_np = img.detach().cpu().numpy()
        else:
            img_np = np.asarray(img)

        # Remove channel dimension: [1, H, W] -> [H, W]
        img_np = np.squeeze(img_np)

        # CLAHE expects uint8 input
        if img_np.dtype != np.uint8:
            img_np = img_np.astype(np.uint8)

        # Apply CLAHE
        img_np = self.clahe.apply(img_np)

        # Convert back to original format
        if is_tensor:
            return torch.from_numpy(img_np).unsqueeze(0)
        else:
            return img_np


    



            
    
    
    