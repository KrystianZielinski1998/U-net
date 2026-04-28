import cv2
import numpy as np
import torch


class CLAHEPreprocessor:
    """
    CLAHE contrast enhancement preprocessing module.

    Assumes input is always a torch.Tensor of shape [1, H, W].
    """

    def __init__(self, clahe_clip_limit=2.0, tile_grid_size=(8, 8)):
        # OpenCV CLAHE operator
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=tile_grid_size
        )

    def __call__(self, img: torch.Tensor):
        """
        Applies CLAHE to a torch image.

        Args:
            img (torch.Tensor): [1, H, W] grayscale image

        Returns:
            torch.Tensor: CLAHE-enhanced image [1, H, W]
        """

        # remove channel dim: [H, W]
        img_np = img.squeeze(0).cpu().numpy()

        # apply CLAHE
        img_np = self.clahe.apply(img_np)

        # back to torch tensor
        return torch.from_numpy(img_np).unsqueeze(0).float()

    


    



            
    
    
    