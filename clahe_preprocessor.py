import cv2
import numpy as np
import torch


class CLAHEPreprocessor:
    """
    CLAHE contrast enhancement preprocessing module.

    Assumes input is always a torch.Tensor of shape [1, H, W].
    """

    def __init__(self, clahe_clip_limit=2.0, tile_grid_size=(4, 4)):
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

        is_tensor = isinstance(img, torch.Tensor)

        if is_tensor:
            img_np = img.detach().cpu().numpy()
        else:
            img_np = np.asarray(img)

        img_np = np.squeeze(img_np)

        if img_np.dtype != np.uint8:
            img_np = img_np.astype(np.uint8)

        img_np = self.clahe.apply(img_np)

        if is_tensor:
            return torch.from_numpy(img_np).unsqueeze(0)
        else:
            return img_np
        


    



            
    
    
    