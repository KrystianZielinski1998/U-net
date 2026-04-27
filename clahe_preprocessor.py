import cv2
import numpy as np
import torch


class CLAHEPreprocessor:
    def __init__(self, clahe_clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=tile_grid_size
        )

    def __call__(self, img):
        """
        Supports:
        - torch.Tensor (1, H, W)
        - numpy.ndarray (H, W)

        Returns:
        - same type as input
        """

        is_tensor = isinstance(img, torch.Tensor)

        # --- convert to numpy ---
        if is_tensor:
            img_np = img.squeeze(0).cpu().numpy()
        else:
            img_np = img

        # --- ensure uint8 ---
        if img_np.dtype != np.uint8:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        # --- apply CLAHE ---
        img_np = self.clahe.apply(img_np)

        # --- convert back if needed ---
        if is_tensor:
            return torch.from_numpy(img_np).unsqueeze(0).float()
        else:
            return img_np

    


    



            
    
    
    