import cv2
import numpy as np


class CLAHEPreprocessor:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img: uint8 or float in range [0, 255]
        
        Returns:
            : uint8 image after CLAHE
        """

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        return self.clahe.apply(img)


    


    



            
    
    
    