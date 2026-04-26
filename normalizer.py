import albumentations as A
import cv2
from torch.utils.data import Dataset, DataLoader, Subset
import torch

class ZScoreNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None
        self.eps = 1e-8

    def fit(self, dataset, clahe_preprocessor, batch_size=16):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        total_sum = 0.0
        total_sum_sq = 0.0
        total_pixels = 0

        for imgs, _ in loader:
            # Apply CLAHE per image if provided
            if clahe_preprocessor is not None:
                processed = []
                for img in imgs:
                    img = clahe_preprocessor(img)  # keep same tensor shape
                    processed.append(img)
                imgs = torch.stack(processed)
                
            imgs = imgs / 255.0

            total_sum += imgs.sum()
            total_sum_sq += (imgs ** 2).sum()
            total_pixels += imgs.numel()

        self.mean = total_sum / total_pixels
        self.std = (total_sum_sq / total_pixels - self.mean ** 2) ** 0.5

        return self

    def __call__(self, img):
        img = img / 255.0
        return (img - self.mean) / (self.std + self.eps)




    



            
    
    
    