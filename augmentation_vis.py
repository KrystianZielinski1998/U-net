import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random


import numpy as np
import matplotlib.pyplot as plt
import torch


class AugmentationVis:
    """
    Reads images and masks directly from paths and visualizes it with its augmented version.

    Rows:
    1 → original 
    2 → augmented 
    3 → original mask
    4 → augmented mask
    """

    def __init__(self, image_paths, mask_paths, transform, device="cpu"):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.device = device

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # -------------------------
    # load image (keep in 0-255 range)
    # -------------------------
    def load_image(self, path):
        """Load image from path, returns numpy array in 0-255 range (uint8)"""
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img  # Keep as uint8, 0-255 range

    def load_mask(self, path):
        """Load mask from path, returns binary numpy array (0 or 1)"""
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)  # Binary mask: 0 or 1
        return mask

    # -------------------------
    # denormalize (for augmented images that were normalized)
    # -------------------------
    def denormalize(self, img):
        """
        img: CHW normalized tensor or numpy array (values ~ N(0,1))
        Returns: HWC numpy array in [0,1] range for visualization
        """
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        # Convert CHW to HWC if needed
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        img = img * self.std + self.mean
        return np.clip(img, 0, 1)

    # -------------------------
    # main
    # -------------------------
    def __call__(self, num_samples=4, save_path="aug_vis.png"):
        
        # Select random samples
        indices = np.random.choice(len(self.image_paths), num_samples, replace=False)
        
        fig, axes = plt.subplots(4, num_samples, figsize=(4 * num_samples, 10))

        for i, idx in enumerate(indices):
            
            # -------------------------
            # LOAD ORIGINAL (0-255 range)
            # -------------------------
            img_path = self.image_paths[idx]
            mask_path = self.mask_paths[idx]
            
            # Load original image (uint8, 0-255) and mask (0 or 1)
            orig_img = self.load_image(img_path)
            orig_mask = self.load_mask(mask_path)
            
            # Original image for display (just convert to float [0,1] for matplotlib)
            orig_img_display = orig_img.astype(np.float32) / 255.0
            
            # -------------------------
            # APPLY AUGMENTATION (includes normalization)
            # -------------------------
            if self.transform:
                # Transform expects image in 0-255 range (uint8) or 0-1 float?
                # Albumentations with Normalize() typically expects uint8 or float in [0,1]
                # Since we have uint8, convert to float32 [0,1] first
                
                aug_img, aug_mask = self.transform(
                    image=img_float,
                    mask=orig_mask,
                    mode="train"
                )
                
                # aug_img is now normalized (mean=0, std=1 approximately)
                # Denormalize for visualization
                aug_img_viz = self.denormalize(aug_img)
                aug_mask_viz = aug_mask
            else:
                aug_img_viz = orig_img_display
                aug_mask_viz = orig_mask
            
            # -------------------------
            # masks cleanup (ensure binary)
            # -------------------------
            orig_mask_viz = (orig_mask > 0.5).astype(np.float32)
            aug_mask_viz = (aug_mask_viz > 0.5).astype(np.float32)
            
            # -------------------------
            # row 1: Original image (0-255 range, displayed as [0,1])
            # -------------------------
            axes[0, i].imshow(orig_img_display)
            axes[0, i].set_title("Original")
            axes[0, i].axis("off")
            
            # -------------------------
            # row 2: Augmented image (denormalized)
            # -------------------------
            axes[1, i].imshow(aug_img_viz)
            axes[1, i].set_title("Augmented")
            axes[1, i].axis("off")
            
            # -------------------------
            # row 3: Original mask
            # -------------------------
            axes[2, i].imshow(orig_mask_viz, cmap="gray")
            axes[2, i].set_title("Mask")
            axes[2, i].axis("off")
            
            # -------------------------
            # row 4: Augmented mask
            # -------------------------
            axes[3, i].imshow(aug_mask_viz, cmap="gray")
            axes[3, i].set_title("Aug Mask")
            axes[3, i].axis("off")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
 








    



            
    
    
    