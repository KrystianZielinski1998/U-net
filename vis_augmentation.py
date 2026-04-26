import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


class VisAugmentation:
    def __init__(self, images_path, masks_path, clahe_preprocessor=None, augmenter=None):
        self.images_path = sorted(list(Path(images_path).glob("*.png")))
        self.masks_path = sorted(list(Path(masks_path).glob("*.png")))
        self.clahe_preprocessor = clahe_preprocessor
        self.augmenter = augmenter

    def load_image(self, path):
        return np.array(Image.open(path).convert("L"))

    def load_mask(self, path):
        mask = np.array(Image.open(path).convert("L"))
        return (mask > 127).astype(np.float32)

    def __call__(self, num_samples=4, save_path="aug_vis.png", intensity:float=0.0):

        indices = np.random.choice(len(self.images_path), num_samples, replace=False)
        fig, axes = plt.subplots(5, num_samples, figsize=(5 * num_samples, 10))

        for i, idx in enumerate(indices):

            # Load image and mask
            orig_img = self.load_image(self.images_path[idx])
            orig_mask = self.load_mask(self.masks_path[idx])

            if self.clahe_preprocessor:
                clahe_img = self.clahe_preprocessor(orig_img)

            # Apply online augmentation 
            if self.augmenter:
                aug_img, aug_mask= self.augmenter(image=clahe_img, mask=orig_mask, intensity=intensity)

            # Plot images 
            axes[0, i].imshow(orig_img, cmap="gray")
            axes[0, i].set_title("Original")
            axes[0, i].axis("off")

            axes[1, i].imshow(clahe_img, cmap="gray")
            axes[1, i].set_title("CLAHE Enhanced")
            axes[1, i].axis("off")

            axes[2, i].imshow(aug_img, cmap="gray")
            axes[2, i].set_title("Augmented")
            axes[2, i].axis("off")

            axes[3, i].imshow(orig_mask, cmap="gray")
            axes[3, i].set_title("Mask")
            axes[3, i].axis("off")

            axes[4, i].imshow(aug_mask, cmap="gray")
            axes[4, i].set_title("Aug Mask")
            axes[4, i].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
 








    



            
    
    
    