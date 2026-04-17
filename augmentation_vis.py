import matplotlib.pyplot as plt
import numpy as np
import torch


from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import os


class AugmentationVis:
    """
    Visualizes augmentation effects on a plot and saves it.

    Row 1: original images
    Row 2: augmented images
    """

    def __init__(self, dataset, transform, mode="train"):
        self.dataset = dataset
        self.transform = transform
        self.mode = mode

    def _to_numpy_img(self, img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        if img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        return img

    def _normalize(self, x):
        x = x - x.min()
        return x / (x.max() + 1e-8)

    def __call__(self, num_samples=4, save_path="aug_vis.png", figsize=(12, 6)):
        """
        Visualize and save augmentation comparison grid.
        """

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, num_samples, figsize=figsize)

        for i in range(num_samples):
            img, mask = self.dataset[i]

            orig_img = self._to_numpy_img(img)

            if self.transform:
                aug = self.transform(
                    image=orig_img,
                    mask=mask,
                    mode=self.mode
                )
                aug_img = aug["image"]
            else:
                aug_img = orig_img

            aug_img = self._to_numpy_img(aug_img)

            orig_img = self._normalize(orig_img)
            aug_img = self._normalize(aug_img)

            axes[0, i].imshow(orig_img)
            axes[0, i].set_title("Original")
            axes[0, i].axis("off")

            axes[1, i].imshow(aug_img)
            axes[1, i].set_title("Augmented")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        






    



            
    
    
    