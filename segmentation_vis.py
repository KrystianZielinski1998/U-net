import torch
import matplotlib.pyplot as plt
import random
from IPython.display import display

import torch
import matplotlib.pyplot as plt
import numpy as np


class SegmentationVis:
    def __init__(self, val_loader, device):
        self.val_loader = val_loader
        self.device = device
        self.start_idx = 0  

    def _get_sequential_samples(self, num_samples):
        dataset = self.val_loader.dataset
        n = len(dataset)

        indices = []
        for i in range(num_samples):
            idx = (self.start_idx + i) % n
            indices.append(idx)

        self.start_idx = (self.start_idx + num_samples) % n

        return [dataset[i] for i in indices]

    def _prepare(self, img, mask):
        img = img.to(self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(img)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).long()

        pred = pred.squeeze().cpu().numpy()
        mask = mask.squeeze().cpu().numpy()
        img = img.squeeze().cpu().permute(1, 2, 0).numpy()

        return img, mask, pred

    def __call__(self, model, num_samples=6):
        self.model = model.to(self.device)

        samples = self._get_sequential_samples(num_samples)

        fig, axes = plt.subplots(3, num_samples, figsize=(5 * num_samples, 12))

        if num_samples == 1:
            axes = axes.reshape(3, 1)

        for i, (img, mask) in enumerate(samples):
            img, mask, pred = self._prepare(img, mask)

            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Image {self.start_idx - num_samples + i}")
            axes[0, i].axis("off")

            axes[1, i].imshow(img)
            axes[1, i].imshow(mask, alpha=0.3, cmap="Greens")
            axes[1, i].imshow(pred, alpha=0.3, cmap="Reds")
            axes[1, i].set_title("GT vs Pred")
            axes[1, i].axis("off")

            combined = mask + 2 * pred
            axes[2, i].imshow(combined, cmap="viridis")
            axes[2, i].set_title("Masks")
            axes[2, i].axis("off")

        plt.tight_layout()
        plt.show()
        plt.close()






    



            
    
    
    