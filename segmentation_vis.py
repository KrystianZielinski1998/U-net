import os
import torch
import numpy as np
import matplotlib.pyplot as plt


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SegmentationVis:
    def __init__(self, val_loader, device, save_dir="vis"):
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.start_idx = 0

        os.makedirs(self.save_dir, exist_ok=True)

    # -------------------------
    # denormalization
    # -------------------------
    def denormalize(self, img):
        img = img.astype(np.float32)

        # CHW → HWC
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        img = img * IMAGENET_STD + IMAGENET_MEAN
        return np.clip(img, 0, 1)

    # -------------------------
    # sequential sampling
    # -------------------------
    def _get_sequential_samples(self, num_samples):
        dataset = self.val_loader.dataset
        n = len(dataset)

        indices = []
        for i in range(num_samples):
            idx = (self.start_idx + i) % n
            indices.append(idx)

        self.start_idx = (self.start_idx + num_samples) % n

        return [dataset[i] for i in indices]

    # -------------------------
    # inference + prepare
    # -------------------------
    def _prepare(self, img, mask):
        img_in = img.to(self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(img_in)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).long()

        pred = pred.squeeze().cpu().numpy()
        mask = mask.squeeze().cpu().numpy()

        img = img_in.squeeze().cpu().numpy()
        img = self.denormalize(img)

        return img, mask, pred

    # -------------------------
    # main
    # -------------------------
    def __call__(self, model, epoch, num_samples=6):

        self.model = model.to(self.device)
        self.model.eval()

        samples = self._get_sequential_samples(num_samples)

        fig, axes = plt.subplots(3, num_samples, figsize=(5 * num_samples, 12))

        if num_samples == 1:
            axes = axes.reshape(3, 1)

        for i, (img, mask) in enumerate(samples):

            img, mask, pred = self._prepare(img, mask)

            # -------------------------
            # row 1: image
            # -------------------------
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Image {self.start_idx - num_samples + i}")
            axes[0, i].axis("off")

            # -------------------------
            # row 2: overlay GT + pred
            # -------------------------
            axes[1, i].imshow(img)
            axes[1, i].imshow(mask, alpha=0.3, cmap="Greens")
            axes[1, i].imshow(pred, alpha=0.3, cmap="Reds")
            axes[1, i].set_title("GT (green) vs Pred (red)")
            axes[1, i].axis("off")

            # -------------------------
            # row 3: mask comparison
            # -------------------------
            combined = mask + 2 * pred
            axes[2, i].imshow(combined, cmap="viridis")
            axes[2, i].set_title("Mask comparison")
            axes[2, i].axis("off")

        plt.tight_layout()

        # -------------------------
        # SAVE ONLY (no show)
        # -------------------------
        save_path = os.path.join(self.save_dir, f"epoch_{epoch:04d}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")






    



            
    
    
    