import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class VisSegmentation:
    def __init__(self, val_loader, device, save_dir="vis"):
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.start_idx = 0

        os.makedirs(self.save_dir, exist_ok=True)

    # -------------------------
    # sequential sampling (wrap-safe)
    # -------------------------
    def _get_sequential_samples(self, num_samples):
        dataset = self.val_loader.dataset
        n = len(dataset)

        indices = [(self.start_idx + i) % n for i in range(num_samples)]
        self.start_idx = (self.start_idx + num_samples) % n

        return [dataset[i] for i in indices]

    # -------------------------
    # ALWAYS safe visualization
    # -------------------------
    def to_display(self, img):
        """
        Converts ANY input (0-255, 0-1, z-score, tensor) → [0,1]
        """

        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        img = img.squeeze().astype(np.float32)

        # always normalize for display
        img = img - img.min()
        img = img / (img.max() + 1e-8)

        return img

    # -------------------------
    # TP / FP / FN visualization
    # -------------------------
    def _create_colored_mask(self, true_mask, pred_mask):
        h, w = true_mask.shape
        colored = np.zeros((h, w, 3), dtype=np.float32)

        tp = (true_mask == 1) & (pred_mask == 1)
        fp = (true_mask == 0) & (pred_mask == 1)
        fn = (true_mask == 1) & (pred_mask == 0)

        colored[tp] = [0.0, 1.0, 0.0]   # Green
        colored[fp] = [1.0, 0.0, 0.0]   # Red
        colored[fn] = [1.0, 1.0, 1.0]   # White

        return colored

    # -------------------------
    # model inference + prep
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
        img = self.to_display(img)

        return img, mask, pred

    # -------------------------
    # main visualization
    # -------------------------
    def __call__(self, model, epoch, num_samples=6):

        self.model = model.to(self.device)
        self.model.eval()

        samples = self._get_sequential_samples(num_samples)

        fig, axes = plt.subplots(3, num_samples, figsize=(5 * num_samples, 12))

        if num_samples == 1:
            axes = axes.reshape(3, 1)

        for i, (img, mask) in enumerate(samples):

            img, true_mask, pred_mask = self._prepare(img, mask)

            # -------------------------
            # Row 1: Image
            # -------------------------
            axes[0, i].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[0, i].set_title("Image")
            axes[0, i].axis("off")

            # -------------------------
            # Row 2: Overlay GT + Pred
            # -------------------------
            axes[1, i].imshow(img, cmap="gray", vmin=0, vmax=1)

            axes[1, i].imshow(true_mask, alpha=0.4, cmap="gray")
            axes[1, i].imshow(pred_mask, alpha=0.4, cmap="Reds")

            axes[1, i].set_title("GT (white) vs Pred (Red)")
            axes[1, i].axis("off")

            # -------------------------
            # Row 3: TP / FP / FN map
            # -------------------------
            colored_mask = self._create_colored_mask(true_mask, pred_mask)

            axes[2, i].imshow(colored_mask)
            axes[2, i].set_title("TP=Green | FP=Red | FN=White")
            axes[2, i].axis("off")

        plt.tight_layout()

        save_path = os.path.join(self.save_dir, f"epoch_{epoch:03d}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")






    



            
    
    
    