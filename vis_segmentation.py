import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class VisSegmentation:
    def __init__(self, val_dataset, device, save_dir):
        self.val_dataset = val_dataset
        self.device = device
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        self.val_index_ptr = 0

    def _get_sequential_samples(self, num_samples):
        samples = []
        dataset_size = len(self.val_dataset)

        for _ in range(num_samples):
            idx = self.val_index_ptr % dataset_size
            sample = self.val_dataset[idx]  # (img, mask)

            samples.append((sample, idx))
            self.val_index_ptr += 1

        return samples

    def __call__(self, model, epoch, num_samples=6):
        self.model = model.to(self.device)
        self.model.eval()

        samples = self._get_sequential_samples(num_samples)

        fig, axes = plt.subplots(3, num_samples, figsize=(5 * num_samples, 12))

        if num_samples == 1:
            axes = axes.reshape(3, 1)

        for i, ((img, mask), idx) in enumerate(samples):

            img, true_mask, pred_mask = self._prepare(img, mask)

            # -------------------------
            # Row 1: Image + IDX
            # -------------------------
            axes[0, i].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[0, i].set_title(f"Image idx: {idx}")
            axes[0, i].axis("off")

            # -------------------------
            # Row 2: Overlay GT + Pred
            # -------------------------
            axes[1, i].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[1, i].imshow(true_mask, alpha=0.4, cmap="gray")
            axes[1, i].imshow(pred_mask, alpha=0.4, cmap="Reds")
            axes[1, i].set_title(f"GT vs Pred (idx: {idx})")
            axes[1, i].axis("off")

            # -------------------------
            # Row 3: TP / FP / FN map
            # -------------------------
            colored_mask = self._create_colored_mask(true_mask, pred_mask)

            axes[2, i].imshow(colored_mask)
            axes[2, i].set_title(f"TP/FP/FN (idx: {idx})")
            axes[2, i].axis("off")

        plt.tight_layout()

        save_path = os.path.join(self.save_dir, f"epoch_{epoch:03d}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _prepare(self, img, mask):
        """Prepare image, GT mask and prediction."""
        with torch.no_grad():
            img = img.to(self.device).unsqueeze(0)
            pred = self.model(img)

        img = img.squeeze().cpu().numpy()
        true_mask = mask.cpu().numpy()
        pred_mask = (pred.squeeze().cpu().numpy() > 0.5).astype(float)

        return img, true_mask, pred_mask

    def _create_colored_mask(self, true_mask, pred_mask):
        """Create TP/FP/FN RGB mask."""
        import numpy as np

        tp = (true_mask == 1) & (pred_mask == 1)
        fp = (true_mask == 0) & (pred_mask == 1)
        fn = (true_mask == 1) & (pred_mask == 0)

        colored = np.zeros((*true_mask.shape, 3))

        colored[tp] = [0, 1, 0]     # green
        colored[fp] = [1, 0, 0]     # red
        colored[fn] = [1, 1, 1]     # white

        return colored






    



            
    
    
    