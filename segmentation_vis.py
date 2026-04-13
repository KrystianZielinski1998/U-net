import torch
import matplotlib.pyplot as plt
import random
from IPython.display import display

class SegmentationVis:
    def __init__(self, val_loader, device):
        self.val_loader = val_loader
        self.device = device

    def _get_random_samples(self, num_samples):
        dataset = self.val_loader.dataset
        indices = random.sample(range(len(dataset)), num_samples)
        samples = [dataset[i] for i in indices]
        return samples

    def _prepare(self, img, mask):
        img = img.to(self.device).unsqueeze(0)  # [1,C,H,W]
        mask = mask.to(self.device)

        with torch.no_grad():
            logits = self.model(img)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).long()

        pred = pred.squeeze().cpu().numpy()
        mask = mask.squeeze().cpu().numpy()
        img = img.squeeze().cpu().permute(1, 2, 0).numpy()

        return img, mask, pred

    def __call__(self, model, num_samples=6):
        self.model = model.to(device)

        samples = self._get_random_samples(num_samples)

        fig, axes = plt.subplots(3, num_samples, figsize=(5*num_samples, 12))

        if num_samples == 1:
            axes = axes.reshape(3, 1)

        for i, (img, mask) in enumerate(samples):
            img, mask, pred = self._prepare(img, mask)

            # --- 1. Original image ---
            axes[0, i].imshow(img)
            axes[0, i].set_title("Image")
            axes[0, i].axis("off")

            # --- 2. Overlay ---
            axes[1, i].imshow(img)
            axes[1, i].imshow(mask, alpha=0.3, cmap="Greens")  # GT
            axes[1, i].imshow(pred, alpha=0.3, cmap="Reds")    # Prediction
            axes[1, i].set_title("Overlay (GT=green, Pred=red)")
            axes[1, i].axis("off")

            # --- 3. Masks ---
            combined = mask + 2 * pred  # 0=bg,1=GT,2=Pred,3=both
            axes[2, i].imshow(combined, cmap="viridis")
            axes[2, i].set_title("Masks (GT & Pred)")
            axes[2, i].axis("off")

        plt.tight_layout()
        display(plt.gcf())
        plt.close()






    



            
    
    
    