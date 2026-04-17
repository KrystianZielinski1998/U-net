from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from augmentation_vis import AugmentationVis

class SegmentationDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None, img_size=224, mode="train"):
        self.images_path = sorted(list(Path(images_path).glob("*.png")))
        self.masks_path = sorted(list(Path(masks_path).glob("*.png")))
        self.transform = transform
        self.img_size = img_size
        self.mode = mode

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img = Image.open(self.images_path[idx]).convert("RGB")
        mask = Image.open(self.masks_path[idx]).convert("L")

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        img = np.array(img, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)

        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=img, mask=mask, mode=self.mode)
            img = augmented["image"]
            mask = augmented["mask"]

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).float()

        return img, mask


class GetLoaders:
    """
    Creates training and validation loaders for segmentation datasets.
    """
    def __init__(self, 
            images_path, 
            masks_path, 
            batch_size=16, 
            val_split=0.1, 
            transform=None,
            seed=42
        ):

        self.images_path = images_path
        self.masks_path = masks_path
        self.batch_size = batch_size
        self.val_split = val_split
        self.transform = transform
        self.seed = seed

    def __call__(self):
        # List of all indices
        all_indices = list(range(len(list(Path(self.images_path).glob("*.png")))))
        rng = np.random.default_rng(self.seed)  
        rng.shuffle(all_indices)

        split = int(len(all_indices) * (1 - self.val_split))
        train_idx = all_indices[:split]
        val_idx = all_indices[split:]

        # Datasets
        train_dataset = torch.utils.data.Subset(
            SegmentationDataset(
                self.images_path,
                self.masks_path,
                transform=self.transform,
                mode="train"
            ),
            train_idx
        )

        val_dataset = torch.utils.data.Subset(
            SegmentationDataset(
                self.images_path,
                self.masks_path,
                transform=self.transform,
                mode="val"
            ),
            val_idx
        )

        # Loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        
        # Visualize 
        visualizer = AugmentationVis(
            dataset=train_dataset.dataset,   
            transform=self.transform,
            mode="train"
        )

        visualizer(
            num_samples=6,
            save_path="augmentation_preview.png"
        )

        return train_loader, val_loader







    



            
    
    
    