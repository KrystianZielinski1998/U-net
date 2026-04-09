from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class SegmentationDataset(Dataset):
    """
    Dataset for 2D image segmentation.
    Expects images and masks as separate folders.
    """
    def __init__(self, 
            images_path, 
            masks_path, 
            transform=None
        ):

        self.images_path = sorted(list(Path(images_path).glob("*.png")))  
        self.masks_path = sorted(list(Path(masks_path).glob("*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        # Load image and mask
        img = np.array(Image.open(self.images_path[idx]).convert("RGB"))
        mask = np.array(Image.open(self.masks_path[idx]).convert("L"))

        img = img.resize((224, 224))
        mask = mask.resize((224, 224), resample=Image.NEAREST)  

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # Convert to torch tensors
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [C,H,W] normalized to 0-1
        mask = torch.from_numpy(mask).long()  # [H,W]

        return img, mask


class GetLoaders:
    """
    Creates training and validation loaders for segmentation datasets.
    """
    def __init__(self, 
            images_path, 
            masks_path, 
            batch_size=64, 
            val_split=0.1, 
            transform=None
        ):

        self.images_path = images_path
        self.masks_path = masks_path
        self.batch_size = batch_size
        self.val_split = val_split
        self.transform = transform

    def __call__(self):
        # List of all indices
        all_indices = list(range(len(list(Path(self.images_path).glob("*.png")))))
        np.random.shuffle(all_indices)
        split = int(len(all_indices) * (1 - self.val_split))
        train_idx = all_indices[:split]
        val_idx = all_indices[split:]

        # Datasets
        train_dataset = torch.utils.data.Subset(
            SegmentationDataset(self.images_path, self.masks_path, transform=self.transform),
            train_idx
        )
        val_dataset = torch.utils.data.Subset(
            SegmentationDataset(self.images_path, self.masks_path, transform=self.transform),
            val_idx
        )

        # Loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        return train_loader, val_loader







    



            
    
    
    