from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import numpy as np


class DatasetReader(Dataset):
    def __init__(self, images_path, masks_path, img_size=224):
        self.images = sorted(Path(images_path).glob("*.png"))
        self.masks = sorted(Path(masks_path).glob("*.png"))
        self.img_size = img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("L")
        mask = Image.open(self.masks[idx]).convert("L")

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        img = np.array(img, dtype=np.float32)
        mask = (np.array(mask) > 127).astype(np.float32)

        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask)

        return img, mask


class DatasetProcessor(Dataset):
    def __init__(self, base_dataset, normalizer=None, clahe_preprocessor=None, augmenter=None, augmentation_scheduler=None):
        self.dataset = base_dataset
        self.clahe_preprocessor = clahe_preprocessor
        self.normalizer = normalizer
        self.augmenter = augmenter
        self.augmentation_scheduler = augmentation_scheduler

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        
        # CLAHE contrast enhancement
        if self.clahe_preprocessor is not None:
            img = self.clahe_preprocessor(img)

        # Normalization 
        if self.normalizer is not None:
            img = self.normalizer(img)

        # Online augmentation for training set
        if self.augmenter is not None and self.augmentation_scheduler is not None:
          
            if self.augmentation_scheduler.current_epoch < self.augmentation_scheduler.start_epoch:
                pass  
            else:
                intensity = self.augmentation_scheduler.intensity

                if intensity > 0:
                    img_np = img.squeeze(0).numpy()
                    mask_np = mask.numpy()

                    img_np, mask_np = self.augmenter(img_np, mask_np, intensity)

                    img = torch.from_numpy(img_np).unsqueeze(0)
                    mask = torch.from_numpy(mask_np)

        return img, mask


class DataModule:
    def __init__(
        self,
        images_path,
        masks_path,
        img_size,
        batch_size=16,
        val_split=0.1,
        clahe_preprocessor=None,
        normalizer=None,
        augmenter=None,
        augmentation_scheduler=None,
        num_workers=2,
        seed=42
    ):
        self.images_path = images_path
        self.masks_path = masks_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.val_split = val_split

        self.clahe_preprocessor = clahe_preprocessor
        self.normalizer = normalizer
        self.augmenter = augmenter
        self.augmentation_scheduler = augmentation_scheduler
        self.num_workers = num_workers
        self.seed = seed

    # Split dataset
    def _split(self, dataset):
        n = len(dataset)
        indices = np.random.default_rng(self.seed).permutation(n)

        split = int(n * (1 - self.val_split))
        train_idx, val_idx = indices[:split], indices[split:]

        return (
            Subset(dataset, train_idx),
            Subset(dataset, val_idx),
        )

    # Build dataset
    def setup(self):
        # Get base dataset 
        base_dataset = DatasetReader(
            self.images_path,
            self.masks_path,
            self.img_size
        )

        # Split base dataset into train and val
        train_base, val_base = self._split(base_dataset)

        # Calculate avg and std of train set images for normalization parameters
        self.normalizer.fit(train_base, clahe_preprocessor)

        # Wrap datasets
        train_dataset = DatasetProcessor(
            train_base,
            clahe_preprocessor=self.clahe_preprocessor
            normalizer=self.normalizer,
            augmenter=self.augmenter,
            augmentation_scheduler=self.augmentation_scheduler   
        )

        val_dataset = DatasetProcessor(
            val_base,
            clahe_preprocessor=self.clahe_preprocessor
            normalizer=self.normalizer,
            augmenter=None,
            augmentation_scheduler=None             
        )

        # Get train loader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        
        # Get val loader
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        return self

    # Get loaders func
    def get_loaders(self):
        return self.train_loader, self.val_loader







    



            
    
    
    