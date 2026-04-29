from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import numpy as np
import random


class DatasetReader(Dataset):
    """
    PyTorch dataset for loading grayscale images and binary segmentation masks.

    This dataset:
    - Loads image/mask pairs from disk
    - Resizes them to a fixed resolution
    - Converts images to torch tensors
    - Converts masks to binary format (0/1)
    """

    def __init__(self, images_path: str, masks_path: str, img_size: int=224):
        """
        Initializes dataset.

        Args:
            images_path (str): Path to image directory
            masks_path (str): Path to mask directory
            img_size (int): Output image resolution 

        Returns:
            None
        """

        # Collect all image file paths and sort them for alignment
        self.images = sorted(Path(images_path).glob("*.png"))
        
        # Collect all mask file paths and sort them for alignment
        self.masks = sorted(Path(masks_path).glob("*.png"))
        
        # Target resize resolution
        self.img_size = img_size

    def __len__(self):
        """
        Returns dataset size.

        Returns:
            int: Number of image-mask pairs
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Loads one image-mask pair.

        Args:
            idx (int): Index of sample

        Returns:
            tuple:
                torch.Tensor: Image tensor [1, H, W]
                torch.Tensor: Mask tensor [H, W]
        """

        # Load image and mask
        img = Image.open(self.images[idx]).convert("L")   # grayscale image
        mask = Image.open(self.masks[idx]).convert("L")   # grayscale mask


        # Resize both to fixed size
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)


        # Convert to numpy arrays
        img = np.array(img, dtype=np.float32)

        # Convert mask to binary (0 or 1)
        mask = (np.array(mask) > 127).astype(np.float32)

        # Convert to torch tensors
        img = torch.from_numpy(img).unsqueeze(0)  
        mask = torch.from_numpy(mask)             

        return img, mask


class DatasetProcessor(Dataset):
    """
    Wrapper dataset that applies preprocessing, normalization and augmentation on top of a base dataset.

    This class:
    1. Loads sample from base dataset
    2. Applies Optional CLAHE contrast enhancement as preprocessing for training and validation images
    3. Applies normalization as preprocessing for training and validation images
    4. Applies option online data augmentation on training data
    """

    def __init__(self, base_dataset, normalizer=None, clahe_preprocessor=None, augmenter=None, augmentation_scheduler=None):
        """
        Initializes dataset wrapper.

        Args:
            base_dataset (Dataset): Base dataset returning (image, mask)
            normalizer (callable, optional): Image normalization function
            clahe_preprocessor (callable, optional): CLAHE preprocessing module
            augmenter (callable, optional): Augmentation function
            augmentation_scheduler (object, optional): Controls augmentation intensity over epochs

        Returns:
            None
        """

        # Base dataset (raw data source)
        self.dataset = base_dataset

        # Optional CLAHE preprocessing module
        self.clahe_preprocessor = clahe_preprocessor

        # Optional normalization function (e.g. mean/std scaling)
        self.normalizer = normalizer

        # Optional augmentation function
        self.augmenter = augmenter

        # Scheduler controlling augmentation intensity over training
        self.augmentation_scheduler = augmentation_scheduler

    def __len__(self):
        """
        Returns dataset size.

        Returns:
            int: number of samples in dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieves and processes one sample.

        Args:
            idx (int): Sample index

        Returns:
            tuple:
                torch.Tensor: processed image [1, H, W]
                torch.Tensor: mask [H, W]
        """

        # Load raw sample from base dataset
        img, mask = self.dataset[idx]

        # CLAHE preprocessing
        if self.clahe_preprocessor is not None:
            img = self.clahe_preprocessor(img)

        # Normalization step
        if self.normalizer is not None:
            img = self.normalizer(img)

        # Online data augmentation 
        if self.augmenter is not None and self.augmentation_scheduler is not None:

            # Skip if the training is not using augmentation yet
            if self.augmentation_scheduler.current_epoch < self.augmentation_scheduler.start_epoch:
                pass

            else:
                # Get current augmentation intensity
                intensity = self.augmentation_scheduler.intensity

                # Convert tensors to numpy for augmentation
                img_np = img.squeeze(0).numpy()
                mask_np = mask.numpy()

                # Apply augmentation
                img_np, mask_np = self.augmenter(img_np, mask_np, intensity)

                # Convert back to torch tensors
                img = torch.from_numpy(img_np).unsqueeze(0)
                mask = torch.from_numpy(mask_np)

        # Return processed sample
        return img, mask


class DataModule:
    """
    Organizes whole data pipeline for segmentation training.

    This class:
    - Creates dataset
    - Splits dataset into train and val sets
    - Gets preprocessed images 
    - Gets train images processed with online data augmentation
    - Creates train and val loaders used in training model
    """

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
        """
        Initializes DataModule configuration.

        Args:
            images_path (str): Path to images directory
            masks_path (str): Path to masks directory
            img_size (int): Image resize resolution (square)
            batch_size (int): Batch size for training
            val_split (float): Fraction of dataset used for validation
            clahe_preprocessor (callable, optional): CLAHE preprocessing module
            normalizer (callable, optional): Normalization module
            augmenter (callable, optional): Augmentation function
            augmentation_scheduler (object, optional): Controls augmentation intensity over epochs
            num_workers (int): Number of DataLoader workers
            seed (int): Random seed for reproducibility

        Returns:
            None
        """

        # Dataset paths
        self.images_path = images_path
        self.masks_path = masks_path

        # Image size configuration
        self.img_size = img_size

        # Training configuration
        self.batch_size = batch_size
        self.val_split = val_split

        # Preprocessing modules
        self.clahe_preprocessor = clahe_preprocessor
        self.normalizer = normalizer
        self.augmenter = augmenter
        self.augmentation_scheduler = augmentation_scheduler

        # DataLoader config
        self.num_workers = num_workers

        # Random seed for reproducibility
        self.seed = seed


    def _split(self, dataset):
        """
        Splits dataset into training and validation subsets.

        Args:
            dataset (Dataset): Full dataset

        Returns:
            tuple:
                Subset: train dataset
                Subset: validation dataset
        """

        # Dataset size
        n = len(dataset)

        # Deterministic shuffled indices
        indices = np.random.default_rng(self.seed).permutation(n)

        # Split point
        split = int(n * (1 - self.val_split))

        # Train / validation indices
        train_idx, val_idx = indices[:split], indices[split:]

        return (
            Subset(dataset, train_idx),
            Subset(dataset, val_idx),
        )


    def _worker_init_fn(self, worker_id):
        """
        Ensures deterministic behavior for each DataLoader worker.

        Args:
            worker_id (int): Worker ID

        Returns:
            None
        """

        # Unique seed per worker
        worker_seed = self.seed + worker_id

        # Seed all random generators
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)


    def setup(self):
        """
        Builds datasets and DataLoaders for training and validation.

        Steps:
        1. Create base dataset
        2. Split into train/validation
        3. Fit normalization on training set
        4. Wrap datasets with preprocessing + augmentation
        5. Create DataLoaders

        Returns:
            self: configured DataModule
        """

        # Create base dataset
        base_dataset = DatasetReader(
            self.images_path,
            self.masks_path,
            self.img_size
        )

        # Split into train and validation sets
        train_base, val_base = self._split(base_dataset)

        # Fit normalization parameters on training data only
        self.normalizer.fit(train_base, self.clahe_preprocessor)

        # Wrap training dataset 
        train_dataset = DatasetProcessor(
            train_base,
            clahe_preprocessor=self.clahe_preprocessor,
            normalizer=self.normalizer,
            augmenter=self.augmenter,
            augmentation_scheduler=self.augmentation_scheduler
        )

        # Wrap validation dataset
        val_dataset = DatasetProcessor(
            val_base,
            clahe_preprocessor=self.clahe_preprocessor,
            normalizer=self.normalizer,
            augmenter=None,
            augmentation_scheduler=None
        )

        # Deterministic DataLoader 
        g = torch.Generator()
        g.manual_seed(self.seed)

        # Training DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=self._worker_init_fn,
            generator=g
        )

        # Validation DataLoader
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return self


    def get_loaders(self):
        """
        Returns training and validation DataLoaders.

        Returns:
            tuple:
                DataLoader: training loader
                DataLoader: validation loader
        """
        return self.train_loader, self.val_loader







    



            
    
    
    