import albumentations as A
import cv2


class AugmentationScheduler:
    """
    Augmentation scheduler with train/val mode support.
    """

    def __init__(self, max_epochs: int, no_aug_epochs: int = 0):
        self.max_epochs = max_epochs
        self.no_aug_epochs = no_aug_epochs
        self.epoch = 0

        # Augmentation pipeline
        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5),

            A.OneOf([
                A.Compose([
                    A.Rotate(limit=5, p=1),
                    A.Affine(
                        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                        scale=(0.95, 1.05),
                        p=1
                    )
                ]),
                A.Affine(shear=(-5, 5), p=1)
            ], p=0.5),

            A.OneOf([
                A.CLAHE(
                    clip_limit=(1.0, 2.0),
                    tile_grid_size=(8, 8),
                    p=1.0
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=1.0
                )
            ], p=0.5),

            A.Normalize()
        ])

        # No augmentation 
        self.no_aug = A.Compose([
            A.Normalize()
        ])

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def get_transform(self, mode: str = "train"):
        """
        Args:
            mode (str): "train" or "val"
        """

        # Validation / test → always no augmentation
        match mode: 
            case "val":
                return self.no_aug

            case "train":
                # Training scheduled augmentation
                if self.epoch < self.max_epochs - self.no_aug_epochs:
                    return self.aug
                else:
                    return self.no_aug
        

    def __call__(self, image, mask, mode: str = "train"):
        transform = self.get_transform(mode)
        augmented = transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"]





    



            
    
    
    