import albumentations as A
import cv2


class AugmentationScheduler:
    def __init__(self, max_epochs: int = 120, no_aug_epochs: int = 0):
        self.max_epochs = max_epochs
        self.no_aug_epochs = no_aug_epochs
        self.epoch = 0

        # Always applied (normalization)
        self.base = [
            A.Lambda(image=lambda x, **kwargs: x / 255.0)
        ]

        # Augmentations only
        self.aug_only = [
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
        ]

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def get_transform(self, mode: str = "train"):
        if mode == "val":
            return A.Compose(self.base)

        if self.epoch < self.max_epochs - self.no_aug_epochs:
            return A.Compose(self.aug_only + self.base)
        else:
            return A.Compose(self.base)

    def __call__(self, image, mask, mode: str = "train"):
        transform = self.get_transform(mode)
        augmented = transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"]






    



            
    
    
    