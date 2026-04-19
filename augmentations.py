import albumentations as A
import cv2


class Augmenter:
    def __init__(self, scheduler=None):
        self.scheduler = scheduler

        self.augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),

            A.OneOf([
                A.Compose([
                    A.Rotate(limit=10, p=1),
                    A.Affine(
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                        scale=(0.9, 1.1),
                        p=1
                    )
                ]),
                A.Affine(shear=(-5, 5), p=1)
            ], p=0.5),
        ])

    def __call__(self, image, mask):
        if self.scheduler is not None and not self.scheduler.is_active():
            return image, mask

        augmented = self.augmentation(image=image, mask=mask)
        return augmented["image"], augmented["mask"]

class AugmentationScheduler:
    def __init__(self, max_epochs=120, no_aug_epochs=0):
        self.max_epochs = max_epochs
        self.no_aug_epochs = no_aug_epochs
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def is_active(self):
        return self.epoch < self.max_epochs - self.no_aug_epochs


    


    



            
    
    
    