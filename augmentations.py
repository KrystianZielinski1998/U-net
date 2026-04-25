import albumentations as A
import cv2


class Augmenter:
    def __call__(self, image, mask, intensity: float):

        
        rotate_min, rotate_max = 5, 15
        translate_min, translate_max = 0.05, 0.15
        scale_min, scale_max = 0.05, 0.15
        shear_min, shear_max = 2, 6

        prob_min, prob_max = 0.5, 0.8

        rotate = rotate_min + intensity * (rotate_max - rotate_min)
        translate = translate_min + intensity * (translate_max - translate_min)
        scale = scale_min + intensity * (scale_max - scale_min)
        shear = shear_min + intensity * (shear_max - shear_min)
        prob = prob_min + intensity * (prob_max - prob_min)

        transform = A.Compose([
            A.HorizontalFlip(p=0.5),

            A.OneOf([
                A.Compose([
                    A.Rotate(limit=rotate, p=1.0),
                    A.Affine(
                        translate_percent={
                            "x": (-translate, translate),
                            "y": (-translate, translate),
                        },
                        scale=(1 - scale, 1 + scale),
                        p=1.0
                    ),
                ]),

                A.Affine(
                    shear=(-shear, shear),
                    p=1.0
                ),
            ], p=prob),
        ])

        out = transform(image=image, mask=mask)
        return out["image"], out["mask"]


class AugmentationScheduler:
    def __init__(self, start_epoch: int=10, end_epoch: int=90):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.current_epoch = 0

        self._intensity = 0.0

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        self._intensity = self._compute_intensity()

    def _compute_intensity(self) -> float:

        if self.current_epoch >= self.end_epoch:
            return 1.0

        intensity = (self.current_epoch - self.start_epoch) / (
            self.end_epoch - self.start_epoch
        )

        return intensity

    @property
    def intensity(self) -> float:
        return self._intensity


    


    



            
    
    
    