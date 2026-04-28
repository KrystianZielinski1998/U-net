import albumentations as A
import cv2


class Augmenter:
    """
    This class defines online data augmentation pipeline with intensity-controlled randomness.

    This class applies spatial augmentations (rotation, translation, scaling, shearing)
    using a continuous intensity parameter in the range [0, 1].
    """

    def __call__(self, image, mask, intensity: float):
        """
        Applies augmentation to image and mask.

        Args:
            image (np.ndarray): Input image
            mask (np.ndarray): Corresponding segmentation mask
            intensity (float): Augmentation strength in range [0, 1]

        Returns:
            tuple:
                np.ndarray: Augmented image
                np.ndarray: Augmented mask
        """

        # Define augmentation ranges
        rotate_min, rotate_max = 5, 15            
        translate_min, translate_max = 0.05, 0.15 
        scale_min, scale_max = 0.05, 0.15        
        shear_min, shear_max = 2, 6               

        # probability range for applying strong augmentations
        prob_min, prob_max = 0.5, 0.8

        # Interpolate values using intensity
        rotate = rotate_min + intensity * (rotate_max - rotate_min)
        translate = translate_min + intensity * (translate_max - translate_min)
        scale = scale_min + intensity * (scale_max - scale_min)
        shear = shear_min + intensity * (shear_max - shear_min)
        prob = prob_min + intensity * (prob_max - prob_min)

        # Build augmentation pipeline
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),  

            A.OneOf([
                # Path 1: rotation + translation & zoom
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

                # Path 2: shear-only transform
                A.Affine(
                    shear=(-shear, shear),
                    p=1.0
                ),
            ], p=prob),
        ])

        # Apply transform to image and mask simultaneously
        out = transform(image=image, mask=mask)

        # Return augmented data
        return out["image"], out["mask"]


class AugmentationScheduler:
    """
    Controls the progression of augmentation intensity during training.

    The intensity increases linearly from `start_epoch` to `end_epoch`,
    enabling curriculum-style augmentation scheduling.
    """

    def __init__(self, start_epoch: int = 10, end_epoch: int = 90):
        """
        Initializes augmentation schedule.

        Args:
            start_epoch (int): Epoch when augmentation starts increasing
            end_epoch (int): Epoch when augmentation reaches full intensity

        Returns:
            None
        """

        # Epoch boundaries
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

        # Current epoch tracker
        self.current_epoch = 0

        # Current computed intensity
        self._intensity = 0.0

    def set_epoch(self, epoch: int):
        """
        Updates current epoch and recomputes augmentation intensity.

        Args:
            epoch (int): Current training epoch

        Returns:
            None
        """
        # Update epoch state
        self.current_epoch = epoch

        # Recompute intensity based on schedule
        self._intensity = self._compute_intensity()

    def _compute_intensity(self) -> float:
        """
        Computes augmentation intensity based on current epoch.

        Returns:
            float: Normalized intensity in range [0, 1]
        """

        # If past end epoch use highest intensity
        if self.current_epoch >= self.end_epoch:
            return 1.0

        # Linear interpolation between start and end epoch
        intensity = (self.current_epoch - self.start_epoch) / (
            self.end_epoch - self.start_epoch
        )

        return intensity

    @property
    def intensity(self) -> float:
        """
        Returns current augmentation intensity.

        Returns:
            float: Current intensity value [0, 1]
        """
        return self._intensity


    


    



            
    
    
    