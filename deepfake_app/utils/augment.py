# utils/augment.py

import albumentations as A
import imgaug.augmenters as iaa  # type: ignore
import numpy as np
from PIL import Image

def albumentations_augment(image_np):
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=25, p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.HueSaturationValue(p=0.3),
        A.Resize(256, 256)
    ])
    augmented = transform(image=image_np)
    return augmented['image']

def imgaug_augment(image_np):
    seq = iaa.Sequential([
        iaa.AdditiveGaussianNoise(scale=(10, 20)),
        iaa.GaussianBlur(sigma=(0.0, 1.0)),
        iaa.Affine(rotate=(-20, 20), scale=(0.8, 1.2)),
        iaa.Flipud(0.5),
        iaa.Fliplr(0.5),
        iaa.Resize((256, 256))
    ])
    return seq(image=image_np)

def apply_augmentation(image_path, output_path, augmenter='albumentations'):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    if augmenter == 'albumentations':
        augmented_np = albumentations_augment(image_np)
    else:
        augmented_np = imgaug_augment(image_np)

    augmented_img = Image.fromarray(augmented_np)
    augmented_img.save(output_path)

    return output_path
