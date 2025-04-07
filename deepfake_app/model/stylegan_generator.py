# model/stylegan_generator.py

import os
import numpy as np
from PIL import Image
from tqdm import tqdm # type: ignore

def generate_fake_images(output_dir='data/fake/', num_images=10, image_size=256):
    """
    Generates placeholder synthetic fake images using random noise.
    Simulates StyleGAN3 output without requiring heavy computation.

    Args:
        output_dir (str): Directory where generated images are saved.
        num_images (int): Number of fake images to generate.
        image_size (int): Width and height of generated square images.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Generating {num_images} synthetic fake images in '{output_dir}'...")

    for i in tqdm(range(num_images), desc="Generating Images"):
        # Create a synthetic RGB image from noise
        image_array = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        image.save(os.path.join(output_dir, f"fake_{i:04d}.jpg"))

    print("[DONE] Fake image generation complete.")
