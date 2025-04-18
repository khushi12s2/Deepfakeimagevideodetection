# run.py

import os
import requests
from model.stylegan_generator import generate_fake_images
from utils.augment import apply_augmentation
from model.train import train_model
from model.predict import predict_image

API_BASE_URL = "http://localhost:8000"

def generate_and_augment(real_dir="data/real/", fake_dir="data/fake/"):
    print("[1] Generating synthetic fake images...")
    generate_fake_images(output_dir=fake_dir, num_images=10)

    print("[2] Augmenting real images...")
    for file in os.listdir(real_dir):
        if file.endswith(".jpg") or file.endswith(".png"):
            input_path = os.path.join(real_dir, file)
            output_path = os.path.join(real_dir, f"aug_{file}")
            apply_augmentation(input_path, output_path, augmenter="albumentations")

    print("[3] Augmenting fake images...")
    for file in os.listdir(fake_dir):
        if file.endswith(".jpg") or file.endswith(".png"):
            input_path = os.path.join(fake_dir, file)
            output_path = os.path.join(fake_dir, f"aug_{file}")
            apply_augmentation(input_path, output_path, augmenter="imgaug")


def call_backend_training():
    print("[API] Triggering model training from backend API...")
    try:
        response = requests.post(f"{API_BASE_URL}/train")
        print("[API] Training Triggered:", response.json())
    except Exception as e:
        print("[ERROR] Failed to call training endpoint:", e)


def main():
    print("[PIPELINE] Running full setup...\n")

    generate_and_augment()

    print("[4] Training CNN model locally...")
    train_model()

    print("[5] Running test prediction...")
    result = predict_image("temp.jpg", threshold=0.7)
    print("Prediction Result:", result)

    print("[6] Triggering backend training API...")
    call_backend_training()


if __name__ == "__main__":
    main()
    # Uncomment the following line to run the script directly
    # main()        