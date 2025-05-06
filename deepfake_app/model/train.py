# utils/dataset_loader.py

import kagglehub
import zipfile
import shutil
import os
from keras.models import load_model

MODEL_PATH = "saved_model/deepfake_cnn.h5"

# Load model safely
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = None
    print(f"[WARN] Model file not found at {MODEL_PATH}. Prediction endpoints will be inactive.")

def find_subfolder_containing(target_folder_name, base_folder="unzipped_data"):
    for root, dirs, _ in os.walk(base_folder):
        if target_folder_name in dirs:
            return os.path.join(root, target_folder_name)
    raise FileNotFoundError(f"Folder '{target_folder_name}' not found in {base_folder}")

def download_and_prepare():
    print("[INFO] Downloading dataset from Kaggle...")

    # Step 1: Download the dataset
    path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")
    print("[INFO] Dataset downloaded to:", path)

    # Step 2: Extract all ZIP files into unzipped_data/
    zip_files = [f for f in os.listdir(path) if f.endswith(".zip")]
    for zip_file in zip_files:
        zip_path = os.path.join(path, zip_file)
        print(f"[INFO] Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("unzipped_data")

    # Step 3: Locate and move real/fake images to data folder
    for label in ["real", "fake"]:
        try:
            src_folder = find_subfolder_containing(label)
            dst_folder = os.path.join("data", label)
            os.makedirs(dst_folder, exist_ok=True)

            for file in os.listdir(src_folder):
                src_path = os.path.join(src_folder, file)
                dst_path = os.path.join(dst_folder, file)
                shutil.move(src_path, dst_path)

            print(f"[INFO] Moved {label} images to {dst_folder}")

        except FileNotFoundError as e:
            print(f"[ERROR] {e}")

    print("[INFO] Dataset setup complete.")

# Optional: run from CLI or FastAPI backend
if __name__ == "__main__":
    download_and_prepare()
    # Ensure the model is loaded for CLI usage
    if model is None:
        print("[ERROR] Model not loaded. Please check the model path.")
    else:
        print("[INFO] Model loaded successfully.")  