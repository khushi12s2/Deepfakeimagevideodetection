# utils/dataset_loader.py

import kagglehub # type: ignore
import zipfile
import shutil
import os

def download_and_prepare():
    print("[INFO] Downloading dataset from Kaggle...")

    # Step 1: Download the dataset
    path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")
    print("[INFO] Dataset downloaded to:", path)

    # Step 2: Extract if zipped
    zip_files = [f for f in os.listdir(path) if f.endswith(".zip")]
    for zip_file in zip_files:
        zip_path = os.path.join(path, zip_file)
        print(f"[INFO] Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("unzipped_data")

    # Step 3: Move files to data/real and data/fake
    for label in ["real", "fake"]:
        src_folder = os.path.join("unzipped_data", label)
        dst_folder = os.path.join("data", label)
        os.makedirs(dst_folder, exist_ok=True)
        for file in os.listdir(src_folder):
            src_path = os.path.join(src_folder, file)
            dst_path = os.path.join(dst_folder, file)
            shutil.move(src_path, dst_path)
        print(f"[INFO] Moved {label} images to {dst_folder}")

    print("[INFO] Dataset setup complete.")

# Optional: run from CLI or FastAPI backend
if __name__ == "__main__":
    download_and_prepare()
    print("[INFO] Dataset downloaded and prepared.")
    print("[INFO] All tasks completed successfully.")   