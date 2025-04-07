from glob import glob
import os
from tqdm import tqdm
import pandas as pd

# Define paths
base_path = "D:/Merged Datasets"
attacked_images_path = os.path.join(base_path, "attacked_images")
original_images_path = os.path.join(base_path, "images")

# Get attacked image file paths
attacked_train_images = glob(os.path.join(attacked_images_path, "train", "*"))
attacked_val_images = glob(os.path.join(attacked_images_path, "val", "*"))

train_dict = {"attacked_images": [], "orig_images": []}
val_dict = {"attacked_images": [], "orig_images": []}

def process_images(attacked_images, dataset_dict, split):
    for image_path in tqdm(attacked_images, desc=f"Processing {split} set"):
        image_name = os.path.basename(image_path)
        base_name, ext = os.path.splitext(image_name)
        original_id = base_name.split("_")[0]  # Extract original image ID
        original_image_name = f"{original_id}{ext}"  # Reconstruct original image name
        # print(original_image_name)
        # print(image_name)
        original_image_path = os.path.join(original_images_path, split, original_image_name)
        if os.path.exists(original_image_path):
            dataset_dict['attacked_images'].append(image_name)
            dataset_dict['orig_images'].append(original_image_name)
        else:
            print(f"Warning: Original image not found for {image_name}")

# Process train and validation images
process_images(attacked_train_images, train_dict, "train")
process_images(attacked_val_images, val_dict, "val")

# Save to CSV
pd.DataFrame(train_dict).to_csv(os.path.join(attacked_images_path, "train.csv"), index=False)
pd.DataFrame(val_dict).to_csv(os.path.join(attacked_images_path, "val.csv"), index=False)

print("CSV files generated successfully!")
