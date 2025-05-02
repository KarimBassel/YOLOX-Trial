import os
import json
import random
import shutil
import numpy as np
from collections import defaultdict

# Paths
GTSDB_PATH = r"D:\YOLOX Dataset (GTSDB)\images"
GTSDB_ANNOTATIONS = r"D:\YOLOX Dataset (GTSDB)\gt.txt"
TSINGHUA_PATH = r"D:\Tsinnghua Dataset\data"
TSINGHUA_ANNOTATIONS = r"D:\Tsinnghua Dataset\data\annotations.json"
OUTPUT_PATH = r"D:\Merged Datasets"

# Ensure output directories exist
os.makedirs(os.path.join(OUTPUT_PATH, "images/train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "images/val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "annotations"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "images/val_attacked"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "images/train_attacked"), exist_ok=True)

# Define category mapping
categories_dic = {
    "pl100": 0, "pl120": 1, "pl20": 2, "pl30": 3, "pl40": 4,
    "pl15": 5, "pl50": 6, "pl60": 7, "pl70": 8, "pl80": 9,
    "7": 0, "8": 1, "0": 2, "1": 3, "2": 6, "3": 7, "4": 8, "5": 9
}

categories = ["pl100", "pl120", "pl20", "pl30", "pl40", "pl15", "pl50", "pl60", "pl70", "pl80"]

COCO_TEMPLATE = {
    "images": [],
    "annotations": [],
    "categories": [{"id": i, "name": cat} for i, cat in enumerate(categories)]
}

def process_gtsdb():
    images = {}
    annotations = []
    annotation_id = 1
    relevant_images = set()

    with open(GTSDB_ANNOTATIONS, "r") as f:
        lines = f.readlines()

    img_id_offset = 100000  # Unique offset for GTSDB images

    for line in lines:
        parts = line.strip().split(";")
        filename, xmin, ymin, xmax, ymax, category = parts
        if category not in categories_dic:
            continue
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        relevant_images.add(filename)

        if filename not in images:
            img_id = len(images) + img_id_offset
            images[filename] = img_id

        annotations.append({
            "id": annotation_id,
            "image_id": images[filename],
            "category_id": categories_dic[category],
            "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
            "area": (xmax - xmin) * (ymax - ymin),
            "iscrowd": 0
        })
        annotation_id += 1

    return images, annotations, relevant_images

def process_tsinghua():
    with open(TSINGHUA_ANNOTATIONS, "r") as f:
        data = json.load(f)

    images = {}
    annotations = []
    annotation_id = 1
    relevant_images = set()

    img_id_offset = 200000  # Unique offset for Tsinghua images

    for img_id, img_data in data.get("imgs", {}).items():
        filename = os.path.basename(img_data["path"])
        valid = False

        for obj in img_data.get("objects", []):
            category = obj.get("category")
            if category in categories_dic:
                valid = True
                bbox = obj["bbox"]
                annotations.append({
                    "id": annotation_id,
                    "image_id": int(img_id) + img_id_offset,
                    "category_id": categories_dic[category],
                    "bbox": [bbox["xmin"], bbox["ymin"], bbox["xmax"] - bbox["xmin"], bbox["ymax"] - bbox["ymin"]],
                    "area": (bbox["xmax"] - bbox["xmin"]) * (bbox["ymax"] - bbox["ymin"]),
                    "iscrowd": 0
                })
                annotation_id += 1

        if valid:
            images[filename] = int(img_id) + img_id_offset
            relevant_images.add(filename)

    return images, annotations, relevant_images

gtsdb_images, gtsdb_annotations, gtsdb_relevant = process_gtsdb()
tsinghua_images, tsinghua_annotations, tsinghua_relevant = process_tsinghua()

# Combine datasets
all_relevant_images = list(gtsdb_relevant | tsinghua_relevant)
random.shuffle(all_relevant_images)
split_index = int(0.8 * len(all_relevant_images))
train_images, val_images = set(all_relevant_images[:split_index]), set(all_relevant_images[split_index:])

def get_coco_images(image_set, image_dict, dataset_name):
    height, width = (800, 1360) if dataset_name == "GTSDB" else (2048, 2048)
    return [
        {
            "id": image_dict[img],
            "file_name": img,
            "height": height,
            "width": width
        }
        for img in image_set if img in image_dict
    ]

train_coco_images = get_coco_images(train_images, gtsdb_images, "GTSDB") + get_coco_images(train_images, tsinghua_images, "Tsinghua")
val_coco_images = get_coco_images(val_images, gtsdb_images, "GTSDB") + get_coco_images(val_images, tsinghua_images, "Tsinghua")

train_annotations = [ann for ann in (gtsdb_annotations + tsinghua_annotations) if ann["image_id"] in {img["id"] for img in train_coco_images}]
val_annotations = [ann for ann in (gtsdb_annotations + tsinghua_annotations) if ann["image_id"] in {img["id"] for img in val_coco_images}]

def save_annotations(file_path, images, annotations):
    with open(file_path, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": COCO_TEMPLATE["categories"]}, f, indent=4)

# save_annotations(os.path.join(OUTPUT_PATH, "annotations/train.json"), train_coco_images, train_annotations)
# save_annotations(os.path.join(OUTPUT_PATH, "annotations/val.json"), val_coco_images, val_annotations)

def generate_attacked_images_and_annotations(image_set, image_dict, annotations, output_folder, img_id_offset=0):
    attacked_images = []
    attacked_annotations = []
    attacked_image_ids = set()

    for img_file in image_set:
        for i in range(1, 6):  # Create 5 variants of each image
            new_img_file = f"{os.path.splitext(img_file)[0]}_{i}.jpg"
            new_image_id = f"{image_dict[img_file] + img_id_offset}_{i}"  # Offset the image id
            attacked_images.append({
                "id": new_image_id,
                "file_name": new_img_file,
                "height": 2048,
                "width": 2048
            })
            attacked_image_ids.add(new_image_id)

            for ann in annotations:
                if ann["image_id"] == image_dict[img_file]:
                    attacked_annotations.append({
                        "id": len(attacked_annotations) + 1,
                        "image_id": new_image_id,
                        "category_id": ann["category_id"],
                        "bbox": ann["bbox"],
                        "area": ann["area"],
                        "iscrowd": ann["iscrowd"]
                    })

            # Copy image variants
            shutil.copy(os.path.join(GTSDB_PATH if img_file in gtsdb_images else TSINGHUA_PATH, img_file),
                        os.path.join(output_folder, new_img_file))

    return attacked_images, attacked_annotations

# Generate attacked images for both train and validation sets
train_attacked_coco_images, train_attacked_annotations = generate_attacked_images_and_annotations(
    train_images, {**gtsdb_images, **tsinghua_images}, train_annotations, os.path.join(OUTPUT_PATH, "images/train_attacked"), img_id_offset=10000)

val_attacked_coco_images, val_attacked_annotations = generate_attacked_images_and_annotations(
    val_images, {**gtsdb_images, **tsinghua_images}, val_annotations, os.path.join(OUTPUT_PATH, "images/val_attacked"), img_id_offset=10000)

# Save attacked images annotations
save_annotations(os.path.join(OUTPUT_PATH, "annotations/train_attacked.json"), train_attacked_coco_images, train_attacked_annotations)
save_annotations(os.path.join(OUTPUT_PATH, "annotations/val_attacked.json"), val_attacked_coco_images, val_attacked_annotations)

def copy_images(image_set, destination_folder, source_paths):
    os.makedirs(destination_folder, exist_ok=True)
    for img_file in image_set:
        found = False
        for src_path in source_paths:
            src_img_path = os.path.join(src_path, img_file)
            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, os.path.join(destination_folder, img_file))
                found = True
                break
        if not found:
            print(f"Warning: Image {img_file} not found in source directories!")

copy_images(train_images, os.path.join(OUTPUT_PATH, "images/train"), [GTSDB_PATH, TSINGHUA_PATH])
copy_images(val_images, os.path.join(OUTPUT_PATH, "images/val"), [GTSDB_PATH, TSINGHUA_PATH])

print("Train, validation, and attacked annotation files created successfully.")
