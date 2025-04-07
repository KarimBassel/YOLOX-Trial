import json
import os
import random
import shutil
from tqdm import tqdm

# Only these 10 classes will be included
CATEGORIES_DICT = {"pl100": 0, "pl120": 1, "pl20": 2, "pl30": 3, "pl40": 4, 
                   "pl15": 5, "pl50": 6, "pl60": 7, "pl70": 8, "pl80": 9}

def parse_tsinghua(annotations_path, image_dir, output_train_json, output_val_json, train_ratio=0.8):
    """
    Parses the Tsinghua dataset JSON and converts it to COCO format, 
    keeping only the specified 10 categories and ignoring missing images.
    """

    with open(annotations_path, "r") as f:
        data = json.load(f)

    # Create COCO format category list
    coco_categories = [{"id": cat_id, "name": cat_name, "supercategory": "traffic sign"} 
                       for cat_name, cat_id in CATEGORIES_DICT.items()]
    
    images, annotations = [], []
    annotation_id = 0
    valid_image_paths = []

    for img_id, img_data in tqdm(data['imgs'].items(), desc="Parsing dataset"):
        img_path = img_data['path']

        # Only process images in '' or 'other/'
        if not (img_path.startswith("") or img_path.startswith("other/")):
            continue
        
        full_img_path = os.path.join(image_dir, img_path.replace("\\", "/"))  # Normalize path
        
        if not os.path.exists(full_img_path):
            print(f"Warning: Skipping missing image {full_img_path}")
            continue

        img_annotations = []
        valid = False

        for obj in img_data['objects']:
            category_name = obj['category']

            # Only keep annotations from the specified 10 categories
            if category_name in CATEGORIES_DICT:
                valid = True
                bbox = obj['bbox']
                img_annotations.append({
                    "id": annotation_id,
                    "image_id": img_data['id'],
                    "category_id": CATEGORIES_DICT[category_name],
                    "bbox": [bbox['xmin'], bbox['ymin'], bbox['xmax'] - bbox['xmin'], bbox['ymax'] - bbox['ymin']],
                    "area": (bbox['xmax'] - bbox['xmin']) * (bbox['ymax'] - bbox['ymin']),
                    "iscrowd": 0,
                    "segmentation": [[]]
                })
                annotation_id += 1

        if valid:
            images.append({
                "id": img_data['id'],
                "file_name": img_path,
                "width": 2048,
                "height": 2048,
            })
            annotations.extend(img_annotations)
            valid_image_paths.append(full_img_path)

    # Shuffle and split dataset
    combined = list(zip(images, annotations, valid_image_paths))
    random.shuffle(combined)
    images, annotations, valid_image_paths = zip(*combined) if combined else ([], [], [])

    split_idx = int(train_ratio * len(images))
    train_data = {"images": images[:split_idx], "annotations": annotations[:split_idx], "categories": coco_categories}
    val_data = {"images": images[split_idx:], "annotations": annotations[split_idx:], "categories": coco_categories}

    # Save JSON files
    with open(output_train_json, "w") as f:
        json.dump(train_data, f, indent=4)
    
    with open(output_val_json, "w") as f:
        json.dump(val_data, f, indent=4)

    # Move images to train2017 and val2017 folders
    os.makedirs(os.path.join(image_dir, "train2017"), exist_ok=True)
    os.makedirs(os.path.join(image_dir, "val2017"), exist_ok=True)

    for i, img_path in tqdm(enumerate(valid_image_paths), desc="Copying images"):
        dest_folder = "train2017" if i < split_idx else "val2017"
        dest_path = os.path.join(image_dir, dest_folder, os.path.basename(img_path))
        shutil.copy(img_path, dest_path)

    print(f"COCO datasets saved: {output_train_json}, {output_val_json}")
    print("Images successfully copied into train2017/ and val2017/")

# Example usage
annotations_path = r"D:\Tsinnghua Dataset\data\annotations.json"
image_dir = r"D:\Tsinnghua Dataset\data"
output_train_json = "train.json"
output_val_json = "val.json"
parse_tsinghua(annotations_path, image_dir, output_train_json, output_val_json)
print("Done!")
