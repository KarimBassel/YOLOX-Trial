import json
from rich.progress import track
import os

def load_txt(file_name, labels_to_keep):
    """Load and filter gt.txt data based on allowed labels."""
    with open(file_name, 'r') as file:
        data = [line.strip() for line in file if line.strip().split(";")[-1] in labels_to_keep]
    return data

def parse_gtsdb(data):
    """Parse annotations from GTSDB and save val JSON files."""
    train_data = {"images": [], "annotations": [], "categories": []}
    val_data = {"images": [], "annotations": [], "categories": []}

    # Define the 10 specific classes to keep
    categories_dic = {"7": 0, "8": 1, "0": 2, "1": 3, "2": 6, "3": 7, "4": 8, "5": 9}
    category_set = set(categories_dic.values())  # New category IDs

    for annotation in track(data, "Parsing GTSDB..."):
        s = annotation.split(';')
        img_id = int(s[0][:5])
        img_name = s[0]
        xmin, ymin, xmax, ymax = map(int, s[1:5])
        original_class_id = s[5]
        
        # Ignore classes not in categories_dic
        if original_class_id not in categories_dic:
            continue
        
        class_id = categories_dic[original_class_id]  # Remap class ID
        anno_id = len(train_data['annotations']) + len(val_data['annotations'])

        img_dict = {
            "license": 1,
            "file_name": os.path.join("..", "images", "train" if img_id <= 539 else "val", img_name),
            "height": 800,
            "width": 1360,
            "id": img_id
        }

        anno_dict = {
            "segmentation": [[]],
            "area": (xmax - xmin) * (ymax - ymin),
            "iscrowd": 0,
            "image_id": img_id,
            "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
            "category_id": class_id,
            "id": anno_id
        }

        # Assign to train or validation set
        if img_id <= 450:
            if not any(img['id'] == img_id for img in train_data['images']):
                train_data['images'].append(img_dict)
            train_data['annotations'].append(anno_dict)
        elif 451 <= img_id <= 599:
            if not any(img['id'] == img_id for img in val_data['images']):
                val_data['images'].append(img_dict)
            val_data['annotations'].append(anno_dict)

    # Add categories
    categories = [{"id": cat_id, "name": f"class_{cat_id}"} for cat_id in sorted(category_set)]
    train_data["categories"] = categories
    val_data["categories"] = categories

    # Save JSON files
    with open("train.json", "w") as train_file:
        json.dump(train_data, train_file, indent=4)

    with open("val.json", "w") as val_file:
        json.dump(val_data, val_file, indent=4)

    print("✅ train.json and val.json saved successfully.")

# Load the dataset and parse it
file_path = "D:/YOLOX Dataset/gt.txt"  # Change this to the actual path of gt.txt
labels_to_keep = set(["7", "8", "0", "1", "2", "3", "4", "5"])  # Only keep the specified 10 classes

data_list = load_txt(file_path, labels_to_keep)
parse_gtsdb(data_list)
