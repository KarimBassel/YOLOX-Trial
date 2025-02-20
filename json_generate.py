import json
import os

# Path to your gt.txt file and output directory for the COCO format JSON files
gt_file = 'D:\Ain Shams University\Graduation Project\Siemens\YOLOX-Dataset Trial2/gt.txt'  # Modify this to the location of your gt.txt
train_json_file = 'train.json'  # The output COCO JSON file for training
val_json_file = 'val.json'  # The output COCO JSON file for validation
image_dir = 'D:\Ain Shams University\Graduation Project\Siemens\YOLOX-Dataset\TrainIJCNN2013\TrainIJCNN2013'  # Directory containing the .ppm images

# Initialize the COCO format data
train_data = {
    "images": [],
    "annotations": [],
    "categories": []
}

val_data = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Categories (you can add your category names here)
categories = {
    0: "category_0",  # Replace these with actual category names
    1: "category_1",
    2: "category_2",
    3: "category_3",
    4: "category_4",
    5: "category_5",
    6: "category_6",
    7: "category_7",
    8: "category_8",
    9: "category_9",
    10: "category_10",
    11: "category_11",
    12: "category_12",
    13: "category_13",
    14: "category_14",
    15: "category_15",
    16: "category_16",
    17: "category_17",
    18: "category_18",
    19: "category_19",
    20: "category_20",
    21: "category_21",
    22: "category_22",
    23: "category_23",
    24: "category_24",
    25: "category_25",
    26: "category_26",
    27: "category_27",
    28: "category_28",
    29: "category_29",
    30: "category_30",
    31: "category_31",
    32: "category_32",
    33: "category_33",
    34: "category_34",
    35: "category_35",
    36: "category_36",
    37: "category_37",
    38: "category_38",
    39: "category_39",
    40: "category_40",
    41: "category_41",
    42: "category_42"
}

# Add categories info to COCO data
for category_id, category_name in categories.items():
    train_data["categories"].append({
        "id": category_id,
        "name": category_name,
        "supercategory": category_name
    })
    val_data["categories"].append({
        "id": category_id,
        "name": category_name,
        "supercategory": category_name
    })

# Initialize image ID and annotation ID counters
image_id = 0
annotation_id = 0

# Open the gt.txt file and start processing the bounding box data
with open(gt_file, 'r') as file:
    for line in file:
        parts = line.strip().split(';')
        filename = parts[0]  # Image filename (e.g., 00556.ppm)
        x1, y1, x2, y2, class_id = map(int, parts[1:])

        # Get the image path and check if it's already in the coco_data
        image_path = os.path.join(image_dir, filename)
        image_entry = {
            "id": image_id,
            "file_name": filename,
            "width": 1360,  # Replace with actual image width
            "height": 800  # Replace with actual image height
        }

        # Determine if the image should go into the train or validation set
        if image_id <= 539:
            coco_data = train_data
        else:
            coco_data = val_data

        if not any(image['file_name'] == filename for image in coco_data["images"]):
            # Add the image to the selected COCO data (train or val)
            coco_data["images"].append(image_entry)
            image_id += 1

        # Add annotation (bounding box info) to COCO data
        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": image_id - 1,  # The image id is one less than the current image id
            "category_id": class_id,
            "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO format [x, y, width, height]
            "area": (x2 - x1) * (y2 - y1),
            "iscrowd": 0
        })
        annotation_id += 1

# Save the final COCO format JSON for both training and validation sets
with open(train_json_file, 'w') as train_json:
    json.dump(train_data, train_json, indent=4)

with open(val_json_file, 'w') as val_json:
    json.dump(val_data, val_json, indent=4)

print(f"COCO annotations saved to {train_json_file} and {val_json_file}")
