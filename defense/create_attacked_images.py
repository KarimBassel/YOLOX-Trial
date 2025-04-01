from attack.fgsm import FGSM
from custom_yolo import yolox_loss, yolox_target_generator
import cv2
import os
from defense.generate_FPN import get_model
import torch
from defense.hgd_trainer import COCODataset
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from defense.hgd_trainer import Preprocessor
import shutil
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def generate_attacked_samples(dataloader, split_name, eps=4):
    attack = FGSM(yolox_target_generator, yolox_loss, get_model(device))
    
    for idx, (input, targets) in enumerate(tqdm(dataloader)):
        # print(f"Processing batch {idx}: Input shape {input.shape}, Targets: {targets}")  # Debugging line

        input = input.to(device)
        # print("Generating attack for input batch")  # Debugging line
        outputs = attack.generate_attack(input, eps=eps, return_numpy=True)
        
        if outputs is None:
            print(f"Error: attack.generate_attack returned None for eps={eps}")
            continue
        
        # print(f"Generated outputs shape: {outputs.shape}")  # Debugging line

        for pic_idx, target in enumerate(targets):
            # print(f"Processing image {pic_idx} with target: {target}")  # Debugging line
            splits = str(target).split("/")
            splitpath = splits[-1]
            sp = str(splitpath).split(".")
            img_name, img_extension = sp[0], sp[-1]
            output_path = os.path.join("D:/Merged Datasets", 'attacked_images', split_name, f"{img_name}_{eps}.{img_extension}")
            
            # print(f"Saving image: {output_path}")  # Debugging line
            success = cv2.imwrite(output_path, outputs[pic_idx].transpose((1, 2, 0)))
            if not success:
                print(f"Failed to write image: {output_path}")

class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, transforms=None):
        self.coco = COCO(annotation_file)
        self.root_dir = root_dir
        self.transforms = transforms
        self.ids = list(sorted(self.coco.imgs.keys()))
        print(f"Loaded {len(self.ids)} images from annotations")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        
        # print(f"Loading image: {img_path}")  # Debugging line
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Error loading image: {img_path}")
            return None, None
        
        # print(f"Original image shape: {img.shape}")  # Debugging line
        img = Preprocessor().preprocess_model_input(img)  # Apply any other necessary preprocessing here
        
        # print(f"Preprocessed image shape: {img.shape}")  # Debugging line

        if self.transforms is not None:
            img = self.transforms(img)
        return img, img_info['file_name']

if __name__ == "__main__":
    datasets_path = "D:/Merged Datasets"
    batch_size = 1

    # Delete previous attacked images
    try:
        shutil.rmtree(os.path.join(datasets_path, 'attacked_images', 'train'))
        shutil.rmtree(os.path.join(datasets_path, 'attacked_images', 'val'))
    except Exception as e:
        print(f"Error deleting previous attacked images: {e}")

    os.makedirs(os.path.join(datasets_path, 'attacked_images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(datasets_path, 'attacked_images', 'val'), exist_ok=True)

    train_dataset = COCODataset(
        os.path.join(datasets_path, 'train'),
        os.path.join(datasets_path, 'annotations', 'train.json')
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)

    val_dataset = COCODataset(
        os.path.join(datasets_path, 'val'),
        os.path.join(datasets_path, 'annotations', 'val.json')
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

    eps_values = [0, 1, 2, 3, 4, 5]

    for eps in eps_values:
        print(f"Generating attacked samples for eps={eps}")
        generate_attacked_samples(train_dataloader, 'train', eps=eps)
        generate_attacked_samples(val_dataloader, 'val', eps=eps)
