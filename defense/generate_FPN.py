import os
import torch
import h5py
import hdf5plugin
import shutil
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from yolox.exp import get_exp
# from yolox import yolox_custom
from yolox.data import COCODataset
import logging
from torchvision import transforms

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def custom_collate_fn(batch):
    images = []
    targets = []
    infos = []
    image_ids = []

    for b in batch:
        img, target, info, image_id = b
        
        # Convert numpy image to tensor if not already tensor
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        images.append(img)
        targets.append(target)   # This should be a tensor already (shape [num_objects, 5])
        infos.append(info)
        image_ids.append(image_id)

    # Now we can stack images
    images = torch.stack(images, dim=0)

    return images, targets, infos, image_ids



@torch.no_grad()
def dump_split_features(model, data_loader, split, datasets_path, device):
    output_path = os.path.join(datasets_path, 'model_features', f'{split}.h5')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, 'w') as hf:
        for batch in tqdm(data_loader, desc=f"Dumping features for {split}"):
            inputs, targets, info, image_ids = batch  # Proper unpacking

            # Move inputs to device and adjust channel order (if in HWC format)
            inputs = inputs.to(device)
            #inputs = inputs.permute(0, 1, 2, 3).contiguous()  # (B, C, H, W)
            inputs = inputs.float()

            print(f"Input shape: {inputs.shape}")
            # Pass through model to get multi-scale FPN features (P3, P4, P5)
            features = model(inputs, return_fpn=True)

            for i in range(inputs.shape[0]):
                # Fix image_name extraction and ensure it's a string
                image_name = str(image_ids[i].item())  # Convert to string for HDF5 group name
                print(f"Processing image: {image_name}")

                # Extract and convert features
                feature_p3 = features[0][i].cpu().numpy()
                feature_p4 = features[1][i].cpu().numpy()
                feature_p5 = features[2][i].cpu().numpy()

                # Create HDF5 group and datasets
                group = hf.create_group(image_name)

                group.create_dataset('p3', data=feature_p3,
                                    **hdf5plugin.Zstd(clevel=22),
                                    dtype=np.float32, shape=feature_p3.shape, chunks=feature_p3.shape)
                group.create_dataset('p4', data=feature_p4,
                                    **hdf5plugin.Zstd(clevel=22),
                                    dtype=np.float32, shape=feature_p4.shape, chunks=feature_p4.shape)
                group.create_dataset('p5', data=feature_p5,
                                    **hdf5plugin.Zstd(clevel=22),
                                    dtype=np.float32, shape=feature_p5.shape, chunks=feature_p5.shape)



def get_model(device, model_name="best_ckpt.pth"):
    # Get the relative path of the current script to the working directory
    
    # dir_relative_path = os.path.relpath(
    #     os.path.dirname(__file__), os.getcwd())
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    YOLOX_OUTPUTS_DIR = os.path.join(ROOT_DIR, "YOLOX_outputs" , "yolox_base")
    # Get the absolute path to the model weights
    model_path = os.path.join(YOLOX_OUTPUTS_DIR, model_name)
    
    # Load experiment file and initialize model
    exp = get_exp(None, "yolox-s")  # You can change "yolox-s" if needed
    exp.num_classes = 10  # Adjust the number of classes to match your dataset
    model = exp.get_model()
    model.eval()

    # Load the trained weights
    logger.info(f"Loading checkpoint from {model_path}")
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info("Loaded checkpoint successfully")

    return model.to(device)

#will be passed to the dataloader
from PIL import Image

def preprocess_fn(img, target, input_dim=(1024, 1024)):
    # Convert NumPy array to PIL image if necessary
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(input_dim),  # Resize image to input dimensions
        transforms.ToTensor(),  # Convert image to PyTorch tensor
    ])
    img = transform(img)  # Apply transformations to the image
    return img, target

def prepare_dataloaders(dataset_path):
    annotations_path = os.path.join(dataset_path, 'annotations')
    #unify image size
    transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((1024, 1024)),  
    ])
    # Load train and validation datasets
    train_dataset = COCODataset(dataset_path,
                                os.path.join(annotations_path, 'train.json'),img_size=(1024,1024),preproc=preprocess_fn)
    val_dataset = COCODataset(dataset_path,
                              os.path.join(annotations_path, 'val.json'),img_size=(1024,1024),preproc=preprocess_fn)

    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=1, pin_memory=True,collate_fn=custom_collate_fn)
    
    val_dataloader = DataLoader(val_dataset, batch_size=1, pin_memory=True,collate_fn=custom_collate_fn)

    return train_dataloader, val_dataloader


def clean_directories(datasets_path):
    # Clean and recreate model_features directories for train and val
    for split in ['train', 'val']:
        split_path = os.path.join(datasets_path, 'model_features', split)
        try:
            shutil.rmtree(split_path)
        except FileNotFoundError:
            pass  # Ignore if not exist
        os.makedirs(split_path, exist_ok=True)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device,r"YOLOX_outputs\yolox_base\best_ckpt.pth")

    # Define dataset path
    # datasets_path = os.path.join(os.path.dirname(os.getcwd()), 'model', 'datasets')
    dataset_path = r"D:\Merged Datasets"

    # Prepare dataset directories
    #clean_directories(dataset_path)

    # Prepare DataLoaders
    train_dataloader, val_dataloader = prepare_dataloaders(dataset_path)

    # Dump features for both splits
    dump_split_features(model, train_dataloader, "train", dataset_path, device)
    dump_split_features(model, val_dataloader, "val", dataset_path, device)
