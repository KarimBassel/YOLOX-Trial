import os
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
from yolox.exp import get_exp
from yolox.utils import postprocess
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from defense.generate_FPN import get_model
from defense.hgd_trainer import get_HGD_model

# Your 10 class names (indexed from 0)
class_names = ["pl100", "pl120", "pl20", "pl30", "pl40", "pl15", "pl50", "pl60", "pl70", "pl80"]

val_images_dir = r"D:\Merged Datasets"
annotation_path = r"D:/Merged Datasets/annotations/val_attacked.json"
weights_path = r"YOLOX_outputs\yolox_base\best_ckpt.pth"
input_size = (1024, 1024)

# Preprocess function
def preprocess(img, input_size):
    h, w = img.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    image_resized = cv2.resize(img, (nw, nh))
    image_padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    image_padded[:nh, :nw] = image_resized
    image_padded = image_padded.astype(np.float32).transpose(2, 0, 1)
    return torch.from_numpy(image_padded).unsqueeze(0), scale

def denoise_image(img, hgd_model, device):
    img_tensor = (
        torch.from_numpy(img.astype(np.float32))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
        .float()
    )
    with torch.no_grad():
        noise = hgd_model(img_tensor)
        denoised_tensor = img_tensor - noise
        denoised_tensor = torch.clamp(denoised_tensor, 0.0, 255.0)
    denoised_img = denoised_tensor.squeeze(0).cpu().byte().permute(1, 2, 0).numpy()
    return denoised_img

def convert_xyxy_to_xywh(bbox):
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]

# Run inference and save predictions
def run_inference_and_save_predictions(model, coco,defense_model=None):
    detections = []
    image_ids = coco.getImgIds()

    for img_id in tqdm(image_ids, desc="Evaluating"):
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        img_path = os.path.join(r"D:/Merged Datasets/annotations/", file_name)

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Error reading {img_path}")
            continue

        if defense_model:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #img2=img
            img = denoise_image(img, defense_model, device)
            #print(img-img2)


        img_input, scale = preprocess(img, input_size)
        img_input = img_input.float()

        


        if torch.cuda.is_available():
            img_input = img_input.cuda()

        with torch.no_grad():
            outputs = model(img_input)
            outputs = postprocess(outputs, num_classes=10, conf_thre=0.4, nms_thre=0.3)

        if outputs[0] is None:
            continue

        output = outputs[0].cpu().numpy()
        bboxes = output[:, 0:4] / scale
        scores = output[:, 4] * output[:, 5]
        cls_ids = output[:, 6].astype(np.int32)

        for bbox, score, cls_id in zip(bboxes, scores, cls_ids):
            x, y, w, h = convert_xyxy_to_xywh(bbox)
            det = {
                "image_id": str(img_id),
                "category_id": int(cls_id),  # COCO expects category_id starting from 1
                "bbox": [float(x), float(y), float(w), float(h)],  # Convert to float
                "area" : float(w)*float(h),
                "score": float(score)  # Convert to float
            }
            detections.append(det)

    # Save predictions as JSON
    with open("predictions.json", "w") as f:
        json.dump(detections, f)


# Evaluate the predictions
def evaluate():
    coco_gt = COCO(annotation_path)
    coco_dt = coco_gt.loadRes("predictions.json")
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    # Validate paths
    if not os.path.exists(val_images_dir):
        print(f"Error: Validation images directory not found: {val_images_dir}")
    elif not os.path.exists(annotation_path):
        print(f"Error: Annotation file not found: {annotation_path}")
    elif not os.path.exists(weights_path):
        print(f"Error: Weights file not found: {weights_path}")
    else:
        # Model and evaluation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_model(device)
        defense = get_HGD_model(device)
        coco = COCO(annotation_path)
        run_inference_and_save_predictions(model, coco,defense)
        evaluate()
