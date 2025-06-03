import os
import json
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from yolox.utils import postprocess
from defense.generate_FPN import get_model
from defense.hgd_trainer import get_HGD_model
import matplotlib.pyplot as plt


# =========================
# ⚙️ CONFIGURATION
# =========================
class_names = ["pl100", "pl120", "pl20", "pl30", "pl40", "pl15", "pl50", "pl60", "pl70", "pl80"]
input_size = (1024, 1024)
val_annotation_path = r"D:/Merged Datasets/annotations/val.json"
attacked_images_dir = r"D:/Merged Datasets/attacked_images1024/val"
original_images_dir = r"D:/Merged Datasets/images/val"
val_csv_path = r"D:/Merged Datasets/attacked_images/val.csv"
os.makedirs("visualizations", exist_ok=True)


# =========================
# 🔧 PREPROCESSOR CLASS
# =========================
class Preprocessor:
    def __init__(self, input_size=(1024, 1024), mean_val=114):
        self.input_size = input_size
        self.mean_val = mean_val

    def __call__(self, img):
        h, w = img.shape[:2]
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        nh, nw = int(h * scale), int(w * scale)
        image_resized = cv2.resize(img, (nw, nh))
        image_padded = np.full((self.input_size[0], self.input_size[1], 3), self.mean_val, dtype=np.uint8)
        image_padded[:nh, :nw] = image_resized
        image_padded = image_padded.astype(np.float32).transpose(2, 0, 1)
        return torch.from_numpy(image_padded).unsqueeze(0), scale


# =========================
# 📦 UTILS
# =========================
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
          (255, 255, 0), (255, 0, 255), (0, 255, 255),
          (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)]

def convert_xyxy_to_xywh(bbox):
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]

def visualize_predictions(img, bboxes, scores, cls_ids, class_names, image_title=None, threshold=0.3, save_path=None):
    img_vis = img.copy()
    for bbox, score, cls_id in zip(bboxes, scores, cls_ids):
        if score < threshold:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        color = COLORS[cls_id % len(COLORS)]
        label = f"{class_names[cls_id]}: {score:.2f}"

        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_vis, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if save_path:
        cv2.imwrite(save_path, img_vis)
    else:
        img_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 8))
        plt.imshow(img_rgb)
        plt.axis("off")
        if image_title:
            plt.title(image_title)
        plt.show()

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

def run_inference_on_images(model, coco, image_dir, output_json, filename_map=None, use_filename_key=False, defense_model=None, preprocessor=None):
    detections = []

    for img_id in tqdm(coco.getImgIds(), desc="Running inference"):
        img_info = coco.loadImgs(img_id)[0]
        original_name = img_info["file_name"].split("/")[-1]

        if filename_map and original_name not in filename_map:
            continue

        img_name = filename_map[original_name] if filename_map and use_filename_key else original_name
        img_path = os.path.join(image_dir, img_name)

        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Error reading image: {img_path}")
            continue

        if defense_model:
            img = denoise_image(img, defense_model, model.device)

        img_input, scale = preprocessor(img)
        img_input = img_input.float().to(model.device)

        with torch.no_grad():
            outputs = model(img_input)
            outputs = postprocess(outputs, num_classes=10, conf_thre=0.6, nms_thre=0.3)

        if outputs[0] is None:
            continue

        output = outputs[0].cpu().numpy()

        # Scale back by preprocessor scale
        bboxes = output[:, 0:4]

        scores = output[:, 4] * output[:, 5]
        cls_ids = output[:, 6].astype(np.int32)

        # === VISUALIZATION WITHOUT EXTENSION-BASED SCALING ===
        visualize_predictions(img, bboxes, scores, cls_ids, class_names, save_path=os.path.join("visualizations", f"{img_id}_{img_name}"))

        # ======= Extension based scaling for output JSON =======
        ext = os.path.splitext(original_name)[1].lower()

        if ext == ".png":
            scale_w, scale_h = 1360 / input_size[1], 1360 / input_size[0]
        elif ext == ".jpg":
            scale_w, scale_h = 2048 / input_size[1], 2048 / input_size[0]
        else:
            scale_w, scale_h = 1.0, 1.0

        # Apply extension based scaling only here
        bboxes[:, [0, 2]] *= scale_w
        bboxes[:, [1, 3]] *= scale_h
        # ======================================

        for bbox, score, cls_id in zip(bboxes, scores, cls_ids):
            x, y, w, h = convert_xyxy_to_xywh(bbox)
            det = {
                "image_id": img_id,
                "category_id": int(cls_id),
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w) * float(h),
                "score": float(score)
            }
            detections.append(det)

    with open(output_json, "w") as f:
        json.dump(detections, f)


def evaluate(predictions_json, annotation_path):
    coco_gt = COCO(annotation_path)
    coco_dt = coco_gt.loadRes(predictions_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


# =========================
# 🚀 MAIN
# =========================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolox_model = get_model(device)
    yolox_model.eval()
    yolox_model.device = device

    preprocessor = Preprocessor(input_size=input_size)

    coco = COCO(val_annotation_path)
    df = pd.read_csv(val_csv_path)

    # Build mappings
    attacked_to_original = dict(zip(df["attacked_images"], df["orig_images"]))
    original_to_attacked = {}

    for orig in df["orig_images"].unique():
        originals_attacks = df[df["orig_images"] == orig]["attacked_images"].tolist()
        original_to_attacked[orig] = originals_attacks[4]

    print(f"Original to Attacked Mapping: {original_to_attacked}")

    # 1. Clean Images
    # print("\n✅ Evaluating on Original (Clean) Images...")
    # run_inference_on_images(
    #     yolox_model,
    #     coco,
    #     original_images_dir,
    #     "pred_clean.json",
    #     preprocessor=preprocessor
    # )
    # evaluate("pred_clean.json", val_annotation_path)

    # 2. Attacked Images
    print("\n🧪 Evaluating on Attacked Images...")
    run_inference_on_images(
        yolox_model,
        coco,
        attacked_images_dir,
        "pred_attacked.json",
        filename_map=original_to_attacked,
        use_filename_key=True,
        preprocessor=preprocessor
    )
    evaluate("pred_attacked.json", val_annotation_path)

    # 3. Denoised Images
    print("\n🔧 Denoising with HGD and Evaluating...")
    hgd_model = get_HGD_model(device)
    hgd_model.eval()
    run_inference_on_images(
        yolox_model,
        coco,
        attacked_images_dir,
        "pred_denoised.json",
        filename_map=original_to_attacked,
        use_filename_key=True,
        defense_model=hgd_model,
        preprocessor=preprocessor
    )
    evaluate("pred_denoised.json", val_annotation_path)