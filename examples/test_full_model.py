import torch
import cv2
import numpy as np
from yolox.exp import get_exp
from yolox.utils import postprocess
from defense.hgd_trainer import get_HGD_model  # same as your trainer file
from loguru import logger

# Define your 10 class names
class_names = [
    "pl100", "pl120", "pl20", "pl30", "pl40",
    "pl15", "pl50", "pl60", "pl70", "pl80"
]

def preprocess(img, input_size):
    h, w = img.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    image_resized = cv2.resize(img, (nw, nh))
    image_padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    image_padded[:nh, :nw] = image_resized
    image_padded = image_padded.astype(np.float32)
    image_padded = image_padded.transpose(2, 0, 1)
    return torch.from_numpy(image_padded).unsqueeze(0), scale

def visualize(img, bboxes, scores, cls_ids, class_names, conf_threshold=0.0001):
    h, w = img.shape[:2]
    for bbox, score, cls_id in zip(bboxes, scores, cls_ids):
        if score < conf_threshold:
            continue
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
        color = (0, 255, 0)
        label = f"{class_names[cls_id]}: {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def detect_and_visualize(model, image, input_size, class_names, tag):
    img_input, scale = preprocess(image, input_size)
    img_input = img_input.float()

    with torch.no_grad():
        outputs = model(img_input)
        outputs = postprocess(outputs, num_classes=10, conf_thre=0.4, nms_thre=0.3)
    result_img = image.copy()
    if outputs[0] is not None:
        output = outputs[0].cpu().numpy()
        bboxes = output[:, 0:4] / scale
        scores = output[:, 4] * output[:, 5]
        cls_ids = output[:, 6].astype(np.int32)

        valid_idx = np.all(np.isfinite(bboxes), axis=1) & np.all(bboxes >= 0, axis=1)
        bboxes = bboxes[valid_idx]
        scores = scores[valid_idx]
        cls_ids = cls_ids[valid_idx]

        bboxes = bboxes.astype(np.int32)
        logger.info(f"{tag} - Valid detections: {len(bboxes)}")

        result_img = visualize(image.copy(), bboxes, scores, cls_ids, class_names, conf_threshold=0.00001)
        cv2.imshow(f"{tag} Detection", result_img)
        cv2.imwrite(f"{tag.lower()}_output.jpg", result_img)
        return result_img
    else:
        cv2.imshow(f"{tag} Detection", result_img)
        cv2.imwrite(f"{tag.lower()}_output.jpg", result_img)
        logger.info(f"{tag} - No detections found.")
        return image

def main():
    weights_path = r"YOLOX_outputs\yolox_base\best_ckpt.pth"
    hgd_ckpt_path = r"best_ckpt.pt"  # path to trained HGD model
    input_size = (1024, 1024)
    image_path = r"D:\Merged Datasets\attacked_images1024\val\7000_5.jpg"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_type = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load YOLOX model
    exp = get_exp(None, "yolox-s")
    exp.num_classes = 10
    model = exp.get_model()
    model.eval()

    logger.info("Loading YOLOX checkpoint...")
    ckpt = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info("YOLOX model loaded successfully")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Image {image_path} not found!")
        return

    # Detect on perturbed image
    detect_and_visualize(model, img, input_size, class_names, tag="Perturbed Image")

    # Load HGD model (from hgd_trainer.get_HGD_model)
    logger.info("Loading HGD model...")
    hgd_model = get_HGD_model(device, hgd_ckpt_path)
    hgd_model.eval()

    # Preprocess for HGD (no normalization)
    img_tensor = (
        torch.from_numpy(img.astype(np.float32))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
        .float()  # Ensures compatibility
    )

    img_tensor = img_tensor.to(device).float()

    with torch.no_grad():
        noise = hgd_model(img_tensor)
        denoised_tensor = img_tensor - noise
        denoised_tensor = torch.clamp(denoised_tensor, 0.0, 255.0)

    # Use denoised tensor directly for visualization
    denoised_tensor = denoised_tensor.squeeze(0).cpu().byte().permute(1, 2, 0).numpy()

    # Detect on denoised image
    detect_and_visualize(model, denoised_tensor, input_size, class_names, tag="Denoised Image")

    # Optional: visualize noise (no normalization)
    noise_vis = noise.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    noise_vis = np.clip(noise_vis, 0, 255).astype(np.uint8)

    detect_and_visualize(model, noise_vis, input_size, class_names, tag="Noise")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
