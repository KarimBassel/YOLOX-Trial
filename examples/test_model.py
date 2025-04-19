import torch
import cv2
import numpy as np
from yolox.exp import get_exp
from yolox.models import YOLOX
from yolox.utils import postprocess
from defense.hgd_trainer import get_HGD_model
from defense.generate_FPN import get_model
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
# Function to visualize detection results
def visualize(img, bboxes, scores, cls_ids, class_names, conf_threshold=0.0001):
    h, w = img.shape[:2]  # Get image dimensions (height, width)
    
    for bbox, score, cls_id in zip(bboxes, scores, cls_ids):
        print(score)
        if score < conf_threshold:
            continue  # Skip low-confidence detections

        # Unpack bounding box coordinates
        x1, y1, x2, y2 = bbox.astype(int)  

        # Clip coordinates to stay within image boundaries
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)

        # Set color and label for the bounding box
        color = (0, 255, 0)  # Green color in BGR
        label = f"{class_names[cls_id]}: {score:.2f}"

        # Draw rectangle and label on image
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img


# Main inference function
def main():
    # Path to model weights
    weights_path = r"YOLOX_outputs\yolox_base\best_ckpt.pth" 
    input_size = (640, 640) 

    # Initialize YOLOX experiment and model
    exp = get_exp(None, "yolox-s")  
    #number of classes for our dataset
    exp.num_classes = 10 
    model = exp.get_model()
    model.eval()

    # Load trained weights
    logger.info("Loading checkpoint")
    ckpt = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info("Loaded checkpoint successfully")

    # Load and preprocess image
    #image_path = r"D:\Merged Datasets\images\val\00185.png"  
    #image_path = r"D:\Merged Datasets\attacked_images\val\00185_1.png"
    image_path = r"D:\Merged Datasets\attacked_images\train\2_3.jpg"
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Image {image_path} not found!")
        return

    img_input, scale = preprocess(img, input_size)
    img_input = img_input.float()  # Ensure float32 type

    # --- Inference ---
    with torch.no_grad():
        outputs = model(img_input)  # Model forward pass
        print("Decoded outputs shape:", outputs.shape)  

        # Extract objectness and class scores for debugging
        objectness = outputs[..., 4]  # [1, 21504]
        class_scores = outputs[..., 5:]  # [1, 21504, 10]

        max_obj_score = objectness.max().item()
        max_cls_score = class_scores.max().item()
        print("Max objectness score in batch:", max_obj_score)
        print("Max class score in batch:", max_cls_score)

        # Post-process outputs (low threshold for debugging detections)
        outputs = postprocess(outputs, num_classes=10, conf_thre=0.4, nms_thre=0.3)

    # --- Output Handling ---
    if outputs[0] is not None:
        output = outputs[0].cpu().numpy()
        
        bboxes = output[:, 0:4] / scale  # Rescale boxes to original image size
        scores = output[:, 4] * output[:, 5]  # Multiply objectness with class confidence
        cls_ids = output[:, 6].astype(np.int32)

        # Filter invalid boxes (NaN, Inf, negative)
        valid_idx = np.all(np.isfinite(bboxes), axis=1) & np.all(bboxes >= 0, axis=1)
        bboxes = bboxes[valid_idx]
        scores = scores[valid_idx]
        cls_ids = cls_ids[valid_idx]

        # Cast to int for OpenCV
        bboxes = bboxes.astype(np.int32)

        logger.info(f"Valid detections: {len(bboxes)}")

        # Draw boxes
        result_img = visualize(img, bboxes, scores, cls_ids, class_names, conf_threshold=0.00001)  # adjust threshold

        # Show and save output
        cv2.imshow("Detection", result_img)
        cv2.imwrite("output.jpg", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        logger.info("No detections found.")


if __name__ == "__main__":
    main()
