import json
import pandas as pd
import numpy as np

def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (list): [x, y, width, height] for the first bounding box.
        box2 (list): [x, y, width, height] for the second bounding box.

    Returns:
        float: The IoU value.
    """
    # Extract coordinates for box1 (ground truth)
    x1_val, y1_val, w1_val, h1_val = box1
    x2_val, y2_val = x1_val + w1_val, y1_val + h1_val

    # Extract coordinates for box2 (prediction)
    x1_pred, y1_pred, w1_pred, h1_pred = box2
    x2_pred, y2_pred = x1_pred + w1_pred, y1_pred + h1_pred

    # Calculate intersection coordinates
    x_left = max(x1_val, x1_pred)
    y_top = max(y1_val, y1_pred)
    x_right = min(x2_val, x2_pred)
    y_bottom = min(y2_val, y2_pred)

    # Calculate intersection area
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Calculate union area
    area_val = w1_val * h1_val
    area_pred = w1_pred * h1_pred
    union_area = area_val + area_pred - intersection_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0
    return intersection_area / union_area

def evaluate_detections(ground_truth_path, predictions_path, iou_threshold=0.5):
    """
    Computes precision and recall for object detection results.

    Args:
        ground_truth_path (str): Path to the ground truth JSON file (e.g., val.json).
        predictions_path (str): Path to the predictions JSON file (e.g., pred_attacked.json).
        iou_threshold (float): The Intersection over Union (IoU) threshold for considering a detection a True Positive.

    Returns:
        tuple: A tuple containing (precision, recall).
    """
    try:
        with open(ground_truth_path, 'r') as f:
            val_data = json.load(f)

        with open(predictions_path, 'r') as f:
            pred_attacked_data = json.load(f)
    except FileNotFoundError:
        print("Error: One or both input files not found. Please check the paths.")
        return 0.0, 0.0
    except json.JSONDecodeError:
        print("Error: Could not decode JSON from one or both files. Please check file content.")
        return 0.0, 0.0

    val_annotations = val_data.get('annotations', [])
    pred_detections = pred_attacked_data # Assuming pred_attacked.json is a list of detections

    # Create DataFrames for easier manipulation
    val_df = pd.DataFrame(val_annotations)
    pred_df = pd.DataFrame(pred_detections)

    # Filter ground truth to remove 'iscrowd' annotations (if present and 1)
    if 'iscrowd' in val_df.columns:
        val_df = val_df[val_df['iscrowd'] == 0].reset_index(drop=True)

    # Get unique image_ids from both dataframes
    val_image_ids = val_df['image_id'].unique()
    pred_image_ids = pred_df['image_id'].unique()

    # Find image_ids present in both dataframes to ensure consistent evaluation
    common_image_ids = np.intersect1d(val_image_ids, pred_image_ids)

    # Filter both dataframes to include only common image_ids
    val_df_filtered = val_df[val_df['image_id'].isin(common_image_ids)].copy()
    pred_df_filtered = pred_df[pred_df['image_id'].isin(common_image_ids)].copy()

    # Initialize metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Iterate through unique images to compute metrics
    for img_id in common_image_ids:
        # Ground truth boxes for the current image
        gt_boxes_for_img = val_df_filtered[val_df_filtered['image_id'] == img_id].copy()
        
        # Predicted boxes for the current image
        pred_boxes_for_img = pred_df_filtered[pred_df_filtered['image_id'] == img_id].copy()

        # Keep track of which ground truth boxes have been matched
        matched_gt_indices = set()

        # Sort predictions by score in descending order (optional, but good practice for AP)
        pred_boxes_for_img = pred_boxes_for_img.sort_values(by='score', ascending=False)

        # Iterate through predicted boxes for the current image
        for idx_pred, pred_row in pred_boxes_for_img.iterrows():
            best_iou = 0.0
            best_gt_idx = -1

            # Iterate through ground truth boxes for the current image
            for idx_gt, gt_row in gt_boxes_for_img.iterrows():
                # Only consider unmatched ground truth boxes
                if idx_gt not in matched_gt_indices:
                    # Check if category_id matches
                    if gt_row['category_id'] == pred_row['category_id']:
                        iou = calculate_iou(gt_row['bbox'], pred_row['bbox'])
                        if iou >= iou_threshold and iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx_gt

            # If a match is found, increment true positives and mark ground truth as matched
            if best_gt_idx != -1:
                true_positives += 1
                matched_gt_indices.add(best_gt_idx)
            else:
                false_positives += 1 # No match found for this prediction

        # False negatives are ground truth boxes that were not matched by any prediction
        false_negatives += len(gt_boxes_for_img) - len(matched_gt_indices)

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    return precision, recall

if __name__ == "__main__":
    # Example Usage:
    # Replace 'val.json' and 'pred_attacked.json' with your actual file paths
    ground_truth_file = r"D:\Merged Datasets\annotations\val.json"
    clean_predictions_file = r"D:\YoloX trial code\pred_clean.json"
    attacked_predictions_file = r"D:\YoloX trial code\pred_attacked.json"
    denoised_predictions_file = r"D:\YoloX trial code\pred_denoised.json"
    precision_score, recall_score = evaluate_detections(ground_truth_file, clean_predictions_file)
    print("Clean Images Evaluation:")
    print(f"Computed Precision: {precision_score:.4f}")
    print(f"Computed Recall: {recall_score:.4f}")

    print()
    precision_score, recall_score = evaluate_detections(ground_truth_file, attacked_predictions_file)
    print("Attacked Images Evaluation:")
    print(f"Computed Precision: {precision_score:.4f}")
    print(f"Computed Recall: {recall_score:.4f}")

    print()
    precision_score, recall_score = evaluate_detections(ground_truth_file, denoised_predictions_file)
    print("Denoised Images Evaluation:")
    print(f"Computed Precision: {precision_score:.4f}")
    print(f"Computed Recall: {recall_score:.4f}")



    # You can also test with a different IoU threshold
    # precision_score_0_7, recall_score_0_7 = evaluate_detections(ground_truth_file, predictions_file, iou_threshold=0.7)
    # print(f"\nComputed Precision (IoU=0.7): {precision_score_0_7:.4f}")
    # print(f"Computed Recall (IoU=0.7): {recall_score_0_7:.4f}")
