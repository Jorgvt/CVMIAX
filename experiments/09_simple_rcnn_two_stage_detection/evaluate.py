"""
Evaluation, Inference Pipeline, Non-Maximum Suppression (NMS), and Metrics for R-CNN.
"""

from typing import Dict, List, Tuple
import numpy as np
from tensorflow import keras

try:
    from .dataset import CLASS_NAMES
    from .proposals import (
        clip_boxes,
        compute_iou_matrix,
        decode_bbox_deltas,
        extract_and_warp_crops,
        generate_sliding_window_proposals,
    )
except ImportError:
    from dataset import CLASS_NAMES
    from proposals import (
        clip_boxes,
        compute_iou_matrix,
        decode_bbox_deltas,
        extract_and_warp_crops,
        generate_sliding_window_proposals,
    )


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.3,
    max_output_size: int = 50,
) -> List[int]:
    """
    Greedy Non-Maximum Suppression (NMS) algorithm from scratch.

    Pedagogical Step-by-Step Logic:
    1. Sort candidate bounding boxes by their classification confidence scores in descending order.
    2. Select the box with the highest score and add its index to the 'keep' list.
    3. Compute the IoU overlap between this chosen box and all remaining candidate boxes.
    4. Suppress (discard) any candidate box whose IoU overlap exceeds `iou_thresh`.
    5. Repeat steps 2-4 on the remaining boxes until no candidates remain or max_output_size is reached.

    Args:
        boxes: np.ndarray of shape (N, 4) in [x1, y1, x2, y2].
        scores: np.ndarray of shape (N,) confidence scores.
        iou_thresh: Overlap threshold above which redundant boxes are suppressed.
        max_output_size: Maximum number of boxes to retain.

    Returns:
        keep: List of integer indices of retained boxes.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]  # Highest confidence first

    keep = []
    while len(order) > 0 and len(keep) < max_output_size:
        i = order[0]
        keep.append(int(i))

        if len(order) == 1:
            break

        # Compute IoU of box i with all remaining boxes in order[1:]
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        union = np.maximum(union, 1e-8)
        iou = inter / union

        # Keep only boxes with IoU <= threshold
        remaining_indices = np.where(iou <= iou_thresh)[0]
        order = order[remaining_indices + 1]

    return keep


def detect_objects_rcnn(
    model: keras.Model,
    image: np.ndarray,
    conf_thresh: float = 0.6,
    nms_iou_thresh: float = 0.3,
    crop_size: Tuple[int, int] = (32, 32),
) -> Dict[str, np.ndarray]:
    """
    End-to-End Two-Stage Detection Inference on a single image.

    Pipeline:
    1. Propose candidate bounding boxes across the image.
    2. Extract & warp all candidate crops to canonical 32x32 size.
    3. Run CNN forward pass -> class probabilities + delta regression offsets.
    4. Filter out background proposals and proposals below confidence threshold.
    5. Refine proposal coordinates using predicted deltas.
    6. Apply Non-Maximum Suppression (NMS) per class.

    Returns dict containing:
        'proposals': Raw candidate boxes (K, 4)
        'raw_boxes': Refined candidate boxes before NMS (M, 4)
        'raw_scores': Confidence scores before NMS (M,)
        'raw_classes': Class IDs before NMS (M,)
        'nms_boxes': Final detected boxes after NMS (D, 4)
        'nms_scores': Final confidence scores after NMS (D,)
        'nms_classes': Final class IDs after NMS (D,)
    """
    img_size = image.shape[0]
    proposals = generate_sliding_window_proposals(img_size=img_size)

    # Crop & warp
    crops = extract_and_warp_crops(image, proposals, crop_size=crop_size)

    # CNN inference
    pred_cls_probs, pred_deltas = model(crops, training=False)
    pred_cls_probs = pred_cls_probs.numpy()
    pred_deltas = pred_deltas.numpy()

    # Determine highest scoring non-background class for each proposal
    fg_probs = pred_cls_probs[:, 1:]  # Exclude background (col 0)
    best_fg_class_offset = np.argmax(fg_probs, axis=1)  # 0..(C-1)
    best_fg_classes = best_fg_class_offset + 1          # 1..C
    best_fg_scores = np.max(fg_probs, axis=1)

    # Filter by confidence threshold
    valid_mask = best_fg_scores >= conf_thresh
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return {
            "proposals": proposals,
            "raw_boxes": np.zeros((0, 4), dtype=np.float32),
            "raw_scores": np.zeros((0,), dtype=np.float32),
            "raw_classes": np.zeros((0,), dtype=np.int32),
            "nms_boxes": np.zeros((0, 4), dtype=np.float32),
            "nms_scores": np.zeros((0,), dtype=np.float32),
            "nms_classes": np.zeros((0,), dtype=np.int32),
        }

    raw_props = proposals[valid_indices]
    raw_deltas = pred_deltas[valid_indices]
    raw_scores = best_fg_scores[valid_indices]
    raw_classes = best_fg_classes[valid_indices]

    # Refine proposal coordinates using predicted deltas
    refined_boxes = decode_bbox_deltas(raw_props, raw_deltas)
    refined_boxes = clip_boxes(refined_boxes, img_size=img_size)

    # Multi-class NMS (run NMS separately per foreground class)
    final_boxes_list = []
    final_scores_list = []
    final_classes_list = []

    for cls_id in np.unique(raw_classes):
        cls_mask = raw_classes == cls_id
        cls_boxes = refined_boxes[cls_mask]
        cls_scores = raw_scores[cls_mask]

        keep_idx = non_max_suppression(cls_boxes, cls_scores, iou_thresh=nms_iou_thresh)

        final_boxes_list.append(cls_boxes[keep_idx])
        final_scores_list.append(cls_scores[keep_idx])
        final_classes_list.append(np.full((len(keep_idx),), cls_id, dtype=np.int32))

    if len(final_boxes_list) > 0:
        final_boxes = np.concatenate(final_boxes_list, axis=0)
        final_scores = np.concatenate(final_scores_list, axis=0)
        final_classes = np.concatenate(final_classes_list, axis=0)
    else:
        final_boxes = np.zeros((0, 4), dtype=np.float32)
        final_scores = np.zeros((0,), dtype=np.float32)
        final_classes = np.zeros((0,), dtype=np.int32)

    return {
        "proposals": proposals,
        "raw_boxes": refined_boxes,
        "raw_scores": raw_scores,
        "raw_classes": raw_classes,
        "nms_boxes": final_boxes,
        "nms_scores": final_scores,
        "nms_classes": final_classes,
    }


def evaluate_dataset_map(
    model: keras.Model,
    images: np.ndarray,
    annotations: List[Dict[str, np.ndarray]],
    conf_thresh: float = 0.6,
    nms_iou_thresh: float = 0.3,
    iou_eval_thresh: float = 0.5,
) -> Dict[str, float]:
    """
    Compute mean Average Precision (mAP@0.5), Precision, and Recall on a dataset.
    """
    total_gt = 0
    total_tp = 0
    total_fp = 0

    per_class_stats = {cls_name: {"tp": 0, "fp": 0, "gt": 0} for cls_name in CLASS_NAMES[1:]}

    for i in range(len(images)):
        img = images[i]
        gt_boxes = annotations[i]["boxes"]
        gt_classes = annotations[i]["classes"]

        res = detect_objects_rcnn(
            model, img, conf_thresh=conf_thresh, nms_iou_thresh=nms_iou_thresh
        )
        pred_boxes = res["nms_boxes"]
        pred_classes = res["nms_classes"]

        # Track GT counts
        for c in gt_classes:
            cls_name = CLASS_NAMES[c]
            per_class_stats[cls_name]["gt"] += 1
            total_gt += 1

        matched_gt = set()
        for p_box, p_cls in zip(pred_boxes, pred_classes):
            cls_name = CLASS_NAMES[p_cls]
            # Find matching GT of same class
            matching_gt_indices = [
                idx for idx, c in enumerate(gt_classes)
                if c == p_cls and idx not in matched_gt
            ]

            if len(matching_gt_indices) == 0:
                total_fp += 1
                per_class_stats[cls_name]["fp"] += 1
                continue

            candidate_gt_boxes = gt_boxes[matching_gt_indices]
            ious = compute_iou_matrix(p_box[np.newaxis, :], candidate_gt_boxes)[0]
            best_idx_within = np.argmax(ious)
            best_iou = ious[best_idx_within]

            if best_iou >= iou_eval_thresh:
                matched_gt.add(matching_gt_indices[best_idx_within])
                total_tp += 1
                per_class_stats[cls_name]["tp"] += 1
            else:
                total_fp += 1
                per_class_stats[cls_name]["fp"] += 1

    overall_precision = total_tp / (total_tp + total_fp + 1e-8)
    overall_recall = total_tp / (total_gt + 1e-8)

    class_aps = {}
    for cls_name, stats in per_class_stats.items():
        p = stats["tp"] / (stats["tp"] + stats["fp"] + 1e-8)
        r = stats["tp"] / (stats["gt"] + 1e-8)
        class_aps[f"AP_{cls_name}"] = p * r  # Approximation for clean pedagogical summary

    return {
        "precision": float(overall_precision),
        "recall": float(overall_recall),
        "mAP_approx": float(overall_precision * overall_recall),
        **class_aps,
    }
