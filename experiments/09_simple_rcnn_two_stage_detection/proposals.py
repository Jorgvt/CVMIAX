"""
Region Proposal Generation, IoU Matching, Bounding Box Parameterization,
and Crop & Warp utilities for R-CNN Two-Stage Detection.
"""

from typing import List, Tuple
import numpy as np
from PIL import Image


def generate_sliding_window_proposals(
    img_size: int = 128,
    scales: List[int] = (24, 36, 48, 64),
    aspect_ratios: List[float] = (0.75, 1.0, 1.33),
    stride: int = 16,
) -> np.ndarray:
    """
    Generate systematic candidate bounding boxes using multi-scale sliding windows.

    Args:
        img_size: Height and width of canvas in pixels.
        scales: Base box dimensions in pixels.
        aspect_ratios: Width-to-height aspect ratios (w / h).
        stride: Grid step size in pixels.

    Returns:
        proposals: np.ndarray of shape (K, 4) in [x1, y1, x2, y2] format.
    """
    proposals = []

    # Grid centers
    grid_x = np.arange(stride // 2, img_size, stride)
    grid_y = np.arange(stride // 2, img_size, stride)

    for cy in grid_y:
        for cx in grid_x:
            for scale in scales:
                for ar in aspect_ratios:
                    w = scale * np.sqrt(ar)
                    h = scale / np.sqrt(ar)

                    x1 = cx - w / 2.0
                    y1 = cy - h / 2.0
                    x2 = cx + w / 2.0
                    y2 = cy + h / 2.0

                    # Keep proposal if at least 70% is inside the image canvas
                    clipped_x1 = max(0.0, x1)
                    clipped_y1 = max(0.0, y1)
                    clipped_x2 = min(float(img_size), x2)
                    clipped_y2 = min(float(img_size), y2)

                    orig_area = max(1.0, (x2 - x1) * (y2 - y1))
                    clipped_area = max(0.0, clipped_x2 - clipped_x1) * max(0.0, clipped_y2 - clipped_y1)

                    if clipped_area / orig_area >= 0.7:
                        proposals.append([clipped_x1, clipped_y1, clipped_x2, clipped_y2])

    return np.array(proposals, dtype=np.float32)


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Intersection-over-Union (IoU) matrix between two sets of boxes.

    Args:
        boxes_a: np.ndarray of shape (M, 4) in [x1, y1, x2, y2].
        boxes_b: np.ndarray of shape (N, 4) in [x1, y1, x2, y2].

    Returns:
        iou_matrix: np.ndarray of shape (M, N) with values in [0.0, 1.0].
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    # Expand dims for broadcasting: (M, 1, 4) vs (1, N, 4)
    a = np.expand_dims(boxes_a, axis=1)
    b = np.expand_dims(boxes_b, axis=0)

    # Intersection coordinates
    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Areas
    area_a = np.maximum(0.0, a[..., 2] - a[..., 0]) * np.maximum(0.0, a[..., 3] - a[..., 1])
    area_b = np.maximum(0.0, b[..., 2] - b[..., 0]) * np.maximum(0.0, b[..., 3] - b[..., 1])

    union_area = area_a + area_b - inter_area
    union_area = np.maximum(union_area, 1e-8)

    return (inter_area / union_area).astype(np.float32)


def corners_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from [x1, y1, x2, y2] to [cx, cy, w, h]."""
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    w = np.maximum(1e-4, x2 - x1)
    h = np.maximum(1e-4, y2 - y1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return np.stack([cx, cy, w, h], axis=-1)


def xywh_to_corners(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.stack([x1, y1, x2, y2], axis=-1)


def encode_bbox_deltas(proposals: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Encode ground-truth target boxes relative to proposals using R-CNN delta parameterization:
        t_x = (cx_gt - cx_prop) / w_prop
        t_y = (cy_gt - cy_prop) / h_prop
        t_w = log(w_gt / w_prop)
        t_h = log(h_gt / h_prop)

    Args:
        proposals: np.ndarray of shape (N, 4) in [x1, y1, x2, y2].
        targets: np.ndarray of shape (N, 4) in [x1, y1, x2, y2].

    Returns:
        deltas: np.ndarray of shape (N, 4) containing [t_x, t_y, t_w, t_h].
    """
    prop_xywh = corners_to_xywh(proposals)
    tgt_xywh = corners_to_xywh(targets)

    tx = (tgt_xywh[..., 0] - prop_xywh[..., 0]) / prop_xywh[..., 2]
    ty = (tgt_xywh[..., 1] - prop_xywh[..., 1]) / prop_xywh[..., 3]
    tw = np.log(np.maximum(1e-4, tgt_xywh[..., 2] / prop_xywh[..., 2]))
    th = np.log(np.maximum(1e-4, tgt_xywh[..., 3] / prop_xywh[..., 3]))

    return np.stack([tx, ty, tw, th], axis=-1).astype(np.float32)


def decode_bbox_deltas(proposals: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """
    Decode predicted deltas applied to candidate proposals back into [x1, y1, x2, y2]:
        cx_pred = cx_prop + w_prop * t_x
        cy_pred = cy_prop + h_prop * t_y
        w_pred = w_prop * exp(t_w)
        h_pred = h_prop * exp(t_h)

    Args:
        proposals: np.ndarray of shape (N, 4) in [x1, y1, x2, y2].
        deltas: np.ndarray of shape (N, 4) containing [t_x, t_y, t_w, t_h].

    Returns:
        pred_boxes: np.ndarray of shape (N, 4) in [x1, y1, x2, y2].
    """
    prop_xywh = corners_to_xywh(proposals)

    # Clip deltas to avoid numerical overflow in exp
    clipped_tw = np.clip(deltas[..., 2], -4.0, 4.0)
    clipped_th = np.clip(deltas[..., 3], -4.0, 4.0)

    pred_cx = prop_xywh[..., 0] + prop_xywh[..., 2] * deltas[..., 0]
    pred_cy = prop_xywh[..., 1] + prop_xywh[..., 3] * deltas[..., 1]
    pred_w = prop_xywh[..., 2] * np.exp(clipped_tw)
    pred_h = prop_xywh[..., 3] * np.exp(clipped_th)

    pred_xywh = np.stack([pred_cx, pred_cy, pred_w, pred_h], axis=-1)
    return xywh_to_corners(pred_xywh)


def clip_boxes(boxes: np.ndarray, img_size: int = 128) -> np.ndarray:
    """Clip box coordinates to remain strictly within [0, img_size]."""
    clipped = np.copy(boxes)
    clipped[..., [0, 2]] = np.clip(clipped[..., [0, 2]], 0.0, float(img_size))
    clipped[..., [1, 3]] = np.clip(clipped[..., [1, 3]], 0.0, float(img_size))
    return clipped


def match_proposals_to_ground_truth(
    proposals: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    pos_thresh: float = 0.5,
    neg_thresh: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match candidate proposals to ground truth bounding boxes based on IoU overlap.

    Labeling Logic:
        - Positive (label in 1..C): IoU >= pos_thresh (assigned GT shape class and GT bbox target).
        - Negative (label = 0): IoU < neg_thresh (assigned background, target deltas = 0).
        - Neutral/Ignore (label = -1): neg_thresh <= IoU < pos_thresh (ambiguous, skipped during training).

    Args:
        proposals: np.ndarray of shape (K, 4) in [x1, y1, x2, y2].
        gt_boxes: np.ndarray of shape (N, 4) in [x1, y1, x2, y2].
        gt_classes: np.ndarray of shape (N,) class integers (1..C).
        pos_thresh: Minimum IoU to qualify as positive.
        neg_thresh: Maximum IoU to qualify as background.

    Returns:
        matched_labels: np.ndarray of shape (K,) with class IDs (0=bg, -1=ignore).
        target_deltas: np.ndarray of shape (K, 4) with regression delta targets.
        max_ious: np.ndarray of shape (K,) with best IoU score achieved by each proposal.
    """
    num_props = len(proposals)
    matched_labels = np.full((num_props,), fill_value=-1, dtype=np.int32)
    target_deltas = np.zeros((num_props, 4), dtype=np.float32)
    max_ious = np.zeros((num_props,), dtype=np.float32)

    if len(gt_boxes) == 0:
        # No objects in image: all proposals are background
        matched_labels[:] = 0
        return matched_labels, target_deltas, max_ious

    iou_matrix = compute_iou_matrix(proposals, gt_boxes)  # Shape (K, N)
    best_gt_indices = np.argmax(iou_matrix, axis=1)        # Shape (K,)
    max_ious = np.max(iou_matrix, axis=1)                  # Shape (K,)

    # Assign positives
    pos_mask = max_ious >= pos_thresh
    for idx in np.where(pos_mask)[0]:
        gt_idx = best_gt_indices[idx]
        matched_labels[idx] = gt_classes[gt_idx]
        target_deltas[idx] = encode_bbox_deltas(
            proposals[idx:idx + 1],
            gt_boxes[gt_idx:gt_idx + 1]
        )[0]

    # Assign negatives (background)
    neg_mask = max_ious < neg_thresh
    matched_labels[neg_mask] = 0

    return matched_labels, target_deltas, max_ious


def extract_and_warp_crops(
    image: np.ndarray,
    boxes: np.ndarray,
    crop_size: Tuple[int, int] = (32, 32),
) -> np.ndarray:
    """
    Extract regions of interest (ROIs) from the image array and warp (resize)
    them to canonical dimensions (crop_size x crop_size x 3).

    Args:
        image: np.ndarray of shape (H, W, 3) in [0, 1] float32.
        boxes: np.ndarray of shape (K, 4) in [x1, y1, x2, y2].
        crop_size: Output canonical resolution tuple (H_crop, W_crop).

    Returns:
        crops: np.ndarray of shape (K, crop_size[0], crop_size[1], 3) in float32.
    """
    if len(boxes) == 0:
        return np.zeros((0, crop_size[0], crop_size[1], 3), dtype=np.float32)

    img_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    h_img, w_img = image.shape[:2]

    crops = []
    for box in boxes:
        x1, y1, x2, y2 = box
        # Clip to image bounds
        x1 = max(0, min(int(round(x1)), w_img - 1))
        y1 = max(0, min(int(round(y1)), h_img - 1))
        x2 = max(x1 + 1, min(int(round(x2)), w_img))
        y2 = max(y1 + 1, min(int(round(y2)), h_img))

        crop_pil = pil_img.crop((x1, y1, x2, y2)).resize(
            (crop_size[1], crop_size[0]), Image.Resampling.BILINEAR
        )
        crop_arr = np.array(crop_pil, dtype=np.float32) / 255.0
        crops.append(crop_arr)

    return np.array(crops, dtype=np.float32)
