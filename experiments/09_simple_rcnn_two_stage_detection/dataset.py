"""
Synthetic Multi-Shape Dataset Generator for Pedagogical Object Detection.

Generates synthetic 2D RGB canvas images containing geometric shapes (Circles,
Squares/Rectangles, Triangles) with exact ground-truth bounding boxes and class labels.
"""

from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw


CLASS_NAMES = ["background", "circle", "rectangle", "triangle"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

# Distinct color palette for shapes (RGB)
SHAPE_COLORS = [
    (220, 50, 50),    # Crimson Red
    (30, 144, 255),   # Dodger Blue
    (46, 204, 113),   # Emerald Green
    (241, 196, 15),   # Sun Yellow
    (155, 89, 182),   # Amethyst Purple
    (230, 126, 34),   # Carrot Orange
    (26, 188, 156),   # Turquoise
]


def _compute_box_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def generate_single_sample(
    img_size: int = 128,
    min_objects: int = 1,
    max_objects: int = 3,
    min_shape_size: int = 22,
    max_shape_size: int = 42,
    bg_color: Tuple[int, int, int] = (245, 245, 247),
    max_overlap_iou: float = 0.05,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a single synthetic image with random geometric shapes.

    Args:
        img_size: Canvas width and height in pixels.
        min_objects: Minimum number of objects on canvas.
        max_objects: Maximum number of objects on canvas.
        min_shape_size: Minimum width/height of a shape.
        max_shape_size: Maximum width/height of a shape.
        bg_color: Background RGB tuple.
        max_overlap_iou: Maximum allowed ground-truth box overlap.
        rng: Numpy random Generator instance.

    Returns:
        image: np.ndarray of shape (H, W, 3) with float32 values in [0, 1].
        boxes: np.ndarray of shape (N, 4) with [x1, y1, x2, y2] in pixel coords.
        classes: np.ndarray of shape (N,) with integer class IDs (1..3).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Create canvas
    canvas = Image.new("RGB", (img_size, img_size), bg_color)
    draw = ImageDraw.Draw(canvas)

    num_objects = rng.integers(min_objects, max_objects + 1)
    boxes: List[List[float]] = []
    classes: List[int] = []

    attempts = 0
    max_attempts = 200

    while len(boxes) < num_objects and attempts < max_attempts:
        attempts += 1
        shape_type = rng.choice(["circle", "rectangle", "triangle"])
        color_idx = rng.integers(0, len(SHAPE_COLORS))
        color = SHAPE_COLORS[color_idx]

        w = int(rng.integers(min_shape_size, max_shape_size + 1))
        h = int(rng.integers(min_shape_size, max_shape_size + 1))

        # Random top-left corner keeping shape inside canvas margin
        margin = 4
        if img_size - w - margin <= margin or img_size - h - margin <= margin:
            continue

        x1 = int(rng.integers(margin, img_size - w - margin))
        y1 = int(rng.integers(margin, img_size - h - margin))
        x2 = x1 + w
        y2 = y1 + h
        candidate_box = [float(x1), float(y1), float(x2), float(y2)]

        # Check overlap with existing objects
        overlap = False
        for existing_box in boxes:
            if _compute_box_iou(candidate_box, existing_box) > max_overlap_iou:
                overlap = True
                break

        if overlap:
            continue

        # Draw shape
        if shape_type == "circle":
            draw.ellipse([x1, y1, x2, y2], fill=color, outline=(20, 20, 20), width=1)
            class_id = CLASS_TO_ID["circle"]
        elif shape_type == "rectangle":
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(20, 20, 20), width=1)
            class_id = CLASS_TO_ID["rectangle"]
        elif shape_type == "triangle":
            # Draw triangle inside bbox
            p1 = (x1 + w // 2, y1)
            p2 = (x1, y2)
            p3 = (x2, y2)
            draw.polygon([p1, p2, p3], fill=color, outline=(20, 20, 20), width=1)
            class_id = CLASS_TO_ID["triangle"]
        else:
            continue

        boxes.append(candidate_box)
        classes.append(class_id)

    # Convert image to float32 [0, 1]
    img_array = np.array(canvas, dtype=np.float32) / 255.0
    boxes_array = np.array(boxes, dtype=np.float32) if len(boxes) > 0 else np.zeros((0, 4), dtype=np.float32)
    classes_array = np.array(classes, dtype=np.int32) if len(classes) > 0 else np.zeros((0,), dtype=np.int32)

    return img_array, boxes_array, classes_array


def generate_dataset(
    num_samples: int = 300,
    img_size: int = 128,
    min_objects: int = 1,
    max_objects: int = 3,
    seed: int = 42,
) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]]]:
    """
    Generate a full synthetic dataset for training/testing.

    Returns:
        images: np.ndarray of shape (num_samples, img_size, img_size, 3).
        annotations: list of dicts with 'boxes' (N, 4) and 'classes' (N,).
    """
    rng = np.random.default_rng(seed)
    images_list = []
    annotations = []

    for _ in range(num_samples):
        img, boxes, classes = generate_single_sample(
            img_size=img_size,
            min_objects=min_objects,
            max_objects=max_objects,
            rng=rng,
        )
        images_list.append(img)
        annotations.append({"boxes": boxes, "classes": classes})

    images = np.stack(images_list, axis=0)
    return images, annotations
