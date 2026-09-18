"""
Synthetic Geometric Shapes Segmentation Dataset.

Generates synthetic 2D RGB images containing geometric shapes (Rectangles, Circles, Triangles)
paired with exact pixel-wise categorical ground-truth masks for semantic segmentation.

Pedagogical Purpose:
- Highlights the difference between models with and without skip connections.
- Shapes have crisp geometric boundaries, sharp corners (rectangles, triangles), and curves (circles)
  where spatial resolution loss in bottleneck-only models becomes immediately visible as edge blur.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf


# Class definitions
CLASS_NAMES = ["Background", "Rectangle", "Circle", "Triangle"]
CLASS_COLORS = {
    0: (240, 242, 245),   # Background: Soft light grey
    1: (231, 76, 60),     # Rectangle: Coral red
    2: (52, 152, 219),    # Circle: Sky blue
    3: (46, 204, 113),    # Triangle: Emerald green
}
NUM_CLASSES = len(CLASS_NAMES)


def draw_regular_polygon(
    draw: ImageDraw.ImageDraw,
    center: Tuple[int, int],
    radius: int,
    num_vertices: int,
    rotation_deg: float,
    fill_color: Tuple[int, int, int],
) -> List[Tuple[int, int]]:
    """Draw a regular polygon with a given number of vertices and rotation."""
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False) + np.radians(rotation_deg)
    vertices = [(int(cx + radius * np.cos(a)), int(cy + radius * np.sin(a))) for a in angles]
    draw.polygon(vertices, fill=fill_color)
    return vertices


def generate_single_sample(
    img_size: int = 128,
    min_shapes: int = 2,
    max_shapes: int = 4,
    min_size: int = 20,
    max_size: int = 40,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single RGB image and corresponding ground-truth integer mask.

    Args:
        img_size: Dimension of the square canvas (H=W=img_size).
        min_shapes: Minimum number of shapes to draw.
        max_shapes: Maximum number of shapes to draw.
        min_size: Minimum radius / half-dimension of shapes.
        max_size: Maximum radius / half-dimension of shapes.
        rng: Numpy random Generator instance for reproducibility.

    Returns:
        image: np.ndarray of shape (img_size, img_size, 3), float32 in [0, 1].
        mask: np.ndarray of shape (img_size, img_size), int32 with class IDs (0..3).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Base background color
    bg_rgb = CLASS_COLORS[0]
    img_canvas = Image.new("RGB", (img_size, img_size), bg_rgb)
    mask_canvas = Image.new("L", (img_size, img_size), 0)

    img_draw = ImageDraw.Draw(img_canvas)
    mask_draw = ImageDraw.Draw(mask_canvas)

    num_shapes = rng.integers(min_shapes, max_shapes + 1)
    
    # Shuffle order of shape types to prevent bias
    shape_types = rng.choice([1, 2, 3], size=num_shapes, replace=True)

    for class_id in shape_types:
        size = int(rng.integers(min_size, max_size + 1))
        margin = size + 4
        if margin >= img_size - margin:
            margin = 8
        cx = int(rng.integers(margin, img_size - margin + 1))
        cy = int(rng.integers(margin, img_size - margin + 1))
        
        # Color with slight random brightness variation for realistic RGB variance
        base_color = CLASS_COLORS[class_id]
        jitter = rng.integers(-15, 16, size=3)
        color = tuple(int(np.clip(base_color[i] + jitter[i], 0, 255)) for i in range(3))

        if class_id == 1:  # Rectangle
            half_w = int(rng.integers(size // 2, size + 1))
            half_h = int(rng.integers(size // 2, size + 1))
            x0, y0 = cx - half_w, cy - half_h
            x1, y1 = cx + half_w, cy + half_h
            img_draw.rectangle([x0, y0, x1, y1], fill=color)
            mask_draw.rectangle([x0, y0, x1, y1], fill=int(class_id))

        elif class_id == 2:  # Circle
            r = size
            x0, y0 = cx - r, cy - r
            x1, y1 = cx + r, cy + r
            img_draw.ellipse([x0, y0, x1, y1], fill=color)
            mask_draw.ellipse([x0, y0, x1, y1], fill=int(class_id))

        elif class_id == 3:  # Triangle (3-vertex polygon)
            rot = float(rng.uniform(0, 360))
            draw_regular_polygon(img_draw, (cx, cy), size, 3, rot, color)
            draw_regular_polygon(mask_draw, (cx, cy), size, 3, rot, int(class_id))

    # Convert to numpy arrays
    image_np = np.array(img_canvas, dtype=np.float32) / 255.0
    mask_np = np.array(mask_canvas, dtype=np.int32)

    # Add subtle Gaussian noise to image for realistic gradient features
    noise = rng.normal(0.0, 0.02, image_np.shape).astype(np.float32)
    image_np = np.clip(image_np + noise, 0.0, 1.0)

    return image_np, mask_np


def generate_dataset(
    num_samples: int = 500,
    img_size: int = 128,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a full dataset of images and segmentation masks.

    Args:
        num_samples: Total number of samples.
        img_size: Canvas size (img_size x img_size).
        seed: Random seed for deterministic generation.

    Returns:
        images: np.ndarray of shape (N, img_size, img_size, 3), float32.
        masks: np.ndarray of shape (N, img_size, img_size), int32.
    """
    rng = np.random.default_rng(seed)
    images = np.empty((num_samples, img_size, img_size, 3), dtype=np.float32)
    masks = np.empty((num_samples, img_size, img_size), dtype=np.int32)

    for i in range(num_samples):
        images[i], masks[i] = generate_single_sample(img_size=img_size, rng=rng)

    return images, masks


def get_dataset_splits(
    num_train: int = 400,
    num_val: int = 100,
    num_test: int = 100,
    img_size: int = 128,
    seed: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Generate Train, Validation, and Test dataset splits."""
    x_train, y_train = generate_dataset(num_train, img_size=img_size, seed=seed)
    x_val, y_val = generate_dataset(num_val, img_size=img_size, seed=seed + 1)
    x_test, y_test = generate_dataset(num_test, img_size=img_size, seed=seed + 2)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def create_tf_dataset(
    images: np.ndarray,
    masks: np.ndarray,
    batch_size: int = 16,
    shuffle: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """Create a high-performance tf.data.Dataset pipeline."""
    # Masks have shape (N, H, W), add channel dim (N, H, W, 1) or keep (N, H, W) for sparse categorical CE
    ds = tf.data.Dataset.from_tensor_slices((images, masks))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images), seed=seed)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == "__main__":
    print("Testing synthetic dataset generation...")
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = get_dataset_splits(50, 10, 10, img_size=128)
    print(f"Train shapes: Images {x_tr.shape} (dtype: {x_tr.dtype}, min: {x_tr.min():.2f}, max: {x_tr.max():.2f})")
    print(f"Train masks:  {y_tr.shape} (dtype: {y_tr.dtype}, unique classes: {np.unique(y_tr)})")
    print("Dataset generation successful!")
