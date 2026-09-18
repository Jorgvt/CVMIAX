"""Dataset generator and preprocessing pipeline for synthetic multimodal CLIP experiment.

Generates colored geometric shapes (circles, squares, triangles, hexagons, stars, diamonds)
paired with natural language descriptions, supporting in-distribution training,
compositional zero-shot splits, and novel shape generalization splits.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Available visual attributes
COLORS: Dict[str, Tuple[int, int, int]] = {
    "red": (230, 57, 70),
    "blue": (69, 123, 157),
    "green": (42, 157, 143),
    "yellow": (233, 196, 106),
    "purple": (155, 93, 229),
    "cyan": (0, 180, 216),
    "orange": (244, 162, 97),
}

KNOWN_SHAPES: List[str] = ["circle", "square", "triangle", "hexagon"]
NOVEL_SHAPES: List[str] = ["star", "diamond"]
ALL_SHAPES: List[str] = KNOWN_SHAPES + NOVEL_SHAPES

# Caption prompt templates for natural language variability
PROMPT_TEMPLATES: List[str] = [
    "a photo of a {color} {shape}",
    "a {color} {shape}",
    "an image showing a {color} {shape}",
    "a centered {color} {shape}",
    "a {shape} colored {color}",
    "a clean {color} {shape} on canvas",
]

# Specifically held out (color, shape) pairs for compositional zero-shot testing
COMPOSITIONAL_HOLDOUT: List[Tuple[str, str]] = [
    ("yellow", "square"),
    ("cyan", "triangle"),
    ("purple", "circle"),
    ("orange", "hexagon"),
]


def draw_geometric_shape(
    shape: str,
    color_rgb: Tuple[int, int, int],
    image_size: int = 64,
    jitter: bool = True,
) -> np.ndarray:
    """Draw a single colored geometric shape on a dark neutral canvas."""
    img = Image.new("RGB", (image_size, image_size), color=(24, 24, 28))
    draw = ImageDraw.Draw(img)

    # Base coordinates with optional spatial jitter
    cx = image_size / 2.0
    cy = image_size / 2.0
    if jitter:
        cx += np.random.uniform(-3, 3)
        cy += np.random.uniform(-3, 3)

    radius = np.random.uniform(18, 24) if jitter else 21.0

    if shape == "circle":
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        draw.ellipse(bbox, fill=color_rgb)

    elif shape == "square":
        side = radius * 1.5
        bbox = [cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0]
        draw.rectangle(bbox, fill=color_rgb)

    elif shape == "triangle":
        pts = [
            (cx, cy - radius * 1.2),
            (cx - radius * 1.1, cy + radius * 0.9),
            (cx + radius * 1.1, cy + radius * 0.9),
        ]
        draw.polygon(pts, fill=color_rgb)

    elif shape == "hexagon":
        pts = []
        for i in range(6):
            angle = i * (2 * np.pi / 6)
            px = cx + radius * np.cos(angle)
            py = cy + radius * np.sin(angle)
            pts.append((px, py))
        draw.polygon(pts, fill=color_rgb)

    elif shape == "star":
        # 5-pointed star
        pts = []
        outer_r = radius * 1.2
        inner_r = radius * 0.52
        for i in range(10):
            r = outer_r if (i % 2 == 0) else inner_r
            angle = i * (np.pi / 5) - (np.pi / 2)
            px = cx + r * np.cos(angle)
            py = cy + r * np.sin(angle)
            pts.append((px, py))
        draw.polygon(pts, fill=color_rgb)

    elif shape == "diamond":
        # Rhombus / diamond
        rx = radius * 0.9
        ry = radius * 1.3
        pts = [(cx, cy - ry), (cx + rx, cy), (cx, cy + ry), (cx - rx, cy)]
        draw.polygon(pts, fill=color_rgb)

    else:
        raise ValueError(f"Unknown shape: {shape}")

    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def generate_caption(color: str, shape: str, template_idx: Optional[int] = None) -> str:
    """Generate a descriptive prompt for the given color and shape."""
    if template_idx is not None:
        template = PROMPT_TEMPLATES[template_idx % len(PROMPT_TEMPLATES)]
    else:
        template = np.random.choice(PROMPT_TEMPLATES)
    return template.format(color=color, shape=shape)


def create_multimodal_dataset(
    num_samples_per_combo: int = 100,
    seed: int = 42,
    image_size: int = 64,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    """Generate train, val, compositional zero-shot test, and novel shape test datasets."""
    np.random.seed(seed)

    train_images, train_texts, train_labels = [], [], []
    val_images, val_texts, val_labels = [], [], []
    comp_images, comp_texts, comp_labels = [], [], []
    novel_images, novel_texts, novel_labels = [], [], []

    color_names = list(COLORS.keys())

    # 1. Generate Known Shapes (Train / Val / Compositional Zero-Shot)
    for shape in KNOWN_SHAPES:
        for color in color_names:
            is_compositional_holdout = (color, shape) in COMPOSITIONAL_HOLDOUT

            for i in range(num_samples_per_combo):
                img = draw_geometric_shape(shape, COLORS[color], image_size=image_size, jitter=True)
                caption = generate_caption(color, shape)
                label_id = f"{color}_{shape}"

                if is_compositional_holdout:
                    comp_images.append(img)
                    comp_texts.append(caption)
                    comp_labels.append(label_id)
                else:
                    # 85% train, 15% validation
                    if np.random.rand() < 0.85:
                        train_images.append(img)
                        train_texts.append(caption)
                        train_labels.append(label_id)
                    else:
                        val_images.append(img)
                        val_texts.append(caption)
                        val_labels.append(label_id)

    # 2. Generate Novel Shapes (Completely novel geometric figures held out from training)
    for shape in NOVEL_SHAPES:
        for color in color_names:
            for i in range(num_samples_per_combo):
                img = draw_geometric_shape(shape, COLORS[color], image_size=image_size, jitter=True)
                caption = generate_caption(color, shape)
                label_id = f"{color}_{shape}"
                novel_images.append(img)
                novel_texts.append(caption)
                novel_labels.append(label_id)

    # Convert to numpy arrays
    def to_dict(imgs, txts, lbls):
        idx = np.random.permutation(len(imgs))
        return {
            "images": np.array(imgs, dtype=np.float32)[idx],
            "texts": np.array(txts, dtype=object)[idx],
            "labels": np.array(lbls, dtype=object)[idx],
        }

    train_data = to_dict(train_images, train_texts, train_labels)
    val_data = to_dict(val_images, val_texts, val_labels)
    comp_data = to_dict(comp_images, comp_texts, comp_labels)
    novel_data = to_dict(novel_images, novel_texts, novel_labels)

    return train_data, val_data, comp_data, novel_data


def build_text_vectorizer(
    train_texts: np.ndarray,
    max_tokens: int = 100,
    output_sequence_length: int = 8,
) -> layers.TextVectorization:
    """Create and adapt TextVectorization layer on training texts."""
    vectorizer = layers.TextVectorization(
        max_tokens=max_tokens,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        output_mode="int",
        output_sequence_length=output_sequence_length,
    )
    # Ensure all color and shape words are present in vocabulary
    all_vocab_words = list(COLORS.keys()) + ALL_SHAPES + [
        "a", "photo", "of", "an", "image", "showing", "centered", "colored", "clean", "on", "canvas"
    ]
    vectorizer.adapt(tf.constant(all_vocab_words + list(train_texts)))
    return vectorizer


def create_tf_dataset(
    data: Dict[str, np.ndarray],
    vectorizer: layers.TextVectorization,
    batch_size: int = 64,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset yielding (images, tokenized_texts)."""
    tokenized_texts = vectorizer(tf.constant(data["texts"])).numpy()
    ds = tf.data.Dataset.from_tensor_slices((data["images"], tokenized_texts))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(data["images"]), seed=42)
    ds = ds.batch(batch_size, drop_remainder=shuffle)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
