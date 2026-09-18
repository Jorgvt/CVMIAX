"""
Visualization of Data Augmentations for Semantic Segmentation (Separated Image & Mask).
Pedagogical objective:
Display input images and discrete segmentation masks in separate rows, visually demonstrating:
1. Photometric transformations: Image changes, Mask is identical/untouched.
2. Geometric transformations (Correct): Image and Mask are co-transformed synchronously.
3. Geometric transformations (Bug): Image is transformed, but Mask remains untransformed (obvious mismatch).
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def set_plot_style():
    plt.rcParams.update({
        "font.sans-serif": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
        "figure.dpi": 300,
    })


# -------------------------------------------------------------------------
# Synthetic Segmentation Sample Generator
# -------------------------------------------------------------------------

def create_segmentation_scene(size=(320, 320)):
    """
    Creates a distinct, asymmetric synthetic scene with multi-class segmentation mask.
    Classes:
      0: Background (Sky)
      1: Terrain / Ground
      2: Vehicle / Object Body
      3: Pedestrian
      4: Tree / Foliage
    """
    w, h = size
    img = Image.new("RGB", (w, h), color=(220, 235, 245))  # Light sky
    mask = Image.new("L", (w, h), color=0)                 # Background = 0
    
    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)
    
    # 1. Ground / Terrain (Class 1)
    ground_poly = [(0, int(h * 0.65)), (w, int(h * 0.65)), (w, h), (0, h)]
    draw_img.polygon(ground_poly, fill=(110, 150, 100))
    draw_mask.polygon(ground_poly, fill=1)
    
    # Grass trim
    draw_img.rectangle([0, int(h * 0.65), w, int(h * 0.68)], fill=(95, 135, 85))
    
    # 2. Tree / Obstacle on the LEFT (Class 4) - asymmetric anchor
    tree_trunk = [(35, int(h * 0.45)), (55, int(h * 0.45)), (55, int(h * 0.72)), (35, int(h * 0.72))]
    draw_img.polygon(tree_trunk, fill=(120, 75, 40))
    draw_mask.polygon(tree_trunk, fill=4)
    
    tree_foliage = [10, int(h * 0.20), 80, int(h * 0.50)]
    draw_img.ellipse(tree_foliage, fill=(35, 120, 50))
    draw_mask.ellipse(tree_foliage, fill=4)
    
    # 3. Vehicle / Car in the CENTER-RIGHT (Class 2)
    car_box = [(125, int(h * 0.55)), (285, int(h * 0.55)), (285, int(h * 0.72)), (125, int(h * 0.72))]
    draw_img.polygon(car_box, fill=(220, 60, 50))  # Red car body
    draw_mask.polygon(car_box, fill=2)
    
    # Car cabin
    cabin_box = [(165, int(h * 0.45)), (255, int(h * 0.45)), (270, int(h * 0.55)), (150, int(h * 0.55))]
    draw_img.polygon(cabin_box, fill=(180, 40, 35))
    draw_mask.polygon(cabin_box, fill=2)
    
    # Car windows (light cyan)
    draw_img.polygon([(170, int(h * 0.47)), (205, int(h * 0.47)), (205, int(h * 0.54)), (157, int(h * 0.54))], fill=(190, 230, 255))
    draw_img.polygon([(210, int(h * 0.47)), (250, int(h * 0.47)), (263, int(h * 0.54)), (210, int(h * 0.54))], fill=(190, 230, 255))
    
    # Car wheels
    draw_img.ellipse([145, int(h * 0.68), 180, int(h * 0.79)], fill=(40, 40, 40))
    draw_img.ellipse([230, int(h * 0.68), 265, int(h * 0.79)], fill=(40, 40, 40))
    draw_mask.ellipse([145, int(h * 0.68), 180, int(h * 0.79)], fill=2)
    draw_mask.ellipse([230, int(h * 0.68), 265, int(h * 0.79)], fill=2)
    
    # 4. Pedestrian on the MID-LEFT (Class 3)
    ped_head = [92, int(h * 0.48), 108, int(h * 0.54)]
    ped_body = [(90, int(h * 0.54)), (110, int(h * 0.54)), (110, int(h * 0.68)), (90, int(h * 0.68))]
    draw_img.ellipse(ped_head, fill=(240, 190, 150))
    draw_img.polygon(ped_body, fill=(50, 90, 200))  # Blue clothes
    draw_mask.ellipse(ped_head, fill=3)
    draw_mask.polygon(ped_body, fill=3)
    
    # Legs
    draw_img.line([(95, int(h * 0.68)), (93, int(h * 0.75))], fill=(30, 30, 30), width=3)
    draw_img.line([(105, int(h * 0.68)), (107, int(h * 0.75))], fill=(30, 30, 30), width=3)
    draw_mask.rectangle([92, int(h * 0.68), 108, int(h * 0.75)], fill=3)

    return np.array(img), np.array(mask)


def render_mask_rgb(mask):
    """Converts discrete integer class mask into high-contrast RGB visualization."""
    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    palette = [
        [38, 50, 56],     # 0: Background dark blue-grey
        [76, 175, 80],    # 1: Ground green
        [229, 57, 53],    # 2: Vehicle red
        [30, 136, 229],   # 3: Pedestrian blue
        [255, 160, 0],    # 4: Tree orange
    ]
    for cid, col in enumerate(palette):
        mask_rgb[mask == cid] = col
    return mask_rgb


def overlay_mask(img, mask, alpha=0.5):
    """
    Overlays a discrete segmentation mask onto an RGB image using alpha blending.
    
    Parameters:
        img: np.ndarray of shape (H, W, 3), dtype uint8
        mask: np.ndarray of shape (H, W), integer class IDs
        alpha: float, blending weight for mask (0.0 = image only, 1.0 = mask only)
        
    Returns:
        blended: np.ndarray of shape (H, W, 3), dtype uint8
    """
    mask_rgb = render_mask_rgb(mask)
    blended = (img.astype(np.float32) * (1.0 - alpha) + mask_rgb.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return blended


# -------------------------------------------------------------------------
# Augmentation Transformations
# -------------------------------------------------------------------------

def apply_color_jitter(img, mask):
    """Photometric: Modifies pixel values, leaves mask untouched."""
    hsv = Image.fromarray(img).convert("HSV")
    h, s, v = hsv.split()
    v = v.point(lambda p: min(255, int(p * 1.25 + 15)))
    s = s.point(lambda p: min(255, int(p * 1.35)))
    aug_img = np.array(Image.merge("HSV", (h, s, v)).convert("RGB"))
    return aug_img, mask.copy()


def apply_gaussian_blur(img, mask):
    """Photometric: Blurs image, leaves mask untouched."""
    pil_img = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=2.5))
    aug_img = np.array(pil_img)
    return aug_img, mask.copy()


def apply_horizontal_flip(img, mask, flip_mask=True):
    """Spatial: Flips image horizontally. flip_mask controls synchronization."""
    aug_img = np.fliplr(img)
    aug_mask = np.fliplr(mask) if flip_mask else mask.copy()
    return aug_img, aug_mask


def apply_random_crop(img, mask, crop_box=(35, 75, 265, 305), crop_mask=True):
    """Spatial: Crops region and resizes back. crop_mask controls synchronization."""
    ymin, xmin, ymax, xmax = crop_box
    h, w, _ = img.shape
    
    cropped_img = img[ymin:ymax, xmin:xmax]
    aug_img = np.array(Image.fromarray(cropped_img).resize((w, h), Image.Resampling.BILINEAR))
    
    if crop_mask:
        cropped_mask = mask[ymin:ymax, xmin:xmax]
        # CRITICAL: Nearest neighbor interpolation for segmentation masks
        aug_mask = np.array(Image.fromarray(cropped_mask).resize((w, h), Image.Resampling.NEAREST))
    else:
        aug_mask = mask.copy()
        
    return aug_img, aug_mask


# -------------------------------------------------------------------------
# Figure Generator
# -------------------------------------------------------------------------

def create_segmentation_augmentation_figure(output_path="figures/06_segmentation_joint_augmentations.png"):
    set_plot_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    base_img, base_mask = create_segmentation_scene((320, 320))
    
    # Transformations list:
    transforms = [
        (
            "Original Reference",
            "Pair ($X, Y$)",
            "#263238",
            base_img,
            base_mask
        ),
        (
            "Color Jitter",
            "Photometric (Mask $Y$ Unchanged ✓)",
            "#0d47a1",
            *apply_color_jitter(base_img, base_mask)
        ),
        (
            "Gaussian Blur",
            "Photometric (Mask $Y$ Unchanged ✓)",
            "#0d47a1",
            *apply_gaussian_blur(base_img, base_mask)
        ),
        (
            "Horizontal Flip",
            "Spatial (Co-Transformed ✓)",
            "#1b5e20",
            *apply_horizontal_flip(base_img, base_mask, flip_mask=True)
        ),
        (
            "Horizontal Flip",
            "BUG (Mask NOT Flipped ✗)",
            "#b71c1c",
            *apply_horizontal_flip(base_img, base_mask, flip_mask=False)
        ),
        (
            "Random Crop",
            "Spatial (Co-Transformed ✓)",
            "#1b5e20",
            *apply_random_crop(base_img, base_mask, crop_mask=True)
        ),
        (
            "Random Crop",
            "BUG (Mask NOT Cropped ✗)",
            "#b71c1c",
            *apply_random_crop(base_img, base_mask, crop_mask=False)
        ),
    ]
    
    num_cols = len(transforms)
    fig, axes = plt.subplots(
        2, num_cols,
        figsize=(20, 6.8),
        gridspec_kw={"hspace": 0.16, "wspace": 0.18, "left": 0.06, "right": 0.98, "top": 0.88, "bottom": 0.12}
    )
    
    # Row labels on the left
    fig.text(0.025, 0.70, "INPUT\nIMAGE ($X$)", fontsize=11, fontweight="bold", ha="center", va="center", color="#37474f")
    fig.text(0.025, 0.28, "TARGET\nMASK ($Y$)", fontsize=11, fontweight="bold", ha="center", va="center", color="#37474f")

    for col_idx, (title, subtitle, color, img_t, mask_t) in enumerate(transforms):
        # Top row: Image
        ax_img = axes[0, col_idx]
        ax_img.imshow(img_t)
        ax_img.set_title(f"{title}\n{subtitle}", color=color, fontsize=9.5, pad=6)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        
        # Bottom row: Discrete Class Mask
        ax_mask = axes[1, col_idx]
        ax_mask.imshow(render_mask_rgb(mask_t))
        ax_mask.set_xticks([])
        ax_mask.set_yticks([])
        
        # Border styling
        for ax in [ax_img, ax_mask]:
            for spine in ax.spines.values():
                if color == "#b71c1c":
                    spine.set_edgecolor("#e53935")
                    spine.set_linewidth(2.6)
                elif color == "#1b5e20":
                    spine.set_edgecolor("#43a047")
                    spine.set_linewidth(2.2)
                elif color == "#0d47a1":
                    spine.set_edgecolor("#1e88e5")
                    spine.set_linewidth(1.8)
                else:
                    spine.set_edgecolor("#90a4ae")
                    spine.set_linewidth(1.2)

    # Class Color Legend along the bottom
    classes = [
        ("Background", [38, 50, 56]),
        ("Ground", [76, 175, 80]),
        ("Vehicle", [229, 57, 53]),
        ("Pedestrian", [30, 136, 229]),
        ("Tree", [255, 160, 0]),
    ]
    
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label=name,
                   markerfacecolor=np.array(col)/255.0, markersize=12)
        for name, col in classes
    ]
    
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=5,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cfd8dc",
        fontsize=10.5,
        bbox_to_anchor=(0.52, 0.01)
    )

    fig.suptitle(
        "Data Augmentations in Semantic Segmentation: Separated Image ($X$) & Ground Truth Mask ($Y$)",
        fontsize=14.5,
        fontweight="bold",
        y=0.96
    )
    
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[✓] Saved publication figure to: {output_path}")


if __name__ == "__main__":
    out_file = "/Users/jorgvt/Developer/CVMIAX/experiments/02_data_augmentation_guide/figures/06_segmentation_joint_augmentations.png"
    create_segmentation_augmentation_figure(out_file)
