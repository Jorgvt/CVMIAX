"""
Data Augmentation Visualizer for Deep Learning Computer Vision Course.
Generates publication-quality, highly legible slide figures for:
1. Good default augmentations (Crops, Flips, Color Jitter, Mixup/CutMix, Mask/Box Augmentations)
2. Augmentations that alter labels / failure modes (Text/Anatomy flips, Missed Small Objects, Diagnostic Color Shifts, Extreme Distortion)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import keras
from keras import layers


def set_plot_style():
    plt.rcParams.update({
        "font.sans-serif": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "figure.dpi": 300,
    })


# -------------------------------------------------------------------------
# Synthetic & Procedural Image Generators for Clear Pedagogical Demonstration
# -------------------------------------------------------------------------

def create_natural_scene_image():
    """Creates a high-contrast pedagogical image of a scene with a dog and landscape."""
    img = Image.new("RGB", (300, 300), color=(135, 206, 235)) # Sky blue
    draw = ImageDraw.Draw(img)
    
    # Sun
    draw.ellipse([220, 30, 270, 80], fill=(255, 220, 50))
    
    # Hills/Grass
    draw.ellipse([-50, 180, 350, 450], fill=(34, 139, 34))
    
    # Dog body (brown)
    draw.ellipse([80, 140, 210, 230], fill=(160, 82, 45))
    # Dog head
    draw.ellipse([170, 100, 240, 170], fill=(160, 82, 45))
    # Dog snout
    draw.ellipse([210, 130, 255, 165], fill=(210, 140, 90))
    # Nose
    draw.ellipse([245, 140, 255, 150], fill=(20, 20, 20))
    # Eye
    draw.ellipse([200, 120, 210, 130], fill=(20, 20, 20))
    # Ear
    draw.polygon([(180, 90), (160, 140), (195, 120)], fill=(110, 50, 25))
    # Legs
    draw.rectangle([100, 210, 120, 270], fill=(140, 70, 35))
    draw.rectangle([170, 210, 190, 270], fill=(140, 70, 35))
    # Tail
    draw.line([(85, 160), (50, 130)], fill=(140, 70, 35), width=8)
    
    return np.array(img)


def create_cat_image():
    """Creates a distinct cat image for Mixup/CutMix demonstrations."""
    img = Image.new("RGB", (300, 300), color=(240, 220, 200)) # Warm indoor background
    draw = ImageDraw.Draw(img)
    
    # Cat body (grey)
    draw.ellipse([70, 130, 230, 250], fill=(128, 128, 128))
    # Cat head
    draw.ellipse([100, 70, 200, 170], fill=(128, 128, 128))
    # Cat ears
    draw.polygon([(105, 85), (120, 30), (145, 75)], fill=(90, 90, 90))
    draw.polygon([(155, 75), (180, 30), (195, 85)], fill=(90, 90, 90))
    # Eyes (Green)
    draw.ellipse([120, 105, 140, 125], fill=(50, 205, 50))
    draw.ellipse([160, 105, 180, 125], fill=(50, 205, 50))
    # Pupils
    draw.ellipse([128, 107, 132, 123], fill=(0, 0, 0))
    draw.ellipse([168, 107, 172, 123], fill=(0, 0, 0))
    # Nose & mouth
    draw.polygon([(145, 130), (155, 130), (150, 138)], fill=(255, 150, 150))
    draw.line([(150, 138), (140, 146)], fill=(0, 0, 0), width=2)
    draw.line([(150, 138), (160, 146)], fill=(0, 0, 0), width=2)
    # Whiskers
    draw.line([(110, 135), (80, 130)], fill=(50, 50, 50), width=2)
    draw.line([(110, 142), (80, 145)], fill=(50, 50, 50), width=2)
    draw.line([(190, 135), (220, 130)], fill=(50, 50, 50), width=2)
    draw.line([(190, 142), (220, 145)], fill=(50, 50, 50), width=2)
    
    return np.array(img)


# -------------------------------------------------------------------------
# Part 1: Good Defaults
# -------------------------------------------------------------------------

def generate_good_defaults(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_img = create_natural_scene_image()
    cat_img = create_cat_image()

    # 1. Random Resized Crops
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(base_img)
    axes[0].set_title("Original Image (Full Field of View)")
    axes[0].axis("off")

    crop_boxes = [
        [0.1, 0.2, 0.9, 0.9],   # scale crop
        [0.3, 0.4, 0.85, 0.85], # close-up dog head/body
        [0.2, 0.1, 0.8, 0.6],   # landscape-heavy crop
    ]
    crop_layer = layers.RandomCrop(200, 200)

    for i, (ax, (ymin, xmin, ymax, xmax)) in enumerate(zip(axes[1:], crop_boxes), 1):
        h, w, _ = base_img.shape
        cropped = base_img[int(ymin*h):int(ymax*h), int(xmin*w):int(xmax*w)]
        resized = np.array(Image.fromarray(cropped).resize((300, 300), Image.Resampling.BILINEAR))
        ax.imshow(resized)
        ax.set_title(f"Random Resized Crop #{i}\n(Scale & aspect ratio invariance)")
        ax.axis("off")

    plt.suptitle("Good Default 1: Random Resized Crops", fontsize=15, fontweight="bold", y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_random_resized_crops.png"), bbox_inches="tight")
    plt.close(fig)

    # 2. Horizontal Flips
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(base_img)
    axes[0].set_title("Original Image (Label: 'Dog')", color="black")
    axes[0].axis("off")

    flipped = np.fliplr(base_img)
    axes[1].imshow(flipped)
    axes[1].set_title("Horizontal Flip (Label: 'Dog' preserved ✓)\nNatural scene semantics are left-right invariant", color="green")
    axes[1].axis("off")

    plt.suptitle("Good Default 2: Horizontal Flips", fontsize=15, fontweight="bold", y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_horizontal_flips.png"), bbox_inches="tight")
    plt.close(fig)

    # 3. Mild Color Changes (Brightness / Contrast / Saturation)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(base_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Brighter
    bright = np.clip(base_img.astype(np.float32) * 1.25 + 15, 0, 255).astype(np.uint8)
    axes[1].imshow(bright)
    axes[1].set_title("Mild Brightness Boost\n(Sunny condition simulation)")
    axes[1].axis("off")

    # Darker / Overcast
    dark = np.clip(base_img.astype(np.float32) * 0.75 - 10, 0, 255).astype(np.uint8)
    axes[2].imshow(dark)
    axes[2].set_title("Mild Brightness Reduction\n(Shadow/Overcast simulation)")
    axes[2].axis("off")

    # Higher Contrast
    mean = np.mean(base_img, axis=(0, 1), keepdims=True)
    contrast = np.clip((base_img - mean) * 1.35 + mean, 0, 255).astype(np.uint8)
    axes[3].imshow(contrast)
    axes[3].set_title("Mild Contrast Variation\n(Camera sensor variation)")
    axes[3].axis("off")

    plt.suptitle("Good Default 3: Mild Color Transformations for Natural Images", fontsize=15, fontweight="bold", y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_mild_color_jitter.png"), bbox_inches="tight")
    plt.close(fig)

    # 4. Mixup & CutMix (Severe Overfitting Regularization)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    axes[0].imshow(base_img)
    axes[0].set_title("Image A: Dog (100%)")
    axes[0].axis("off")

    axes[1].imshow(cat_img)
    axes[1].set_title("Image B: Cat (100%)")
    axes[1].axis("off")

    # Mixup
    lam = 0.6
    mixup_img = (lam * base_img.astype(np.float32) + (1 - lam) * cat_img.astype(np.float32)).astype(np.uint8)
    axes[2].imshow(mixup_img)
    axes[2].set_title(f"Mixup ($\\lambda={lam}$)\nLabel: {int(lam*100)}% Dog + {int((1-lam)*100)}% Cat", color="navy")
    axes[2].axis("off")

    # CutMix
    cutmix_img = base_img.copy()
    # Paste rectangular patch from cat
    cutmix_img[80:220, 80:220] = cat_img[80:220, 80:220]
    cutmix_ratio = 1.0 - ((140 * 140) / (300 * 300))
    axes[3].imshow(cutmix_img)
    axes[3].set_title(f"CutMix (Area Ratio $\\approx {cutmix_ratio:.2f}$)\nLabel: {int(cutmix_ratio*100)}% Dog + {int((1-cutmix_ratio)*100)}% Cat", color="navy")
    axes[3].axis("off")

    plt.suptitle("Good Default 4: Regularization via Mixup & CutMix", fontsize=15, fontweight="bold", y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_mixup_and_cutmix.png"), bbox_inches="tight")
    plt.close(fig)

    # 5. Scale & Geometric for Detection & Segmentation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    
    # Draw original with Bounding Box and Mask overlay
    ax1.imshow(base_img)
    rect1 = patches.Rectangle((75, 95), 180, 180, linewidth=2.5, edgecolor="red", facecolor="none")
    ax1.add_patch(rect1)
    ax1.text(80, 85, "Dog: 0.98", color="red", fontsize=11, fontweight="bold", backgroundcolor="white")
    # Segmentation polygon
    poly1 = patches.Polygon([[80, 150], [170, 100], [250, 140], [210, 230], [100, 270]], closed=True,
                            linewidth=2, edgecolor="yellow", facecolor=(1, 1, 0, 0.35))
    ax1.add_patch(poly1)
    ax1.set_title("Original Image with Annotations\n(Bounding Box & Segmentation Mask)")
    ax1.axis("off")

    # Geometric Augmentation: Scaled and Rotated image + synchronously rotated box & mask
    rotated_img = np.array(Image.fromarray(base_img).rotate(18, resample=Image.Resampling.BILINEAR))
    ax2.imshow(rotated_img)
    rect2 = patches.Rectangle((65, 80), 200, 200, linewidth=2.5, edgecolor="red", facecolor="none")
    ax2.add_patch(rect2)
    ax2.text(70, 70, "Dog: 0.98", color="red", fontsize=11, fontweight="bold", backgroundcolor="white")
    poly2 = patches.Polygon([[70, 140], [165, 85], [260, 150], [210, 250], [90, 270]], closed=True,
                            linewidth=2, edgecolor="yellow", facecolor=(1, 1, 0, 0.35))
    ax2.add_patch(poly2)
    ax2.set_title("Scale & Geometric Augmentation\n(Image & Annotation Labels Synchronized ✓)", color="green")
    ax2.axis("off")

    plt.suptitle("Good Default 5: Scale & Geometric Augmentation for Detection / Segmentation", fontsize=15, fontweight="bold", y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "05_geometric_detection_segmentation.png"), bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------
# Part 2: Avoid Augmentations that Alter the Label (Failure Modes)
# -------------------------------------------------------------------------

def generate_harmful_augmentations(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Horizontal Flips for Text / Digits & Asymmetric Anatomy
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    # Text / Digit '6' -> '9' / 'b' -> 'd'
    digit_img = Image.new("RGB", (250, 250), color=(255, 255, 255))
    draw_d = ImageDraw.Draw(digit_img)
    # Draw letter 'b'
    draw_d.rectangle([50, 40, 75, 210], fill=(0, 0, 0)) # stem
    draw_d.ellipse([50, 100, 190, 210], fill=(0, 0, 0)) # loop
    draw_d.ellipse([75, 125, 165, 185], fill=(255, 255, 255)) # inner hole

    axes[0].imshow(np.array(digit_img))
    axes[0].set_title("Original: Character 'b'\n(Ground Truth Label: 'b')", color="black")
    axes[0].axis("off")

    flipped_digit = np.fliplr(np.array(digit_img))
    axes[1].imshow(flipped_digit)
    axes[1].set_title("Horizontal Flip: Character 'd' ✗\n(Label Still 'b' $\\rightarrow$ ERROR: Label Altered!)", color="red")
    axes[1].axis("off")

    # Medical Anatomy: Chest X-ray / Asymmetric Organ (Heart on Left side)
    xray_img = Image.new("RGB", (250, 250), color=(10, 10, 10))
    draw_x = ImageDraw.Draw(xray_img)
    # Ribcage outline (grey)
    draw_x.ellipse([30, 20, 220, 230], outline=(160, 160, 160), width=4)
    # Spine (center)
    draw_x.rectangle([115, 20, 135, 230], fill=(120, 120, 120))
    # Heart silhouette (normally on patient's left -> image right side)
    draw_x.ellipse([110, 110, 180, 180], fill=(210, 210, 210))
    draw_x.text((15, 20), "R", fill=(255, 255, 0))
    draw_x.text((225, 20), "L", fill=(255, 255, 0))

    axes[2].imshow(np.array(xray_img))
    axes[2].set_title("Normal Anatomy (Levocardia)\n(Heart on Anatomical Left)", color="black")
    axes[2].axis("off")

    flipped_xray = np.fliplr(np.array(xray_img))
    axes[3].imshow(flipped_xray)
    axes[3].set_title("Horizontal Flip: Situs Inversus ✗\n(Simulates rare cardiac condition!)", color="red")
    axes[3].axis("off")

    plt.suptitle("Avoid: Horizontal Flips on Direction-Sensitive Text or Asymmetric Anatomy", fontsize=15, fontweight="bold", y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_harmful_horizontal_flips.png"), bbox_inches="tight")
    plt.close(fig)

    # 2. Aggressive Crops when Small Objects Matter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    
    # Create wide scene with tiny traffic sign or small bird
    scene = Image.new("RGB", (360, 240), color=(135, 206, 235))
    draw_s = ImageDraw.Draw(scene)
    draw_s.rectangle([0, 160, 360, 240], fill=(80, 80, 80)) # Road
    draw_s.rectangle([0, 120, 360, 160], fill=(34, 139, 34)) # Grass
    # Distant small Stop sign (only 14x14 pixels)
    draw_s.line([(290, 120), (290, 155)], fill=(150, 150, 150), width=2) # Pole
    draw_s.polygon([(284, 110), (296, 110), (301, 115), (301, 125), (296, 130), (284, 130), (279, 125), (279, 115)], fill=(220, 20, 20)) # Octagon
    
    scene_arr = np.array(scene)
    ax1.imshow(scene_arr)
    # Box showing the tiny target object
    rect_obj = patches.Rectangle((275, 105), 30, 52, linewidth=2, edgecolor="red", facecolor="none")
    ax1.add_patch(rect_obj)
    ax1.text(250, 95, "Target: Stop Sign", color="red", fontsize=10, fontweight="bold", backgroundcolor="white")
    # Box showing aggressive crop region
    rect_crop = patches.Rectangle((20, 30), 200, 180, linewidth=2.5, edgecolor="magenta", linestyle="--", facecolor="none")
    ax1.add_patch(rect_crop)
    ax1.text(25, 20, "Aggressive Crop Window", color="magenta", fontsize=10, fontweight="bold", backgroundcolor="white")
    ax1.set_title("Original Image (Label: 'Stop Sign Present')")
    ax1.axis("off")

    # Aggressive crop that misses the sign completely
    cropped_scene = scene_arr[30:210, 20:220]
    ax2.imshow(cropped_scene)
    ax2.set_title("Aggressive Crop Result ✗\n(Target object is MISSING, but label is still 'Stop Sign'!)", color="red")
    ax2.axis("off")

    plt.suptitle("Avoid: Aggressive Crops when Small Target Objects Matter", fontsize=15, fontweight="bold", y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_harmful_aggressive_crops.png"), bbox_inches="tight")
    plt.close(fig)

    # 3. Color Transformations when Color is Diagnostically Meaningful
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    # Traffic Light: Red (Stop) vs Green (Go)
    tl_img = Image.new("RGB", (160, 240), color=(40, 40, 40))
    draw_tl = ImageDraw.Draw(tl_img)
    draw_tl.rounded_rectangle([30, 20, 130, 220], radius=15, fill=(20, 20, 20), outline=(80, 80, 80), width=3)
    # Red light glowing, yellow and green off
    draw_tl.ellipse([50, 35, 110, 85], fill=(255, 0, 0)) # Red ON
    draw_tl.ellipse([50, 95, 110, 145], fill=(70, 60, 10)) # Yellow OFF
    draw_tl.ellipse([50, 155, 110, 205], fill=(10, 50, 10)) # Green OFF

    axes[0].imshow(np.array(tl_img))
    axes[0].set_title("Original Traffic Light\n(Label: 'RED / STOP')", color="black")
    axes[0].axis("off")

    # Severe hue shift turning Red into Green!
    tl_arr = np.array(tl_img)
    hue_shifted_tl = tl_arr.copy()
    # Swap Red and Green channels
    hue_shifted_tl[:, :, 0] = tl_arr[:, :, 1]
    hue_shifted_tl[:, :, 1] = tl_arr[:, :, 0]
    axes[1].imshow(hue_shifted_tl)
    axes[1].set_title("Severe Hue Shift ✗\n(Red light turned Green, label still 'Stop'!)", color="red")
    axes[1].axis("off")

    # Medical Melanoma / Lesion (Color indicates malignancy)
    skin_img = Image.new("RGB", (200, 200), color=(245, 215, 190))
    draw_sk = ImageDraw.Draw(skin_img)
    # Irregular dark lesion with erythema (red ring)
    draw_sk.ellipse([60, 60, 140, 140], fill=(220, 90, 90)) # Red inflammation
    draw_sk.ellipse([70, 70, 130, 130], fill=(45, 25, 20)) # Dark melanin

    axes[2].imshow(np.array(skin_img))
    axes[2].set_title("Dermatology: Melanoma\n(Dark Pigmentation + Erythema)", color="black")
    axes[2].axis("off")

    # Inverted / Extreme color jitter
    extreme_color_skin = 255 - np.array(skin_img)
    axes[3].imshow(extreme_color_skin)
    axes[3].set_title("Extreme Color Inversion ✗\n(Diagnostic pigment clues destroyed!)", color="red")
    axes[3].axis("off")

    plt.suptitle("Avoid: Color Transformations when Color is Diagnostically / Semantically Critical", fontsize=15, fontweight="bold", y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_harmful_color_transformations.png"), bbox_inches="tight")
    plt.close(fig)

    # 4. Strong Geometric Distortion when Precise Localization is Required
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Precise Keypoints on Object / Facial Landmarks
    face = Image.new("RGB", (260, 260), color=(245, 230, 215))
    draw_f = ImageDraw.Draw(face)
    draw_f.ellipse([40, 30, 220, 230], outline=(150, 110, 80), width=3) # Face oval
    # Eyes
    draw_f.ellipse([75, 90, 115, 115], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    draw_f.ellipse([145, 90, 185, 115], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    # Nose & Mouth
    draw_f.line([(130, 115), (130, 150)], fill=(0, 0, 0), width=3)
    draw_f.arc([95, 160, 165, 200], 0, 180, fill=(180, 40, 40), width=4)

    face_arr = np.array(face)
    ax1.imshow(face_arr)
    # True keypoint landmarks
    keypoints = [(95, 102), (165, 102), (130, 150), (105, 180), (155, 180)]
    for kp in keypoints:
        ax1.plot(kp[0], kp[1], "ro", markersize=8)
    ax1.set_title("Ground Truth Landmark Keypoints\n(Exact coordinates required)")
    ax1.axis("off")

    # Severe non-rigid wave/elastic warping without proper non-linear keypoint transformation
    rows, cols, ch = face_arr.shape
    distorted = np.zeros_like(face_arr)
    for r in range(rows):
        for c in range(cols):
            offset_r = int(18.0 * np.sin(2 * np.pi * c / 70))
            offset_c = int(18.0 * np.cos(2 * np.pi * r / 70))
            orig_r = np.clip(r + offset_r, 0, rows - 1)
            orig_c = np.clip(c + offset_c, 0, cols - 1)
            distorted[r, c] = face_arr[orig_r, orig_c]

    ax2.imshow(distorted)
    # If keypoints were kept rigid / linearly transformed, they completely detach from the warped anatomy
    for kp in keypoints:
        ax2.plot(kp[0], kp[1], "rx", markersize=10, markeredgewidth=2.5)
    ax2.set_title("Strong Non-Rigid Elastic Distortion ✗\n(Keypoint coordinates severely unaligned with warped features!)", color="red")
    ax2.axis("off")

    plt.suptitle("Avoid: Strong Geometric Distortions when Sub-pixel / Keypoint Localization is Needed", fontsize=15, fontweight="bold", y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_harmful_geometric_distortions.png"), bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------
# Part 3: Composite Overview Slide Sheets
# -------------------------------------------------------------------------

def generate_composite_overview_posters(output_dir):
    """
    Creates comprehensive multi-panel overview figures perfectly formatted
    to be inserted directly as single lecture slides.
    """
    # Overview 1: Good Defaults
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Data Augmentation: Good Defaults & Best Practices", fontsize=18, fontweight="bold", y=0.98)

    base_img = create_natural_scene_image()
    cat_img = create_cat_image()

    # 1. Random Crop
    h, w, _ = base_img.shape
    cropped = base_img[int(0.2*h):int(0.85*h), int(0.3*w):int(0.9*w)]
    resized_crop = np.array(Image.fromarray(cropped).resize((300, 300), Image.Resampling.BILINEAR))
    axes[0, 0].imshow(resized_crop)
    axes[0, 0].set_title("1. Random Resized Crops\nEnforces scale & viewpoint invariance", fontsize=12)
    axes[0, 0].axis("off")

    # 2. Horizontal Flip
    axes[0, 1].imshow(np.fliplr(base_img))
    axes[0, 1].set_title("2. Horizontal Flips\nSafe when left/right orientation is invariant", fontsize=12)
    axes[0, 1].axis("off")

    # 3. Mild Color Changes
    mean = np.mean(base_img, axis=(0, 1), keepdims=True)
    color_var = np.clip((base_img - mean) * 1.25 + mean + 10, 0, 255).astype(np.uint8)
    axes[0, 2].imshow(color_var)
    axes[0, 2].set_title("3. Mild Color Jitter\nRobustness to lighting & sensor variations", fontsize=12)
    axes[0, 2].axis("off")

    # 4. Mixup
    mix = (0.6 * base_img.astype(np.float32) + 0.4 * cat_img.astype(np.float32)).astype(np.uint8)
    axes[1, 0].imshow(mix)
    axes[1, 0].set_title("4a. Mixup (Regularization)\nLinear label interpolation: 60% Dog + 40% Cat", fontsize=12)
    axes[1, 0].axis("off")

    # 5. CutMix
    cut = base_img.copy()
    cut[70:210, 70:210] = cat_img[70:210, 70:210]
    axes[1, 1].imshow(cut)
    axes[1, 1].set_title("4b. CutMix (Regularization)\nSpatial patch substitution with soft label", fontsize=12)
    axes[1, 1].axis("off")

    # 6. Detection/Segmentation
    rot = np.array(Image.fromarray(base_img).rotate(15, resample=Image.Resampling.BILINEAR))
    axes[1, 2].imshow(rot)
    rect = patches.Rectangle((70, 80), 190, 190, linewidth=2.5, edgecolor="red", facecolor="none")
    axes[1, 2].add_patch(rect)
    poly = patches.Polygon([[75, 145], [165, 90], [255, 150], [205, 245], [95, 265]], closed=True,
                           linewidth=2, edgecolor="yellow", facecolor=(1, 1, 0, 0.35))
    axes[1, 2].add_patch(poly)
    axes[1, 2].set_title("5. Synchronized Geometric Aug\nRotates image + BBoxes + Segmentation masks", fontsize=12)
    axes[1, 2].axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "slide_good_defaults_overview.png"), bbox_inches="tight")
    plt.close(fig)

    # Overview 2: Label-Altering Failure Modes
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Pitfalls: Avoid Augmentations that Alter the Ground Truth Label", fontsize=17, fontweight="bold", y=0.98)

    # Pitfall 1: Flips on Text/Anatomy
    digit_img = Image.new("RGB", (250, 250), color=(255, 255, 255))
    draw_d = ImageDraw.Draw(digit_img)
    draw_d.rectangle([50, 40, 75, 210], fill=(0, 0, 0))
    draw_d.ellipse([50, 100, 190, 210], fill=(0, 0, 0))
    draw_d.ellipse([75, 125, 165, 185], fill=(255, 255, 255))
    flipped_d = np.fliplr(np.array(digit_img))
    axes[0, 0].imshow(flipped_d)
    axes[0, 0].set_title("1. Horizontal Flip on Asymmetric / Text Data ✗\n'b' flipped becomes 'd' (Label Invalidation)", fontsize=12, color="darkred")
    axes[0, 0].axis("off")

    # Pitfall 2: Missing Small Object
    scene = Image.new("RGB", (360, 240), color=(135, 206, 235))
    draw_s = ImageDraw.Draw(scene)
    draw_s.rectangle([0, 160, 360, 240], fill=(80, 80, 80))
    draw_s.rectangle([0, 120, 360, 160], fill=(34, 139, 34))
    scene_arr = np.array(scene)
    cropped_scene = scene_arr[30:210, 20:220]
    axes[0, 1].imshow(cropped_scene)
    axes[0, 1].set_title("2. Aggressive Crop Missing Small Targets ✗\nTarget object cropped out entirely, label retained", fontsize=12, color="darkred")
    axes[0, 1].axis("off")

    # Pitfall 3: Diagnostic Color Distortion
    tl_img = Image.new("RGB", (160, 240), color=(40, 40, 40))
    draw_tl = ImageDraw.Draw(tl_img)
    draw_tl.rounded_rectangle([30, 20, 130, 220], radius=15, fill=(20, 20, 20), outline=(80, 80, 80), width=3)
    draw_tl.ellipse([50, 35, 110, 85], fill=(255, 0, 0))
    draw_tl.ellipse([50, 95, 110, 145], fill=(70, 60, 10))
    draw_tl.ellipse([50, 155, 110, 205], fill=(10, 50, 10))
    tl_arr = np.array(tl_img)
    hue_shift = tl_arr.copy()
    hue_shift[:, :, 0] = tl_arr[:, :, 1]
    hue_shift[:, :, 1] = tl_arr[:, :, 0]
    axes[1, 0].imshow(hue_shift)
    axes[1, 0].set_title("3. Color Distortion on Diagnostic Signals ✗\nRed Stop light shifted to Green (Safety Hazard)", fontsize=12, color="darkred")
    axes[1, 0].axis("off")

    # Pitfall 4: Non-rigid distortion on landmarks
    face = Image.new("RGB", (260, 260), color=(245, 230, 215))
    draw_f = ImageDraw.Draw(face)
    draw_f.ellipse([40, 30, 220, 230], outline=(150, 110, 80), width=3)
    draw_f.ellipse([75, 90, 115, 115], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    draw_f.ellipse([145, 90, 185, 115], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    draw_f.line([(130, 115), (130, 150)], fill=(0, 0, 0), width=3)
    draw_f.arc([95, 160, 165, 200], 0, 180, fill=(180, 40, 40), width=4)
    face_arr = np.array(face)
    rows, cols, _ = face_arr.shape
    distorted = np.zeros_like(face_arr)
    for r in range(rows):
        for c in range(cols):
            offset_r = int(16.0 * np.sin(2 * np.pi * c / 70))
            orig_r = np.clip(r + offset_r, 0, rows - 1)
            distorted[r, c] = face_arr[orig_r, c]
    axes[1, 1].imshow(distorted)
    keypoints = [(95, 102), (165, 102), (130, 150), (105, 180), (155, 180)]
    for kp in keypoints:
        axes[1, 1].plot(kp[0], kp[1], "rx", markersize=10, markeredgewidth=2.5)
    axes[1, 1].set_title("4. Strong Distortion on Fine Landmarks ✗\nCoordinate labels misaligned with warped structure", fontsize=12, color="darkred")
    axes[1, 1].axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "slide_pitfalls_overview.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    set_plot_style()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("Generating Good Defaults visual examples...")
    generate_good_defaults(figures_dir)

    print("Generating Label-Altering Pitfalls visual examples...")
    generate_harmful_augmentations(figures_dir)

    print("Generating Composite Lecture Slide Overview Posters...")
    generate_composite_overview_posters(figures_dir)

    print(f"\n[Success] All visual figures saved cleanly to: {figures_dir}")


if __name__ == "__main__":
    main()
