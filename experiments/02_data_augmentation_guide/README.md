# Experiment 02: Visual Guide to Data Augmentation (Good Defaults & Label-Preservation Pitfalls)

This module provides visual demonstrations and pedagogical examples for designing effective data augmentation pipelines in computer vision, contrasting **Good Defaults** against **Label-Altering Failure Modes**.

---

## 1. Good Defaults for Data Augmentation

Data augmentation is domain-dependent regularization that synthesizes new training data by applying label-preserving transformations.

### 1.1 Random Resized Crops
- **Principle**: Extracts crops of varying area (e.g. 50% to 100%) and aspect ratio (e.g. 3:4 to 4:3), then rescales them back to standard model input dimensions.
- **Why it works**: Forces the model to recognize semantic objects at various scales, viewpoints, and partial occlusions without overfitting to fixed spatial locations.
- **Figure**: [`01_random_resized_crops.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/02_data_augmentation_guide/figures/01_random_resized_crops.png)

```python
# Keras Implementation
data_augmentation = keras.Sequential([
    layers.RandomCrop(height=224, width=224),
    layers.Rescaling(1.0 / 255),
])
```

---

### 1.2 Horizontal Flips (When Left/Right Orientation is Irrelevant)
- **Principle**: Mirrored reflections along the vertical axis ($x \to W - x$).
- **Why it works**: In natural scenes (dogs, cats, cars, landscapes), semantic classes are symmetric under horizontal reflection. Effectively doubles effective sample diversity for free.
- **Figure**: [`02_horizontal_flips.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/02_data_augmentation_guide/figures/02_horizontal_flips.png)

```python
# Keras Implementation
layers.RandomFlip(mode="horizontal")
```

---

### 1.3 Mild Color Jitter for Natural-Image Classification
- **Principle**: Subtle perturbations in brightness, contrast, hue, and saturation ($\pm 10-25\%$).
- **Why it works**: Real-world camera sensors, white balance shifts, and ambient illumination (sunny vs cloudy vs indoor) vary widely. Mild perturbations prevent the network from memorizing rigid pixel intensities.
- **Figure**: [`03_mild_color_jitter.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/02_data_augmentation_guide/figures/03_mild_color_jitter.png)

```python
# Keras Implementation
layers.RandomBrightness(factor=0.2),
layers.RandomContrast(factor=0.2)
```

---

### 1.4 Mixup & CutMix (Regularization for Severe Overfitting)
- **Principle**:
  - **Mixup**: $\tilde{x} = \lambda x_i + (1-\lambda) x_j$, with targets $\tilde{y} = \lambda y_i + (1-\lambda) y_j$.
  - **CutMix**: Replaces a spatial bounding box in image $A$ with a patch from image $B$, weighting targets by the area ratio $\lambda = 1 - \frac{W_{\text{box}} H_{\text{box}}}{W H}$.
- **Why it works**: Softens hard decision boundaries, acts as an inductive bias for linearity, and eliminates over-confident calibration errors.
- **Figure**: [`04_mixup_and_cutmix.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/02_data_augmentation_guide/figures/04_mixup_and_cutmix.png)

---

### 1.5 Scale & Geometric Augmentation for Detection and Segmentation
- **Principle**: Affine transformations (rotation, shearing, translation, scaling) applied **synchronously** to both the input image and all target annotations (bounding boxes, segmentation masks, keypoints).
- **Figure**: [`05_geometric_detection_segmentation.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/02_data_augmentation_guide/figures/05_geometric_detection_segmentation.png)

---

## 2. Avoid Augmentations that Alter the Label (Failure Modes)

> **Golden Rule of Data Augmentation**: An augmentation is only valid if a human expert would assign the exact same ground-truth label to the augmented image.

### 2.1 Horizontal Flips for Text or Asymmetric Anatomy
- **Failure Mode**: Flipping the letter `'b'` produces `'d'`; flipping `'6'` produces a mirror invalid digit. In medical chest radiography, flipping mirrors the heart from normal *levocardia* (left) into *dextrocardia / situs inversus*, creating severe clinical misdiagnosis.
- **Figure**: [`01_harmful_horizontal_flips.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/02_data_augmentation_guide/figures/01_harmful_horizontal_flips.png)

---

### 2.2 Aggressive Crops when Small Objects Matter
- **Failure Mode**: When detecting or classifying small objects (e.g. distant traffic signs, small birds, microcalcifications in mammography), aggressive cropping frequently cuts the target object out entirely, leaving an empty background with a positive ground-truth label.
- **Figure**: [`02_harmful_aggressive_crops.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/02_data_augmentation_guide/figures/02_harmful_aggressive_crops.png)

---

### 2.3 Color Transformations when Color is Diagnostically Meaningful
- **Failure Mode**: In traffic control, hue-shifting turns a **Red** stop light into **Green** go light while retaining the label `'Stop'`. In dermatology / histology, color signifies melanin density, oxygenation, and tissue stain absorption; corrupting color destroys diagnostic validity.
- **Figure**: [`03_harmful_color_transformations.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/02_data_augmentation_guide/figures/03_harmful_color_transformations.png)

---

### 2.4 Strong Geometric Distortion for Precise Localization
- **Failure Mode**: Non-rigid elastic distortions or excessive shearing warp fine anatomical boundaries and facial landmark keypoints, detaching annotations from real physical geometry.
- **Figure**: [`04_harmful_geometric_distortions.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/02_data_augmentation_guide/figures/04_harmful_geometric_distortions.png)

---

## 3. Lecture Slide Ready Posters

- **All Good Defaults on a Single Slide**: [`slide_good_defaults_overview.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/02_data_augmentation_guide/figures/slide_good_defaults_overview.png)
- **All Failure Modes / Pitfalls on a Single Slide**: [`slide_pitfalls_overview.png`](file:///Users/jorgvt/Developer/CVMIAX/experiments/02_data_augmentation_guide/figures/slide_pitfalls_overview.png)

---

## 4. Generating the Visual Assets

```bash
uv run python experiments/02_data_augmentation_guide/generate_visualizations.py
```
