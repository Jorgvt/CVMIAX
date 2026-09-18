"""
Feature Extraction Module for Traditional Computer Vision Pipeline.

Implements classic hand-crafted visual feature descriptors:
1. Histogram of Oriented Gradients (HOG) (Dalal & Triggs, 2005)
2. Local Binary Patterns (LBP) (Ojala et al., 2002)
3. Hu Moment Invariants (Hu, 1962)
4. Raw Pixel Intensity Baseline
"""

from typing import Tuple, Optional, Union
import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern


def compute_gradients(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute horizontal and vertical gradients, magnitude, and orientation.
    
    Formulation:
    ------------
    G_x(x, y) = I(x + 1, y) - I(x - 1, y)  (Sobel or 1D central difference)
    G_y(x, y) = I(x, y + 1) - I(x, y - 1)
    
    Magnitude:
    M(x, y) = sqrt(G_x^2 + G_y^2)
    
    Unsigned Orientation (0 to 180 degrees):
    theta(x, y) = arctan2(G_y, G_x) mod 180
    
    Parameters
    ----------
    img : np.ndarray
        2D grayscale image (H, W).
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (Gx, Gy, magnitude, orientation_degrees)
    """
    img_f = img.astype(np.float32)
    # 1D Sobel / derivative filters
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=1)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    
    return gx, gy, magnitude, orientation


def extract_hog_features(
    img: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (7, 7),
    cells_per_block: Tuple[int, int] = (2, 2),
    visualize: bool = False,
    block_norm: str = 'L2-Hys'
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract Histogram of Oriented Gradients (HOG) descriptor.
    
    HOG captures local object appearance and edge directions through 
    the distribution of local intensity gradients.
    
    Parameters
    ----------
    img : np.ndarray
        2D grayscale image (H, W).
    orientations : int
        Number of gradient orientation bins (default 9).
    pixels_per_cell : Tuple[int, int]
        Spatial size of each cell in pixels (default (7, 7)).
    cells_per_block : Tuple[int, int]
        Number of cells per normalization block (default (2, 2)).
    visualize : bool
        If True, return also the HOG visualization image.
    block_norm : str
        Block normalization scheme ('L2-Hys' or 'L2').
        
    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        1D feature vector of shape (D,), or (features, hog_image) if visualize=True.
    """
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-7)
    
    if visualize:
        features, hog_img = hog(
            img_norm,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm=block_norm,
            visualize=True,
            feature_vector=True
        )
        return features.astype(np.float32), hog_img
    else:
        features = hog(
            img_norm,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm=block_norm,
            visualize=False,
            feature_vector=True
        )
        return features.astype(np.float32)


def extract_lbp_features(
    img: np.ndarray,
    num_points: int = 8,
    radius: int = 1,
    method: str = 'uniform',
    visualize: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract Local Binary Patterns (LBP) texture descriptor.
    
    Mathematical Formulation:
    -------------------------
    LBP_{P, R}(x_c, y_c) = sum_{p=0}^{P-1} s(g_p - g_c) * 2^p
    where s(x) = 1 if x >= 0 else 0.
    
    Parameters
    ----------
    img : np.ndarray
        2D grayscale image.
    num_points : int
        Number of circularly symmetric neighbor points P (default 8).
    radius : int
        Radius of circle R (default 1).
    method : str
        LBP method ('uniform', 'default', 'ror').
    visualize : bool
        Whether to return the 2D LBP pattern representation.
        
    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        Normalized LBP histogram (1D feature vector) or (hist, lbp_image).
    """
    if img.dtype != np.uint8:
        u8_img = (cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
    else:
        u8_img = img
        
    lbp_map = local_binary_pattern(u8_img, P=num_points, R=radius, method=method)
    
    # For 'uniform' method, number of bins is P + 2
    n_bins = num_points + 2 if method == 'uniform' else int(2**num_points)
    hist, _ = np.histogram(lbp_map.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    hist = hist.astype(np.float32)
    
    if visualize:
        return hist, lbp_map
    return hist


def extract_hu_moments(img: np.ndarray) -> np.ndarray:
    """
    Extract 7 Hu Moment Invariants (invariant to translation, scale, and rotation).
    
    Log-transformed for numerical stability:
    h_i' = -sign(h_i) * log10(|h_i|)
    
    Parameters
    ----------
    img : np.ndarray
        2D grayscale image.
        
    Returns
    -------
    np.ndarray
        1D feature vector of shape (7,).
    """
    moments = cv2.moments(img.astype(np.float32))
    hu = cv2.HuMoments(moments).flatten()
    
    # Log transform to compress exponential scale
    log_hu = np.zeros_like(hu)
    for i in range(len(hu)):
        val = hu[i]
        if abs(val) > 1e-12:
            log_hu[i] = -1.0 * np.sign(val) * np.log10(np.abs(val))
        else:
            log_hu[i] = 0.0
    return log_hu.astype(np.float32)


def extract_raw_pixels(img: np.ndarray) -> np.ndarray:
    """
    Baseline raw flattened pixel intensities normalized to [0, 1].
    """
    return img.astype(np.float32).ravel()


def extract_features_dataset(
    images: np.ndarray,
    feature_type: str = 'hog',
    **kwargs
) -> np.ndarray:
    """
    Extract feature vectors across an entire dataset of images.
    
    Parameters
    ----------
    images : np.ndarray
        Array of images with shape (N, H, W).
    feature_type : str
        One of 'hog', 'lbp', 'hu', 'raw', or 'hog+lbp'.
        
    Returns
    -------
    np.ndarray
        2D matrix of shape (N, D) where D is feature dimension.
    """
    n_samples = len(images)
    features_list = []
    
    for i in range(n_samples):
        img = images[i]
        if feature_type == 'hog':
            feat = extract_hog_features(img, **kwargs)
        elif feature_type == 'lbp':
            feat = extract_lbp_features(img, **kwargs)
        elif feature_type == 'hu':
            feat = extract_hu_moments(img)
        elif feature_type == 'raw':
            feat = extract_raw_pixels(img)
        elif feature_type == 'hog+lbp':
            f_hog = extract_hog_features(img)
            f_lbp = extract_lbp_features(img)
            feat = np.concatenate([f_hog, f_lbp])
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
            
        features_list.append(feat)
        
    return np.array(features_list, dtype=np.float32)
