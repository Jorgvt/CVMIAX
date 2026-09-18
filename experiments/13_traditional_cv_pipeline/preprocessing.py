"""
Image Preprocessing Module for Traditional Computer Vision Pipeline.

This module provides essential classical preprocessing techniques:
1. Deskewing via Central Image Moments (affine shear correction).
2. Gaussian Noise Filtering (spatial convolution smoothing).
3. Contrast Normalization and Otsu Thresholding.
"""

from typing import Tuple, Optional, Union
import numpy as np
import cv2


def compute_image_moments(img: np.ndarray) -> dict:
    """
    Compute spatial and central image moments for a 2D grayscale image.
    
    Parameters
    ----------
    img : np.ndarray
        2D grayscale image (H, W).
        
    Returns
    -------
    dict
        Dictionary of raw spatial moments (m00, m10, m01) and 
        central moments (mu20, mu11, mu02).
    """
    moments = cv2.moments(img)
    return moments


def deskew_image(img: np.ndarray, threshold: float = 1e-2) -> np.ndarray:
    """
    Deskew a grayscale digit or shape using second-order central image moments.
    
    Mathematical Formulation:
    -------------------------
    1. Raw moments:
       m_pq = sum_{x, y} x^p y^q I(x, y)
    2. Centroid (center of mass):
       x_bar = m_10 / m_00,  y_bar = m_01 / m_00
    3. Central moments:
       mu_pq = sum_{x, y} (x - x_bar)^p (y - y_bar)^q I(x, y)
    4. Skew angle tangent (covariance ratio):
       alpha = mu_11 / mu_02
    5. Affine shear transformation matrix:
       M = [[1, alpha, -0.5 * W * alpha],
            [0,     1,                 0]]
            
    Parameters
    ----------
    img : np.ndarray
        2D grayscale image uint8 or float in [0, 255] or [0, 1].
    threshold : float
        Minimum variance threshold to avoid division by zero on blank images.
        
    Returns
    -------
    np.ndarray
        Deskewed image with upright alignment.
    """
    # Ensure float32 for computation
    src = img.astype(np.float32)
    h, w = src.shape[:2]
    
    moments = cv2.moments(src)
    mu02 = moments.get('mu02', 0.0)
    mu11 = moments.get('mu11', 0.0)
    
    if abs(mu02) < threshold:
        return src
    
    # Skew orientation slope
    skew = mu11 / mu02
    
    # Affine transformation matrix for horizontal shear
    # We shear x relative to the vertical center h / 2
    M = np.float32([
        [1.0, skew, -0.5 * h * skew],
        [0.0, 1.0, 0.0]
    ])
    
    # Apply warp affine with border replicate/constant zero
    deskewed = cv2.warpAffine(
        src, M, (w, h), 
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return deskewed


def gaussian_filter(img: np.ndarray, kernel_size: int = 3, sigma: float = 0.5) -> np.ndarray:
    """
    Apply 2D Gaussian smoothing filter to suppress high-frequency sensor noise.
    
    Kernel: G(x, y) = 1 / (2*pi*sigma^2) * exp(-(x^2 + y^2) / (2*sigma^2))
    
    Parameters
    ----------
    img : np.ndarray
        2D image array.
    kernel_size : int
        Odd integer size of the kernel (default 3).
    sigma : float
        Standard deviation of Gaussian kernel (default 0.5).
        
    Returns
    -------
    np.ndarray
        Smoothed image.
    """
    if kernel_size <= 1:
        return img
    return cv2.GaussianBlur(img.astype(np.float32), (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)


def otsu_threshold(img: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Apply Otsu's adaptive global thresholding to binarize an image.
    
    Minimizes intra-class variance:
    sigma_w^2(t) = q1(t) * sigma_1^2(t) + q2(t) * sigma_2^2(t)
    
    Parameters
    ----------
    img : np.ndarray
        Grayscale image (uint8 or float).
        
    Returns
    -------
    Tuple[float, np.ndarray]
        (optimal_threshold, binary_mask)
    """
    # Scale to uint8 [0, 255]
    if img.dtype != np.uint8:
        norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        u8_img = norm.astype(np.uint8)
    else:
        u8_img = img
        
    thresh_val, binary = cv2.threshold(u8_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh_val, binary


def normalize_intensity(img: np.ndarray) -> np.ndarray:
    """
    Normalize image intensities to standard range [0, 1].
    """
    img_f = img.astype(np.float32)
    max_val = np.max(img_f)
    if max_val > 1.0:
        img_f /= 255.0
    return np.clip(img_f, 0.0, 1.0)


def preprocess_single_image(
    img: np.ndarray, 
    deskew: bool = True, 
    blur: bool = True, 
    kernel_size: int = 3, 
    sigma: float = 0.5,
    normalize: bool = True
) -> np.ndarray:
    """
    Apply the complete classical preprocessing pipeline to a single image.
    
    Parameters
    ----------
    img : np.ndarray
        Input 2D grayscale image (H, W).
    deskew : bool
        Whether to perform moments-based deskewing.
    blur : bool
        Whether to apply Gaussian noise reduction.
    kernel_size : int
        Gaussian filter kernel size.
    sigma : float
        Gaussian standard deviation.
    normalize : bool
        Whether to normalize intensity to [0, 1].
        
    Returns
    -------
    np.ndarray
        Preprocessed image.
    """
    res = img.copy()
    if normalize:
        res = normalize_intensity(res)
    if deskew:
        res = deskew_image(res)
    if blur:
        res = gaussian_filter(res, kernel_size=kernel_size, sigma=sigma)
    if normalize:
        res = normalize_intensity(res)
    return res


def preprocess_batch(
    images: np.ndarray,
    deskew: bool = True,
    blur: bool = True,
    kernel_size: int = 3,
    sigma: float = 0.5,
    normalize: bool = True
) -> np.ndarray:
    """
    Vectorized batch preprocessing across a 3D array of images (N, H, W).
    
    Parameters
    ----------
    images : np.ndarray
        Input array of shape (N, H, W).
        
    Returns
    -------
    np.ndarray
        Array of preprocessed images with identical shape (N, H, W).
    """
    processed = np.empty_like(images, dtype=np.float32)
    for i in range(len(images)):
        processed[i] = preprocess_single_image(
            images[i],
            deskew=deskew,
            blur=blur,
            kernel_size=kernel_size,
            sigma=sigma,
            normalize=normalize
        )
    return processed
