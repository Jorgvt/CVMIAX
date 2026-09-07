"""
Embedding Space Collapse Simulation & Metrics.

Provides utilities to:
1. Generate synthetic healthy vs collapsed embedding distributions (Constant Collapse, Subspace Collapse, Uniform/Healthy).
2. Compute diagnostic collapse metrics:
   - Covariance Matrix & Singular Value Decomposition (SVD Spectrum / Effective Rank).
   - Intra-pair Alignment vs Inter-sample Uniformity (Wang & Isola, 2020).
   - Embedding Variance per Dimension.
"""

import numpy as np


def generate_healthy_embeddings(num_samples=600, num_classes=5, dim=2, radius=1.0, noise_std=0.08, seed=42):
    """
    Generates a healthy, well-dispersed embedding distribution on the unit circle/sphere
    with distinct class clusters and high variance across all dimensions.
    """
    np.random.seed(seed)
    samples_per_class = num_samples // num_classes
    embeddings = []
    labels = []

    if dim == 2:
        angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
        for class_idx, center_angle in enumerate(angles):
            # Cluster around center angle
            cluster_angles = center_angle + np.random.normal(0, noise_std * 2.5, samples_per_class)
            radii = radius + np.random.normal(0, noise_std * 0.3, samples_per_class)
            x = radii * np.cos(cluster_angles)
            y = radii * np.sin(cluster_angles)
            pts = np.stack([x, y], axis=1)
            # Normalize to unit sphere
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            embeddings.append(pts)
            labels.append(np.full(samples_per_class, class_idx))

    elif dim == 3:
        # Distribute cluster centers on 3D sphere
        for class_idx in range(num_classes):
            phi = np.random.uniform(0, 2 * np.pi)
            costheta = np.random.uniform(-1, 1)
            theta = np.arccos(costheta)
            center = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            pts = center + np.random.normal(0, noise_std, size=(samples_per_class, 3))
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            embeddings.append(pts)
            labels.append(np.full(samples_per_class, class_idx))
    else:
        # High dimensional Gaussian mixture normalized
        for class_idx in range(num_classes):
            center = np.random.normal(0, 1, size=(dim,))
            center = center / np.linalg.norm(center)
            pts = center + np.random.normal(0, noise_std, size=(samples_per_class, dim))
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            embeddings.append(pts)
            labels.append(np.full(samples_per_class, class_idx))

    return np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0)


def generate_constant_collapsed_embeddings(num_samples=600, num_classes=5, dim=2, seed=42):
    """
    Simulates Complete / Constant Collapse:
    The network outputs a constant vector for all inputs regardless of class or content.
    """
    np.random.seed(seed)
    samples_per_class = num_samples // num_classes
    labels = np.concatenate([np.full(samples_per_class, i) for i in range(num_classes)])

    # Target collapse point on unit sphere
    target = np.zeros(dim)
    target[0] = 1.0  # e.g. (1.0, 0.0)

    # All samples collapse into a tiny numerical noise delta around target
    noise = np.random.normal(0, 0.015, size=(num_samples, dim))
    pts = target + noise
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)

    return pts, labels


def generate_subspace_collapsed_embeddings(num_samples=600, num_classes=5, dim=2, seed=42):
    """
    Simulates Dimensional / Subspace Collapse:
    Embeddings span only a 1D line or narrow degenerate manifold, ignoring other available dimensions.
    """
    np.random.seed(seed)
    samples_per_class = num_samples // num_classes
    labels = np.concatenate([np.full(samples_per_class, i) for i in range(num_classes)])

    if dim == 2:
        # Spread along x axis between -1 and 1, but y axis is collapsed to near zero
        x_vals = np.linspace(-0.95, 0.95, num_samples)
        y_vals = np.random.normal(0, 0.02, size=num_samples)
        pts = np.stack([x_vals, y_vals], axis=1)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    else:
        # All energy in first 1-2 principal components
        pts = np.zeros((num_samples, dim))
        pts[:, 0] = np.random.normal(0, 1.0, num_samples)
        pts[:, 1:] = np.random.normal(0, 0.01, size=(num_samples, dim - 1))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)

    return pts, labels


def compute_svd_spectrum(embeddings):
    """
    Computes singular values of the zero-centered embedding matrix Z.
    Singular values reveal the effective dimensionality and variance distribution.
    """
    centered = embeddings - np.mean(embeddings, axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    # Normalize singular values to sum to 1 (relative variance explained)
    s_norm = s / np.sum(s) if np.sum(s) > 0 else s
    return cov, s_norm
