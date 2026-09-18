"""
Dataset loader and procedural 3D data generator for NeRF experiments.

Provides seamless loading and caching of the standard Tiny NeRF Lego dataset
(Mildenhall et al.), with a self-contained synthetic geometric 3D scene fallback.
"""

import os
import urllib.request
from typing import Dict, Optional, Tuple, Union
import numpy as np


TINY_NERF_URLS = [
    "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz",
    "https://raw.githubusercontent.com/yenchenlin/nerf-pytorch/master/data/tiny_nerf_data.npz",
]


def load_tiny_nerf_dataset(
    data_dir: str = "data",
    cache_filename: str = "tiny_nerf_data.npz",
    val_split_index: int = 100,
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Load the Tiny NeRF synthetic Lego dataset. Downloads and caches locally if needed.

    Args:
        data_dir: Directory where data file is stored/cached.
        cache_filename: Name of the npz cache file.
        val_split_index: Index threshold separating train and test views.

    Returns:
        Dictionary containing:
            - 'images_train': np.ndarray [N_train, H, W, 3] (values in [0, 1])
            - 'poses_train': np.ndarray [N_train, 4, 4] (camera-to-world matrices)
            - 'images_val': np.ndarray [N_val, H, W, 3]
            - 'poses_val': np.ndarray [N_val, 4, 4]
            - 'focal': float (focal length in pixels)
            - 'height': int (image height)
            - 'width': int (image width)
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, cache_filename)

    if not os.path.exists(cache_path):
        print(f"Dataset not found at {cache_path}. Attempting download...")
        downloaded = False
        for url in TINY_NERF_URLS:
            try:
                print(f"Downloading from {url}...")
                urllib.request.urlretrieve(url, cache_path)
                print(f"Successfully downloaded to {cache_path} ({os.path.getsize(cache_path)} bytes)")
                downloaded = True
                break
            except Exception as e:
                print(f"Download from {url} failed: {e}")

        if not downloaded:
            print("Warning: Online download failed. Generating fallback procedural 3D dataset...")
            return generate_synthetic_3d_dataset(n_views=106, height=100, width=100)

    try:
        data = np.load(cache_path)
        images = data["images"].astype(np.float32)
        poses = data["poses"].astype(np.float32)
        focal = float(data["focal"])
    except Exception as e:
        print(f"Error loading {cache_path}: {e}. Falling back to procedural generator.")
        return generate_synthetic_3d_dataset(n_views=106, height=100, width=100)

    n_images, height, width, _ = images.shape
    val_split = min(val_split_index, n_images - 1)

    train_images = images[:val_split]
    train_poses = poses[:val_split]
    val_images = images[val_split:]
    val_poses = poses[val_split:]

    return {
        "images_train": train_images,
        "poses_train": train_poses,
        "images_val": val_images,
        "poses_val": val_poses,
        "focal": focal,
        "height": height,
        "width": width,
    }


def pose_spherical(theta_deg: float, phi_deg: float, radius: float) -> np.ndarray:
    """
    Compute a 4x4 camera-to-world (c2w) transformation matrix on a sphere.

    Args:
        theta_deg: Azimuth angle in degrees.
        phi_deg: Elevation angle in degrees.
        radius: Distance from camera to world origin.

    Returns:
        4x4 c2w extrinsic matrix.
    """
    def trans_t(t: float) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ], dtype=np.float32)

    def rot_phi(phi: float) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

    def rot_theta(th: float) -> np.ndarray:
        return np.array([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    c2w = trans_t(radius)
    c2w = rot_phi(phi) @ c2w
    c2w = rot_theta(theta) @ c2w
    # Convert from OpenGL / NeRF coordinate convention (x right, y up, z back)
    c2w = np.array([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32) @ c2w
    return c2w


def generate_synthetic_3d_dataset(
    n_views: int = 60,
    height: int = 64,
    width: int = 64,
    focal: float = 60.0,
    radius: float = 4.0,
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Generate an analytical procedural 3D geometric scene (colored spheres & cube)
    with exact ray-traced ground truth renderings and camera poses.

    Ensures 100% offline reproducibility without external network downloads.
    """
    thetas = np.linspace(0, 360, n_views, endpoint=False)
    phis = np.full(n_views, -30.0)

    images = []
    poses = []

    # Camera intrinsics
    cx = width / 2.0
    cy = height / 2.0
    u = np.arange(width, dtype=np.float32)
    v = np.arange(height, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    # Unit camera frame rays (pointing along -z)
    dirs_cam = np.stack([(uu - cx) / focal, -(vv - cy) / focal, -np.ones_like(uu)], axis=-1)

    for th, ph in zip(thetas, phis):
        c2w = pose_spherical(th, ph, radius)
        poses.append(c2w)

        # Transform rays to world frame: dirs = dirs_cam @ R^T
        rot = c2w[:3, :3]
        origin = c2w[:3, 3]
        dirs_world = dirs_cam @ rot.T
        dirs_norm = dirs_world / np.linalg.norm(dirs_world, axis=-1, keepdims=True)

        # Analytical sphere intersection: Sphere 1 (Center (0, 0, 0), Radius 0.8, Color [0.9, 0.2, 0.2])
        # Sphere 2 (Center (0.7, 0.7, 0), Radius 0.4, Color [0.2, 0.8, 0.3])
        # Sphere 3 (Center (-0.6, -0.6, 0.3), Radius 0.4, Color [0.2, 0.4, 0.9])
        spheres = [
            {"c": np.array([0.0, 0.0, 0.0]), "r": 0.8, "rgb": np.array([0.9, 0.25, 0.25])},
            {"c": np.array([0.7, 0.5, 0.2]), "r": 0.4, "rgb": np.array([0.2, 0.85, 0.3])},
            {"c": np.array([-0.6, -0.5, 0.3]), "r": 0.4, "rgb": np.array([0.25, 0.45, 0.95])},
        ]

        img = np.ones((height, width, 3), dtype=np.float32)  # White background
        closest_t = np.full((height, width), np.inf, dtype=np.float32)

        for s in spheres:
            # Quadratic intersection: |o + t*d - c|^2 = r^2
            oc = origin - s["c"]
            b = 2.0 * np.sum(dirs_norm * oc, axis=-1)
            c = np.sum(oc * oc) - s["r"] ** 2
            discriminant = b**2 - 4.0 * c
            hit = discriminant >= 0

            t0 = (-b - np.sqrt(np.maximum(discriminant, 0.0))) / 2.0
            mask = hit & (t0 > 0) & (t0 < closest_t)

            if np.any(mask):
                closest_t[mask] = t0[mask]
                hit_pts = origin + t0[mask, None] * dirs_norm[mask]
                normals = (hit_pts - s["c"]) / s["r"]
                # Diffuse shading from light at camera origin
                light_dir = -dirs_norm[mask]
                diffuse = np.maximum(0.2, np.sum(normals * light_dir, axis=-1, keepdims=True))
                img[mask] = s["rgb"] * diffuse

        images.append(img)

    images = np.stack(images, axis=0).astype(np.float32)
    poses = np.stack(poses, axis=0).astype(np.float32)

    val_idx = int(0.85 * n_views)
    return {
        "images_train": images[:val_idx],
        "poses_train": poses[:val_idx],
        "images_val": images[val_idx:],
        "poses_val": poses[val_idx:],
        "focal": float(focal),
        "height": height,
        "width": width,
    }
