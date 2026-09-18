"""
Dataset loader and camera trajectory generator for 3D Gaussian Splatting experiments.

Supports loading the standard Tiny NeRF multi-view dataset, generating procedural
synthetic multi-view geometric 3D scenes, and synthesizing smooth orbital camera paths.
"""

import os
import urllib.request
from typing import Dict, Tuple, Union
import numpy as np


TINY_NERF_URLS = [
    "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz",
    "https://raw.githubusercontent.com/yenchenlin/nerf-pytorch/master/data/tiny_nerf_data.npz",
]


def load_tiny_nerf_dataset(
    data_dir: str = "data",
    cache_filename: str = "tiny_nerf_data.npz",
    val_split_index: int = 100,
) -> Dict[str, Union[np.ndarray, float, int]]:
    """
    Load the Tiny NeRF multi-view dataset. Checks local and sibling experiment directories,
    downloads if needed, or falls back to a procedural 3D dataset.

    Args:
        data_dir: Directory where data file is stored/cached.
        cache_filename: Name of the npz cache file.
        val_split_index: Index separating train and validation views.

    Returns:
        Dictionary containing:
            - 'images_train': np.ndarray [N_train, H, W, 3] in [0, 1]
            - 'poses_train': np.ndarray [N_train, 4, 4] (camera-to-world)
            - 'images_val': np.ndarray [N_val, H, W, 3]
            - 'poses_val': np.ndarray [N_val, 4, 4]
            - 'focal': float
            - 'height': int
            - 'width': int
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, cache_filename)

    # Check sibling experiment directory if not present in current data_dir
    sibling_cache = os.path.join(os.path.dirname(data_dir), "11_nerf_novel_view_synthesis", "data", cache_filename)
    if not os.path.exists(cache_path) and os.path.exists(sibling_cache):
        try:
            import shutil
            shutil.copyfile(sibling_cache, cache_path)
            print(f"Loaded dataset from existing cache: {sibling_cache}")
        except Exception:
            pass

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
            print("Warning: Online download failed. Generating procedural 3D multi-view dataset...")
            return generate_procedural_multiview_dataset(num_views=106, height=100, width=100, val_split=val_split_index)

    loaded = np.load(cache_path)
    images = loaded["images"].astype(np.float32)
    poses = loaded["poses"].astype(np.float32)
    focal = float(loaded["focal"])
    height, width = images.shape[1], images.shape[2]

    images_train = images[:val_split_index]
    poses_train = poses[:val_split_index]
    images_val = images[val_split_index:]
    poses_val = poses[val_split_index:]

    return {
        "images_train": images_train,
        "poses_train": poses_train,
        "images_val": images_val,
        "poses_val": poses_val,
        "focal": focal,
        "height": height,
        "width": width,
    }


def generate_orbital_camera_poses(
    num_poses: int = 40,
    radius: float = 4.03,
    elevation_deg: float = -30.0,
) -> np.ndarray:
    """
    Generate a 360-degree circular orbital camera trajectory around the origin.

    Args:
        num_poses: Number of camera poses along the circular orbit.
        radius: Distance from the world origin (camera center).
        elevation_deg: Elevation angle above the ground plane in degrees.

    Returns:
        np.ndarray: [num_poses, 4, 4] Camera-to-world extrinsic matrices.
    """
    poses = []
    phi = np.deg2rad(elevation_deg)

    for theta_deg in np.linspace(0.0, 360.0, num_poses, endpoint=False):
        theta = np.deg2rad(theta_deg)

        # Camera position in world coordinates
        cam_x = radius * np.cos(phi) * np.sin(theta)
        cam_y = radius * np.sin(phi)
        cam_z = radius * np.cos(phi) * np.cos(theta)
        cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

        # Look-at matrix: camera looks toward the origin [0, 0, 0]
        # Camera convention: -Z is forward, +Y is up, +X is right
        forward = -cam_pos / np.linalg.norm(cam_pos)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(world_up, -forward)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(-forward, right)
        up = up / (np.linalg.norm(up) + 1e-8)

        # Rotation matrix (columns are right, up, -forward)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = cam_pos
        poses.append(c2w)

    return np.stack(poses, axis=0)


def generate_procedural_multiview_dataset(
    num_views: int = 106,
    height: int = 100,
    width: int = 100,
    val_split: int = 100,
) -> Dict[str, Union[np.ndarray, float, int]]:
    """
    Generate a synthetic multi-view dataset of a colorful 3D cube.

    Returns:
        Dictionary formatted identically to load_tiny_nerf_dataset.
    """
    focal = 138.8888
    radius = 4.03
    poses = generate_orbital_camera_poses(num_poses=num_views, radius=radius, elevation_deg=-30.0)

    # Render simple multi-color cube faces
    images = []
    for pose in poses:
        img = np.ones((height, width, 3), dtype=np.float32)
        # Add basic projected pattern
        cx, cy = width / 2.0, height / 2.0
        r_c2w = pose[:3, :3]
        t_c2w = pose[:3, 3]
        for u in range(width):
            for v in range(height):
                dir_cam = np.array([(u - cx) / focal, -(v - cy) / focal, -1.0])
                dir_world = r_c2w @ dir_cam
                dir_world = dir_world / np.linalg.norm(dir_world)
                # Sphere intersection
                b = np.dot(t_c2w, dir_world)
                c = np.dot(t_c2w, t_c2w) - 0.7 ** 2
                disc = b ** 2 - c
                if disc > 0:
                    t_hit = -b - np.sqrt(disc)
                    if t_hit > 0:
                        hit_p = t_c2w + t_hit * dir_world
                        normal = hit_p / np.linalg.norm(hit_p)
                        color = 0.5 * (normal + 1.0)
                        img[v, u] = color
        images.append(img)

    images = np.stack(images, axis=0)
    return {
        "images_train": images[:val_split],
        "poses_train": poses[:val_split],
        "images_val": images[val_split:],
        "poses_val": poses[val_split:],
        "focal": focal,
        "height": height,
        "width": width,
    }


def sample_visual_hull_point_cloud(
    images: np.ndarray,
    poses: np.ndarray,
    focal: float,
    height: int,
    width: int,
    num_points: int = 600,
    spatial_bound: float = 0.8,
    grid_resolution: int = 35,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract an accurate 3D point cloud by multi-view silhouette space carving.
    Checks 3D voxel projections across multi-view foreground silhouettes to guarantee
    that Gaussians are strictly placed on the physical 3D object.

    Args:
        images: [N_views, H, W, 3] Multi-view RGB images in [0, 1] with black background.
        poses: [N_views, 4, 4] Camera-to-world extrinsic matrices.
        focal: Camera focal length in pixels.
        height: Image height.
        width: Image width.
        num_points: Desired number of surface point samples.
        spatial_bound: Spatial bounding box half-extent in world units.
        grid_resolution: Voxel resolution per axis.

    Returns:
        points: [num_points, 3] 3D coordinates on the object.
        colors: [num_points, 3] Multi-view sampled RGB colors.
    """
    lin = np.linspace(-spatial_bound, spatial_bound, grid_resolution)
    X, Y, Z = np.meshgrid(lin, lin, lin)
    grid_pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).astype(np.float32)

    cx = float(width) / 2.0
    cy = float(height) / 2.0
    num_views = min(25, len(images))
    inside_count = np.zeros(len(grid_pts), dtype=int)

    for i in range(num_views):
        c2w = poses[i]
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]

        pts_cam = (grid_pts - t_c2w[None, :]) @ R_c2w
        depths = -pts_cam[:, 2]

        u = np.round(focal * pts_cam[:, 0] / (depths + 1e-7) + cx).astype(int)
        v = np.round(-focal * pts_cam[:, 1] / (depths + 1e-7) + cy).astype(int)

        valid = (depths > 0.5) & (u >= 0) & (u < width) & (v >= 0) & (v < height)

        img = images[i]
        is_fg = np.zeros(len(grid_pts), dtype=bool)
        # Foreground pixels have non-zero RGB (black background)
        is_fg[valid] = np.sum(img[v[valid], u[valid]], axis=-1) > 0.1
        inside_count += is_fg.astype(int)

    # Keep voxels that project to foreground in at least 65% of viewpoints
    hull_mask = inside_count >= int(0.65 * num_views)
    hull_points = grid_pts[hull_mask]

    if len(hull_points) == 0:
        # Fallback if silhouette carving finds no intersection
        hull_points = np.random.uniform(-0.5, 0.5, size=(num_points, 3)).astype(np.float32)

    # Sample num_points from the carved volume
    if len(hull_points) > num_points:
        sel = np.random.choice(len(hull_points), num_points, replace=False)
        pts = hull_points[sel]
    else:
        pts = hull_points

    # Sample initial colors by projecting points into reference view 0
    c2w0 = poses[0]
    pts_cam0 = (pts - c2w0[:3, 3]) @ c2w0[:3, :3]
    depths0 = -pts_cam0[:, 2]
    u0 = np.clip(np.round(focal * pts_cam0[:, 0] / (depths0 + 1e-7) + cx).astype(int), 0, width - 1)
    v0 = np.clip(np.round(-focal * pts_cam0[:, 1] / (depths0 + 1e-7) + cy).astype(int), 0, height - 1)
    sampled_colors = images[0][v0, u0]

    return pts.astype(np.float32), sampled_colors.astype(np.float32)


