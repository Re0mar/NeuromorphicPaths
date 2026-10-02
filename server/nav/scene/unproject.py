"""
Depth pixels to a 3D point cloud in the camera frame.

Camera axes follow OpenCV: x right, y down, z forward. A pixel (u, v) at depth z lands at
((u - cx) z / fx, (v - cy) z / fy, z).

Pixels that are not finite or outside the usable depth range are dropped here, before anything
downstream could average them into a plane or a clearance.
"""

# Third party imports
import numpy as np
import open3d

# Local package imports
from nav.scene.config import SceneConfig


def unproject_depth(
    depth_meters: np.ndarray,
    intrinsics: np.ndarray,
    stride: int,
    config: SceneConfig,
) -> np.ndarray:
    """
    Turn a depth image into camera-frame points, dropping invalid pixels first.

    :param depth_meters: (H, W) depth in meters. NaN, zero and out-of-range values are invalid.
    :param intrinsics: (3, 3) camera matrix at the depth image's resolution.
    :param stride: Take every stride-th pixel in both directions.
    :param config: Supplies the usable depth range.
    :return: (N, 3) float64 points in the camera frame. N may be zero.
    :rtype: np.ndarray
    """
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")

    depth = np.asarray(depth_meters, dtype=np.float64)[::stride, ::stride]
    rows, columns = np.mgrid[0 : depth_meters.shape[0] : stride, 0 : depth_meters.shape[1] : stride]

    # The old file's range was 0.1 to 30 meters. Closer than the floor is a smudged lens, further
    # is sky, and either one unprojected would be a point the planner takes seriously.
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(depth) & (depth >= config.min_depth_meters) & (depth <= config.max_depth_meters)

    z = depth[valid]
    u = columns[valid].astype(np.float64)
    v = rows[valid].astype(np.float64)
    focal_x, focal_y = intrinsics[0, 0], intrinsics[1, 1]
    principal_x, principal_y = intrinsics[0, 2], intrinsics[1, 2]

    return np.column_stack(((u - principal_x) * z / focal_x, (v - principal_y) * z / focal_y, z))


def downsample(points_camera: np.ndarray, voxel_size_meters: float) -> np.ndarray:
    """
    Thin the cloud to one point per voxel.

    The only Open3D call in the scene layer. Everything else is numpy, so a swap to another
    downsampler is a change here and nowhere else.

    :param points_camera: (N, 3) points.
    :param voxel_size_meters: Edge of the voxel each surviving point represents.
    :return: (M, 3) points, M at most N.
    :rtype: np.ndarray
    """
    if voxel_size_meters <= 0:
        raise ValueError(f"voxel size must be positive, got {voxel_size_meters}")
    if len(points_camera) == 0:
        return np.empty((0, 3), dtype=np.float64)

    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(np.asarray(points_camera, dtype=np.float64))
    return np.asarray(cloud.voxel_down_sample(voxel_size_meters).points)
