"""
Synthetic depth images with analytic ground truth, for the scene tests.

Two halves, kept separable on purpose. clean_scene() makes a depth image whose floor plane, box
position and box height are known exactly, so a test can assert against a number it did not read
out of the system. The degrade_* functions then inject holes, zero rows and a missing floor into
a clean image, so degenerate input is a test and not an accident of the fixture.

Camera axes: x right, y down, z forward. The camera sits CAMERA_HEIGHT above a flat floor and is
pitched down by PITCH_DEGREES, so the floor is not simply "y equals a constant" and a floor fit
has real work to do.
"""

# Standard library imports
from dataclasses import dataclass

# Third party imports
import numpy as np

# Local package imports
from nav.types import Plane

HEIGHT = 96
WIDTH = 128
CAMERA_HEIGHT_METERS = 1.6
PITCH_DEGREES = 20.0
FOCAL_PIXELS = 100.0


@dataclass(frozen=True)
class SyntheticScene:
    """A depth image and everything that is true about it."""

    depth_meters: np.ndarray
    intrinsics: np.ndarray
    floor_plane_camera: Plane
    box_lateral_meters: float | None
    box_forward_meters: float | None
    box_height_meters: float | None


def intrinsics(height: int = HEIGHT, width: int = WIDTH, focal: float = FOCAL_PIXELS) -> np.ndarray:
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])


def pitch_rotation(pitch_degrees: float) -> np.ndarray:
    """Rotation taking level-camera coordinates to a camera pitched down by the angle."""
    angle = np.radians(pitch_degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    # Pitching the camera down rotates the world up around the camera's x axis.
    return np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])


def floor_plane_in_camera(camera_height: float = CAMERA_HEIGHT_METERS, pitch_degrees: float = PITCH_DEGREES) -> Plane:
    """The floor as the pitched camera sees it, normal up, normal . p + offset == height above floor."""
    rotation = pitch_rotation(pitch_degrees)
    level_up = np.array([0.0, -1.0, 0.0])
    normal = rotation @ level_up
    # The camera is at the origin, camera_height above the floor, so the origin's height is the offset.
    return Plane(normal=normal, offset_meters=camera_height)


def clean_scene(
    box_lateral_meters: float | None = 0.5,
    box_forward_meters: float | None = 3.0,
    box_height_meters: float | None = 1.0,
    box_half_width_meters: float = 0.25,
    camera_height: float = CAMERA_HEIGHT_METERS,
    pitch_degrees: float = PITCH_DEGREES,
) -> SyntheticScene:
    """
    Ray cast a flat floor and one upright box into a depth image.

    :param box_lateral_meters: Box centre to the right of the camera, on the floor. None for no box.
    :param box_forward_meters: Box centre ahead of the camera along the level ground.
    :param box_height_meters: Top of the box above the floor.
    :param box_half_width_meters: Half the box's lateral and forward extent.
    :return: The scene with its analytic truth.
    :rtype: SyntheticScene
    """
    camera_matrix = intrinsics()
    rotation = pitch_rotation(pitch_degrees)
    rows, columns = np.mgrid[0:HEIGHT, 0:WIDTH]
    rays_camera = np.stack(
        (
            (columns - camera_matrix[0, 2]) / camera_matrix[0, 0],
            (rows - camera_matrix[1, 2]) / camera_matrix[1, 1],
            np.ones_like(columns, dtype=np.float64),
        ),
        axis=-1,
    )
    # Express rays in level-ground coordinates: x right, y down, z forward, camera at height h.
    rays_level = rays_camera @ rotation  # inverse rotation, rotation is orthonormal

    depth = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float64)

    # Floor: y_level = camera_height. A ray hits it at t = camera_height / ray_y when ray_y > 0.
    ray_y = rays_level[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(ray_y > 1e-6, camera_height / ray_y, np.inf)
    depth = np.where(np.isfinite(t_floor), t_floor * rays_camera[..., 2], depth)

    if box_lateral_meters is not None and box_forward_meters is not None and box_height_meters is not None:
        # The box's front face is a vertical plane at z_level = box_forward - half_width, between
        # the floor and box_height, within the lateral extent. Enough for a depth image.
        face_z = box_forward_meters - box_half_width_meters
        ray_z = rays_level[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_face = np.where(ray_z > 1e-6, face_z / ray_z, np.inf)
        hit_x = t_face * rays_level[..., 0]
        hit_y = t_face * ray_y
        on_face = (
            np.isfinite(t_face)
            & (np.abs(hit_x - box_lateral_meters) <= box_half_width_meters)
            & (hit_y <= camera_height)
            & (hit_y >= camera_height - box_height_meters)
        )
        face_depth = t_face * rays_camera[..., 2]
        depth = np.where(on_face & (face_depth < np.nan_to_num(depth, nan=np.inf)), face_depth, depth)

    return SyntheticScene(
        depth_meters=depth.astype(np.float32),
        intrinsics=camera_matrix,
        floor_plane_camera=floor_plane_in_camera(camera_height, pitch_degrees),
        box_lateral_meters=box_lateral_meters,
        box_forward_meters=box_forward_meters,
        box_height_meters=box_height_meters,
    )


def level_floor_depth(heights_meters: np.ndarray | None = None) -> np.ndarray:
    """
    The floor seen by a level camera, CAMERA_HEIGHT_METERS up with intrinsics(), optionally lifted.

    No pitch, so a floor pixel's depth has a closed form a test can check by hand: row v reads the
    floor at FOCAL_PIXELS * CAMERA_HEIGHT_METERS / (v - HEIGHT / 2). Rows at and above the horizon
    see no floor and read NaN. With the defaults the bottom row is floor 3.40 m out.

    :param heights_meters: (HEIGHT, WIDTH) height to lift each pixel's reading off the floor, or None.
    :return: (HEIGHT, WIDTH) float32 depth.
    :rtype: np.ndarray
    """
    horizon = HEIGHT / 2.0
    rows = np.arange(HEIGHT, dtype=np.float64)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(rows > horizon, FOCAL_PIXELS * CAMERA_HEIGHT_METERS / (rows - horizon), np.nan) * np.ones((1, WIDTH))
    if heights_meters is not None:
        # On a pixel's own ray, height above the floor is camera height * (1 - reading / floor
        # depth), so a reading this much nearer stands that much higher.
        depth = depth * (1.0 - heights_meters / CAMERA_HEIGHT_METERS)
    return depth.astype(np.float32)


def with_box_on_level_floor(depth: np.ndarray, bottom_meters: float = 0.0) -> np.ndarray:
    """
    A box face painted over level_floor_depth: 3.6 m out, 0.6 m wide, from bottom_meters to 0.8 m up.

    With the defaults it covers columns 64 +- 100 * 0.3 / 3.6, so 56 to 72. Its top, 0.8 m up, is
    row 48 + 100 * 0.8 / 3.6 = 70.2. Standing on the floor its foot is row 48 + 160 / 3.6 = 92.4.
    So rows 71 to 92, fewer when it starts above the floor.
    """
    distance, half_width, top = 3.6, 0.3, 0.8
    first_column = int(np.ceil(WIDTH / 2.0 - FOCAL_PIXELS * half_width / distance))
    last_column = int(np.floor(WIDTH / 2.0 + FOCAL_PIXELS * half_width / distance))
    first_row = int(np.ceil(HEIGHT / 2.0 + FOCAL_PIXELS * (CAMERA_HEIGHT_METERS - top) / distance))
    last_row = int(np.floor(HEIGHT / 2.0 + FOCAL_PIXELS * (CAMERA_HEIGHT_METERS - bottom_meters) / distance))
    boxed = depth.copy()
    boxed[first_row : last_row + 1, first_column : last_column + 1] = distance
    return boxed


def degrade_with_holes(depth: np.ndarray, fraction: float, seed: int = 0) -> np.ndarray:
    """Set a random fraction of pixels to NaN, the way a confidence filter would."""
    generator = np.random.default_rng(seed)
    holed = depth.copy()
    holed[generator.random(depth.shape) < fraction] = np.nan
    return holed


def degrade_with_depth_noise(depth: np.ndarray, sigma_meters: float, seed: int = 0) -> np.ndarray:
    """Add seeded Gaussian noise to every depth, the way a real sensor jitters. Holes stay holes."""
    # A clean floor is a perfect plane, and on one every RANSAC draw refits to the same answer.
    # Noise is what makes a fit's randomness show, so a repeatability test can fail at all.
    generator = np.random.default_rng(seed)
    return depth + generator.normal(0.0, sigma_meters, depth.shape)


def degrade_with_zero_rows(depth: np.ndarray, rows: slice) -> np.ndarray:
    """Zero a band of rows, the way ARCore reports pixels it has no estimate for."""
    zeroed = depth.copy()
    zeroed[rows, :] = 0.0
    return zeroed


def degrade_without_floor(depth: np.ndarray, plane: Plane, intrinsics_matrix: np.ndarray) -> np.ndarray:
    """Remove every pixel that lies on the floor plane, leaving only what stands on it."""
    rows, columns = np.mgrid[0 : depth.shape[0], 0 : depth.shape[1]]
    z = depth.astype(np.float64)
    x = (columns - intrinsics_matrix[0, 2]) * z / intrinsics_matrix[0, 0]
    y = (rows - intrinsics_matrix[1, 2]) * z / intrinsics_matrix[1, 1]
    heights = np.stack((x, y, z), axis=-1) @ plane.normal + plane.offset_meters
    floorless = depth.copy()
    floorless[np.abs(heights) < 0.05] = np.nan
    return floorless
