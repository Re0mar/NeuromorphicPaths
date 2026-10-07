"""
Finding the ground in a point cloud, and measuring heights against it.

A Plane here is written so that normal . point + offset is the height above the floor, with the
normal pointing up. That convention is what lets the height band be two comparisons.

Every function that needs to know which way is up takes it as a camera-frame unit vector, because
the camera is not level. A phone held in portrait sends a depth image in the sensor's landscape
orientation, so image-up points sideways and the floor's normal sits 90 degrees from it. Gravity
from the pose is up whenever the source says its orientation is gravity aligned, which the Pixel
and the Neon both do, the Neon with no position. Image-up, CAMERA_UP, is all a plain video file can
offer.

The floor fit is a RANSAC seeded from the config, so the same points always give the same floor
and a replayed recording gives the same numbers every run.
"""

# Standard library imports
import math

# Third party imports
import numpy as np

# Local package imports
from nav.scene.config import SceneConfig
from nav.types import Plane

# Image-up, in camera axes where y points down. The right "up" only when the camera is level.
CAMERA_UP = np.array([0.0, -1.0, 0.0])

# How many RANSAC planes are scored at once. Bounds memory to points times this, and changes
# nothing about which plane wins.
_PLANES_PER_CHUNK = 32


def normalize_plane(plane: Plane, up_camera: np.ndarray) -> Plane:
    """
    The same plane with a unit normal pointing toward up, and the offset scaled to match.

    Every plane the scene judges goes through here first, the fit's and a source's alike, so a
    phone that sends its normal pointing down is read the same way a fit that came out upside
    down is.

    :param plane: Any plane with a non-zero normal.
    :param up_camera: Unit vector pointing up, in the camera frame.
    :return: The plane in the convention the rest of the scene assumes.
    :rtype: Plane
    :raises ValueError: When the normal has zero length, which is not a plane at all.
    """
    normal = np.asarray(plane.normal, dtype=np.float64)
    length = float(np.linalg.norm(normal))
    if length == 0.0:
        raise ValueError("a plane's normal cannot have zero length")
    normal, offset = normal / length, plane.offset_meters / length
    if normal @ up_camera < 0:
        normal, offset = -normal, -offset
    return Plane(normal=normal, offset_meters=float(offset))


def plane_is_a_floor(plane: Plane, config: SceneConfig, up_camera: np.ndarray) -> str | None:
    """
    Judge a normalized plane against where a floor can be, and say which rule it broke.

    Three rules: level enough, not too close to the camera, and not too far below it. The first
    two are the old file's. The third came from the first Pixel walk, where ARCore handed over a
    plane 2.3 m down, a meter below the real floor, and nothing refused it. This is the one place
    the rules live, so the fit and a supplied plane cannot drift apart on what a floor is.

    :param plane: A plane from normalize_plane.
    :param config: The tilt limit, the minimum and the maximum camera height.
    :param up_camera: Unit vector pointing up, in the camera frame. Level is measured against it.
    :return: None when the plane passes, otherwise the rule it broke, with the numbers.
    :rtype: str | None
    """
    tilt_degrees = float(np.degrees(np.arccos(np.clip(plane.normal @ up_camera, -1.0, 1.0))))
    if tilt_degrees > config.floor_max_tilt_degrees:
        return f"leans {tilt_degrees:.1f} deg from up, limit {config.floor_max_tilt_degrees:.1f}"
    if plane.offset_meters <= config.floor_min_offset_meters:
        return f"camera {plane.offset_meters:.2f} m above it, under the minimum {config.floor_min_offset_meters:.2f}"
    if plane.offset_meters > config.floor_max_offset_meters:
        return f"camera {plane.offset_meters:.2f} m above it, over the maximum {config.floor_max_offset_meters:.2f}"
    return None


def fit_floor(points_camera: np.ndarray, previous: Plane | None, config: SceneConfig, up_camera: np.ndarray) -> Plane:
    """
    RANSAC a plane through the points that could plausibly be floor.

    Every candidate goes through plane_is_a_floor. A plane that is not roughly level, that sits
    too close to the camera, or that lies further below it than a held or worn camera can be is
    not the floor, however many points agree with it, and in that case the previous frame's plane
    is better than a wrong one.

    :param points_camera: (N, 3) camera-frame points.
    :param previous: Last frame's plane, returned when this frame has no believable floor.
    :param config: Candidate selection and sanity thresholds.
    :param up_camera: Unit vector pointing up, in the camera frame. Gravity from the pose when
        the source has one, CAMERA_UP otherwise. "Below" and "level" are both measured against it.
    :return: The floor, normal pointing up.
    :rtype: Plane
    :raises ValueError: When the RANSAC seed or success probability is out of range, or when no
        floor is found and there is no previous plane to fall back on.
    """
    # Checked on every call, before the cloud decides whether the search runs at all. Otherwise a
    # bad value only surfaces on the first frame that happens to reach the search.
    _check_ransac_config(config)

    # Only points clearly below the camera can be floor. Without this, a wall straight ahead
    # with enough points wins the vote.
    depth_below_camera = -(points_camera @ up_camera)
    candidates = points_camera[depth_below_camera > config.floor_candidate_min_below_camera_meters]
    fitted = None
    if len(candidates) >= config.floor_min_candidate_points:
        fitted = _ransac_plane(candidates, config, up_camera)

    if fitted is not None:
        return fitted
    if previous is not None:
        return previous
    raise ValueError(
        f"no floor found in {len(points_camera)} points, {len(candidates)} below the camera, and no previous plane"
    )


def _check_ransac_config(config: SceneConfig) -> None:
    if config.floor_ransac_seed < 0:
        raise ValueError(f"floor_ransac_seed must be non-negative, got {config.floor_ransac_seed}")
    probability = config.floor_ransac_success_probability
    if not 0.0 < probability <= 1.0:
        raise ValueError(f"floor_ransac_success_probability must be above 0 and at most 1, got {probability}")


def _ransac_plane(candidates: np.ndarray, config: SceneConfig, up_camera: np.ndarray) -> Plane | None:
    # A fresh generator from the config's seed on every call, so the same cloud always draws the
    # same planes and a frame's floor never depends on the frames before it. Open3D's RANSAC
    # ignored its seed, and a replay came out different every run.
    generator = np.random.default_rng(config.floor_ransac_seed)
    points = np.asarray(candidates, dtype=np.float64)
    triples = generator.integers(0, len(points), size=(config.floor_ransac_iterations, 3))
    first, second, third = points[triples[:, 0]], points[triples[:, 1]], points[triples[:, 2]]
    normals = np.cross(second - first, third - first)
    lengths = np.linalg.norm(normals, axis=1)

    # A triple that repeats a point has no normal and is dropped here. A nearly collinear one keeps
    # a normal pointing anywhere, and it loses on inlier count rather than being filtered.
    has_normal = lengths > 0
    if not has_normal.any():
        return None
    normals = normals[has_normal] / lengths[has_normal, None]
    first = first[has_normal]
    offsets = -(normals[:, 0] * first[:, 0] + normals[:, 1] * first[:, 1] + normals[:, 2] * first[:, 2])

    x, y, z = (np.ascontiguousarray(column) for column in points.T)
    best = _best_plane(x, y, z, normals, offsets, config)

    plane = _refit_on_inliers(points, x, y, z, normals[best], offsets[best], config.floor_ransac_distance_meters)
    plane = normalize_plane(plane, up_camera)

    # Not level enough, too close, or too far down means this is not the floor.
    if plane_is_a_floor(plane, config, up_camera) is not None:
        return None
    return plane


def _best_plane(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
    config: SceneConfig,
) -> int:
    # Planes are tried in the order they were drawn, the way Open3D tries them. Most inliers wins,
    # and a tie goes to the tighter fit. Once the best plane so far makes a better one unlikely
    # enough, the search stops. Trying them in draw order keeps that stop the same for any chunk size.
    best_index, best_count, best_rmse = 0, -1, np.inf
    tries_needed = float(len(normals))
    for start in range(0, len(normals), _PLANES_PER_CHUNK):
        chunk = slice(start, start + _PLANES_PER_CHUNK)
        counts, rmse = _score_planes(x, y, z, normals[chunk], offsets[chunk], config.floor_ransac_distance_meters)
        for index, (count, error) in enumerate(zip(counts.tolist(), rmse.tolist()), start=start):
            if count > best_count or (count == best_count and error < best_rmse):
                best_index, best_count, best_rmse = index, count, error
                tries_needed = min(tries_needed, _tries_needed(best_count / len(x), config.floor_ransac_success_probability))
            if index + 1 >= tries_needed:
                return best_index
    return best_index


def _tries_needed(inlier_fraction: float, probability: float) -> float:
    # How many draws it takes to have hit three inliers at least once with this probability.
    # Open3D's stopping rule. A floor that is 90 percent of the points needs about 14 draws.
    all_three_inliers = inlier_fraction**3
    if probability >= 1.0 or all_three_inliers <= 0.0:
        return np.inf
    if all_three_inliers >= 1.0:
        return 0.0
    return math.log(1.0 - probability) / math.log(1.0 - all_three_inliers)


def _distances_to_planes(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    # One row per plane, written out rather than as normals @ points.T. A matrix product goes
    # through BLAS, which can split the sum differently by thread count and change the last bits,
    # and the fit has to give the same plane in any process.
    distances = np.multiply.outer(normals[:, 0], x)
    distances += np.multiply.outer(normals[:, 1], y)
    distances += np.multiply.outer(normals[:, 2], z)
    distances += offsets[:, None]
    return np.abs(distances, out=distances)


def _score_planes(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
    threshold_meters: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Each plane's distances are one contiguous row, scored on its own, so how many planes share
    # the array changes memory and nothing about the scores.
    distances = _distances_to_planes(x, y, z, normals, offsets)
    # Strictly under the threshold counts, as in Open3D.
    is_inlier = distances < threshold_meters
    counts = np.count_nonzero(is_inlier, axis=1)
    np.multiply(distances, distances, out=distances)
    distances[~is_inlier] = 0.0
    rmse = np.sqrt(distances.sum(axis=1) / np.maximum(counts, 1))
    return counts, rmse


def _refit_on_inliers(
    points: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    normal: np.ndarray,
    offset: float,
    threshold_meters: float,
) -> Plane:
    # The winning plane runs through three noisy points. The least-squares plane through all its
    # inliers is the better floor, and it's what Open3D returns too.
    members = points[_distances_to_planes(x, y, z, normal[None, :], np.array([offset]))[0] < threshold_meters]
    centroid = members.mean(axis=0)
    x, y, z = (members - centroid).T
    # The covariance by hand for the same reason as the distances: no BLAS in the sum.
    covariance = np.array(
        [
            [(x * x).sum(), (x * y).sum(), (x * z).sum()],
            [(x * y).sum(), (y * y).sum(), (y * z).sum()],
            [(x * z).sum(), (y * z).sum(), (z * z).sum()],
        ]
    )
    # eigh sorts eigenvalues ascending, so the first eigenvector is the direction the inliers
    # spread least along, which is the plane's normal.
    _, eigenvectors = np.linalg.eigh(covariance)
    refitted_normal = eigenvectors[:, 0]
    refitted_offset = -(
        refitted_normal[0] * centroid[0] + refitted_normal[1] * centroid[1] + refitted_normal[2] * centroid[2]
    )
    return Plane(normal=refitted_normal, offset_meters=float(refitted_offset))


def ground_axes(plane: Plane, forward_hint: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Unit axes on the floor: lateral, positive to the walker's right, and forward.

    Forward is the hint projected onto the plane, so "ahead" means where the camera points, not
    where the body happens to face. The hint is the camera's z axis in whatever frame the plane
    is in: (0, 0, 1) in the camera frame, the rotated z axis in the world frame.

    :param plane: The floor.
    :param forward_hint: (3,) direction to project. The camera frame's z axis when None.
    :return: (lateral_axis, forward_axis), each (3,).
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    camera_forward = np.array([0.0, 0.0, 1.0]) if forward_hint is None else np.asarray(forward_hint, dtype=np.float64)
    normal = plane.normal
    forward = camera_forward - (camera_forward @ normal) * normal
    if np.linalg.norm(forward) < 1e-3:
        # Looking straight down. Any horizontal direction is as good as another, so take the one
        # the plane's normal is least aligned with.
        fallback = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        forward = fallback - (fallback @ normal) * normal
    forward = forward / np.linalg.norm(forward)
    lateral = np.cross(forward, normal)
    return lateral, forward


def height_above_floor(points: np.ndarray, plane: Plane) -> np.ndarray:
    """Signed height of each point above the plane, in meters."""
    return np.asarray(points) @ plane.normal + plane.offset_meters
