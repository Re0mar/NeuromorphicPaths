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

Two ways to find the floor. With gravity, fit_level_floor picks among the level surfaces below the
camera: the floor, tier platforms, stair treads, seats. Without it, "level" has nothing to be
measured against, and fit_floor_with_refusal fits one plane by RANSAC. Neither depends on chance.
The level choice draws nothing, and the RANSAC is seeded from the config, so a replayed recording
gives the same floor every run.
"""

# Standard library imports
import math
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum

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


class FloorRefusalCause(Enum):
    """Why a plane, or a frame's whole search, did not give a floor."""

    LEANS = "leans"  # Tilted from up past the limit.
    TOO_CLOSE = "too close"  # Camera at or under the minimum height above it.
    TOO_FAR = "too far"  # Camera over the maximum height above it.
    TOO_FEW_CANDIDATES = "too few candidates"  # Not enough points below the camera to search.
    NO_PLANE = "no plane"  # Every drawn triple was degenerate, so there was nothing to judge.
    NO_LEVEL_SURFACE = "no level surface"  # No height below the camera held enough points to be a surface.


@dataclass(frozen=True)
class FloorRefusal:
    """
    One refusal, with the number that broke the rule and the rule's limit.

    measured is degrees for LEANS, meters for the two height causes, and a point count for
    TOO_FEW_CANDIDATES and NO_LEVEL_SURFACE. NO_PLANE has neither.
    """

    cause: FloorRefusalCause
    measured: float | None
    limit: float | None

    def __str__(self) -> str:
        # The three plane rules keep the exact wording the debug log has always printed.
        match self.cause:
            case FloorRefusalCause.LEANS:
                return f"leans {self.measured:.1f} deg from up, limit {self.limit:.1f}"
            case FloorRefusalCause.TOO_CLOSE:
                return f"camera {self.measured:.2f} m above it, under the minimum {self.limit:.2f}"
            case FloorRefusalCause.TOO_FAR:
                return f"camera {self.measured:.2f} m above it, over the maximum {self.limit:.2f}"
            case FloorRefusalCause.TOO_FEW_CANDIDATES:
                return f"{self.measured:.0f} candidate points below the camera, under the minimum {self.limit:.0f}"
            case FloorRefusalCause.NO_PLANE:
                return "every drawn triple was degenerate, so there was no plane to judge"
            case FloorRefusalCause.NO_LEVEL_SURFACE:
                return f"the most points at one height below the camera was {self.measured:.0f}, under the {self.limit:.0f} a surface needs"
            case _:
                # Unreachable while every member is handled. Loud, so a new cause can't print nothing.
                raise ValueError(f"no message for {self.cause}")


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


def plane_is_a_floor(plane: Plane, config: SceneConfig, up_camera: np.ndarray) -> FloorRefusal | None:
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
    :rtype: FloorRefusal | None
    """
    tilt_degrees = float(np.degrees(np.arccos(np.clip(plane.normal @ up_camera, -1.0, 1.0))))
    if tilt_degrees > config.floor_max_tilt_degrees:
        return FloorRefusal(FloorRefusalCause.LEANS, tilt_degrees, config.floor_max_tilt_degrees)
    if plane.offset_meters <= config.floor_min_offset_meters:
        return FloorRefusal(FloorRefusalCause.TOO_CLOSE, plane.offset_meters, config.floor_min_offset_meters)
    if plane.offset_meters > config.floor_max_offset_meters:
        return FloorRefusal(FloorRefusalCause.TOO_FAR, plane.offset_meters, config.floor_max_offset_meters)
    return None


def fit_floor(points_camera: np.ndarray, previous: Plane | None, config: SceneConfig, up_camera: np.ndarray) -> Plane:
    """
    RANSAC a plane through the points that could plausibly be floor.

    The floor alone, for callers that don't need to know why a fit fell back. Same search and same
    answer as fit_floor_with_refusal.

    :param points_camera: (N, 3) camera-frame points.
    :param previous: Last frame's plane, returned when this frame has no believable floor.
    :param config: Candidate selection and sanity thresholds.
    :param up_camera: Unit vector pointing up, in the camera frame.
    :return: The floor, normal pointing up.
    :rtype: Plane
    :raises ValueError: As fit_floor_with_refusal.
    """
    floor, _ = fit_floor_with_refusal(points_camera, previous, config, up_camera)
    return floor


def fit_floor_with_refusal(
    points_camera: np.ndarray,
    previous: Plane | None,
    config: SceneConfig,
    up_camera: np.ndarray,
) -> tuple[Plane, FloorRefusal | None]:
    """
    RANSAC a plane through the points that could plausibly be floor, and say why when none was.

    Every candidate goes through plane_is_a_floor. A plane that is not roughly level, that sits
    too close to the camera, or that lies further below it than a held or worn camera can be is
    not the floor, however many points agree with it, and in that case the previous frame's plane
    is better than a wrong one.

    :param points_camera: (N, 3) camera-frame points.
    :param previous: Last frame's plane, returned when this frame has no believable floor.
    :param config: Candidate selection and sanity thresholds.
    :param up_camera: Unit vector pointing up, in the camera frame. Gravity from the pose when
        the source has one, CAMERA_UP otherwise. "Below" and "level" are both measured against it.
    :return: The floor, normal pointing up, and None when it was fitted from these points. On a
        fall back, the previous plane itself, so callers can test identity, and why this frame's
        search gave nothing.
    :rtype: tuple[Plane, FloorRefusal | None]
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
    if len(candidates) >= config.floor_min_candidate_points:
        fitted = _ransac_plane(candidates, config, up_camera)
    else:
        fitted = FloorRefusal(FloorRefusalCause.TOO_FEW_CANDIDATES, float(len(candidates)), float(config.floor_min_candidate_points))

    if isinstance(fitted, Plane):
        return fitted, None
    if previous is not None:
        return previous, fitted
    raise ValueError(
        f"no floor found in {len(points_camera)} points, {len(candidates)} below the camera, and no previous plane"
    )


@dataclass(frozen=True)
class LevelFloorChoice:
    """
    One frame's floor from the level surfaces, and how it was chosen.

    `refusal` is None when the floor came from this frame. Otherwise `floor` is the previous plane
    itself and `refusal` says why no surface was taken. `passed_over_deeper` means a deeper surface
    stood and the one near the recent floor was taken. `reset` means the deeper one had been passed
    over long enough that it was taken, so the history starts again from it.
    """

    floor: Plane
    refusal: FloorRefusal | None
    passed_over_deeper: bool = False
    reset: bool = False


class LevelFloorHistory:
    """
    The camera heights of the last few accepted floors, and how long a deeper surface has been passed over.

    Held by the scene between frames. `fit_level_floor` only reads it. The scene records each frame's
    choice afterwards, so one frame's choice never depends on its own outcome.
    """

    def __init__(self, length: int) -> None:
        if length < 1:
            raise ValueError(f"floor_level_history must be at least 1, got {length}")
        self._heights: deque[float] = deque(maxlen=length)
        self._passed_over_frames = 0

    @property
    def reference_meters(self) -> float | None:
        """The median camera height over the recent accepted floors. None with no history yet."""
        return float(np.median(self._heights)) if self._heights else None

    @property
    def passed_over_frames(self) -> int:
        """How many frames in a row a deeper surface stood and the one near the recent floor was taken."""
        return self._passed_over_frames

    def record(self, choice: LevelFloorChoice) -> None:
        """
        Fold one frame's choice in. A frame that kept the previous floor adds nothing.

        :param choice: What `fit_level_floor` returned for the frame.
        """
        if choice.refusal is not None:
            return
        if choice.reset:
            self.clear()
        self._heights.append(choice.floor.offset_meters)
        self._passed_over_frames = self._passed_over_frames + 1 if choice.passed_over_deeper else 0

    def record_supplied(self, floor: Plane) -> None:
        """Fold in a floor the source supplied and the scene accepted. It's the walker's floor as much as a chosen one."""
        self._heights.append(floor.offset_meters)
        self._passed_over_frames = 0

    def clear(self) -> None:
        """Forget every height and the passed-over count, as when gravity is lost or a reset took a deeper surface."""
        self._heights.clear()
        self._passed_over_frames = 0


def fit_level_floor(
    points_camera: np.ndarray,
    previous: Plane | None,
    history: LevelFloorHistory,
    config: SceneConfig,
    up_camera: np.ndarray,
) -> LevelFloorChoice:
    """
    Pick the walker's floor among the level surfaces below the camera.

    A level surface is a height below the camera, measured along up, that holds a large share of the
    points. Each is refit as a plane and has to pass `plane_is_a_floor` with the tighter tilt limit
    `floor_level_max_tilt_degrees`. A plane through stair edges spreads its points over many heights,
    so it never forms one.

    The deepest surface is the floor, because tier platforms, stair treads and seats stand on it.
    The exception is a lower tier seen from a higher one. The camera rides at eye height above
    whatever level the walker is on, so the recent floors sit about eye height down. When a surface
    sits within `floor_level_tolerance_meters` of them and the deepest doesn't, the near one is taken.
    Once the deepest has been passed over like that on `floor_level_reset_frames` - 1 frames in a row,
    the next frame takes it and the history restarts there. That undoes a recent floor locked onto
    the wrong surface, a seat row or a tread. It can also lock onto a lower tier that stays in view.

    Use only with up from gravity. Image-up on a tilted head would turn the floor into a slope.

    :param points_camera: (N, 3) camera-frame points.
    :param previous: Last frame's plane, kept when this frame has no level surface.
    :param history: The recent floors. Read, never written.
    :param config: Candidate selection, the surface rules and the floor limits.
    :param up_camera: Unit vector pointing up, in the camera frame, from gravity.
    :return: The floor, normal pointing up, and how it was chosen.
    :rtype: LevelFloorChoice
    :raises ValueError: When a level-surface setting is out of range, or when no surface is found and
        there is no previous plane to fall back on.
    """
    _check_level_config(config)

    # Same cut as the RANSAC route. Only points clearly below the camera can be floor.
    drops = -(points_camera @ up_camera)
    is_candidate = drops > config.floor_candidate_min_below_camera_meters
    candidates, candidate_drops = points_camera[is_candidate], drops[is_candidate]
    if len(candidates) < config.floor_min_candidate_points:
        refusal = FloorRefusal(FloorRefusalCause.TOO_FEW_CANDIDATES, float(len(candidates)), float(config.floor_min_candidate_points))
        return _keep_previous(previous, refusal, len(points_camera), len(candidates))

    surfaces, refusal = _level_surfaces(candidates, candidate_drops, config, up_camera)
    if not surfaces:
        return _keep_previous(previous, refusal, len(points_camera), len(candidates))

    deepest = surfaces[-1]
    reference = history.reference_meters
    tolerance = config.floor_level_tolerance_meters
    if reference is None or abs(deepest.offset_meters - reference) <= tolerance:
        return LevelFloorChoice(deepest, None)
    near_recent = [surface for surface in surfaces if abs(surface.offset_meters - reference) <= tolerance]
    if not near_recent:
        # Nothing near the recent floor, so this frame can't see the walker's own level. The deepest
        # is the best guess, and the history isn't cleared, so one such frame can't move the reference.
        return LevelFloorChoice(deepest, None)
    if history.passed_over_frames + 1 >= config.floor_level_reset_frames:
        return LevelFloorChoice(deepest, None, reset=True)
    return LevelFloorChoice(near_recent[-1], None, passed_over_deeper=True)


def _keep_previous(previous: Plane | None, refusal: FloorRefusal, point_count: int, candidate_count: int) -> LevelFloorChoice:
    if previous is not None:
        return LevelFloorChoice(previous, refusal)
    raise ValueError(f"no floor found in {point_count} points, {candidate_count} below the camera, and no previous plane: {refusal}")


def _level_surfaces(
    candidates: np.ndarray,
    candidate_drops: np.ndarray,
    config: SceneConfig,
    up_camera: np.ndarray,
) -> tuple[list[Plane], FloorRefusal]:
    # Surfaces that pass, shallowest first, and the refusal to report when none does: the deepest
    # surface's own, or, when no height held enough points, how many the busiest one held.
    bin_meters = config.floor_level_bin_meters
    # Out to the deepest point, not the height limit, so a surface below the limit still forms and
    # plane_is_a_floor says it's too far, rather than the frame reporting no surface at all.
    edges = np.arange(config.floor_candidate_min_below_camera_meters, float(candidate_drops.max()) + 2.0 * bin_meters, bin_meters)
    counts, _ = np.histogram(candidate_drops, bins=edges)
    # Smoothed over three bins, so a surface whose height straddles a bin edge is one peak, not two.
    smoothed = np.convolve(counts, np.ones(3) / 3.0, mode="same")
    padded = np.concatenate(([-1.0], smoothed, [-1.0]))
    is_peak = (padded[1:-1] >= padded[:-2]) & (padded[1:-1] > padded[2:])
    centers = edges[:-1] + bin_meters / 2.0

    needed = max(config.floor_min_candidate_points, math.ceil(config.floor_level_min_share * len(candidates)))
    # The tighter tilt limit goes through the one place the floor rules live, rather than a copy of them.
    level_config = replace(config, floor_max_tilt_degrees=config.floor_level_max_tilt_degrees)
    x, y, z = (np.ascontiguousarray(column) for column in candidates.T)
    surfaces: list[Plane] = []
    refusal = None
    most_held = 0
    for center in centers[is_peak]:
        # Same width as the RANSAC's inlier distance, so a surface holds what a fit through it would.
        held = int(np.count_nonzero(np.abs(candidate_drops - center) < config.floor_ransac_distance_meters))
        most_held = max(most_held, held)
        if held < needed:
            continue
        # A level plane at this height, refit by least squares on the points near it.
        plane = _refit_on_inliers(candidates, x, y, z, up_camera, float(center), config.floor_ransac_distance_meters)
        plane = normalize_plane(plane, up_camera)
        surface_refusal = plane_is_a_floor(plane, level_config, up_camera)
        if surface_refusal is None:
            surfaces.append(plane)
        else:
            refusal = surface_refusal
    if refusal is None:
        refusal = FloorRefusal(FloorRefusalCause.NO_LEVEL_SURFACE, float(most_held), float(needed))
    surfaces.sort(key=lambda surface: surface.offset_meters)
    return surfaces, refusal


def _check_level_config(config: SceneConfig) -> None:
    if config.floor_level_bin_meters <= 0.0:
        raise ValueError(f"floor_level_bin_meters must be above 0, got {config.floor_level_bin_meters}")
    if not 0.0 <= config.floor_level_min_share <= 1.0:
        raise ValueError(f"floor_level_min_share must be from 0 to 1, got {config.floor_level_min_share}")
    if config.floor_level_tolerance_meters < 0.0:
        raise ValueError(f"floor_level_tolerance_meters must not be negative, got {config.floor_level_tolerance_meters}")
    if config.floor_level_reset_frames < 1:
        raise ValueError(f"floor_level_reset_frames must be at least 1, got {config.floor_level_reset_frames}")


def _check_ransac_config(config: SceneConfig) -> None:
    if config.floor_ransac_seed < 0:
        raise ValueError(f"floor_ransac_seed must be non-negative, got {config.floor_ransac_seed}")
    probability = config.floor_ransac_success_probability
    if not 0.0 < probability <= 1.0:
        raise ValueError(f"floor_ransac_success_probability must be above 0 and at most 1, got {probability}")


def _ransac_plane(candidates: np.ndarray, config: SceneConfig, up_camera: np.ndarray) -> Plane | FloorRefusal:
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
        return FloorRefusal(FloorRefusalCause.NO_PLANE, None, None)
    normals = normals[has_normal] / lengths[has_normal, None]
    first = first[has_normal]
    offsets = -(normals[:, 0] * first[:, 0] + normals[:, 1] * first[:, 1] + normals[:, 2] * first[:, 2])

    x, y, z = (np.ascontiguousarray(column) for column in points.T)
    best = _best_plane(x, y, z, normals, offsets, config)

    plane = _refit_on_inliers(points, x, y, z, normals[best], offsets[best], config.floor_ransac_distance_meters)
    plane = normalize_plane(plane, up_camera)

    # Not level enough, too close, or too far down means this is not the floor. The refusal is
    # handed back rather than dropped, so a replay can say why the previous floor stood.
    refusal = plane_is_a_floor(plane, config, up_camera)
    if refusal is not None:
        return refusal
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
