"""
From a height-filtered cloud to groups of ground points, one group per occupied grid cell.

A cell is the mechanism. A group is the identity the rest of the pipeline sees, carried as an
integer id that the planner uses to take its per-group maximum. The word cell does not leave this
file.
"""

# Standard library imports
from dataclasses import dataclass

# Third party imports
import numpy as np

# Local package imports
from nav.scene.config import SceneConfig
from nav.scene.floor import ground_axes
from nav.types import Plane
from nav.walker import WalkerConfig


@dataclass(frozen=True)
class GroupSummary:
    """One obstacle, as the planner will see it."""

    group_id: int
    point_count: int
    centroid_lateral_meters: float
    centroid_forward_meters: float
    nearest_lateral_meters: float
    nearest_forward_meters: float
    max_height_meters: float
    is_wall: bool
    # The row, in the arrays summarize_groups was given, of the point it chose as nearest. The
    # pipeline uses it to look that point up in the camera frame for the depth view.
    nearest_index: int


def filter_height_band(
    points: np.ndarray,
    heights: np.ndarray,
    config: SceneConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep the points a walker could collide with: above the ankles, below the head.

    :param points: (N, 3) points in whichever frame the heights were measured in.
    :param heights: (N,) height above the floor for each point.
    :param config: The two heights.
    :return: (points, heights, keep) of the survivors, keep being the (N,) mask that chose them so
        a caller can apply the same cut to an array it kept alongside.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    keep = (heights > config.ankle_height_meters) & (heights < config.head_height_meters)
    return points[keep], heights[keep], keep


def flatten_to_ground(
    points: np.ndarray,
    plane: Plane,
    origin: np.ndarray | None = None,
    forward_hint: np.ndarray | None = None,
) -> np.ndarray:
    """
    Project points onto the floor and express them as (lateral, forward) from an origin.

    :param points: (N, 3) points in the plane's frame.
    :param plane: The floor, which fixes the lateral and forward axes.
    :param origin: (3,) point whose ground position is (0, 0). The camera when None.
    :param forward_hint: Where the camera points, in the plane's frame. See ground_axes.
    :return: (N, 2) lateral and forward in meters.
    :rtype: np.ndarray
    """
    lateral_axis, forward_axis = ground_axes(plane, forward_hint)
    relative = np.asarray(points, dtype=np.float64)
    if origin is not None:
        relative = relative - np.asarray(origin, dtype=np.float64)
    return np.column_stack((relative @ lateral_axis, relative @ forward_axis))


def assign_groups(ground_points: np.ndarray, config: SceneConfig) -> np.ndarray:
    """
    Give each ground point the id of the grid cell it falls in, or -1 outside the grid.

    The grid spans lateral -half_width to +half_width and forward 0 to grid_forward, in cells of
    cell_size. The id is row * columns + column, so the same ground position always gets the same
    id, which is what makes a group's history meaningful in the world frame.

    :param ground_points: (N, 2) lateral and forward.
    :param config: Grid extent and cell size.
    :return: (N,) int64 ids, -1 where the point is off the grid.
    :rtype: np.ndarray
    """
    if config.cell_size_meters <= 0:
        raise ValueError(f"cell size must be positive, got {config.cell_size_meters}")
    columns = int(np.ceil(2 * config.grid_half_width_meters / config.cell_size_meters))
    rows = int(np.ceil(config.grid_forward_meters / config.cell_size_meters))

    lateral = ground_points[:, 0]
    forward = ground_points[:, 1]
    inside = (
        (lateral >= -config.grid_half_width_meters)
        & (lateral < config.grid_half_width_meters)
        & (forward >= 0.0)
        & (forward < config.grid_forward_meters)
    )

    ids = np.full(len(ground_points), -1, dtype=np.int64)
    column_index = np.floor((lateral[inside] + config.grid_half_width_meters) / config.cell_size_meters).astype(np.int64)
    row_index = np.floor(forward[inside] / config.cell_size_meters).astype(np.int64)
    # Clip guards the exact upper edge, where floor() of a value just under the bound can still
    # round into the next cell in floating point.
    column_index = np.clip(column_index, 0, columns - 1)
    row_index = np.clip(row_index, 0, rows - 1)
    ids[inside] = row_index * columns + column_index
    return ids


# Room for a grid two million cells wide in each direction before ids collide. A walk is not that long.
WORLD_GRID_INDEX_OFFSET = 1 << 20
WORLD_GRID_ROW_STRIDE = 1 << 21


def assign_world_groups(ground_points: np.ndarray, config: SceneConfig) -> np.ndarray:
    """
    Give each ground point the id of its cell on a grid fixed to the world, with no bounds.

    Same cell size as assign_groups, but anchored at the world origin instead of the walker, so a
    post keeps its id as the walker advances past it. That stability is what the clearance history
    needs to measure N on something that is the same thing frame to frame.

    :param ground_points: (N, 2) lateral and forward in fixed world ground axes.
    :param config: Cell size.
    :return: (N,) int64 ids, always non-negative.
    :rtype: np.ndarray
    """
    if config.cell_size_meters <= 0:
        raise ValueError(f"cell size must be positive, got {config.cell_size_meters}")
    column_index = np.floor(ground_points[:, 0] / config.cell_size_meters).astype(np.int64) + WORLD_GRID_INDEX_OFFSET
    row_index = np.floor(ground_points[:, 1] / config.cell_size_meters).astype(np.int64) + WORLD_GRID_INDEX_OFFSET
    if np.any(column_index < 0) or np.any(row_index < 0) or np.any(column_index >= WORLD_GRID_ROW_STRIDE):
        raise ValueError("a point is further from the world origin than the world grid can index")
    return row_index * WORLD_GRID_ROW_STRIDE + column_index


def within_planning_window(ground_points: np.ndarray, config: SceneConfig) -> np.ndarray:
    """Which walker-relative ground points fall inside the grid the planner looks at."""
    lateral = ground_points[:, 0]
    forward = ground_points[:, 1]
    return (
        (lateral >= -config.grid_half_width_meters)
        & (lateral < config.grid_half_width_meters)
        & (forward >= 0.0)
        & (forward < config.grid_forward_meters)
    )


def summarize_groups(
    ground_points: np.ndarray,
    heights: np.ndarray,
    group_ids: np.ndarray,
    config: SceneConfig,
) -> list[GroupSummary]:
    """
    Reduce every occupied cell to one summary, dropping cells too sparse to trust.

    :param ground_points: (N, 2) lateral and forward.
    :param heights: (N,) height above floor.
    :param group_ids: (N,) from assign_groups.
    :param config: Minimum points per group and the wall height.
    :return: Summaries, sorted by id.
    :rtype: list[GroupSummary]
    """
    summaries: list[GroupSummary] = []
    for group_id in np.unique(group_ids):
        if group_id < 0:
            continue
        members = group_ids == group_id
        if int(members.sum()) < config.min_points_per_cell:
            # One point in a cell is as likely to be depth noise as an object.
            continue

        points = ground_points[members]
        distances = np.hypot(points[:, 0], points[:, 1])
        nearest_member = int(np.argmin(distances))
        nearest = points[nearest_member]
        max_height = float(heights[members].max())
        summaries.append(
            GroupSummary(
                group_id=int(group_id),
                point_count=int(members.sum()),
                centroid_lateral_meters=float(points[:, 0].mean()),
                centroid_forward_meters=float(points[:, 1].mean()),
                nearest_lateral_meters=float(nearest[0]),
                nearest_forward_meters=float(nearest[1]),
                max_height_meters=max_height,
                # The one place a wall is decided. A cell with points this tall is a wall or a
                # tree trunk, and both deserve a wider berth than a bollard.
                is_wall=max_height >= config.wall_cell_min_height_meters,
                nearest_index=int(np.flatnonzero(members)[nearest_member]),
            )
        )
    return summaries


def clearance(group: GroupSummary, walker: WalkerConfig) -> float:
    """
    Distance from the edge of the walker's footprint to the group's nearest point. This is S.

    Floored at zero here. The planner floors it again at its own epsilon before dividing by it.

    :param group: The group, with ground coordinates relative to the walker.
    :param walker: The one home of the footprint radius.
    :return: Clearance in meters, never negative.
    :rtype: float
    """
    center_distance = float(np.hypot(group.nearest_lateral_meters, group.nearest_forward_meters))
    return max(0.0, center_distance - walker.radius_meters)
