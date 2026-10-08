"""
What holds the arrow at its sidestep limit when the nearest thing in the corridor is 3 to 5.32 m ahead.

For every such frame, two things. A breakdown of what each cost term charges along the chosen path
against walking straight on, computed the way the dynamic program adds it up. And five tests, one per
known cause, each asking whether the frame would still sit at the limit without that cause:

- the phone pointing off the walker's line, tested by re-planning with the obstacles turned into the
  walker's direction of travel
- a wall counted once per scene cell, tested by re-planning with touching wall cells joined into one
- the goal acting only on the plan's last row, tested by re-planning with the same goal spread over
  every row
- the prior toward the previous plan holding a sidestep it started earlier, tested by re-planning
  without it
- the band itself, where a full sidestep may simply be the right plan, tested by geometry alone

The prior toward the previous plan can't be rebuilt from one frame, so it is taken from the field the
plan was made from and kept in every re-plan but its own. In the phone-pointing re-plan it stays in the
phone's frame while the obstacles are turned, which leans that re-plan toward the plan it is testing,
so it can only under-count the frames phone pointing explains.

Every re-plan goes through the planner's own functions with one input changed. None of them is an
arrow anyone sees or scores. They measure, and the planner's own arrow stays the only one judged.
"""

# Standard library imports
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum

# Third party imports
import numpy as np

# Local package imports
from nav.evaluation.config import PlannerNumbersConfig
from nav.evaluation.planner_numbers import (
    ClearanceBand,
    ReplayedFrame,
    clearance_band,
    horizon_reach_meters,
    is_pinned,
    nearest_beside_clearance,
    nearest_heading_clearance,
    sidestep_limit_degrees,
)
from nav.planner.config import GoalMode, PlannerConfig
from nav.planner.contact import ContactSurprise
from nav.planner.dynamic_programming import plan
from nav.planner.field import cost_field, lateral_grid
from nav.planner.goal import goal_position, goal_term
from nav.planner.heading import lookahead_heading, lookahead_step_index
from nav.planner.surprise import CollisionSurprise
from nav.planner.units import bits_from_nats
from nav.types import ObstaclePoint, ObstacleSet
from nav.walker import WalkerConfig

TERM_NAMES = ("surprise", "contact", "goal", "previous plan", "kinetic")
# Where the slowest clearing sidestep is looked for, across the planner's grid.
CLEARING_SEARCH_STEP_METERS = 0.05
# Points this far beyond the nearest one still count as part of what has to be passed. Two scene cells.
OBSTACLE_DEPTH_METERS = 0.5


class PinCandidate(Enum):
    """A known cause of the arrow sitting at its limit with something 3 to 5.32 m ahead."""

    PHONE_POINTING = "phone pointing"  # The plan is made along the phone's forward, not the walker's travel.
    WALL_CELLS = "wall cells"  # A wall costs once per 0.25 m cell, so it pushes harder than one obstacle.
    GOAL_LAST_ROW = "goal last row"  # The goal acts only on the plan's last row.
    RIGHT_PLAN = "right plan"  # A full sidestep is simply what clearing it takes.
    PREVIOUS_PLAN = "previous plan"  # The prior toward the previous plan holds a sidestep it started earlier.


# Measured here and fixed elsewhere: planning along the walker's direction of travel, and counting a wall
# once, are both separate pieces of work waiting on other decisions.
ON_HOLD = frozenset({PinCandidate.PHONE_POINTING, PinCandidate.WALL_CELLS})
# The one cause a change to the planner here could remove.
FIXABLE = frozenset({PinCandidate.GOAL_LAST_ROW})
# Right plan and previous plan are the planner doing what it was built to do. They are neither on hold nor
# something to fix, and each is counted under its own name.
# A phone more than this far off the walking direction points sideways or behind, which is the track's
# heading going wrong rather than a phone held off the line, so the test isn't run on it.
MAX_PHONE_OFFSET_DEGREES = 90.0


@dataclass(frozen=True)
class FrameAttribution:
    """One pinned band frame: what each term charged, and which causes explain it."""

    timestamp_seconds: float
    nearest_meters: float
    heading_degrees: float
    # Each term's cost along the chosen path minus along the straight path, in bits like the path's own cost.
    term_difference: dict[str, float]
    explained: frozenset[PinCandidate]
    # Causes that couldn't be tested on this frame, such as the phone offset while standing.
    unknown: frozenset[PinCandidate]
    # The walker's direction of travel to the right of the phone's forward, where it is known.
    phone_offset_degrees: float | None = None
    # Something nearer than the band's low edge within beside_meters of the line: the frame is in the
    # plain band but not the restated one.
    near_beside: bool = False


@dataclass(frozen=True)
class BandAttribution:
    """Every pinned band frame of a walk, and how many each cause explains."""

    band_frames: int
    pinned_band_frames: int
    explained_by: dict[PinCandidate, int]
    unknown_by: dict[PinCandidate, int]
    # Explained, and only by causes on hold.
    only_on_hold: int
    # Explained by the cause a change here could remove.
    fixable: int
    # Explained by more than one cause.
    several: int
    # Explained by none, counting a frame with an untested cause as untested rather than unexplained.
    unexplained: int
    frames: list[FrameAttribution]
    # Over every band frame, pinned or not, so a zero for walls can be read against how many there were.
    band_frames_with_walls: int = 0
    band_wall_points: int = 0
    band_points: int = 0


def term_fields(
    frame: ReplayedFrame,
    obstacles: ObstacleSet,
    planner_config: PlannerConfig,
    walker: WalkerConfig,
    goal_mode: GoalMode,
) -> dict[str, np.ndarray]:
    """
    The frame's field split by term, each (steps, cells), for the given obstacles.

    The previous-plan prior is the kept field minus everything else, since it depends on the frame
    before and can't be rebuilt from this frame alone. It is taken from the real obstacles, and is the
    same in every re-plan.

    :raises ValueError: When the frame was replayed without keep_fields.
    """
    if frame.field is None:
        raise ValueError(f"the frame at {frame.input.timestamp_seconds:.3f} s was replayed without its field, so it can't be split by term")
    grid = lateral_grid(planner_config)
    real = frame.input.obstacles
    surprise = cost_field(obstacles, grid, planner_config, walker, (CollisionSurprise(),))
    contact = cost_field(obstacles, grid, planner_config, walker, (ContactSurprise(),)) if planner_config.contact_term_enabled else np.zeros_like(surprise)
    goal = np.zeros_like(surprise)
    goal[-1] = goal_term(grid, goal_position(goal_mode, planner_config, frame.input.gaze_ground_point), planner_config)
    real_surprise = surprise if obstacles is real else cost_field(real, grid, planner_config, walker, (CollisionSurprise(),))
    if obstacles is real:
        real_contact = contact
    elif planner_config.contact_term_enabled:
        real_contact = cost_field(real, grid, planner_config, walker, (ContactSurprise(),))
    else:
        real_contact = np.zeros_like(surprise)
    previous = frame.field - real_surprise - real_contact - goal
    return {"surprise": surprise, "contact": contact, "goal": goal, "previous plan": previous}


def path_term_costs(fields: dict[str, np.ndarray], offsets: np.ndarray, planner_config: PlannerConfig) -> dict[str, float]:
    """
    What each term costs along a path, in bits, so the terms add up to the path's `cumulative_cost_bits`.

    Added up the way the dynamic program adds them, times the time step, in its natural-log units,
    then converted once at the end the way the planner converts its total.
    """
    grid = lateral_grid(planner_config)
    cells = np.argmin(np.abs(grid[None, :] - np.asarray(offsets)[:, None]), axis=1)
    rows = np.arange(len(offsets))
    dt = planner_config.time_step_seconds
    nats = {name: float(np.sum(field[rows, cells]) * dt) for name, field in fields.items()}
    lateral_speed = np.diff(np.asarray(offsets)) / dt
    nats["kinetic"] = float(np.sum(0.5 * planner_config.lateral_kinetic_weight * lateral_speed**2) * dt)
    return {name: float(bits_from_nats(value)) for name, value in nats.items()}


def replan_pinned(fields: dict[str, np.ndarray], planner_config: PlannerConfig, config: PlannerNumbersConfig) -> bool:
    """Whether the planner's own dynamic program, on this field, puts the arrow at its limit."""
    grid = lateral_grid(planner_config)
    total = sum(fields.values())
    offsets, _ = plan(total, 0.0, grid, planner_config)
    heading = lookahead_heading(offsets, lookahead_step_index(planner_config), planner_config)
    return is_pinned(float(np.degrees(heading)), sidestep_limit_degrees(planner_config), config)


def turned_into_travel(obstacles: ObstacleSet, offset_radians: float) -> ObstacleSet:
    """
    The obstacles as seen from the walker's direction of travel, which is offset_radians to the right of the phone's forward.

    Lateral is positive to the right, so turning the frame right by the offset turns every point left by it.
    """
    cosine, sine = float(np.cos(offset_radians)), float(np.sin(offset_radians))
    points = tuple(
        replace(
            point,
            lateral_meters=point.lateral_meters * cosine - point.forward_meters * sine,
            forward_meters=point.forward_meters * cosine + point.lateral_meters * sine,
        )
        for point in obstacles.points
    )
    return ObstacleSet(obstacles.timestamp_seconds, points, obstacles.groups_in_view)


def walls_joined(obstacles: ObstacleSet, cell_size_meters: float) -> ObstacleSet:
    """
    The obstacles with every run of touching wall cells made one group, as if a wall counted once.

    Cells touch when they share a side or a corner. Wall cells with a free cell between them stay apart,
    so a doorway is never closed. Points that aren't walls keep their groups.
    """
    walls = [index for index, point in enumerate(obstacles.points) if point.is_wall]
    cells = {index: (int(np.floor(obstacles.points[index].lateral_meters / cell_size_meters)), int(np.floor(obstacles.points[index].forward_meters / cell_size_meters))) for index in walls}
    parent = {index: index for index in walls}

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first in walls:
        for second in walls:
            if first < second and max(abs(cells[first][0] - cells[second][0]), abs(cells[first][1] - cells[second][1])) <= 1:
                parent[root(second)] = root(first)
    # One group id per run: the smallest id among its cells, so a run never takes a non-wall group's id.
    run_id: dict[int, int] = {}
    for index in walls:
        run_id[root(index)] = min(run_id.get(root(index), obstacles.points[index].group_id), obstacles.points[index].group_id)
    points = tuple(
        replace(point, group_id=run_id[root(index)]) if point.is_wall else point
        for index, point in enumerate(obstacles.points)
    )
    groups = len({point.group_id for point in points})
    return ObstacleSet(obstacles.timestamp_seconds, points, groups)


def sidestep_is_the_right_plan(obstacles: ObstacleSet, planner_config: PlannerConfig, config: PlannerNumbersConfig) -> bool:
    """
    Whether the slowest sidestep that gets the body past the nearest obstacle still reads as the limit.

    The obstacle is every point within OBSTACLE_DEPTH_METERS behind the nearest heading-corridor point.
    The walker must reach a lateral position at least the body half-width from all of them before
    walking level with the nearest. Moving there at a steady sideways speed over the time left, the
    arrow, a lookahead ahead, points at atan2(sideways distance covered by then, forward distance). If
    that is within the pinned tolerance of the limit, nothing gentler would do, and the limit is right.
    """
    corridor = [
        point for point in obstacles.points
        if point.forward_meters > 0.0 and abs(point.lateral_meters) <= config.heading_corridor_half_width_meters
    ]
    if not corridor:
        return False
    nearest = min(corridor, key=lambda point: point.clearance_meters)
    blocking = [point for point in obstacles.points if nearest.forward_meters <= point.forward_meters <= nearest.forward_meters + OBSTACLE_DEPTH_METERS]
    body = planner_config.body_half_width_meters
    candidates = np.arange(-planner_config.grid_half_width_meters, planner_config.grid_half_width_meters + 1e-9, CLEARING_SEARCH_STEP_METERS)
    clear = [x for x in candidates if all(abs(x - point.lateral_meters) >= body for point in blocking)]
    if not clear:
        return True
    needed = min(abs(x) for x in clear)
    time_left = nearest.forward_meters / planner_config.walking_speed_mps
    speed = min(needed / time_left, planner_config.max_lateral_speed_mps)
    lookahead = planner_config.heading_lookahead_seconds
    covered = min(speed * lookahead, needed)
    heading = float(np.degrees(np.arctan2(covered, planner_config.walking_speed_mps * lookahead)))
    return is_pinned(heading, sidestep_limit_degrees(planner_config), config) or speed >= planner_config.max_lateral_speed_mps


def band_attribution(
    frames: Sequence[ReplayedFrame],
    travel_offset_at: Callable[[float], float | None],
    planner_config: PlannerConfig,
    walker: WalkerConfig,
    goal_mode: GoalMode,
    cell_size_meters: float,
    config: PlannerNumbersConfig,
) -> BandAttribution:
    """
    Attribute every frame in the band with the arrow at its limit to the causes that explain it.

    :param frames: Replayed with keep_fields.
    :param travel_offset_at: The walker's direction of travel to the right of the phone's forward, in
        radians, at a timestamp, or None where it isn't known (standing, near a tracking break).
    :param cell_size_meters: The scene's group cell, for joining wall cells.
    :return: The counts, and each attributed frame.
    :rtype: BandAttribution
    """
    limit = sidestep_limit_degrees(planner_config)
    attributed = []
    band_frames = 0
    walls = [0, 0, 0]  # band frames with a wall point, wall points, points
    for frame in frames:
        obstacles = frame.input.obstacles
        if clearance_band(nearest_heading_clearance(obstacles, config), planner_config, config) is not ClearanceBand.BAND:
            continue
        band_frames += 1
        wall_points = sum(1 for point in obstacles.points if point.is_wall)
        walls[0] += int(wall_points > 0)
        walls[1] += wall_points
        walls[2] += len(obstacles.points)
        heading = float(np.degrees(frame.path.lookahead_heading_radians))
        if not is_pinned(heading, limit, config):
            continue
        fields = term_fields(frame, obstacles, planner_config, walker, goal_mode)
        chosen = path_term_costs(fields, frame.path.lateral_offsets_meters, planner_config)
        straight = path_term_costs(fields, np.zeros_like(frame.path.lateral_offsets_meters), planner_config)
        difference = {name: chosen[name] - straight[name] for name in TERM_NAMES}

        explained = set()
        unknown = set()
        offset = travel_offset_at(frame.input.timestamp_seconds)
        if offset is None or abs(np.degrees(offset)) > MAX_PHONE_OFFSET_DEGREES:
            unknown.add(PinCandidate.PHONE_POINTING)
        elif not replan_pinned(term_fields(frame, turned_into_travel(obstacles, offset), planner_config, walker, goal_mode), planner_config, config):
            explained.add(PinCandidate.PHONE_POINTING)
        if not replan_pinned(term_fields(frame, walls_joined(obstacles, cell_size_meters), planner_config, walker, goal_mode), planner_config, config):
            explained.add(PinCandidate.WALL_CELLS)
        spread_goal = dict(fields)
        spread_goal["goal"] = np.broadcast_to(fields["goal"][-1] / len(fields["goal"]), fields["goal"].shape).copy()
        if not replan_pinned(spread_goal, planner_config, config):
            explained.add(PinCandidate.GOAL_LAST_ROW)
        if sidestep_is_the_right_plan(obstacles, planner_config, config):
            explained.add(PinCandidate.RIGHT_PLAN)
        without_memory = dict(fields)
        without_memory["previous plan"] = np.zeros_like(fields["previous plan"])
        if not replan_pinned(without_memory, planner_config, config):
            explained.add(PinCandidate.PREVIOUS_PLAN)
        attributed.append(
            FrameAttribution(
                timestamp_seconds=frame.input.timestamp_seconds,
                nearest_meters=nearest_heading_clearance(obstacles, config),
                heading_degrees=heading,
                term_difference=difference,
                explained=frozenset(explained),
                unknown=frozenset(unknown),
                phone_offset_degrees=None if offset is None else float(np.degrees(offset)),
                near_beside=nearest_beside_clearance(obstacles, config) < config.band_low_meters,
            )
        )
    return summarize_attribution(attributed, band_frames, tuple(walls))


def summarize_attribution(attributed: list[FrameAttribution], band_frames: int, walls: tuple[int, int, int] = (0, 0, 0)) -> BandAttribution:
    """
    Count how many pinned band frames each cause explains. A frame explained by several counts in each.

    :param attributed: Every pinned band frame of a walk.
    :param band_frames: How many band frames the walk had, pinned or not.
    :param walls: Over every band frame: how many held a wall point, the wall points, and all points.
    :rtype: BandAttribution
    """
    return BandAttribution(
        band_frames=band_frames,
        pinned_band_frames=len(attributed),
        explained_by={candidate: sum(candidate in each.explained for each in attributed) for candidate in PinCandidate},
        unknown_by={candidate: sum(candidate in each.unknown for each in attributed) for candidate in PinCandidate},
        only_on_hold=sum(1 for each in attributed if each.explained and each.explained <= ON_HOLD),
        fixable=sum(1 for each in attributed if each.explained & FIXABLE),
        several=sum(1 for each in attributed if len(each.explained) > 1),
        unexplained=sum(1 for each in attributed if not each.explained and not each.unknown),
        frames=attributed,
        band_frames_with_walls=walls[0],
        band_wall_points=walls[1],
        band_points=walls[2],
    )
