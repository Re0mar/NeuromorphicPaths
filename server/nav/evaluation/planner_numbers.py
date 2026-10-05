"""
The planner's whole-walk numbers: how often the arrow sits at its sidestep limit, how many headings it
takes, how the alarm behaves, and how far consecutive plans disagree once both are placed in the world.

These are the definitions every recorded heading and alarm figure was measured by since 2026-10-02.
They are pure arithmetic over replayed frames. Nothing here reads a file, runs the scene or plans.

Three values come from `PlannerConfig` on purpose, because they are about the planner's own reach:
the sidestep limit, the horizon's reach, and how far forward each step of a plan lies. Everything
else is a fixed constant in `PlannerNumbersConfig`, so tuning the planner moves the numbers only
through the arrow, never through what counts.
"""

# Standard library imports
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

# Third party imports
import numpy as np

# Local package imports
from nav.evaluation.config import PlannerNumbersConfig
from nav.planner.alarm import corridor_points
from nav.planner.config import PlannerConfig
from nav.types import ObstacleSet, PlannedPath


class ClearanceBand(Enum):
    """Where the nearest thing in the heading's corridor is, which decides what the heading is judged against."""

    CLOSE = "close"  # Nearer than close_meters, including anything already inside the footprint.
    NEAR = "near"  # From close_meters to band_low_meters.
    BAND = "band"  # From band_low_meters to the horizon's reach. What the sideways weight was tuned against.
    CLEAR = "clear"  # At or past the reach, or nothing in the corridor at all.


@dataclass(frozen=True)
class PlannerInput:
    """What the planner is given for one frame, and where the walker stood in the world when it was taken."""

    timestamp_seconds: float
    obstacles: ObstacleSet
    # The walker's frame in the world. None on a frame with no world pose.
    origin: np.ndarray | None
    lateral_axis: np.ndarray | None
    forward_axis: np.ndarray | None
    gaze_ground_point: np.ndarray | None


@dataclass(frozen=True)
class ReplayedFrame:
    """One planned frame of a replay: the input, the path exactly as `plan` returned it, and the alarm's raw decision."""

    input: PlannerInput
    path: PlannedPath
    # alarm_raised before the hold. The path's alarm is after it.
    raise_decision: bool
    # The field the plan was made from, every term and both priors, when the replay was asked to keep
    # it. A grid the size of the horizon per frame, so only the breakdown that reads it asks.
    field: np.ndarray | None = None


@dataclass(frozen=True)
class PlannerNumbers:
    """Every whole-walk figure, with the frame count behind each one so a share can be read with its base."""

    frames: int
    sidestep_limit_degrees: float
    horizon_reach_meters: float
    pinned_by_band: dict[ClearanceBand, int]
    frames_by_band: dict[ClearanceBand, int]
    # The restated band: band frames with nothing nearer than its low edge within beside_meters.
    restated_band_frames: int
    restated_band_pinned: int
    pinned_frames: int
    distinct_headings: int
    alarm_on_frames: int
    alarm_changes: int
    # Raise decisions with the nearest point in the alarm's own corridor farther than close_meters.
    raised_past_close: int
    # How long the shown alarm stayed up after its raise decision was last true.
    longest_hold_seconds: float
    near_noise_median_meters: float | None
    near_noise_readings: int
    disagreement_median_meters: float | None
    disagreement_p90_meters: float | None
    disagreement_pairs: int
    pairs_without_world: int
    duplicate_pairs: int


def sidestep_limit_degrees(planner_config: PlannerConfig) -> float:
    """
    The widest heading the planner can ask for: a full sidestep at walking pace.

    Borrowed from the planner on purpose. "Pinned" means at the planner's own limit, so the limit has
    to move with it.
    """
    return float(np.degrees(np.arctan2(planner_config.max_lateral_speed_mps, planner_config.walking_speed_mps)))


def horizon_reach_meters(planner_config: PlannerConfig) -> float:
    """How far ahead the plan sees at walking pace. Borrowed on purpose, as the band's high edge."""
    return planner_config.walking_speed_mps * planner_config.horizon_seconds


def plan_forward_distances(planner_config: PlannerConfig) -> np.ndarray:
    """How far forward each step of a plan lies at walking pace, the same steps the planner plans on."""
    steps = int(round(planner_config.horizon_seconds / planner_config.time_step_seconds)) + 1
    return np.arange(steps) * planner_config.time_step_seconds * planner_config.walking_speed_mps


def is_pinned(heading_degrees: float, limit_degrees: float, config: PlannerNumbersConfig) -> bool:
    """Whether a heading sits at the sidestep limit, on either side."""
    return abs(abs(heading_degrees) - limit_degrees) <= config.pinned_tolerance_degrees


def nearest_heading_clearance(obstacles: ObstacleSet, config: PlannerNumbersConfig) -> float:
    """
    The clearance of the nearest point ahead in the heading's corridor, or inf when there is none.

    The heading's corridor, not the alarm's. See `PlannerNumbersConfig.heading_corridor_half_width_meters`.
    """
    return min(
        (
            point.clearance_meters for point in obstacles.points
            if point.forward_meters > 0.0 and abs(point.lateral_meters) <= config.heading_corridor_half_width_meters
        ),
        default=float("inf"),
    )


def clearance_band(nearest_meters: float, planner_config: PlannerConfig, config: PlannerNumbersConfig) -> ClearanceBand:
    """Which band a frame falls in. Each band holds its low edge and stops short of its high edge."""
    if nearest_meters < config.close_meters:
        return ClearanceBand.CLOSE
    if nearest_meters < config.band_low_meters:
        return ClearanceBand.NEAR
    if nearest_meters < horizon_reach_meters(planner_config):
        return ClearanceBand.BAND
    return ClearanceBand.CLEAR


def nearest_beside_clearance(obstacles: ObstacleSet, config: PlannerNumbersConfig) -> float:
    """The clearance of the nearest point ahead within beside_meters of the walker's line, or inf when there is none."""
    return min(
        (
            point.clearance_meters for point in obstacles.points
            if point.forward_meters > 0.0 and abs(point.lateral_meters) <= config.beside_meters
        ),
        default=float("inf"),
    )


def in_restated_band(obstacles: ObstacleSet, planner_config: PlannerConfig, config: PlannerNumbersConfig) -> bool:
    """
    Whether a frame is in the restated band: in the plain band, and nothing nearer than its low edge beside the line.

    The plain band reads only the heading corridor, so a frame with something 2 m away just outside it
    still counted as "3 to 5.32 m ahead". This one doesn't count it.
    """
    in_plain_band = clearance_band(nearest_heading_clearance(obstacles, config), planner_config, config) is ClearanceBand.BAND
    return in_plain_band and nearest_beside_clearance(obstacles, config) >= config.band_low_meters


def share(count: int, total: int, config: PlannerNumbersConfig) -> float | None:
    """count over total, or None when total is too few frames for a share to mean anything."""
    if total < config.min_frames_for_a_share:
        return None
    return count / total


def plan_disagreement(earlier: ReplayedFrame, later: ReplayedFrame, forward_distances: np.ndarray) -> float | None:
    """
    Mean lateral gap between two consecutive plans, compared at the same place on the floor.

    The earlier plan's points go into the world through its own walker frame, then into the later
    frame's axes, and are compared with the later plan at the same forward distance. Comparing at
    the same place rather than the same time keeps the gap between assumed and real walking speed
    out of the number.

    :return: Meters, or None when the plans do not overlap.
    :rtype: float | None
    :raises ValueError: When either frame has no world pose.
    """
    first, second = earlier.input, later.input
    if first.origin is None or second.origin is None:
        raise ValueError(
            f"a plan with no world pose can't be placed in the world: frames at "
            f"{first.timestamp_seconds:.3f} s and {second.timestamp_seconds:.3f} s"
        )
    world = (
        first.origin
        + np.outer(forward_distances, first.forward_axis)
        + np.outer(earlier.path.lateral_offsets_meters, first.lateral_axis)
    )
    relative = world - second.origin
    forward_in_later = relative @ second.forward_axis
    lateral_in_later = relative @ second.lateral_axis
    overlap = (forward_in_later >= 0.0) & (forward_in_later <= forward_distances[-1])
    if not overlap.any():
        return None
    later_lateral = np.interp(forward_in_later[overlap], forward_distances, later.path.lateral_offsets_meters)
    return float(np.mean(np.abs(lateral_in_later[overlap] - later_lateral)))


def whole_walk_numbers(frames: Sequence[ReplayedFrame], planner_config: PlannerConfig, config: PlannerNumbersConfig) -> PlannerNumbers:
    """
    Every whole-walk figure over consecutive replayed frames.

    :param frames: In timestamp order, one planner across all of them, as a live run plans.
    :param planner_config: For the three borrowed values and the alarm's corridor.
    :param config: Everything else that decides what counts.
    :return: The figures with their frame counts.
    :rtype: PlannerNumbers
    :raises ValueError: When frames is empty.
    """
    if not frames:
        raise ValueError("no replayed frames to count, so there are no whole-walk numbers")
    limit = sidestep_limit_degrees(planner_config)
    forward_distances = plan_forward_distances(planner_config)

    headings = np.array([float(np.degrees(frame.path.first_heading_radians)) for frame in frames])
    pinned = np.array([is_pinned(heading, limit, config) for heading in headings])
    bands = [clearance_band(nearest_heading_clearance(frame.input.obstacles, config), planner_config, config) for frame in frames]
    pinned_by_band = {band: 0 for band in ClearanceBand}
    frames_by_band = {band: 0 for band in ClearanceBand}
    for band, at_limit in zip(bands, pinned):
        frames_by_band[band] += 1
        pinned_by_band[band] += int(at_limit)
    restated = [in_restated_band(frame.input.obstacles, planner_config, config) for frame in frames]
    restated_pinned = sum(1 for inside, at_limit in zip(restated, pinned) if inside and at_limit)

    alarms = [frame.path.alarm for frame in frames]
    changes = sum(1 for before, after in zip(alarms, alarms[1:]) if before != after)
    raised_past_close = 0
    for frame in frames:
        in_the_way = corridor_points(frame.input.obstacles, planner_config)
        nearest = min((point.clearance_meters for point in in_the_way), default=float("inf"))
        if frame.raise_decision and nearest > config.close_meters:
            raised_past_close += 1

    longest_hold = 0.0
    last_raised: float | None = None
    for frame in frames:
        stamp = frame.input.timestamp_seconds
        if frame.raise_decision:
            last_raised = stamp
        elif frame.path.alarm and last_raised is not None:
            longest_hold = max(longest_hold, stamp - last_raised)

    near_noise = [
        point.noise_scale_meters
        for frame in frames
        for point in frame.input.obstacles.points
        if point.forward_meters > 0.0
        and abs(point.lateral_meters) <= config.heading_corridor_half_width_meters
        and point.clearance_meters <= config.noise_near_meters
    ]

    gaps: list[float] = []
    without_world = 0
    duplicates = 0
    for earlier, later in zip(frames, frames[1:]):
        if earlier.input.origin is None or later.input.origin is None:
            without_world += 1
            continue
        if earlier.input.timestamp_seconds == later.input.timestamp_seconds:
            duplicates += 1
        gap = plan_disagreement(earlier, later, forward_distances)
        if gap is not None:
            gaps.append(gap)

    return PlannerNumbers(
        frames=len(frames),
        sidestep_limit_degrees=limit,
        horizon_reach_meters=horizon_reach_meters(planner_config),
        pinned_by_band=pinned_by_band,
        frames_by_band=frames_by_band,
        restated_band_frames=sum(restated),
        restated_band_pinned=restated_pinned,
        pinned_frames=int(pinned.sum()),
        distinct_headings=len(np.unique(np.round(headings, config.distinct_heading_decimals))),
        alarm_on_frames=sum(alarms),
        alarm_changes=changes,
        raised_past_close=raised_past_close,
        longest_hold_seconds=longest_hold,
        near_noise_median_meters=float(np.median(near_noise)) if near_noise else None,
        near_noise_readings=len(near_noise),
        disagreement_median_meters=float(np.median(gaps)) if gaps else None,
        disagreement_p90_meters=float(np.percentile(gaps, 90)) if gaps else None,
        disagreement_pairs=len(gaps),
        pairs_without_world=without_world,
        duplicate_pairs=duplicates,
    )


# *******************************************
# What a disagreeing pair of plans is
# *******************************************


class PairCause(Enum):
    """
    Why two consecutive plans disagree, checked in this order with the first match winning.

    The order runs from causes outside the planner to the planner's own choice. A tracker jump moves
    the whole frame and says nothing about the plan, so it is ruled out first. A turning phone swings
    a plan that didn't change through the world, which is about where the plan is drawn rather than
    what it chose. Only then is the plan itself compared with what was in front of it.
    """

    FRAME_JUMP = "frame jump"  # The tracker relocalized between the frames.
    AXIS_TURN = "axis turn"  # The phone turned, and lining up the axes removes most of the gap.
    NEW_OBSTACLE = "new obstacle"  # The plans pass a point on opposite sides, and the earlier frame hadn't seen it.
    SIDE_FLIP = "side flip"  # The plans pass a point both frames saw on opposite sides.
    SAME_SIDE_SHIFT = "same side shift"  # No point is passed on opposite sides, but the plans differ.


@dataclass(frozen=True)
class DiagnosedPair:
    """One pair of consecutive plans at or above the percentile, with what it is."""

    earlier_seconds: float
    later_seconds: float
    disagreement_meters: float
    cause: PairCause
    # Lateral and forward, in the later frame, of the point the plans pass on opposite sides. None
    # unless the cause is a flip or a new obstacle.
    deciding_point: tuple[float, float] | None
    earlier_heading_degrees: float
    later_heading_degrees: float
    # How far the walker frame's forward axis turned between the two frames, signed, positive left.
    axis_turn_degrees: float


def _earlier_plan_in_later_frame(earlier: ReplayedFrame, later: ReplayedFrame, forward_distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The earlier plan's points as forward and lateral distances in the later frame, through the world."""
    first, second = earlier.input, later.input
    world = first.origin + np.outer(forward_distances, first.forward_axis) + np.outer(earlier.path.lateral_offsets_meters, first.lateral_axis)
    relative = world - second.origin
    return relative @ second.forward_axis, relative @ second.lateral_axis


def _aligned_disagreement(earlier: ReplayedFrame, later: ReplayedFrame, forward_distances: np.ndarray) -> float | None:
    """
    The disagreement if the axes hadn't turned: both plans in their own walker frames, the earlier one
    moved by how far the walker went along the later frame's axes.
    """
    moved = later.input.origin - earlier.input.origin
    forward_in_later = forward_distances - float(moved @ later.input.forward_axis)
    lateral_in_later = earlier.path.lateral_offsets_meters - float(moved @ later.input.lateral_axis)
    overlap = (forward_in_later >= 0.0) & (forward_in_later <= forward_distances[-1])
    if not overlap.any():
        return None
    later_lateral = np.interp(forward_in_later[overlap], forward_distances, later.path.lateral_offsets_meters)
    return float(np.mean(np.abs(lateral_in_later[overlap] - later_lateral)))


def _axis_turn_degrees(earlier: ReplayedFrame, later: ReplayedFrame) -> float:
    forward, lateral = later.input.forward_axis, later.input.lateral_axis
    # The earlier forward axis seen in the later frame. Lateral is positive to the right, and a turn to
    # the left leaves the old forward on the right, so a left turn comes out positive.
    return float(np.degrees(np.arctan2(earlier.input.forward_axis @ lateral, earlier.input.forward_axis @ forward)))


def _seen_earlier(earlier: ReplayedFrame, later: ReplayedFrame, lateral: float, forward: float, config: PlannerNumbersConfig) -> bool:
    """Whether the earlier frame had a point within same_point_meters of this later-frame point, matched through the world."""
    first, second = earlier.input, later.input
    target = second.origin + forward * second.forward_axis + lateral * second.lateral_axis
    for point in first.obstacles.points:
        place = first.origin + point.forward_meters * first.forward_axis + point.lateral_meters * first.lateral_axis
        relative = place - target
        # Only the floor plane counts. The two origins sit at camera height and may differ in it.
        gap = np.hypot(relative @ second.forward_axis, relative @ second.lateral_axis)
        if gap <= config.same_point_meters:
            return True
    return False


def pair_cause(
    earlier: ReplayedFrame,
    later: ReplayedFrame,
    planner_config: PlannerConfig,
    config: PlannerNumbersConfig,
) -> tuple[PairCause, tuple[float, float] | None]:
    """
    What a pair of consecutive plans that disagree is.

    A plan's side of a point is the sign of the plan's lateral offset minus the point's, at the
    point's forward distance in the later frame. A plan within side_clearance_meters of the point has
    no side. It went through, which is counted as no flip.

    Obstacles are matched between frames by their place in the world, never by group id. The scene's
    ids are cells of a grid fixed to the walker, so one id names a different patch of floor once the
    walker has moved.

    :return: The cause, and the deciding point's lateral and forward in the later frame when there is one.
    :rtype: tuple[PairCause, tuple[float, float] | None]
    :raises ValueError: When either frame has no world pose.
    """
    first, second = earlier.input, later.input
    if first.origin is None or second.origin is None:
        raise ValueError(
            f"a pair needs both frames placed in the world: frames at {first.timestamp_seconds:.3f} s "
            f"and {second.timestamp_seconds:.3f} s"
        )
    elapsed = second.timestamp_seconds - first.timestamp_seconds
    moved = second.origin - first.origin
    step = float(np.hypot(moved @ second.forward_axis, moved @ second.lateral_axis))
    if elapsed > 0.0 and step / elapsed > config.tracker_jump_speed_mps:
        return PairCause.FRAME_JUMP, None

    forward_distances = plan_forward_distances(planner_config)
    in_world = plan_disagreement(earlier, later, forward_distances)
    aligned = _aligned_disagreement(earlier, later, forward_distances)
    if in_world and aligned is not None and aligned <= (1.0 - config.axis_turn_explained_share) * in_world:
        return PairCause.AXIS_TURN, None

    earlier_forward, earlier_lateral = _earlier_plan_in_later_frame(earlier, later, forward_distances)
    order = np.argsort(earlier_forward)
    for point in sorted(second.obstacles.points, key=lambda each: each.forward_meters):
        if not 0.0 < point.forward_meters <= forward_distances[-1]:
            continue
        if not earlier_forward.min() <= point.forward_meters <= earlier_forward.max():
            continue
        earlier_gap = float(np.interp(point.forward_meters, earlier_forward[order], earlier_lateral[order])) - point.lateral_meters
        later_gap = float(np.interp(point.forward_meters, forward_distances, later.path.lateral_offsets_meters)) - point.lateral_meters
        if min(abs(earlier_gap), abs(later_gap)) <= config.side_clearance_meters:
            continue
        if np.sign(earlier_gap) != np.sign(later_gap):
            deciding = (point.lateral_meters, point.forward_meters)
            if _seen_earlier(earlier, later, point.lateral_meters, point.forward_meters, config):
                return PairCause.SIDE_FLIP, deciding
            return PairCause.NEW_OBSTACLE, deciding
    return PairCause.SAME_SIDE_SHIFT, None


def disagreement_pairs(
    frames: Sequence[ReplayedFrame],
    planner_config: PlannerConfig,
    config: PlannerNumbersConfig,
    percentile: float,
) -> tuple[float | None, list[DiagnosedPair]]:
    """
    Every consecutive pair whose disagreement is at or above the percentile, with its cause.

    Pairs without a world frame or with no overlap have no disagreement and are left out, as
    whole_walk_numbers leaves them out of its percentiles.

    :return: The percentile's value in meters, or None when no pair has one, and the pairs at or above it.
    :rtype: tuple[float | None, list[DiagnosedPair]]
    :raises ValueError: When the percentile is outside 0 to 100.
    """
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"a percentile is from 0 to 100, got {percentile}")
    forward_distances = plan_forward_distances(planner_config)
    scored = []
    for earlier, later in zip(frames, frames[1:]):
        if earlier.input.origin is None or later.input.origin is None:
            continue
        gap = plan_disagreement(earlier, later, forward_distances)
        if gap is not None:
            scored.append((earlier, later, gap))
    if not scored:
        return None, []
    threshold = float(np.percentile([gap for _, _, gap in scored], percentile))
    pairs = []
    for earlier, later, gap in scored:
        if gap < threshold:
            continue
        cause, deciding = pair_cause(earlier, later, planner_config, config)
        pairs.append(
            DiagnosedPair(
                earlier_seconds=earlier.input.timestamp_seconds,
                later_seconds=later.input.timestamp_seconds,
                disagreement_meters=gap,
                cause=cause,
                deciding_point=deciding,
                earlier_heading_degrees=float(np.degrees(earlier.path.first_heading_radians)),
                later_heading_degrees=float(np.degrees(later.path.first_heading_radians)),
                axis_turn_degrees=_axis_turn_degrees(earlier, later),
            )
        )
    return threshold, pairs
