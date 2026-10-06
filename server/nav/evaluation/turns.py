"""
The walker's real turns, how much heading wanders when they walk straight, and whether an obstacle
stood ahead of them before each turn.

Turns and the straight-walking spread read only the pose, so they come out the same on every replay.
The obstacle-ahead tag reads the scene, which is why the replay computes it once per scene pass.
"""

# Standard library imports
from dataclasses import dataclass
from enum import Enum

# Third party imports
import numpy as np

# Local package imports
from nav.evaluation.config import EvaluationConfig
from nav.evaluation.frames import PlannedFrame
from nav.evaluation.track import NO_PIECE, WalkerTrack, floor_heading_radians, heading_direction

class TurnSide(Enum):
    LEFT = "left"
    RIGHT = "right"


class TurnTag(Enum):
    """Whether an obstacle stood ahead of the walker in the seconds before a turn."""

    OBSTACLE_AHEAD = "obstacle ahead"
    OPEN_AHEAD = "open ahead"
    # No frame in the look-back could see the walker's line ahead. Never read as open.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Turn:
    onset_seconds: float
    end_seconds: float
    side: TurnSide
    change_radians: float


@dataclass(frozen=True)
class StraightSpread:
    """How much the heading changes over one turn window while the walker walks straight."""

    # The union of the straight windows' time, so overlapping windows aren't counted twice.
    straight_seconds: float
    window_count: int
    # Raw samples, so several walks can be pooled before any percentile is taken.
    change_samples_degrees: np.ndarray
    # The selection's own bound. No sample can exceed it, so a percentile near it was set by the
    # selection rather than by the walking.
    change_cap_degrees: float


def straight_stretch_spread(track: WalkerTrack, config: EvaluationConfig) -> StraightSpread:
    """
    Measure heading change over the turn window on stretches where the walker went straight.

    Any definition of straight caps the spread it measures, so this one is a single stated rule
    and the cap travels with the result. A window is straight when it is one defined piece and
    overlaps no turn-window change past `straight_max_change_degrees`, counting each such change's
    whole span. That span reaches past both ends of a turn, which keeps out a turn's gentle start
    and end. On their own those are small enough to pass the cap and would fill the spread up
    toward it.

    :param track: The walker's track.
    :param config: The straight window, the change cap and the turn window.
    :return: The raw change samples and how much straight walking they came from.
    :rtype: StraightSpread
    """
    window = config.steps(config.straight_window_seconds)
    turn_window = config.steps(config.turn_window_seconds)
    count = track.times_seconds.shape[0]
    heading = track.heading_radians

    over_cap = np.zeros(count, dtype=bool)
    if count > turn_window:
        changes = np.abs(heading[turn_window:] - heading[:-turn_window])
        over_cap[:-turn_window] = np.nan_to_num(changes, nan=0.0) > np.radians(config.straight_max_change_degrees)
    # Every sample inside such a change's own span, not only where it starts. A window that ends just
    # after a turn starts holds a flagged start. One that begins just before a turn ends holds none,
    # only the tail, which is why the whole span is marked.
    within_change_over_cap = np.convolve(over_cap.astype(float), np.ones(turn_window + 1))[:count] > 0

    covered = np.zeros(count, dtype=bool)
    change_starts = np.zeros(count, dtype=bool)
    window_count = 0
    for start in range(count - window):
        stop = start + window
        if not _one_defined_piece(track, start, stop) or within_change_over_cap[start:stop + 1].any():
            continue
        window_count += 1
        covered[start:stop + 1] = True
        change_starts[start:stop - turn_window + 1] = True

    starts = np.flatnonzero(change_starts)
    samples = np.abs(heading[starts + turn_window] - heading[starts])
    return StraightSpread(
        straight_seconds=float(np.count_nonzero(covered)) * config.resample_step_seconds,
        window_count=window_count,
        change_samples_degrees=np.degrees(samples),
        change_cap_degrees=config.straight_max_change_degrees,
    )


def detect_turns(track: WalkerTrack, config: EvaluationConfig) -> tuple[Turn, ...]:
    """
    Find every place the heading changes past the threshold within the turn window.

    A window counts only when every sample in it is defined and in one piece, so it never spans a
    standing pause or a track break. Overlapping windows of one sign merge into one interval, and
    windows of opposite sign are separate, which is what an S-bend is. Each interval is then trimmed
    to its largest change in its own direction, and starts no earlier than the previous turn's end,
    so a turn's change always has its side's sign and onsets come in order.

    :param track: The walker's track.
    :param config: The turn window and threshold.
    :return: The turns in time order.
    :rtype: tuple[Turn, ...]
    """
    window = config.steps(config.turn_window_seconds)
    threshold = np.radians(config.turn_threshold_degrees)
    heading = track.heading_radians
    intervals: list[list[int]] = []  # [start, end, sign]
    for start in range(heading.shape[0] - window):
        stop = start + window
        if not _one_defined_piece(track, start, stop):
            continue
        change = heading[stop] - heading[start]
        if abs(change) <= threshold:
            continue
        sign = 1 if change > 0 else -1
        if intervals and intervals[-1][2] == sign and start <= intervals[-1][1]:
            intervals[-1][1] = stop
        else:
            intervals.append([start, stop, sign])

    turns = []
    previous_end = -1
    for start, stop, sign in intervals:
        # A turn may not begin before the one before it ended. Windows of opposite sign can overlap,
        # and an overlap scored one arrow window against both sides of the pair.
        start = max(start, previous_end)
        if stop <= start:
            continue
        # Trim to the turn itself: the largest change in the interval's own direction. The interval's
        # endpoints can sit on a swing the other way, which once gave a right turn a net change of
        # -3 degrees and an onset placed while the walker was swinging left.
        oriented = sign * heading[start:stop + 1]
        rises = oriented - np.minimum.accumulate(oriented)
        peak = start + int(np.argmax(rises))
        base = start + int(np.argmin(oriented[:peak - start + 1]))
        change = float(heading[peak] - heading[base])
        if abs(change) <= threshold:
            # What's left after giving way to the turn before it is no longer a turn.
            continue
        turns.append(
            Turn(
                onset_seconds=turn_onset_seconds(track, base, peak),
                end_seconds=float(track.times_seconds[peak]),
                side=TurnSide.RIGHT if sign > 0 else TurnSide.LEFT,
                change_radians=change,
            )
        )
        previous_end = peak
    return tuple(turns)


def turn_onset_seconds(track: WalkerTrack, start_index: int, end_index: int) -> float:
    """
    When the turn began: where a line through its half-change point, at its mean rate over its
    middle half, meets the heading it started from.

    This is exact for a turn at a steady rate, needs no threshold of its own, and reads the middle of
    the turn, where smoothing changes the least. A turn that speeds up lands a little early and one
    that slows down a little late.

    :param track: The walker's track.
    :param start_index: The turn interval's first grid index.
    :param end_index: Its last grid index.
    :return: Seconds, clamped to the interval.
    :rtype: float
    """
    times = track.times_seconds[start_index:end_index + 1]
    heading = track.heading_radians[start_index:end_index + 1]
    total = heading[-1] - heading[0]
    progress = (heading - heading[0]) / total
    quarter_time = _first_crossing(times, progress, 0.25)
    half_time = _first_crossing(times, progress, 0.5)
    three_quarter_time = _first_crossing(times, progress, 0.75)
    if three_quarter_time <= quarter_time:
        return float(times[0])
    # Progress per second across the middle half. The line through the half point at that rate
    # reaches zero progress half a turn's worth of time earlier.
    rate = 0.5 / (three_quarter_time - quarter_time)
    onset = half_time - 0.5 / rate
    return float(min(max(onset, times[0]), times[-1]))


def patch_ahead_visible(frame: PlannedFrame, travel_heading_radians: float, config: EvaluationConfig) -> bool:
    """
    Whether the walker's line ahead, from `view_check_near_meters` to `obstacle_ahead_reach_meters`
    along their direction of travel, was inside the camera's image.

    The scene only reports what the camera saw. When the phone points well off the walker's line,
    "nothing ahead" would be a guess, and this is what tells the two apart.

    :param frame: Its camera pose, intrinsics, image size and floor.
    :param travel_heading_radians: The walker's heading at the frame's time.
    :param config: The near end and the reach.
    :return: True when both ends of the line land in front of the camera and inside the image.
    :rtype: bool
    """
    if frame.camera_position_world is None or frame.camera_rotation_world is None or frame.floor_world is None:
        return False
    normal = np.asarray(frame.floor_world.normal, dtype=np.float64)
    camera = np.asarray(frame.camera_position_world, dtype=np.float64)
    # The walker's foot: the camera dropped onto the floor.
    foot = camera - (normal @ camera + frame.floor_world.offset_meters) * normal
    direction = heading_direction(travel_heading_radians)
    direction = direction - (direction @ normal) * normal
    direction = direction / np.linalg.norm(direction)
    for distance in (config.view_check_near_meters, config.obstacle_ahead_reach_meters):
        point_camera = frame.camera_rotation_world.T @ (foot + distance * direction - camera)
        if point_camera[2] <= 0.0:
            return False
        column = frame.intrinsics[0, 0] * point_camera[0] / point_camera[2] + frame.intrinsics[0, 2]
        row = frame.intrinsics[1, 1] * point_camera[1] / point_camera[2] + frame.intrinsics[1, 2]
        if not (0.0 <= column <= frame.image_width_pixels and 0.0 <= row <= frame.image_height_pixels):
            return False
    return True


def tag_turn(
    turn: Turn,
    track: WalkerTrack,
    frames: list[PlannedFrame],
    config: EvaluationConfig,
) -> TurnTag:
    """
    Whether an obstacle stood ahead of the walker, in their direction of travel, before the turn.

    Obstacle points are relative to the phone's forward axis, so each is rotated by the
    phone-to-travel offset into the walker's travel frame first. A frame is judged only when it has a
    world axis, the walker's heading is known at its time, and their line ahead was in view.

    :param turn: The turn.
    :param track: The walker's track, for their heading at each frame.
    :param frames: Planned frames, in time order.
    :param config: The patch, the look-back and the share.
    :return: The tag.
    :rtype: TurnTag
    """
    judged = 0
    showing = 0
    earliest = turn.onset_seconds - config.obstacle_ahead_look_back_seconds
    for frame in frames:
        if frame.timestamp_seconds < earliest or frame.timestamp_seconds > turn.onset_seconds:
            continue
        if frame.forward_axis_world is None:
            continue
        heading = _heading_at(track, frame.timestamp_seconds)
        if heading is None or not patch_ahead_visible(frame, heading, config):
            continue
        judged += 1
        offset = float(floor_heading_radians(frame.forward_axis_world)) - heading
        if _obstacle_in_patch(frame, offset, config):
            showing += 1
    if judged == 0:
        return TurnTag.UNKNOWN
    if showing / judged >= config.obstacle_ahead_min_frame_share:
        return TurnTag.OBSTACLE_AHEAD
    return TurnTag.OPEN_AHEAD


def _one_defined_piece(track: WalkerTrack, start: int, stop: int) -> bool:
    pieces = track.piece_index[start:stop + 1]
    return (
        pieces[0] != NO_PIECE
        and bool(np.all(pieces == pieces[0]))
        and bool(np.all(np.isfinite(track.heading_radians[start:stop + 1])))
    )


def _first_crossing(times: np.ndarray, progress: np.ndarray, level: float) -> float:
    """The time progress first reaches the level, interpolated between grid samples."""
    reached = np.flatnonzero(progress >= level)
    index = int(reached[0]) if reached.size else progress.shape[0] - 1
    if index == 0:
        return float(times[0])
    before, after = progress[index - 1], progress[index]
    fraction = (level - before) / (after - before) if after != before else 0.0
    return float(times[index - 1] + fraction * (times[index] - times[index - 1]))


def _heading_at(track: WalkerTrack, time_seconds: float) -> float | None:
    index = int(np.argmin(np.abs(track.times_seconds - time_seconds)))
    if abs(track.times_seconds[index] - time_seconds) > _track_step(track):
        return None
    heading = track.heading_radians[index]
    return None if not np.isfinite(heading) else float(heading)


def _track_step(track: WalkerTrack) -> float:
    return float(track.times_seconds[1] - track.times_seconds[0]) if track.times_seconds.shape[0] > 1 else 0.0


def _obstacle_in_patch(frame: PlannedFrame, offset_radians: float, config: EvaluationConfig) -> bool:
    """Any obstacle point, rotated into the travel frame, inside the patch ahead of the walker."""
    if not frame.obstacles.points:
        return False
    lateral = np.array([point.lateral_meters for point in frame.obstacles.points])
    forward = np.array([point.forward_meters for point in frame.obstacles.points])
    # The phone's forward is offset to the right of travel by offset_radians, so a point at angle a
    # from the phone's forward is at a + offset from the walker's.
    cosine, sine = np.cos(offset_radians), np.sin(offset_radians)
    lateral_travel = lateral * cosine + forward * sine
    forward_travel = forward * cosine - lateral * sine
    inside = (forward_travel > 0.0) & (forward_travel <= config.obstacle_ahead_reach_meters)
    inside &= np.abs(lateral_travel) <= config.obstacle_ahead_half_width_meters
    return bool(inside.any())
