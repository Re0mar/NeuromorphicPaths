"""
The walker's heading and turn rate over time, from where they actually went.

The phone is hand-held and points wherever the hand points, so its own yaw is not the walker's
heading. The direction of travel is. It comes from the ARCore position track, projected onto the
floor, put on a uniform time grid and smoothed with a centered window. The track breaks wherever the
recorded position can't be trusted to join up: a tracker jump, a hole in the stream, a lost
position. Nothing is interpolated, smoothed or unwrapped across a break.
"""

# Standard library imports
from dataclasses import dataclass

# Third party imports
import numpy as np

# Local package imports
from nav.evaluation.config import EvaluationConfig
from nav.types import WORLD_UP

# Heading zero points along ARCore's -z, which is where a phone held level at session start looks.
ZERO_HEADING_DIRECTION = np.array([0.0, 0.0, -1.0])
# Forward cross up, the same construction ground_axes uses for the walker's right. Turning toward
# it raises the heading, so positive is right, the same convention as the planner's arrow.
RIGHT_OF_ZERO_HEADING = np.cross(ZERO_HEADING_DIRECTION, WORLD_UP)
# A horizontal part shorter than this has no direction worth reporting.
MIN_HORIZONTAL_LENGTH = 1e-9
# Marks a grid sample that belongs to no piece of the track.
NO_PIECE = -1


@dataclass(frozen=True)
class WalkerTrack:
    """The walker's path on a uniform grid. NaN wherever a value is not known."""

    times_seconds: np.ndarray
    positions_floor_meters: np.ndarray  # (n, 3), smoothed, horizontal part only
    heading_radians: np.ndarray  # unwrapped within each piece
    turn_rate_radians_per_second: np.ndarray  # positive is turning right
    speed_mps: np.ndarray
    piece_index: np.ndarray  # which unbroken piece each sample belongs to, NO_PIECE between pieces
    duplicates_dropped: int
    samples_too_slow: int
    breaks_for_jumps: int
    breaks_for_gaps: int
    poses_without_position: int


def wrap_radians(angles: np.ndarray | float) -> np.ndarray:
    """
    Fold angles into (-pi, pi].

    :param angles: Radians, any shape.
    :return: The same angles, folded.
    :rtype: np.ndarray
    """
    return np.pi - np.mod(np.pi - np.asarray(angles, dtype=np.float64), 2.0 * np.pi)


def floor_heading_radians(vectors_world: np.ndarray) -> np.ndarray:
    """
    The angle of each vector's horizontal part about `WORLD_UP`, positive to the right.

    "Right" is the lateral axis `ground_axes` builds for a walker facing along the vector, so this
    and the planner's arrow agree on which way is positive. A vector with no horizontal part, one
    pointing straight up or down, has no heading and gives NaN.

    :param vectors_world: (3,) or (n, 3) world vectors.
    :return: Radians in (-pi, pi], shape () or (n,).
    :rtype: np.ndarray
    """
    vectors = np.asarray(vectors_world, dtype=np.float64)
    horizontal = vectors - np.multiply.outer(vectors @ WORLD_UP, WORLD_UP)
    along_zero = horizontal @ ZERO_HEADING_DIRECTION
    along_right = horizontal @ RIGHT_OF_ZERO_HEADING
    headings = np.arctan2(along_right, along_zero)
    return np.where(np.hypot(along_zero, along_right) < MIN_HORIZONTAL_LENGTH, np.nan, headings)


def heading_direction(heading_radians: float) -> np.ndarray:
    """
    The horizontal unit vector a heading points along. The inverse of `floor_heading_radians`.

    :param heading_radians: Positive to the right of `ZERO_HEADING_DIRECTION`.
    :return: (3,) world vector.
    :rtype: np.ndarray
    """
    return np.cos(heading_radians) * ZERO_HEADING_DIRECTION + np.sin(heading_radians) * RIGHT_OF_ZERO_HEADING


def walker_track(times_seconds: np.ndarray, positions_world: np.ndarray, config: EvaluationConfig) -> WalkerTrack:
    """
    Build the walker's heading and turn rate from a recorded pose series.

    :param times_seconds: (n,) frame timestamps.
    :param positions_world: (n, 3) positions in a world with `WORLD_UP` up, a row of NaN where the
        pose had no position.
    :param config: The grid, the window, the speeds and gaps that break the track.
    :return: The track on a uniform grid.
    :rtype: WalkerTrack
    :raises ValueError: On mismatched shapes, a non-finite timestamp, an infinite or half-missing
        position, a timestamp earlier than the one before it, or fewer than two positioned samples.
    """
    times = np.asarray(times_seconds, dtype=np.float64)
    positions = np.asarray(positions_world, dtype=np.float64)
    _validate(times, positions)

    # A render-driven sender can re-send a sample before the tracker refreshes it. Dividing by a
    # zero time step would follow, so the repeat goes and the first copy stays.
    keep = np.concatenate([[True], np.diff(times) != 0.0])
    duplicates_dropped = int(np.count_nonzero(~keep))
    times, positions = times[keep], positions[keep]
    backwards = np.flatnonzero(np.diff(times) < 0.0)
    if backwards.size:
        index = int(backwards[0])
        raise ValueError(f"timestamp {times[index + 1]} comes after {times[index]}, the log is out of order")

    positioned = np.isfinite(positions).all(axis=1)
    poses_without_position = int(np.count_nonzero(~positioned))
    if np.count_nonzero(positioned) < 2:
        raise ValueError("need at least two samples with a position and distinct timestamps")

    pieces, breaks_for_jumps, breaks_for_gaps = _split_into_pieces(times, positions, positioned, config)

    step = config.resample_step_seconds
    first_time = times[positioned][0]
    last_time = times[positioned][-1]
    sample_count = int(np.floor((last_time - first_time) / step + 1e-9)) + 1
    grid = first_time + np.arange(sample_count) * step

    positions_floor = np.full((sample_count, 3), np.nan)
    heading = np.full(sample_count, np.nan)
    speed = np.full(sample_count, np.nan)
    piece_index = np.full(sample_count, NO_PIECE, dtype=np.int64)
    half_window = config.steps(config.heading_half_window_seconds)

    for number, (piece_times, piece_positions) in enumerate(pieces):
        on_grid = np.flatnonzero((grid >= piece_times[0] - 1e-9) & (grid <= piece_times[-1] + 1e-9))
        if on_grid.size == 0:
            continue
        piece_index[on_grid] = number
        if on_grid.size < 2 * half_window + 1:
            # Too short for one full window, so every value would be an edge value. Left unknown.
            continue
        horizontal = piece_positions - np.outer(piece_positions @ WORLD_UP, WORLD_UP)
        resampled = np.column_stack(
            [np.interp(grid[on_grid], piece_times, horizontal[:, axis]) for axis in range(3)]
        )
        smoothed = _centered_mean(resampled, half_window)
        velocity = np.gradient(smoothed, step, axis=0)
        positions_floor[on_grid] = smoothed
        speed[on_grid] = np.linalg.norm(velocity, axis=1)
        moving = speed[on_grid] >= config.min_walking_speed_mps
        heading[on_grid] = _unwrap_moving_runs(floor_heading_radians(velocity), moving)

    too_slow = np.isfinite(speed) & (speed < config.min_walking_speed_mps)
    turn_rate = _central_rate(heading, piece_index, step)

    return WalkerTrack(
        times_seconds=grid,
        positions_floor_meters=positions_floor,
        heading_radians=heading,
        turn_rate_radians_per_second=turn_rate,
        speed_mps=speed,
        piece_index=piece_index,
        duplicates_dropped=duplicates_dropped,
        samples_too_slow=int(np.count_nonzero(too_slow)),
        breaks_for_jumps=breaks_for_jumps,
        breaks_for_gaps=breaks_for_gaps,
        poses_without_position=poses_without_position,
    )


def _validate(times: np.ndarray, positions: np.ndarray) -> None:
    if times.ndim != 1:
        raise ValueError(f"times_seconds must be one-dimensional, got shape {times.shape}")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions_world must be (n, 3), got shape {positions.shape}")
    if positions.shape[0] != times.shape[0]:
        raise ValueError(f"{times.shape[0]} timestamps but {positions.shape[0]} positions")
    if not np.isfinite(times).all():
        raise ValueError("times_seconds holds a non-finite timestamp")
    if np.isinf(positions).any():
        raise ValueError("positions_world holds an infinite value, which is a defect upstream, not a lost position")
    missing = np.isnan(positions)
    half_missing = missing.any(axis=1) & ~missing.all(axis=1)
    if half_missing.any():
        raise ValueError(f"position row {int(np.flatnonzero(half_missing)[0])} is partly NaN, a lost position is a whole row of NaN")


def _split_into_pieces(
    times: np.ndarray,
    positions: np.ndarray,
    positioned: np.ndarray,
    config: EvaluationConfig,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int, int]:
    """Cut the series wherever consecutive positions can't be joined, and count why."""
    pieces: list[tuple[np.ndarray, np.ndarray]] = []
    current: list[int] = []
    breaks_for_jumps = 0
    breaks_for_gaps = 0
    previous_positioned = False
    for index in range(times.shape[0]):
        if not positioned[index]:
            previous_positioned = False
            continue
        if current and previous_positioned:
            last = current[-1]
            elapsed = times[index] - times[last]
            distance = float(np.linalg.norm(positions[index] - positions[last]))
            if elapsed > config.max_interpolation_gap_seconds:
                breaks_for_gaps += 1
                pieces.append(_piece(times, positions, current))
                current = []
            elif distance / elapsed > config.max_plausible_step_speed_mps:
                breaks_for_jumps += 1
                pieces.append(_piece(times, positions, current))
                current = []
        elif current:
            # A lost position ended the last piece. Counted in poses_without_position, not here.
            pieces.append(_piece(times, positions, current))
            current = []
        current.append(index)
        previous_positioned = True
    if current:
        pieces.append(_piece(times, positions, current))
    # A one-sample piece has no time span to put on the grid.
    return [piece for piece in pieces if piece[0].shape[0] >= 2], breaks_for_jumps, breaks_for_gaps


def _piece(times: np.ndarray, positions: np.ndarray, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    return times[indices], positions[indices]


def _centered_mean(values: np.ndarray, half_window: int) -> np.ndarray:
    """Mean over 2k + 1 samples around each one, the window shrunk symmetrically at the ends."""
    count = values.shape[0]
    cumulative = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    indices = np.arange(count)
    halves = np.minimum(half_window, np.minimum(indices, count - 1 - indices))
    lows = indices - halves
    highs = indices + halves + 1
    return (cumulative[highs] - cumulative[lows]) / (highs - lows)[:, None]


def _unwrap_moving_runs(headings: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """
    Unwrap heading within each run of moving samples, and start each later run within half a turn
    of where the previous one ended. Heading while standing is noise, so it is never unwrapped
    through, and comes back NaN.
    """
    result = np.full(headings.shape, np.nan)
    defined = moving & np.isfinite(headings)
    previous_end: float | None = None
    run_starts = np.flatnonzero(defined & ~np.concatenate([[False], defined[:-1]]))
    for start in run_starts:
        end = start
        while end + 1 < defined.shape[0] and defined[end + 1]:
            end += 1
        run = np.unwrap(headings[start:end + 1])
        if previous_end is not None:
            run = run + (previous_end + float(wrap_radians(run[0] - previous_end)) - run[0])
        result[start:end + 1] = run
        previous_end = float(run[-1])
    return result


def _central_rate(heading: np.ndarray, piece_index: np.ndarray, step: float) -> np.ndarray:
    """Central difference of heading, only where both neighbors are defined and in the same piece."""
    rate = np.full(heading.shape, np.nan)
    if heading.shape[0] < 3:
        return rate
    before, after = heading[:-2], heading[2:]
    same_piece = (piece_index[:-2] == piece_index[2:]) & (piece_index[:-2] != NO_PIECE)
    usable = np.isfinite(before) & np.isfinite(after) & same_piece
    rate[1:-1] = np.where(usable, (after - before) / (2.0 * step), np.nan)
    return rate
