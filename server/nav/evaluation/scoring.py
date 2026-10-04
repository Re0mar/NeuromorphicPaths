"""
Scores the planner's arrow against the walker's real turns.

The arrow is drawn over the camera image, so it points relative to the phone. The direction it sends
the walker is the phone's direction plus the arrow, and that is what gets compared with where the
walker went. Nothing here re-plans, gates or corrects the arrow. Where the phone leads a turn, its
pointing can lend the arrow agreement, and the phone offset is kept so its size is on the record.

Unknown is never zero. A turn with no arrow before it, or whose lead can't be known, keeps None all the
way into the summary, and the summary counts it separately.
"""

# Standard library imports
from dataclasses import dataclass

# Third party imports
import numpy as np

# Local package imports
from nav.evaluation.config import EvaluationConfig
from nav.evaluation.frames import PlannedFrame
from nav.evaluation.track import NO_PIECE, WalkerTrack, floor_heading_radians, wrap_radians
from nav.evaluation.turns import Turn, TurnSide, TurnTag


@dataclass(frozen=True)
class ArrowSeries:
    times_seconds: np.ndarray
    # The direction the arrow sends the walker, against their direction of travel, positive right.
    # NaN where there is no fresh arrow, no world axis, or no walker heading.
    travel_frame_radians: np.ndarray
    # The phone's forward minus the walker's heading, kept so its size is on the record.
    phone_offset_radians: np.ndarray


@dataclass(frozen=True)
class TurnScore:
    turn: Turn
    tag: TurnTag
    # None when the arrow said carry on, or there was no arrow to read.
    arrow_side: TurnSide | None
    # False when no defined arrow sample fell in the agreement window.
    arrow_read: bool
    # None when not read. False covers both the wrong side and carry on.
    agreed: bool | None
    # None when it didn't agree, or the arrow had let go by onset.
    lead_seconds: float | None
    # The lead ran into the cap, the previous turn's end, the start of the recording, or a stretch
    # where the walker's heading was unknown, so it is "at least" this long.
    lead_at_limit: bool


@dataclass(frozen=True)
class Sidestep:
    start_seconds: float
    end_seconds: float
    side: TurnSide


@dataclass(frozen=True)
class LaggedCorrelation:
    lags_seconds: np.ndarray
    # NaN at a lag with too few pairs or no variance.
    correlation: np.ndarray
    pair_counts: np.ndarray
    peak_lag_seconds: float | None
    peak_correlation: float | None


@dataclass(frozen=True)
class ScoreSummary:
    turns: int
    arrow_read: int
    agreed: int
    wrong_side: int
    carry_on: int
    not_read: int
    # None when no lead is known. known_leads says how many the median is over.
    median_lead_seconds: float | None
    known_leads: int
    leads_at_limit: int


@dataclass(frozen=True)
class WalkSummary:
    by_tag: dict[TurnTag, ScoreSummary]
    sidesteps: int
    sidestep_seconds: float
    # Per minute of walking with an arrow to read. None when there was none.
    sidesteps_per_minute: float | None


def arrow_in_travel_frame(frames: list[PlannedFrame], track: WalkerTrack, config: EvaluationConfig) -> ArrowSeries:
    """
    Put the planner's arrow on the track's grid, as the direction it sends the walker.

    :param frames: Planned frames in time order.
    :param track: The walker's track.
    :param config: How old an arrow may be and still count.
    :return: The arrow and the phone offset on the track's grid.
    :rtype: ArrowSeries
    """
    times = track.times_seconds
    travel = np.full(times.shape, np.nan)
    offset = np.full(times.shape, np.nan)
    if not frames:
        return ArrowSeries(times, travel, offset)
    frame_times = np.array([frame.timestamp_seconds for frame in frames])
    phone_headings = np.array(
        [np.nan if frame.forward_axis_world is None else float(floor_heading_radians(frame.forward_axis_world)) for frame in frames]
    )
    arrows = np.array([frame.arrow_radians for frame in frames])
    latest = np.searchsorted(frame_times, times, side="right") - 1
    has_frame = latest >= 0
    latest = np.clip(latest, 0, None)
    fresh = has_frame & (times - frame_times[latest] <= config.arrow_max_staleness_seconds)
    usable = fresh & np.isfinite(phone_headings[latest]) & np.isfinite(track.heading_radians)
    offset[usable] = wrap_radians(phone_headings[latest][usable] - track.heading_radians[usable])
    travel[usable] = wrap_radians(arrows[latest][usable] + offset[usable])
    return ArrowSeries(times, travel, offset)


def score_turn(
    turn: Turn,
    tag: TurnTag,
    arrow: ArrowSeries,
    previous_turn_end_seconds: float | None,
    config: EvaluationConfig,
) -> TurnScore:
    """
    Whether the arrow pointed to the turn's side just before it started, and for how long it had.

    :param turn: The turn.
    :param tag: Its obstacle-ahead tag, carried into the score.
    :param arrow: The arrow on the track's grid.
    :param previous_turn_end_seconds: Where a lead must stop, so it never counts a lean that
        belonged to the turn before. None for the first turn.
    :param config: The agreement window, the dead band and the cap.
    :return: The score.
    :rtype: TurnScore
    """
    times = arrow.times_seconds
    values = arrow.travel_frame_radians
    in_window = (times >= turn.onset_seconds - config.agreement_window_seconds - 1e-9) & (times <= turn.onset_seconds + 1e-9)
    readable = values[in_window & np.isfinite(values)]
    if readable.size == 0:
        return TurnScore(turn, tag, None, False, None, None, False)

    dead_band = np.radians(config.arrow_dead_band_degrees)
    # A circular mean. A straight mean of +179 and -176 degrees is about 0, which would read an
    # arrow pointing almost straight back as one saying carry on.
    mean = float(np.arctan2(np.mean(np.sin(readable)), np.mean(np.cos(readable))))
    if abs(mean) <= dead_band:
        return TurnScore(turn, tag, None, True, False, None, False)
    side = TurnSide.RIGHT if mean > 0 else TurnSide.LEFT
    if side != turn.side:
        return TurnScore(turn, tag, side, True, False, None, False)

    lead, at_limit = _lead(times, values, turn, side, previous_turn_end_seconds, dead_band, config)
    return TurnScore(turn, tag, side, True, True, lead, at_limit)


def find_sidesteps(arrow: ArrowSeries, turns: tuple[Turn, ...], config: EvaluationConfig) -> tuple[Sidestep, ...]:
    """
    Stretches where the arrow asked for a sidestep and the walker went straight.

    :param arrow: The arrow on the track's grid.
    :param turns: The walker's turns.
    :param config: How far and how long a lean must be, and the quiet time after it.
    :return: The sidesteps in time order.
    :rtype: tuple[Sidestep, ...]
    """
    times = arrow.times_seconds
    values = arrow.travel_frame_radians
    limit = np.radians(config.sidestep_min_degrees)
    step = float(times[1] - times[0]) if times.shape[0] > 1 else 0.0
    defined = np.flatnonzero(np.isfinite(values))
    # +1 for a lean right, -1 for a lean left, 0 for neither, over the defined samples only. An
    # unknown stretch no longer than max_interpolation_gap_seconds doesn't end a lean, because one
    # unknown sample in the middle would otherwise split one sidestep into two.
    sides = np.where(values[defined] > limit, 1, np.where(values[defined] < -limit, -1, 0))
    sidesteps = []
    index = 0
    while index < defined.shape[0]:
        if sides[index] == 0:
            index += 1
            continue
        end = index
        while (
            end + 1 < defined.shape[0]
            and sides[end + 1] == sides[index]
            and times[defined[end + 1]] - times[defined[end]] <= config.max_interpolation_gap_seconds + 1e-9
        ):
            end += 1
        start_time, end_time = float(times[defined[index]]), float(times[defined[end]])
        long_enough = end_time - start_time + step >= config.sidestep_min_hold_seconds - 1e-9
        quiet_until = end_time + config.sidestep_quiet_seconds
        walker_turned = any(turn.onset_seconds <= quiet_until and turn.end_seconds >= start_time for turn in turns)
        if long_enough and not walker_turned:
            sidesteps.append(Sidestep(start_time, end_time, TurnSide.RIGHT if sides[index] > 0 else TurnSide.LEFT))
        index = end + 1
    return tuple(sidesteps)


def lagged_correlation(arrow: ArrowSeries, track: WalkerTrack, config: EvaluationConfig) -> LaggedCorrelation:
    """
    The correlation between the arrow now and the walker's turn rate a lag later, at each lag.

    The lag with the highest correlation is a second estimate of lead time that needs no threshold.

    :param arrow: The arrow on the track's grid.
    :param track: The walker's track.
    :param config: The largest lag and the fewest pairs a correlation may rest on.
    :return: The correlation at each lag, and its peak.
    :rtype: LaggedCorrelation
    """
    lag_count = config.steps(config.correlation_max_lag_seconds) + 1
    count = track.times_seconds.shape[0]
    correlations = np.full(lag_count, np.nan)
    pair_counts = np.zeros(lag_count, dtype=np.int64)
    for lag in range(lag_count):
        if lag >= count:
            break
        now = arrow.travel_frame_radians[:count - lag]
        later = track.turn_rate_radians_per_second[lag:]
        same_piece = (track.piece_index[:count - lag] == track.piece_index[lag:]) & (track.piece_index[lag:] != NO_PIECE)
        paired = same_piece & np.isfinite(now) & np.isfinite(later)
        pair_counts[lag] = int(np.count_nonzero(paired))
        if pair_counts[lag] < config.min_correlation_pairs:
            continue
        x, y = now[paired], later[paired]
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            continue
        correlations[lag] = float(np.corrcoef(x, y)[0, 1])
    lags = np.arange(lag_count) * config.resample_step_seconds
    if np.isfinite(correlations).any():
        peak = int(np.nanargmax(correlations))
        return LaggedCorrelation(lags, correlations, pair_counts, float(lags[peak]), float(correlations[peak]))
    return LaggedCorrelation(lags, correlations, pair_counts, None, None)


def summarize(scores: list[TurnScore], sidesteps: tuple[Sidestep, ...], arrow_seconds: float) -> WalkSummary:
    """
    Count the scores per tag, keeping every unknown out of the medians and in its own count.

    :param scores: Every scored turn.
    :param sidesteps: Sidesteps while walking straight.
    :param arrow_seconds: How much walking had an arrow to read. Sidesteps can only happen there,
        so the rate is per minute of that, and there is no rate without it.
    :return: The summary.
    :rtype: WalkSummary
    """
    by_tag = {}
    for tag in TurnTag:
        tagged = [score for score in scores if score.tag == tag]
        leads = [score.lead_seconds for score in tagged if score.lead_seconds is not None]
        by_tag[tag] = ScoreSummary(
            turns=len(tagged),
            arrow_read=sum(1 for score in tagged if score.arrow_read),
            agreed=sum(1 for score in tagged if score.agreed is True),
            wrong_side=sum(1 for score in tagged if score.arrow_read and score.arrow_side is not None and not score.agreed),
            carry_on=sum(1 for score in tagged if score.arrow_read and score.arrow_side is None),
            not_read=sum(1 for score in tagged if not score.arrow_read),
            median_lead_seconds=float(np.median(leads)) if leads else None,
            known_leads=len(leads),
            leads_at_limit=sum(1 for score in tagged if score.lead_at_limit),
        )
    sidestep_seconds = float(sum(sidestep.end_seconds - sidestep.start_seconds for sidestep in sidesteps))
    per_minute = len(sidesteps) / (arrow_seconds / 60.0) if arrow_seconds > 0 else None
    return WalkSummary(by_tag, len(sidesteps), sidestep_seconds, per_minute)


def _lead(
    times: np.ndarray,
    values: np.ndarray,
    turn: Turn,
    side: TurnSide,
    previous_turn_end_seconds: float | None,
    dead_band: float,
    config: EvaluationConfig,
) -> tuple[float | None, bool]:
    """How long the arrow had held the turn's side, unbroken, at onset."""
    at_or_before = np.flatnonzero(times <= turn.onset_seconds + 1e-9)
    if at_or_before.size == 0:
        return None, False
    sign = 1.0 if side == TurnSide.RIGHT else -1.0

    def on_side(index: int) -> bool:
        return bool(np.isfinite(values[index]) and sign * values[index] > dead_band)

    onset_index = int(at_or_before[-1])
    if not on_side(onset_index):
        # The mean agreed, but the arrow had let go by onset. How long it held isn't known.
        return None, False
    limit = turn.onset_seconds - config.lead_time_cap_seconds
    if previous_turn_end_seconds is not None:
        limit = max(limit, previous_turn_end_seconds)
    earliest = onset_index
    at_limit = False
    while True:
        previous = earliest - 1
        if previous < 0:
            # The grid starts while the arrow is still on side, so the lead runs past what is recorded.
            at_limit = True
            break
        if times[previous] < limit - 1e-9:
            at_limit = True
            break
        if not np.isfinite(values[previous]):
            # Unknown, not let go. The walker's heading wasn't known there, so the arrow may have
            # held on through it. The lead is at least this long, never exactly this long.
            at_limit = True
            break
        if not on_side(previous):
            break
        earliest = previous
    return float(turn.onset_seconds - times[earliest]), at_limit
