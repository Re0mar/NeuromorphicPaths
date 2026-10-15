"""
How often the wearer had looked at the thing that raised the alarm, over a replayed walk.

Counted for a sweep of angular tolerances, time windows and credits, so the planner's own setting
can be picked from the numbers rather than guessed. Each frame where the alarm was up lands in one
of three states: noticed, not noticed, or unknown because no gaze sample fell inside the window and
no earlier look was still in credit. Unknown never goes in the denominator, since "the tracker saw
nothing" isn't "the wearer looked away".

A look counts on the frame it happened and, with a credit, on later frames whose danger sits where
the looked-at one would be by then. Every frame is walked in order for that, alarm or not, since
people look at an obstacle while it is still a few meters off.

A second table is the control. Each frame's danger is paired with the gaze from a frame several
seconds away, credit and all. If people look at the middle of the picture and obstacles sit there
too, the real row is high with nobody looking at anything, and the control row is high with it. A
setting is only worth shipping where the real row stands clear of the control.

Pure arithmetic over replayed frames. Nothing here reads a file, runs the scene or plans.
"""

# Standard library imports
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

# Third party imports
import numpy as np

# Local package imports
from nav.evaluation.planner_numbers import ReplayedFrame
from nav.planner.alarm import nearest_corridor_point
from nav.planner.attention import AttentionCredit, danger_object, gaze_on_outlines, samples_in_window, tolerance_radians
from nav.planner.config import PlannerConfig
from nav.types import GAZE_CARRY_WINDOW_SECONDS, CameraOutline, ObstaclePoint


class Look(Enum):
    """What one alarm frame's gaze says about its danger, under one setting."""

    NOTICED = "noticed"
    NOT_NOTICED = "not noticed"
    # No gaze sample inside the window and no look in credit, so the frame says nothing either way.
    UNKNOWN = "unknown"


class GazeCarried(Enum):
    """How much gaze a log's frames carry, which decides which columns can be counted at all."""

    # Every sample near the stamp, so every window can be counted.
    SAMPLES = "samples"
    # Only the one nearest sample, from a log written before the samples were carried.
    NEAREST_ONLY = "nearest only"
    NONE = "none"


@dataclass(frozen=True)
class AttentionWindow:
    """One column of the table: how far from the frame's stamp a sample may be. None is the one nearest sample."""

    seconds: float | None

    @property
    def label(self) -> str:
        return "nearest" if self.seconds is None else f"+-{self.seconds * 1000:.0f} ms"


NEAREST = AttentionWindow(None)


@dataclass(frozen=True)
class AttentionSetting:
    """One cell's setting: how long a look counts for, how wide the angle, and which samples."""

    credit_seconds: float
    tolerance_degrees: float
    window: AttentionWindow


@dataclass(frozen=True)
class LookCounts:
    """One cell: how many alarm frames landed in each state."""

    noticed: int
    not_noticed: int
    unknown: int

    @property
    def with_a_sample(self) -> int:
        """The denominator. Frames with nothing to go on are left out of it."""
        return self.noticed + self.not_noticed


@dataclass(frozen=True)
class AttentionCount:
    """Both tables of one walk, for every setting, with the counts every share is taken over."""

    frames: int
    # Frames the alarm was up on, the hold included, since every one of them would beep.
    alarm_frames: int
    # Frames the alarm's raise decision was true on, before the hold. For reference beside the above.
    raised_frames: int
    # Alarm frames held up with nothing left in the corridor, so there is no danger to look at.
    # Counted apart and in no cell.
    held_without_danger: int
    gaze: GazeCarried
    credits_seconds: tuple[float, ...]
    tolerances_degrees: tuple[float, ...]
    windows: tuple[AttentionWindow, ...]
    match_meters: float
    link_meters: float
    cap_meters: float
    real: dict[AttentionSetting, LookCounts]
    control: dict[AttentionSetting, LookCounts]
    shift_seconds: float
    # Control pairs that had to look back by the shift because looking forward ran off the end.
    control_looked_back: int
    # Alarm frames with no partner at least half the shift away either side, on a walk too short for
    # one. Their control reads unknown rather than borrowing gaze from a moment ago.
    control_unpaired: int = 0


def check_windows(windows: Sequence[AttentionWindow]) -> None:
    """
    Refuse a window the frames can't answer.

    :raises ValueError: For a window under zero or wider than what the frames carry.
    """
    for window in windows:
        if window.seconds is None:
            continue
        if not np.isfinite(window.seconds) or window.seconds < 0.0:
            raise ValueError(f"a window is zero or more seconds, got {window.seconds}")
        if window.seconds > GAZE_CARRY_WINDOW_SECONDS:
            raise ValueError(
                f"a window of {window.seconds} s is wider than the {GAZE_CARRY_WINDOW_SECONDS} s of gaze each frame carries, "
                f"so the samples past it were never logged"
            )


def gaze_carried(frames: Sequence[ReplayedFrame]) -> GazeCarried:
    """
    Whether a walk's frames carry samples, only the nearest one, or no gaze at all.

    A walk whose every window is empty has no gaze. The live glasses carry empty windows when their
    gaze stream is missing or stopped, and counting those as samples would print tables of unknowns.
    """
    if any(frame.input.gaze_pixels is not None and len(frame.input.gaze_pixels) > 0 for frame in frames):
        return GazeCarried.SAMPLES
    if any(frame.input.gaze_pixel is not None for frame in frames):
        return GazeCarried.NEAREST_ONLY
    return GazeCarried.NONE


def look(
    gaze_frame: ReplayedFrame,
    outlines: Sequence[CameraOutline | None],
    window: AttentionWindow,
    tolerance: float,
) -> Look:
    """
    One frame's gaze against one danger's object, under one window and one tolerance in radians. No credit.

    The gaze's pixels become rays through the camera matrix of the frame they came with. The outlines
    are tangents, which need no matrix, so a partner's gaze is compared fairly with another frame's danger.

    :param gaze_frame: The frame whose gaze is read. The danger's own frame, or the control's partner.
    :param outlines: The danger's object, one outline per piece, as seen in the frame it was found in.
    """
    intrinsics = gaze_frame.input.intrinsics
    if intrinsics is None:
        return Look.UNKNOWN
    if window.seconds is None:
        if gaze_frame.input.gaze_pixel is None:
            return Look.UNKNOWN
        pixels = gaze_frame.input.gaze_pixel.reshape(1, 2)
        offsets = np.zeros(1)
        window_seconds = 0.0
    else:
        pixels = gaze_frame.input.gaze_pixels
        offsets = gaze_frame.input.gaze_sample_offsets_seconds
        if pixels is None or offsets is None or not np.any(samples_in_window(offsets, window.seconds)):
            return Look.UNKNOWN
        window_seconds = window.seconds
    noticed = gaze_on_outlines(pixels, offsets, window_seconds, intrinsics, outlines, tolerance)
    return Look.NOTICED if noticed else Look.NOT_NOTICED


def control_partner(timestamps: np.ndarray, index: int, shift_seconds: float) -> tuple[int | None, bool]:
    """
    The frame nearest this one's stamp plus the shift, or minus it where plus runs off the end.

    :return: The partner's index, or None when neither side has a frame at least half the shift
        away, and whether it looked back.
    :rtype: tuple[int | None, bool]
    """
    stamp = timestamps[index]
    forward = int(np.argmin(np.abs(timestamps - (stamp + shift_seconds))))
    # A partner nearer than half the shift isn't shifted at all, it's the end of the walk.
    if abs(timestamps[forward] - stamp) >= shift_seconds / 2:
        return forward, False
    backward = int(np.argmin(np.abs(timestamps - (stamp - shift_seconds))))
    if abs(timestamps[backward] - stamp) >= shift_seconds / 2:
        return backward, True
    return None, True


class _Cell:
    """One setting's running counts and its own credit, for the real gaze or the control's."""

    def __init__(self, setting: AttentionSetting, match_meters: float, walking_speed_mps: float) -> None:
        self.setting = setting
        self.tolerance = tolerance_radians(setting.tolerance_degrees)
        self.credit = AttentionCredit(setting.credit_seconds, match_meters, walking_speed_mps)
        self.counts = dict.fromkeys(Look, 0)

    def step(
        self,
        gaze_frame: ReplayedFrame | None,
        danger: ObstaclePoint | None,
        outlines: Sequence[CameraOutline | None],
        timestamp: float,
        alarm: bool,
    ) -> None:
        """One frame: the gaze of gaze_frame, None for a control with no partner, against this frame's danger."""
        if danger is None:
            self.credit.update(None, False, timestamp)
            return
        seen = Look.UNKNOWN if gaze_frame is None else look(gaze_frame, outlines, self.setting.window, self.tolerance)
        noticed = self.credit.update(danger, seen is Look.NOTICED, timestamp)
        if not alarm:
            return
        if noticed:
            self.counts[Look.NOTICED] += 1
        elif seen is Look.UNKNOWN:
            self.counts[Look.UNKNOWN] += 1
        else:
            self.counts[Look.NOT_NOTICED] += 1

    def result(self) -> LookCounts:
        return LookCounts(noticed=self.counts[Look.NOTICED], not_noticed=self.counts[Look.NOT_NOTICED], unknown=self.counts[Look.UNKNOWN])


def attention_count(
    frames: Sequence[ReplayedFrame],
    planner_config: PlannerConfig,
    credits_seconds: Sequence[float],
    tolerances_degrees: Sequence[float],
    windows: Sequence[AttentionWindow],
    match_meters: float,
    link_meters: float,
    cap_meters: float,
    shift_seconds: float,
) -> AttentionCount:
    """
    Both tables for one replayed walk.

    The credits, tolerances, windows and match distance are this count's own, never the planner's,
    so tuning the planner can't move what the count measures. The walking speed the credit advances
    by is the planner's, since it is the pace the planner assumes the walker keeps.

    :param frames: In timestamp order, as the replay planned them.
    :param planner_config: The alarm's corridor, to find each frame's danger, and the walking speed.
    :param credits_seconds: How long a look counts for. 0 is the looking frame only.
    :param tolerances_degrees: The rows.
    :param windows: The columns.
    :param match_meters: How near a later danger must be to where the looked-at one would be by then.
    :param link_meters: How near neighboring pieces must be to count as one object with the danger.
    :param cap_meters: The furthest a piece may be from the danger and still be part of its object.
    :param shift_seconds: How far the control's gaze is taken from, above zero.
    :raises ValueError: For any setting or the shift out of range.
    """
    check_windows(windows)
    if not np.isfinite(shift_seconds) or shift_seconds <= 0.0:
        raise ValueError(f"the control's shift is above zero seconds, got {shift_seconds}")
    if not np.isfinite(link_meters) or link_meters <= 0.0:
        raise ValueError(f"the object's link distance is above zero meters, got {link_meters}")
    if not np.isfinite(cap_meters) or cap_meters <= 0.0:
        raise ValueError(f"the object's cap is above zero meters, got {cap_meters}")
    settings = [
        AttentionSetting(credit, degrees, window)
        for credit in credits_seconds
        for degrees in tolerances_degrees
        for window in windows
    ]
    real = [_Cell(setting, match_meters, planner_config.walking_speed_mps) for setting in settings]
    control = [_Cell(setting, match_meters, planner_config.walking_speed_mps) for setting in settings]
    timestamps = np.array([frame.input.timestamp_seconds for frame in frames], dtype=np.float64)
    alarm_frames = held_without_danger = looked_back = unpaired = 0
    for index, frame in enumerate(frames):
        danger = nearest_corridor_point(frame.input.obstacles, planner_config)
        alarm = frame.path.alarm
        if alarm:
            alarm_frames += 1
            if danger is None:
                held_without_danger += 1
        outlines: tuple[CameraOutline | None, ...] = ()
        if danger is not None:
            outlines = tuple(piece.camera_outline for piece in danger_object(danger, frame.input.obstacles.points, link_meters, cap_meters))
        partner, back = control_partner(timestamps, index, shift_seconds)
        if alarm and danger is not None:
            looked_back += back and partner is not None
            unpaired += partner is None
        partner_frame = None if partner is None else frames[partner]
        for real_cell, control_cell in zip(real, control):
            real_cell.step(frame, danger, outlines, frame.input.timestamp_seconds, alarm)
            control_cell.step(partner_frame, danger, outlines, frame.input.timestamp_seconds, alarm)
    return AttentionCount(
        frames=len(frames),
        alarm_frames=alarm_frames,
        raised_frames=sum(1 for frame in frames if frame.raise_decision),
        held_without_danger=held_without_danger,
        gaze=gaze_carried(frames),
        credits_seconds=tuple(credits_seconds),
        tolerances_degrees=tuple(tolerances_degrees),
        windows=tuple(windows),
        match_meters=match_meters,
        link_meters=link_meters,
        cap_meters=cap_meters,
        real={cell.setting: cell.result() for cell in real},
        control={cell.setting: cell.result() for cell in control},
        shift_seconds=shift_seconds,
        control_looked_back=looked_back,
        control_unpaired=unpaired,
    )
