"""
The definitions the evaluation classifies by.

Every value here was fixed before any agreement number existed, and none is chosen by looking at one.
None is read from `PlannerConfig`, `SceneConfig` or `WalkerConfig`, because a definition borrowed from
the system being measured moves whenever that system is tuned.
"""

# Standard library imports
import dataclasses
import math
from dataclasses import dataclass

# A duration counts as a whole number of grid steps when it is this close to one.
WHOLE_STEP_TOLERANCE = 1e-9


@dataclass(frozen=True)
class EvaluationConfig:
    """How the walker's track is built, what a turn is, and what counts as an obstacle ahead."""

    # Every series is put on one uniform grid this far apart.
    resample_step_seconds: float = 0.1
    # The smoothing window is 2k + 1 samples with k from this, so 11 samples at 0.1 s. An odd count
    # is what makes the window centered, and a centered window adds no delay to the middle of a turn.
    heading_half_window_seconds: float = 0.5
    # Below this the walker is standing, and which way they face says nothing about where they go.
    min_walking_speed_mps: float = 0.3
    # Faster than any walker. A step this fast is the tracker relocalizing. pixel_walk_3 has 11
    # steps over this speed across all its segments, one of them 17.48 m in 0.033 s.
    max_plausible_step_speed_mps: float = 3.0
    # A hole in the pose stream longer than this breaks the track rather than being filled with
    # invented straight walking. pixel_display_run has a 4.61 s one.
    max_interpolation_gap_seconds: float = 0.5
    # A pause longer than this starts a new segment of the walk. The value walk_timeline.py used.
    segment_gap_seconds: float = 5.0
    # Shorter segments are not scored. A segment of exactly this length is kept.
    min_segment_seconds: float = 20.0

    # A stretch is straight when no change over the turn window, in it or within a window of it,
    # passes this cap. Any definition of straight caps what the straight-walking spread can measure,
    # so the cap is reported beside the spread, and a spread near it means the cap set the number.
    # On the straight walk that set the threshold (straight_walk_4, 2026-10-03, 65 m in one line),
    # the 99th percentile was 10.39 to 10.40 deg for every cap from 15 to 40, so the walking set
    # the number, not the cap. At 15 the half-cap guard trips on the real tail. At 25 it passes
    # with the same percentile.
    straight_window_seconds: float = 4.0
    straight_max_change_degrees: float = 25.0
    # A turn is a heading change past the threshold within this window.
    turn_window_seconds: float = 2.0
    # Measured 2026-10-03: the 99th percentile of heading change over 2 s on straight walking, 10.39
    # deg, rounded up. From straight_walk_4, a 68 s walk recorded on the phone for this purpose, with
    # 62.4 s of straight walking over 584 windows and 604 change samples. The 50th and 95th
    # percentiles were 1.71 and 6.92 deg. The unknown-heading rule at piece ends moves the window
    # and sample counts slightly but leaves the 99th percentile alone. Reproduce with:
    #   python -m nav.evaluation --spread-only frame_logs/straight_walk_4
    # The three scored walks couldn't set it. They hold under 17 s of straight walking between them,
    # because they loop around rooms.
    turn_threshold_degrees: float = 11.0
    # How the threshold is set from the pooled straight-walking spread, and when that spread can't be
    # trusted to set it. Less straight walking than this, counted once however windows overlap, is
    # too little to measure. A percentile over this share of the change cap was set by the cap.
    min_straight_seconds: float = 60.0
    threshold_percentile: float = 99.0
    max_spread_share_of_cap: float = 0.5

    # "Was there an obstacle ahead of the walker before the turn". The patch runs from the walker
    # along their direction of travel. The half-width is a walker's own, written here rather than
    # read from WalkerConfig or the alarm, so tuning either can't change which turns are tagged.
    obstacle_ahead_reach_meters: float = 3.0
    obstacle_ahead_half_width_meters: float = 0.35
    obstacle_ahead_look_back_seconds: float = 2.0
    # One noisy frame in a look-back of sixty is not an obstacle. A quarter of the frames is.
    obstacle_ahead_min_frame_share: float = 0.25
    # A frame is only judged when the walker's line ahead, from here to the reach, was in the
    # camera's view. Held upright, the depth image spans about 42 degrees side to side, so anything
    # nearer than about a meter is out of view on a normal grip.
    view_check_near_meters: float = 1.0

    # An arrow within this of the walker's direction of travel says "carry on", not left or right.
    arrow_dead_band_degrees: float = 5.0
    # The arrow's side for a turn is its mean over this long before onset.
    agreement_window_seconds: float = 1.0
    # Lead time is counted back from onset at most this far, and never past the previous turn's end.
    lead_time_cap_seconds: float = 5.0
    # An arrow this far to one side for this long is a sidestep. It counts as one "while walking
    # straight" when no turn overlaps it or starts within the quiet time after it.
    sidestep_min_degrees: float = 10.0
    sidestep_min_hold_seconds: float = 0.5
    sidestep_quiet_seconds: float = 2.0
    # The arrow sample used at a grid time must be no older than this, or that time has no arrow.
    arrow_max_staleness_seconds: float = 0.3
    # The arrow is compared with the walker's turn rate this far later, from zero up.
    correlation_max_lag_seconds: float = 3.0
    # A lag with fewer pairs gives no correlation. That's 10 s of walking on the grid. Neighboring
    # samples are strongly correlated, so far fewer would not mean anything.
    min_correlation_pairs: int = 100

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field.name} must be a number, got {value!r}")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field.name} must be a finite positive number, got {value}")
        for name in (
            "heading_half_window_seconds",
            "straight_window_seconds",
            "turn_window_seconds",
            "agreement_window_seconds",
            "correlation_max_lag_seconds",
        ):
            self._require_whole_steps(name, getattr(self, name))
        if self.min_correlation_pairs < 2 or self.min_correlation_pairs != int(self.min_correlation_pairs):
            raise ValueError(f"min_correlation_pairs must be a whole number of at least 2, got {self.min_correlation_pairs}")
        if self.threshold_percentile > 100.0:
            raise ValueError(f"threshold_percentile is a percentile, at most 100, got {self.threshold_percentile}")
        if self.max_spread_share_of_cap > 1.0:
            raise ValueError(f"max_spread_share_of_cap is a share, at most 1, got {self.max_spread_share_of_cap}")
        if self.obstacle_ahead_min_frame_share > 1.0:
            raise ValueError(f"obstacle_ahead_min_frame_share is a share, at most 1, got {self.obstacle_ahead_min_frame_share}")
        if self.turn_window_seconds > self.straight_window_seconds:
            raise ValueError(
                f"turn_window_seconds of {self.turn_window_seconds} s must fit inside "
                f"straight_window_seconds of {self.straight_window_seconds} s, or no straight change can be measured"
            )
        if self.view_check_near_meters >= self.obstacle_ahead_reach_meters:
            raise ValueError(
                f"view_check_near_meters of {self.view_check_near_meters} m must be short of "
                f"obstacle_ahead_reach_meters of {self.obstacle_ahead_reach_meters} m"
            )

    def steps(self, seconds: float) -> int:
        """
        A duration as a count of grid steps. The one place a duration becomes a step count.

        :param seconds: The duration.
        :return: The nearest whole number of steps.
        :rtype: int
        """
        # round, not int: 0.3 / 0.1 is 2.9999999999999996 in floating point, and int would make it 2.
        return int(round(seconds / self.resample_step_seconds))

    def _require_whole_steps(self, name: str, seconds: float) -> None:
        ratio = seconds / self.resample_step_seconds
        if abs(ratio - round(ratio)) > WHOLE_STEP_TOLERANCE * max(1.0, abs(ratio)):
            raise ValueError(
                f"{name} of {seconds} s is not a whole number of {self.resample_step_seconds} s steps"
            )


@dataclass(frozen=True)
class PlannerNumbersConfig:
    """
    What the planner's own whole-walk numbers are counted by: pinned, the clearance bands, the heading's corridor.

    The values are the ones every recorded heading and alarm figure was measured with, from 2026-10-02
    on. Changing one restates those figures, so a change here is a change to the record too.
    """

    # The corridor the heading is judged through: the 0.35 m footprint plus 0.15 m. Fixed here on
    # purpose. The alarm's corridor narrowed later, to the 0.30 m body, and reading the heading through
    # it moved a clear-corridor figure from 9.1 % to 14.0 % with no change to the arrow.
    heading_corridor_half_width_meters: float = 0.50
    # Nothing farther than this from the footprint may raise the alarm. Also the top of the closest band.
    close_meters: float = 1.0
    # The low edge of the band the sideways weight was tuned against. Its high edge is the horizon's
    # reach, borrowed from the planner, because the band is about what the plan can see.
    band_low_meters: float = 3.0
    # Corridor groups this close give the near noise the post safety tests are set from.
    noise_near_meters: float = 2.0
    # A heading this close to the sidestep limit counts as pinned at it.
    pinned_tolerance_degrees: float = 0.5
    # Headings are told apart to this many decimals of a degree.
    distinct_heading_decimals: int = 2
    # A share over fewer frames than this moves by whole percentage points per frame, so it isn't a result.
    min_frames_for_a_share: int = 100
    # At most this share of clear-corridor frames may sit at the limit.
    clear_target_share: float = 0.10
    # At most this share of band frames may sit at the limit. Recorded unmet since 2026-10-03, met since
    # the prior toward the previous plan on 2026-10-05. Judged on the restated band below.
    band_target_share: float = 0.50
    # The restated band, chosen 2026-10-05: a band frame counts only if nothing nearer than band_low_meters
    # sits within this far to either side of the walker's line. The heading corridor alone missed things
    # just beside it. On the classroom walk, 42 of the 59 band frames with the arrow at its limit had
    # something nearer than 3 m 0.5 to 1.0 m to the side. The plain band, on the corridor alone, is still
    # reported beside it, so the figures recorded before this stay comparable.
    beside_meters: float = 1.0

    # What a pair of consecutive plans that disagree is, checked in the order PairCause lists them.
    # The walker's origin moving faster than this between two frames is the tracker relocalizing. The
    # same figure the turn evaluation uses, read from it rather than restated.
    tracker_jump_speed_mps: float = EvaluationConfig.max_plausible_step_speed_mps
    # The phone's turn explains a pair when lining up the two frames' axes removes at least this share of
    # the disagreement. A share, not an angle, so a pair where the phone turned and the plan also flipped
    # sides still reads as a flip.
    axis_turn_explained_share: float = 0.5
    # A plan passing a point closer than this has no side of it. It went through. The body half-width as
    # of 2026-10-04, fixed here so tuning the planner's body can't change how pairs are classified.
    side_clearance_meters: float = 0.30
    # An earlier frame saw a point when one of its points lies within this of it in the world. One scene cell.
    same_point_meters: float = 0.25

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field.name} must be a number, got {value!r}")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field.name} must be a finite number of zero or more, got {value}")
        for name in ("distinct_heading_decimals", "min_frames_for_a_share"):
            if getattr(self, name) != int(getattr(self, name)):
                raise ValueError(f"{name} must be a whole number, got {getattr(self, name)}")
        for name in (
            "heading_corridor_half_width_meters",
            "close_meters",
            "pinned_tolerance_degrees",
            "min_frames_for_a_share",
            "tracker_jump_speed_mps",
            "axis_turn_explained_share",
            "same_point_meters",
        ):
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be above zero")
        for name in ("clear_target_share", "band_target_share", "axis_turn_explained_share"):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} is a share, at most 1, got {getattr(self, name)}")
        if self.close_meters >= self.band_low_meters:
            raise ValueError(f"close_meters of {self.close_meters} m must be short of band_low_meters of {self.band_low_meters} m")
        if self.beside_meters < self.heading_corridor_half_width_meters:
            raise ValueError(
                f"beside_meters of {self.beside_meters} m must reach at least the heading corridor's "
                f"{self.heading_corridor_half_width_meters} m, or the restated band would count frames the plain one doesn't"
            )
