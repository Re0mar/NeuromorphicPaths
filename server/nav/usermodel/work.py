"""
What an avoidance cost the walker, in bits.

An episode opens when the planner asks for a turn, either through the alarm or a heading outside
tolerance, and closes when both the planner and the walker's observed heading are back
inside it. The work of the episode is the planner's cost when it opened minus its cost when it
closed, which is the slides' H at start minus H at threshold.

This layer only reads. It takes the planner's path and the heading the runtime observed, and
hands back finished episodes when asked. It never writes to a display, because a channel whose
reader may be absent cannot carry a rule, and the runtime is the one that knows where episodes go.
"""

# Standard library imports
from dataclasses import dataclass

# Local package imports
from nav.types import PlannedPath
from nav.usermodel.config import UserModelConfig
from nav.usermodel.relaxation import predicted_turn_time


@dataclass(frozen=True)
class AvoidanceEpisode:
    """One avoidance, from the planner asking for a turn to the walker having made it."""

    start_seconds: float
    end_seconds: float
    cost_at_start_bits: float
    cost_at_end_bits: float
    work_bits: float
    observed_turn_seconds: float | None
    predicted_turn_seconds: float


class WorkMeter:
    """Watches paths and headings go by and measures each avoidance as it completes."""

    def __init__(self, config: UserModelConfig) -> None:
        self._config = config
        self._completed: list[AvoidanceEpisode] = []
        self._open_since: float | None = None
        self._cost_at_start: float = 0.0
        self._commanded_error_at_start: float = 0.0
        self._turn_started_at: float | None = None
        self._turn_ended_at: float | None = None

    @property
    def episode_open(self) -> bool:
        return self._open_since is not None

    def observe(self, path: PlannedPath, observed_heading_radians: float, timestamp_seconds: float) -> None:
        """
        Feed one frame's planner output and the heading the walker actually has.

        :param path: What the planner asked for this frame.
        :param observed_heading_radians: The walker's heading relative to straight ahead, from the pose.
        :param timestamp_seconds: This frame's time.
        """
        tolerance = self._config.heading_tolerance_radians
        planner_wants_a_turn = path.alarm or abs(path.first_heading_radians) > tolerance
        walker_is_turned = abs(observed_heading_radians) > tolerance

        if self._open_since is None:
            if planner_wants_a_turn:
                self._open_since = timestamp_seconds
                self._cost_at_start = path.cumulative_cost_bits
                self._commanded_error_at_start = abs(path.first_heading_radians)
                self._turn_started_at = None
                self._turn_ended_at = None
            return

        # Track when the walker actually turned, so observed and predicted can be compared.
        if walker_is_turned and self._turn_started_at is None:
            self._turn_started_at = timestamp_seconds
        if not walker_is_turned and self._turn_started_at is not None and self._turn_ended_at is None:
            self._turn_ended_at = timestamp_seconds

        if planner_wants_a_turn or walker_is_turned:
            return

        # Both settled. The episode is over, and its work is what the field cost to cross.
        observed_turn = None
        if self._turn_started_at is not None and self._turn_ended_at is not None:
            observed_turn = self._turn_ended_at - self._turn_started_at
        self._completed.append(
            AvoidanceEpisode(
                start_seconds=self._open_since,
                end_seconds=timestamp_seconds,
                cost_at_start_bits=self._cost_at_start,
                cost_at_end_bits=path.cumulative_cost_bits,
                work_bits=self._cost_at_start - path.cumulative_cost_bits,
                observed_turn_seconds=observed_turn,
                predicted_turn_seconds=predicted_turn_time(self._commanded_error_at_start, self._config),
            )
        )
        self._open_since = None

    def completed_episodes(self) -> list[AvoidanceEpisode]:
        """Every episode that has closed, oldest first. The open one, if any, is not in it."""
        return list(self._completed)
