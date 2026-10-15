"""
The run's switches: whether the gaze may quiet the alarm, and which of the goal's sources are on.

Run-wide on purpose, like the switch between a live input and a recording. Any page may flip one, and
every page is told. They live here rather than on one planner, because the loop builds a new planner
for each new stretch of video, and the switches have to survive that. The loop reads all three before
every plan.
"""

# Standard library imports
import logging
import threading
from collections.abc import Callable

# Local package imports
from nav.planner.config import GoalSwitches, PlannerConfig, RunSwitchKind

log = logging.getLogger(__name__)


class RunSwitch:
    """One switch: holds its state, takes requests from any thread, and tells every listener of each change."""

    def __init__(self, kind: RunSwitchKind, enabled: bool) -> None:
        """
        :param kind: Which switch this is, for the log.
        :param enabled: The starting state, from the command line.
        """
        self._kind = kind
        self._lock = threading.Lock()
        self._enabled = enabled
        self._listeners: list[Callable[[bool], None]] = []

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def request(self, enabled: bool) -> None:
        """Turn the switch on or off for every plan from the next one. The state already in force changes nothing."""
        with self._lock:
            if enabled == self._enabled:
                return
            self._enabled = enabled
            listeners = list(self._listeners)
        log.info("%s %s from the next plan", self._kind.value, "on" if enabled else "off")
        for listener in listeners:
            listener(enabled)

    def subscribe(self, listener: Callable[[bool], None]) -> Callable[[], None]:
        """
        Call the listener with the state now and on each change, and return the call that stops it.

        Listeners are called on whichever thread changed the state, so one must hand off rather than block.
        """
        with self._lock:
            self._listeners.append(listener)
            current = self._enabled
        listener(current)

        def unsubscribe() -> None:
            with self._lock:
                self._listeners = [existing for existing in self._listeners if existing is not listener]

        return unsubscribe


class RunSwitches:
    """Every run-wide switch, one of each kind, started from the planner's config."""

    def __init__(self, config: PlannerConfig) -> None:
        starts = {
            RunSwitchKind.GAZE_SILENCING: config.gaze_silences_alarm,
            RunSwitchKind.GAZE_GOAL: config.gaze_goal,
            RunSwitchKind.ROUTE_GOAL: config.route_goal,
        }
        # Every kind needs a start, so a switch added to the vocabulary without one fails here at startup.
        missing = [kind.value for kind in RunSwitchKind if kind not in starts]
        if missing:
            raise ValueError(f"no starting state for the run switches {missing}")
        self._switches = {kind: RunSwitch(kind, enabled) for kind, enabled in starts.items()}

    def __getitem__(self, kind: RunSwitchKind) -> RunSwitch:
        return self._switches[kind]

    def as_mapping(self) -> dict[RunSwitchKind, RunSwitch]:
        """Each switch by its kind, for the sink that announces them."""
        return dict(self._switches)

    def goal_switches(self) -> GoalSwitches:
        """The goal's two switches as they stand now."""
        return GoalSwitches(gaze_goal=self[RunSwitchKind.GAZE_GOAL].enabled, route_goal=self[RunSwitchKind.ROUTE_GOAL].enabled)
