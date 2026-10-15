"""
The run's gaze switch: whether the walker's eyes on the danger may quiet the alarm.

Run-wide on purpose, like the switch between a live input and a recording. Any page may flip it, and
every page is told. It lives here rather than on one planner, because the loop builds a new planner
for each new stretch of video, and the switch has to survive that.
"""

# Standard library imports
import logging
import threading
from collections.abc import Callable

log = logging.getLogger(__name__)


class GazeSilencingSwitch:
    """Holds the switch, takes requests from any thread, and tells every listener of each change."""

    def __init__(self, enabled: bool) -> None:
        """
        :param enabled: The starting state, from the command line.
        """
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
        log.info("gaze %s quiet the alarm from the next plan", "may" if enabled else "may no longer")
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
