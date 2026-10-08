"""
One loop, several displays.

A walk puts the arrow on the phone and the depth view in a browser at the same time: one is for
the walker and one is for whoever is watching the laptop. The loop talks to one sink, so this
holds the sinks a run named and hands each of them every path. Nothing upstream learns how many
displays are behind it.

A sink that fails to start ends the run, because a display asked for and silently absent is worse
than a run that says why it stopped. A sink that fails while publishing is dropped and the rest
keep going, because by then a walker is looking at one of them.
"""

# Standard library imports
import logging
from collections.abc import Iterable

# Third party imports
import numpy as np

# Local package imports
from nav.types import DebugSink, DebugView, PathSink, PlannedPath

log = logging.getLogger(__name__)


class FanOutSink:
    """Every path to every sink a run asked for, in the order they were named."""

    def __init__(self, sinks: Iterable[PathSink]) -> None:
        self._all = list(sinks)
        if not self._all:
            raise ValueError("a fan-out sink needs at least one sink to hand paths to")
        # Two lists on purpose. A sink dropped mid-run stops being published to and still holds a
        # socket or a window, so close has to see every sink that was ever started.
        self._serving = list(self._all)

    @property
    def sinks(self) -> tuple[PathSink, ...]:
        """The sinks still being served. A sink that failed while publishing is no longer here."""
        return tuple(self._serving)

    def start(self) -> None:
        """Start every sink, in order. A sink that cannot start ends the run, with its own message."""
        for sink in self._all:
            sink.start()

    def publish(self, path: PlannedPath) -> None:
        """Hand the path to every sink, debug sinks included, with nothing extra for them to draw."""
        for sink in tuple(self._serving):
            self._hand_over(sink, lambda target: target.publish(path))

    def publish_debug(self, path: PlannedPath, field: np.ndarray, grid: np.ndarray, view: DebugView) -> None:
        """
        Hand the path to every sink, and the field and view to the ones that can draw them.

        The loop dispatches on the protocol once, at the top. This dispatches again per sink,
        because a run can name a browser and a phone together and only one of them draws a field.
        """
        for sink in tuple(self._serving):
            if isinstance(sink, DebugSink):
                self._hand_over(sink, lambda target: target.publish_debug(path, field, grid, view))
            else:
                self._hand_over(sink, lambda target: target.publish(path))

    def close(self) -> None:
        """Close every sink, including the ones that were dropped mid-run."""
        for sink in self._all:
            try:
                sink.close()
            except Exception as unexpected_error:
                # Closing is the last thing that happens, and one sink that will not close must
                # not leave the others open.
                log.error(
                    "UNEXPECTED %s closing %s, may need a handler",
                    type(unexpected_error).__name__,
                    type(sink).__name__,
                    exc_info=True,
                )

    def _hand_over(self, sink: PathSink, deliver) -> None:
        try:
            deliver(sink)
        except Exception as unexpected_error:
            # Each sink already handles its own expected failures: a phone that left, a browser
            # that closed. What reaches here is nobody's predicted case, so it is logged as one
            # and that sink is dropped rather than taking down the displays that still work.
            log.error(
                "UNEXPECTED %s from %s, dropping that display for the rest of the run",
                type(unexpected_error).__name__,
                type(sink).__name__,
                exc_info=True,
            )
            self._serving.remove(sink)
