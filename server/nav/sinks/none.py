"""The sink for a run that only wants the log, the frame recording, or the exit code."""

# Local package imports
from nav.types import PlannedPath


class NullSink:
    """Accepts every path and does nothing with it."""

    def start(self) -> None:
        """Nothing to open."""

    def publish(self, path: PlannedPath) -> None:
        """Nothing. Present so the loop can publish without asking what it is talking to."""

    def close(self) -> None:
        """Nothing to release."""
