"""What each sink needs. The null sink needs nothing, so it does not appear here."""

# Standard library imports
from dataclasses import dataclass


@dataclass(frozen=True)
class DebugWindowConfig:
    """The OpenCV window. Smoothing is display only and off, because the planner does not smooth."""

    show_field: bool = True
    smooth_display: bool = False
    smoothing_weight: float = 0.7  # The old file's EMA weight, kept for the eye and nothing else.


@dataclass(frozen=True)
class WebConfig:
    """The websocket server the browser page connects to."""

    port: int = 8765


@dataclass(frozen=True)
class PhoneAppConfig:
    """The port the laptop listens on for the Pixel app's path connection.

    The phone connects here with the same laptop address it sends depth to, so the laptop never
    needs the phone's address. There is no accept timeout: the sink waits for a phone for as long
    as the run lasts, and plans without one.
    """

    port: int = 9100  # Beside the depth port the laptop listens on, so the two never collide.
    bind_address: str = "0.0.0.0"
