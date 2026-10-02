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
    """The Pixel app's listening socket, which the phone sink connects out to."""

    address: str
    port: int = 9100  # Beside the depth port the laptop listens on, so the two never collide.
