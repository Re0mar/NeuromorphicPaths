"""What each sink needs. The debug window and the null sink need nothing, so neither appears here."""

# Standard library imports
from dataclasses import dataclass


@dataclass(frozen=True)
class WebConfig:
    """The websocket server the browser page connects to."""

    port: int = 8765


@dataclass(frozen=True)
class PhoneAppConfig:
    """The Pixel app's listening socket, which the phone sink connects out to."""

    address: str
    port: int
