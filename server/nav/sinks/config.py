"""What each sink needs. The null sink needs nothing, so it does not appear here."""

# Standard library imports
from dataclasses import dataclass
from pathlib import Path

# The self-signed certificate the page is served with, committed beside it. The README there says
# why it is self-signed, why it is committed, and how to remake it.
TLS_DIRECTORY = Path(__file__).parent / "tls"
DEFAULT_CERTIFICATE_PATH = TLS_DIRECTORY / "demo_page.crt"
DEFAULT_KEY_PATH = TLS_DIRECTORY / "demo_page.key"


@dataclass(frozen=True)
class DebugWindowConfig:
    """The OpenCV window. Smoothing is display only and off, because the planner does not smooth."""

    show_field: bool = True
    smooth_display: bool = False
    smoothing_weight: float = 0.7  # The old file's EMA weight, kept for the eye and nothing else.


@dataclass(frozen=True)
class WebConfig:
    """The server the browser page and its websocket come from, over TLS."""

    port: int = 8765
    # The plan view and the depth picture take about 50 ms to build, on a thread that competes with
    # the planner. A person watching a laptop doesn't need more than this. Every path still goes out.
    max_pictures_per_second: float = 10.0
    # The page is HTTPS because the browser's own video decoder, WebCodecs, only exists on a secure
    # origin, and a phone opens the page by the laptop's address. The pair is committed so every
    # laptop serves the same certificate and each browser accepts it once.
    certificate_path: str = str(DEFAULT_CERTIFICATE_PATH)
    key_path: str = str(DEFAULT_KEY_PATH)
    # How many access units may wait for one browser on the video socket. Two seconds at the
    # glasses' 30 frames a second, one keyframe gap, so a browser further behind than that is
    # resynced at the next keyframe rather than fed a growing backlog.
    video_queue_limit: int = 60

    def __post_init__(self) -> None:
        if not self.max_pictures_per_second > 0:
            raise ValueError(f"max_pictures_per_second must be above 0, got {self.max_pictures_per_second}")
        # The resync puts the description back and then one more item, so the queue needs two slots.
        if self.video_queue_limit < 2:
            raise ValueError(f"video_queue_limit must be at least 2, got {self.video_queue_limit}")
        # Checked here rather than at start, so a run with a wrong path fails before the model loads.
        for name, path in (("certificate_path", self.certificate_path), ("key_path", self.key_path)):
            if not Path(path).is_file():
                raise ValueError(f"{name} {path} is not a file")


@dataclass(frozen=True)
class PhoneAppConfig:
    """The port the laptop listens on for the Pixel app's path connection.

    The phone connects here with the same laptop address it sends depth to, so the laptop never
    needs the phone's address. There is no accept timeout: the sink waits for a phone for as long
    as the run lasts, and plans without one.
    """

    port: int = 9100  # Beside the depth port the laptop listens on, so the two never collide.
    bind_address: str = "0.0.0.0"
