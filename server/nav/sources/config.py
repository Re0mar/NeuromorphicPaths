"""
What each source needs, and what the recording tap needs.

One dataclass per source. A config that is None on RunConfig means that source was not chosen, so
a layer reading one it did not expect is reading a mistake rather than a stale default.
"""

# Standard library imports
from dataclasses import dataclass


@dataclass(frozen=True)
class VideoConfig:
    """A recording on disk, or an IP camera app's stream URL. OpenCV opens both the same way."""

    path: str


@dataclass(frozen=True)
class NeonConfig:
    """A Pupil Labs Neon on the network.

    address is None when the device should be discovered over mDNS, which is the normal case.
    Supplying one is the fallback for a network that blocks multicast between subnets, which
    university wifi usually does. That is why the old script carried a hard-coded IP.
    """

    address: str | None = None
    port: int = 8080  # The Neon real-time API's documented default.
    discovery_timeout_seconds: float = 10.0  # The client's own default search duration.


@dataclass(frozen=True)
class ArCoreConfig:
    """The port the laptop listens on for the Pixel app's depth frames."""

    port: int = 9000


@dataclass(frozen=True)
class NeonPluginConfig:
    """A Neon recording folder that the Neon Player depth plugin has already run over."""

    recording_dir: str


@dataclass(frozen=True)
class LoggedConfig:
    """A frame log written by the recording tap on an earlier run."""

    log_dir: str


@dataclass(frozen=True)
class EstimatorConfig:
    """The depth estimator."""

    # Metric depth, so clearance comes out in meters and the old file's cam_height rescale is not
    # needed. 1.3 GB of weights against 6.8 GB for the DA3NESTED-GIANT-LARGE the old script
    # defaulted to, which returns relative depth and does not fit the 4 GB card on this laptop.
    model_name: str = "depth-anything/DA3METRIC-LARGE"
    process_resolution: int = 504  # The old file's --res default. Lower is faster.
    confidence_drop_percentile: float = 30.0  # The old file's conf_pct. Drops the least certain pixels.


@dataclass(frozen=True)
class TapConfig:
    """The recording tap. log_dir None means frames are not written."""

    log_dir: str | None = None
