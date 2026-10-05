"""
What each source needs, and what the recording tap needs.

One dataclass per source. A config that is None on RunConfig means that source was not chosen, so
a layer reading one it did not expect is reading a mistake rather than a stale default.
"""

# Standard library imports
from dataclasses import dataclass
from enum import Enum


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
    # How long without a scene frame before the log says so. It warns and keeps waiting, because a
    # wifi drop in the middle of a walk should not end the walk.
    stall_warning_seconds: float = 5.0
    # How long a Time Echo clock measurement may take before it is abandoned. A hundred round trips
    # on a phone hotspot take about a second, so this only fires when the phone has gone.
    time_echo_timeout_seconds: float = 5.0
    # A capture folder from examples/capture_neon_stream.py, played back in place of the glasses at
    # the pace it was recorded. None means the glasses themselves.
    replay_dir: str | None = None


@dataclass(frozen=True)
class ArCoreConfig:
    """The socket the laptop listens on for the Pixel app's depth frames."""

    port: int = 9000
    bind_address: str = "0.0.0.0"  # Every interface, so the phone reaches it over the hotspot.
    accept_timeout_seconds: float = 30.0  # Long enough to launch the app, short enough to notice it never came.


class NeonPluginModel(Enum):
    """The models the Neon Player depth plugin offers, spelled as it spells its cache file names."""

    METRIC_LARGE = "DA3Metric-Large"
    SMALL = "DA3-Small"
    BASE = "DA3-Base"

    @property
    def is_metric(self) -> bool:
        # Only the metric model's values are meters. The others cache relative inverse depth that
        # the plugin scales to 0 to 255 for display, which the planner cannot use for clearance.
        return self is NeonPluginModel.METRIC_LARGE

    @property
    def cache_stem(self) -> str:
        return self.value.replace(" ", "_")


@dataclass(frozen=True)
class NeonPluginConfig:
    """A Neon recording folder that the Neon Player depth plugin has already run over."""

    recording_dir: str
    model: NeonPluginModel = NeonPluginModel.METRIC_LARGE
    sample_tolerance_seconds: float = 0.05  # How far an IMU or gaze sample may sit from a scene frame.


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
    # Used only when the model returns no intrinsics, which the metric model does for a plain video.
    # The old file assumed about 100 degrees horizontal. A phone camera is nearer 75, so a run on
    # phone footage should set this to match, or distances straight ahead come out short.
    fallback_half_field_of_view_degrees: float = 50.0


@dataclass(frozen=True)
class TapConfig:
    """The recording tap. log_dir None means frames are not written."""

    log_dir: str | None = None
