"""
What each source needs, and what the recording tap needs.

One dataclass per source. A config that is None on RunConfig means that source was not chosen, so
a layer reading one it did not expect is reading a mistake rather than a stale default.
"""

# Standard library imports
from dataclasses import dataclass
from enum import Enum, auto


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
    # How long a Time Echo clock measurement may take before the device process gives up on it. A
    # hundred round trips on a phone hotspot take about a second, so this only fires when the phone
    # has gone.
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


class DepthScale(Enum):
    """How a checkpoint's depth relates to meters."""

    # Meters for a camera with a 300 px focal at the resolution the model ran at. The composed
    # source rescales it with the real camera's focal.
    METERS_AT_CANONICAL_FOCAL = auto()
    # Meters already. The nested model scales its depth to its own metric branch inside the library.
    METERS = auto()
    # Right only up to an unknown scale, so not meters at all.
    RELATIVE = auto()


class DepthCheckpoint(Enum):
    """The Depth Anything 3 checkpoints the pipeline knows, spelled as the Hugging Face hub spells them.

    The value is what DepthAnything3.from_pretrained takes. The set is the library's own registry of
    seven model configs, plus the 1.1 nested release that the library names as its default.
    """

    METRIC_LARGE = "depth-anything/DA3METRIC-LARGE"
    NESTED_GIANT_LARGE = "depth-anything/DA3NESTED-GIANT-LARGE"
    NESTED_GIANT_LARGE_1_1 = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    SMALL = "depth-anything/DA3-SMALL"
    BASE = "depth-anything/DA3-BASE"
    LARGE = "depth-anything/DA3-LARGE"
    GIANT = "depth-anything/DA3-GIANT"
    MONO_LARGE = "depth-anything/DA3MONO-LARGE"

    @property
    def depth_scale(self) -> DepthScale:
        match self:
            case DepthCheckpoint.METRIC_LARGE:
                return DepthScale.METERS_AT_CANONICAL_FOCAL
            case DepthCheckpoint.NESTED_GIANT_LARGE | DepthCheckpoint.NESTED_GIANT_LARGE_1_1:
                return DepthScale.METERS
            case (
                DepthCheckpoint.SMALL
                | DepthCheckpoint.BASE
                | DepthCheckpoint.LARGE
                | DepthCheckpoint.GIANT
                | DepthCheckpoint.MONO_LARGE
            ):
                return DepthScale.RELATIVE
            case _:
                # Unreachable while every member above is handled. Here so a new member that was
                # never given a scale fails loudly instead of being read as meters.
                raise ValueError(f"no depth scale recorded for {self.value}")


def _metric_checkpoint_names() -> str:
    return ", ".join(checkpoint.value for checkpoint in DepthCheckpoint if checkpoint.depth_scale is not DepthScale.RELATIVE)


def require_metric(checkpoint: DepthCheckpoint) -> DepthCheckpoint:
    """
    Refuse a checkpoint whose depth is not meters.

    The planner's clearance is meters. Relative depth would put every surprise value off by an
    unknown scale, which is the same reason the plugin route refuses its relative models.

    :param checkpoint: The checkpoint asked for.
    :return: The same checkpoint, when its depth is meters or converts to them.
    :rtype: DepthCheckpoint
    :raises ValueError: When the checkpoint gives relative depth.
    """
    if checkpoint.depth_scale is DepthScale.RELATIVE:
        raise ValueError(
            f"{checkpoint.value} gives relative depth, and the planner's clearance needs meters. "
            f"Use one of {_metric_checkpoint_names()}"
        )
    return checkpoint


def depth_checkpoint_from_name(name: str) -> DepthCheckpoint:
    """
    Turn a checkpoint name from the command line into a checkpoint the pipeline can run.

    :param name: The hub name, such as "depth-anything/DA3METRIC-LARGE".
    :return: The matching checkpoint.
    :rtype: DepthCheckpoint
    :raises ValueError: When the name is not a known checkpoint, naming the accepted ones, or when
        it is a known one that gives relative depth.
    """
    try:
        checkpoint = DepthCheckpoint(name)
    except ValueError:
        # The enum's own message names only the bad value. The accepted ones are the useful part.
        raise ValueError(f"unknown Depth Anything 3 checkpoint {name!r}. Accepted: {_metric_checkpoint_names()}") from None
    return require_metric(checkpoint)


@dataclass(frozen=True)
class EstimatorConfig:
    """The depth estimator."""

    # Metric depth, so clearance comes out in meters and the old file's cam_height rescale is not
    # needed. 1.3 GB of weights against 6.8 GB for the DA3NESTED-GIANT-LARGE the old script
    # defaulted to. That one gives meters too, but it does not fit the 4 GB card on this laptop.
    model_name: DepthCheckpoint = DepthCheckpoint.METRIC_LARGE
    process_resolution: int = 504  # The old file's --res default. Lower is faster.
    # The old file's conf_pct. Drops the least certain pixels, but only for a checkpoint that gives a
    # confidence map. The metric model gives none, so with the default model this drops nothing,
    # and the composed source logs that once.
    confidence_drop_percentile: float = 30.0
    # Used only when the camera brings no calibration and the model returns no intrinsics, which
    # the metric model does for a plain video. The old file assumed about 100 degrees horizontal.
    # A phone camera is nearer 75, so a run on phone footage should set this to match. A wrong value
    # scales distances straight ahead, and on a camera pitched down it also misreads the camera's
    # height and tilts the floor.
    fallback_half_field_of_view_degrees: float = 50.0

    def __post_init__(self) -> None:
        # A name from the command line is converted once at the parser, so a string here is a
        # caller that skipped that step.
        if not isinstance(self.model_name, DepthCheckpoint):
            raise TypeError(
                f"model_name must be a DepthCheckpoint, got {self.model_name!r}. "
                f"Convert a name with depth_checkpoint_from_name"
            )
        require_metric(self.model_name)


@dataclass(frozen=True)
class TapConfig:
    """The recording tap. log_dir None means frames are not written."""

    log_dir: str | None = None
