"""
Every knob in one place, and the two factories that turn a chosen kind into an object.

Configuration is command line only. Nothing in this package reads an environment variable, so a
run is fully described by the arguments that started it and a log line can repeat them back.

This is also the only module allowed to mention SourceKind or SinkKind. Once the parser has turned
a string into an enum, the rest of the package receives built objects and never asks what kind
they came from. A test enforces that.
"""

# Standard library imports
import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

# Local package imports
from nav.types import DepthFrameSource, PathSink


class SourceKind(Enum):
    """Where frames come from. Values are the spellings the command line accepts."""

    VIDEO_FILE = "video_file"
    NEON_LIVE = "neon_live"
    ARCORE_TCP = "arcore_tcp"
    NEON_PLUGIN = "neon_plugin"
    LOGGED = "logged"


class SinkKind(Enum):
    """Where the planned path goes. Values are the spellings the command line accepts."""

    DEBUG_WINDOW = "debug_window"
    WEB = "web"
    PHONE_APP = "phone_app"
    NONE = "none"


class GoalMode(Enum):
    """What the planner aims at when nothing is in the way."""

    AHEAD = "ahead"
    GAZE = "gaze"


# These two sources carry RGB only, so they need the depth estimator composed in behind them.
# The other three already deliver depth, so asking them for a model name would be meaningless.
ESTIMATOR_BACKED_SOURCES = frozenset({SourceKind.VIDEO_FILE, SourceKind.NEON_LIVE})


@dataclass(frozen=True)
class VideoConfig:
    """A recording on disk, or an IP camera app's stream URL. OpenCV opens both the same way."""

    path: str


@dataclass(frozen=True)
class NeonConfig:
    """A Pupil Labs Neon on the network. The address was a constant in the old script."""

    address: str
    port: int = 8080  # The Neon real-time API's documented default.


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
    """The depth estimator. model_name has no default until the metric checkpoint is verified."""

    model_name: str
    process_resolution: int = 504  # The old file's --res default. Lower is faster.
    confidence_drop_percentile: float = 30.0  # The old file's conf_pct. Drops the least certain pixels.


@dataclass(frozen=True)
class SceneConfig:
    """Depth pixels to grouped ground obstacles. Defaults from the old file where it had one."""

    depth_stride: int = 2  # Old file's stride.
    voxel_size_meters: float = 0.05  # Old file's voxel.
    ankle_height_meters: float = 0.20  # Old file's h_min. Below this is floor, not obstacle.
    head_height_meters: float = 2.00  # Old file's h_max. Above this the walker passes under it.
    cell_size_meters: float = 0.25  # Coarser than the old file's 0.10 grid, because a cell is now one obstacle.
    grid_half_width_meters: float = 3.0  # Old file's x_half.
    grid_forward_meters: float = 6.0  # Old file's z_max.
    min_points_per_cell: int = 2  # One point is as likely to be depth noise as an object.
    noise_window_seconds: float = 0.5  # Half a second of history is what N is measured over.
    min_history_samples: int = 3  # Below this a standard deviation says nothing.
    noise_floor_meters: float = 0.01  # N never goes below this, or surprise divides by almost zero.
    floor_max_tilt_degrees: float = 35.0  # Old file's floor sanity check.
    floor_min_offset_meters: float = 0.3  # Old file's floor sanity check.
    wall_cell_min_height_meters: float = 1.5  # A cell with points this tall is treated as a wall.


@dataclass(frozen=True)
class PlannerConfig:
    """The surprise field and the dynamic program over it.

    Four of these are the professor's values from the lecture and the paper, and are marked. The
    rest are walking values we chose and expect to tune once there are real recordings.
    """

    time_step_seconds: float = 0.1  # His dt.
    horizon_seconds: float = 3.8  # His horizon.
    clearance_epsilon_meters: float = 0.06  # His epsilon, the floor under S.
    lateral_kinetic_weight: float = 0.055  # His weight on the lateral kinetic term.
    surprise_cap: float = 2.0e4  # His cap on a single point's surprise.
    walking_speed_mps: float = 1.4  # Ours. Average walking pace, against his 5.0 for a cyclist.
    grid_half_width_meters: float = 3.0  # Ours. Matches the scene grid.
    grid_spacing_meters: float = 0.1  # Ours.
    max_lateral_speed_mps: float = 1.0  # Ours. How fast a walker can sidestep.
    wall_noise_multiplier: float = 3.0  # Ours. A wall is worth avoiding further out than a post.
    predict_motion: bool = False  # Off until the scene's group velocities are trusted.
    alarm_time_to_contact_seconds: float = 1.0  # Under a second to contact turns the display red.
    goal_distance_meters: float = 4.0  # Old file's goal_dist.
    goal_tolerance_meters: float = 1.5  # Half the corridor width, so the goal term picks among safe paths.


@dataclass(frozen=True)
class UserModelConfig:
    """How the walker is modeled. Measures only, never steers."""

    seconds_per_bit: float = 0.25  # Placeholder until a walker is actually measured. See BUG-004.
    heading_tolerance_radians: float = 0.05  # About three degrees. Inside this, a turn is finished.


@dataclass(frozen=True)
class WalkerConfig:
    """The walker's footprint. The one home of the radius, read by the scene and the planner."""

    radius_meters: float = 0.35  # Shoulder half-width plus a margin.


@dataclass(frozen=True)
class TapConfig:
    """The recording tap. log_dir None means frames are not written."""

    log_dir: str | None = None


@dataclass(frozen=True)
class WebConfig:
    """The websocket server the browser page connects to."""

    port: int = 8765


@dataclass(frozen=True)
class PhoneAppConfig:
    """The Pixel app's listening socket, which the phone sink connects out to."""

    address: str
    port: int


@dataclass(frozen=True)
class RunConfig:
    """One whole run. Everything the factories and the loop need, and nothing they have to find."""

    source_kind: SourceKind
    sink_kind: SinkKind
    goal_mode: GoalMode
    scene: SceneConfig = field(default_factory=SceneConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    usermodel: UserModelConfig = field(default_factory=UserModelConfig)
    walker: WalkerConfig = field(default_factory=WalkerConfig)
    tap: TapConfig = field(default_factory=TapConfig)
    video: VideoConfig | None = None
    neon: NeonConfig | None = None
    arcore: ArCoreConfig | None = None
    neon_plugin: NeonPluginConfig | None = None
    logged: LoggedConfig | None = None
    estimator: EstimatorConfig | None = None
    web: WebConfig | None = None
    phone_app: PhoneAppConfig | None = None
    reconnect: bool = False
    realtime_replay: bool = False
    verbose: bool = False
    estimator_factory: Callable[..., object] | None = None
    """Composition-root hook, set by tests to inject a stub estimator instead of loading a model.

    Deliberately has no command line flag. A run started from the command line always gets the
    real estimator, so a flag here could only ever make a live run quietly fake.
    """


def _positive_float(text: str) -> float:
    """A measurement that is meaningless at or below zero, such as a footprint radius."""
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError(f"must be greater than zero, got {value}")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than zero, got {value}")
    return value


def _port_number(text: str) -> int:
    value = int(text)
    if not 1 <= value <= 65535:
        raise argparse.ArgumentTypeError(f"must be a port between 1 and 65535, got {value}")
    return value


def _percentile(text: str) -> float:
    """A percentage of pixels to drop. At 100 there would be nothing left to unproject."""
    value = float(text)
    if not 0.0 <= value < 100.0:
        raise argparse.ArgumentTypeError(f"must be at least 0 and below 100, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser, one group per layer.

    Per-source and per-sink arguments carry their own prefix, so reading a command line says
    which source it configures without checking what --source was set to.

    :return: The parser. Parsing is done by build_run_config.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(prog="nav", description="Plan a walking path from a depth stream.")

    pipeline = parser.add_argument_group("pipeline")
    pipeline.add_argument("--source", required=True, choices=[kind.value for kind in SourceKind])
    pipeline.add_argument("--sink", required=True, choices=[kind.value for kind in SinkKind])
    pipeline.add_argument("--goal", choices=[mode.value for mode in GoalMode], default=GoalMode.AHEAD.value)

    video = parser.add_argument_group("video_file source")
    video.add_argument("--path", help="recording on disk, or an IP camera stream URL")

    neon = parser.add_argument_group("neon_live source")
    neon.add_argument("--neon-address", help="the Neon's address on the network")
    neon.add_argument("--neon-port", type=_port_number, default=8080)

    arcore = parser.add_argument_group("arcore_tcp source")
    arcore.add_argument("--arcore-port", type=_port_number, default=9000)
    arcore.add_argument("--reconnect", action="store_true", help="keep listening after the phone disconnects")

    neon_plugin = parser.add_argument_group("neon_plugin source")
    neon_plugin.add_argument("--recording-dir", help="a Neon recording the depth plugin has run over")

    logged = parser.add_argument_group("logged source")
    logged.add_argument("--log-dir", help="a frame log written by --record-to")
    logged.add_argument("--realtime", action="store_true", help="replay at the recorded frame rate")

    estimator = parser.add_argument_group("depth estimator")
    estimator.add_argument("--model", help="Depth Anything 3 checkpoint, required for an RGB source")
    estimator.add_argument("--process-resolution", type=_positive_int, default=504)
    estimator.add_argument("--confidence-drop-percentile", type=_percentile, default=30.0)

    tap = parser.add_argument_group("recording tap")
    tap.add_argument("--record-to", help="write every frame to this directory as it passes")

    walker = parser.add_argument_group("walker")
    walker.add_argument("--walker-radius", type=_positive_float, default=0.35, help="footprint radius in meters")

    web = parser.add_argument_group("web sink")
    web.add_argument("--web-port", type=_port_number, default=8765)

    phone = parser.add_argument_group("phone_app sink")
    phone.add_argument("--phone-address", help="the Pixel app's address")
    phone.add_argument("--phone-port", type=_port_number, default=9100)

    logging_group = parser.add_argument_group("logging")
    logging_group.add_argument("--verbose", action="store_true")

    return parser


def build_run_config(argv: list[str] | None = None) -> RunConfig:
    """
    Parse a command line into a RunConfig, rejecting a source or sink that is missing an argument.

    Strings become enums exactly once, here. Every required argument is checked against the chosen
    source and sink rather than being marked required in the parser, because what is required
    depends on what was chosen.

    :param argv: Argument list, or None to read sys.argv.
    :return: A fully populated configuration.
    :rtype: RunConfig
    """
    parser = build_parser()
    arguments = parser.parse_args(argv)

    source_kind = SourceKind(arguments.source)
    sink_kind = SinkKind(arguments.sink)
    goal_mode = GoalMode(arguments.goal)

    video = None
    neon = None
    arcore = None
    neon_plugin = None
    logged = None

    match source_kind:
        case SourceKind.VIDEO_FILE:
            _require(parser, arguments.path, "--path", source_kind)
            video = VideoConfig(path=arguments.path)
        case SourceKind.NEON_LIVE:
            _require(parser, arguments.neon_address, "--neon-address", source_kind)
            neon = NeonConfig(address=arguments.neon_address, port=arguments.neon_port)
        case SourceKind.ARCORE_TCP:
            arcore = ArCoreConfig(port=arguments.arcore_port)
        case SourceKind.NEON_PLUGIN:
            _require(parser, arguments.recording_dir, "--recording-dir", source_kind)
            neon_plugin = NeonPluginConfig(recording_dir=arguments.recording_dir)
        case SourceKind.LOGGED:
            _require(parser, arguments.log_dir, "--log-dir", source_kind)
            logged = LoggedConfig(log_dir=arguments.log_dir)
        case _:
            # Unreachable while every member above is handled. Here so that adding a member and
            # forgetting this function fails loudly instead of leaving every config None.
            raise ValueError(f"no argument handling for {source_kind}")

    estimator = None
    if source_kind in ESTIMATOR_BACKED_SOURCES:
        _require(parser, arguments.model, "--model", source_kind)
        estimator = EstimatorConfig(
            model_name=arguments.model,
            process_resolution=arguments.process_resolution,
            confidence_drop_percentile=arguments.confidence_drop_percentile,
        )

    web = None
    phone_app = None

    match sink_kind:
        case SinkKind.WEB:
            web = WebConfig(port=arguments.web_port)
        case SinkKind.PHONE_APP:
            _require(parser, arguments.phone_address, "--phone-address", sink_kind)
            phone_app = PhoneAppConfig(address=arguments.phone_address, port=arguments.phone_port)
        case SinkKind.DEBUG_WINDOW | SinkKind.NONE:
            pass
        case _:
            raise ValueError(f"no argument handling for {sink_kind}")

    return RunConfig(
        source_kind=source_kind,
        sink_kind=sink_kind,
        goal_mode=goal_mode,
        walker=WalkerConfig(radius_meters=arguments.walker_radius),
        tap=TapConfig(log_dir=arguments.record_to),
        video=video,
        neon=neon,
        arcore=arcore,
        neon_plugin=neon_plugin,
        logged=logged,
        estimator=estimator,
        web=web,
        phone_app=phone_app,
        reconnect=arguments.reconnect,
        realtime_replay=arguments.realtime,
        verbose=arguments.verbose,
    )


def build_source(config: RunConfig) -> DepthFrameSource:
    """
    Turn the chosen source kind into a source.

    :param config: The run configuration.
    :return: A source yielding DepthFrame objects.
    :rtype: DepthFrameSource
    """
    match config.source_kind:
        case SourceKind.VIDEO_FILE:
            raise NotImplementedError("video_file source lands in STEP_02")
        case SourceKind.NEON_LIVE:
            raise NotImplementedError("neon_live source lands in STEP_02")
        case SourceKind.LOGGED:
            raise NotImplementedError("logged source lands in STEP_03")
        case SourceKind.ARCORE_TCP:
            raise NotImplementedError("arcore_tcp source lands in STEP_04")
        case SourceKind.NEON_PLUGIN:
            raise NotImplementedError("neon_plugin source lands in STEP_04")
        case _:
            # Every member must be handled above. A missing case is a bug, not a reason to guess
            # at a source, and guessing is what the old script's hardware detection did.
            raise ValueError(f"no source constructor for {config.source_kind}")


def build_sink(config: RunConfig) -> PathSink:
    """
    Turn the chosen sink kind into a sink.

    :param config: The run configuration.
    :return: A sink accepting PlannedPath objects.
    :rtype: PathSink
    """
    match config.sink_kind:
        case SinkKind.DEBUG_WINDOW:
            raise NotImplementedError("debug_window sink lands in STEP_08")
        case SinkKind.WEB:
            raise NotImplementedError("web sink lands in STEP_08")
        case SinkKind.PHONE_APP:
            raise NotImplementedError("phone_app sink lands in STEP_08")
        case SinkKind.NONE:
            raise NotImplementedError("none sink lands in STEP_08")
        case _:
            raise ValueError(f"no sink constructor for {config.sink_kind}")


def _require(
    parser: argparse.ArgumentParser,
    value: object,
    argument_name: str,
    kind: SourceKind | SinkKind,
) -> None:
    """Exit with a parser error naming the missing argument and what needed it."""
    if value is None:
        parser.error(f"{argument_name} is required for {kind.value}")
