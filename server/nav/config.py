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
from pathlib import Path

# Local package imports
from nav.planner.config import GoalMode, PlannerConfig
from nav.runtime.tap import RecordingTap
from nav.scene.config import SceneConfig
from nav.sinks.config import DebugWindowConfig, PhoneAppConfig, WebConfig
from nav.sinks.debug_window import DebugWindowSink
from nav.sinks.none import NullSink
from nav.sinks.phone_app import PhoneAppSink
from nav.sinks.web import WebSink
from nav.sources.arcore_tcp import ArCoreTcpSource
from nav.sources.config import (
    ArCoreConfig,
    EstimatorConfig,
    LoggedConfig,
    NeonConfig,
    NeonPluginConfig,
    NeonPluginModel,
    TapConfig,
    VideoConfig,
)
from nav.sources.estimated_depth import EstimatedDepthSource
from nav.sources.estimator import DepthEstimator, DepthEstimatorProtocol
from nav.sources.logged import LoggedDepthFrameSource
from nav.sources.neon_plugin import NativeNeonRecordingReader, NeonPluginDepthFrameSource
from nav.sources.rgb import RgbSource
from nav.sources.video_file import URL_MARKER, VideoFileRgbSource
from nav.types import DepthFrameSource, PathSink
from nav.usermodel.config import UserModelConfig
from nav.walker import WalkerConfig


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


# These two sources carry RGB only, so they need the depth estimator composed in behind them.
# The other three already deliver depth, so asking them for a model name would be meaningless.
ESTIMATOR_BACKED_SOURCES = frozenset({SourceKind.VIDEO_FILE, SourceKind.NEON_LIVE})


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
    debug_window: DebugWindowConfig = field(default_factory=DebugWindowConfig)
    reconnect: bool = False
    realtime_replay: bool = False
    verbose: bool = False
    estimator_factory: Callable[[EstimatorConfig], DepthEstimatorProtocol] | None = None
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
    neon.add_argument("--neon-address", help="the Neon's address, or omit it to discover the device")
    neon.add_argument("--neon-port", type=_port_number, default=NeonConfig.port)

    arcore = parser.add_argument_group("arcore_tcp source")
    arcore.add_argument("--arcore-port", type=_port_number, default=ArCoreConfig.port)
    arcore.add_argument("--reconnect", action="store_true", help="keep listening after the phone disconnects")
    arcore.add_argument(
        "--arcore-accept-timeout",
        type=_positive_float,
        default=ArCoreConfig.accept_timeout_seconds,
        help="seconds to wait for the phone to connect. Launching the app by hand takes longer than the default",
    )

    neon_plugin = parser.add_argument_group("neon_plugin source")
    neon_plugin.add_argument("--recording-dir", help="a Neon recording the depth plugin has run over")
    neon_plugin.add_argument(
        "--plugin-model",
        choices=[model.value for model in NeonPluginModel],
        default=NeonPluginModel.METRIC_LARGE.value,
        help="which model's cache to read. Only the metric one gives meters",
    )

    logged = parser.add_argument_group("logged source")
    logged.add_argument("--log-dir", help="a frame log written by --record-to")
    logged.add_argument("--realtime", action="store_true", help="replay at the recorded frame rate")

    estimator = parser.add_argument_group("depth estimator")
    # Every default below is read from its dataclass rather than restated, so each value has one
    # home and a test can check the parser against it.
    estimator.add_argument("--model", default=EstimatorConfig.model_name, help="Depth Anything 3 checkpoint")
    estimator.add_argument("--process-resolution", type=_positive_int, default=EstimatorConfig.process_resolution)
    estimator.add_argument("--confidence-drop-percentile", type=_percentile, default=EstimatorConfig.confidence_drop_percentile)
    estimator.add_argument(
        "--fallback-fov",
        type=_positive_float,
        default=2 * EstimatorConfig.fallback_half_field_of_view_degrees,
        help="horizontal field of view in degrees, used when the model returns no intrinsics. A phone is about 75",
    )

    tap = parser.add_argument_group("recording tap")
    tap.add_argument("--record-to", help="write every frame to this directory as it passes")

    scene = parser.add_argument_group("scene")
    scene.add_argument(
        "--floor-max-tilt",
        type=_positive_float,
        default=SceneConfig.floor_max_tilt_degrees,
        help="degrees from camera up a fitted floor may lean before it is rejected. Head-mounted 35, a hand-held phone pointed down needs more",
    )
    scene.add_argument(
        "--floor-max-height",
        type=_positive_float,
        default=SceneConfig.floor_max_offset_meters,
        help="meters a fitted or supplied floor may lie below the camera before it is refused. The first Pixel walk's false plane was 2.3 down",
    )

    walker = parser.add_argument_group("walker")
    walker.add_argument("--walker-radius", type=_positive_float, default=WalkerConfig.radius_meters, help="footprint radius in meters")

    web = parser.add_argument_group("web sink")
    web.add_argument("--web-port", type=_port_number, default=WebConfig.port)

    phone = parser.add_argument_group("phone_app sink")
    phone.add_argument("--phone-address", help="the Pixel app's address")
    phone.add_argument("--phone-port", type=_port_number, default=PhoneAppConfig.port)

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

    # The loop and the replay act on these flags alone and never name a kind, so the kind checks
    # live here, before anything that opens a path.
    if arguments.reconnect and source_kind is not SourceKind.ARCORE_TCP:
        parser.error(f"--reconnect only applies to {SourceKind.ARCORE_TCP.value}, not {source_kind.value}")
    if arguments.realtime and source_kind is not SourceKind.LOGGED:
        parser.error(f"--realtime only applies to {SourceKind.LOGGED.value}, not {source_kind.value}")

    video = None
    neon = None
    arcore = None
    neon_plugin = None
    logged = None

    match source_kind:
        case SourceKind.VIDEO_FILE:
            _require(parser, arguments.path, "--path", source_kind)
            # Checked here rather than when the source opens, because the estimator loads its
            # 1.3 GB model before the first frame is asked for, and a typo should cost nothing.
            if URL_MARKER not in arguments.path and not Path(arguments.path).is_file():
                parser.error(f"--path {arguments.path} is not a file. A camera stream needs a URL with a scheme")
            video = VideoConfig(path=arguments.path)
        case SourceKind.NEON_LIVE:
            # No address is the normal case. The source discovers the device instead.
            neon = NeonConfig(address=arguments.neon_address, port=arguments.neon_port)
        case SourceKind.ARCORE_TCP:
            arcore = ArCoreConfig(port=arguments.arcore_port, accept_timeout_seconds=arguments.arcore_accept_timeout)
        case SourceKind.NEON_PLUGIN:
            _require(parser, arguments.recording_dir, "--recording-dir", source_kind)
            if not Path(arguments.recording_dir).is_dir():
                parser.error(f"--recording-dir {arguments.recording_dir} is not a directory")
            neon_plugin = NeonPluginConfig(
                recording_dir=arguments.recording_dir,
                model=NeonPluginModel(arguments.plugin_model),
            )
        case SourceKind.LOGGED:
            _require(parser, arguments.log_dir, "--log-dir", source_kind)
            if not Path(arguments.log_dir).is_dir():
                parser.error(f"--log-dir {arguments.log_dir} is not a directory")
            logged = LoggedConfig(log_dir=arguments.log_dir)
        case _:
            # Unreachable while every member above is handled. Here so that adding a member and
            # forgetting this function fails loudly instead of leaving every config None.
            raise ValueError(f"no argument handling for {source_kind}")

    estimator = None
    if source_kind in ESTIMATOR_BACKED_SOURCES:
        estimator = EstimatorConfig(
            model_name=arguments.model,
            process_resolution=arguments.process_resolution,
            confidence_drop_percentile=arguments.confidence_drop_percentile,
            fallback_half_field_of_view_degrees=arguments.fallback_fov / 2.0,
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
        scene=SceneConfig(floor_max_tilt_degrees=arguments.floor_max_tilt, floor_max_offset_meters=arguments.floor_max_height),
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


def build_estimated_depth_source(rgb_source: RgbSource, config: RunConfig) -> EstimatedDepthSource:
    """
    Compose an RGB source with the depth estimator.

    The one place an estimator is constructed. Tests inject a stub through
    RunConfig.estimator_factory rather than patching an import, so the production path and the
    test path differ by one value instead of by a monkeypatch.

    :param rgb_source: The camera to read from.
    :param config: The run configuration, whose estimator field must be set.
    :return: A source of DepthFrame objects.
    :rtype: EstimatedDepthSource
    """
    if config.estimator is None:
        raise ValueError(f"{config.source_kind.value} needs an estimator config and none was built")

    build = config.estimator_factory if config.estimator_factory is not None else DepthEstimator
    return EstimatedDepthSource(rgb_source, build(config.estimator), config.estimator)


def build_source(config: RunConfig) -> DepthFrameSource:
    """
    Turn the chosen source kind into a source, wrapped in the recording tap when asked for.

    The tap wraps whatever was built, so every source records the same way and the replay reads
    one format back.

    :param config: The run configuration.
    :return: A source yielding DepthFrame objects.
    :rtype: DepthFrameSource
    """
    source = _build_inner_source(config)
    if config.tap.log_dir is None:
        return source
    return RecordingTap(source, Path(config.tap.log_dir))


def _build_inner_source(config: RunConfig) -> DepthFrameSource:
    match config.source_kind:
        case SourceKind.VIDEO_FILE:
            if config.video is None:
                raise ValueError("video_file needs a video config and none was built")
            return build_estimated_depth_source(VideoFileRgbSource(config.video.path), config)
        case SourceKind.NEON_LIVE:
            if config.neon is None:
                raise ValueError("neon_live needs a neon config and none was built")
            # Imported here rather than at module scope so this module stays importable without
            # the Pupil Labs client. That is the lazy-import rule's optional-dependency case.
            from nav.sources.neon_live import NeonLiveRgbSource

            return build_estimated_depth_source(NeonLiveRgbSource(config.neon), config)
        case SourceKind.LOGGED:
            if config.logged is None:
                raise ValueError("logged needs a logged config and none was built")
            return LoggedDepthFrameSource(Path(config.logged.log_dir), realtime=config.realtime_replay)
        case SourceKind.ARCORE_TCP:
            if config.arcore is None:
                raise ValueError("arcore_tcp needs an arcore config and none was built")
            return ArCoreTcpSource(config.arcore)
        case SourceKind.NEON_PLUGIN:
            if config.neon_plugin is None:
                raise ValueError("neon_plugin needs a neon_plugin config and none was built")
            return NeonPluginDepthFrameSource(config.neon_plugin, reader_factory=NativeNeonRecordingReader)
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
            return DebugWindowSink(config.debug_window)
        case SinkKind.WEB:
            if config.web is None:
                raise ValueError("web needs a web config and none was built")
            return WebSink(config.web)
        case SinkKind.PHONE_APP:
            if config.phone_app is None:
                raise ValueError("phone_app needs a phone_app config and none was built")
            return PhoneAppSink(config.phone_app)
        case SinkKind.NONE:
            return NullSink()
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
