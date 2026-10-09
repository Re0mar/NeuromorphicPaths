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
import logging
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
from nav.sinks.fan_out import FanOutSink
from nav.sinks.none import NullSink
from nav.sinks.phone_app import PhoneAppSink
from nav.sinks.web import WebSink
from nav.sources.arcore_tcp import ArCoreTcpSource
from nav.sources.config import (
    ArCoreConfig,
    DepthCheckpoint,
    EstimatorConfig,
    LoggedConfig,
    NeonConfig,
    NeonPluginConfig,
    NeonPluginModel,
    NeonRecordingConfig,
    TapConfig,
    VideoConfig,
    depth_checkpoint_from_name,
)
from nav.sources.estimated_depth import EstimatedDepthSource
from nav.sources.estimator import DepthEstimator, DepthEstimatorProtocol
from nav.sources.logged import LoggedDepthFrameSource
from nav.sources.neon_plugin import NeonPluginDepthFrameSource
# Module scope, like neon_plugin: pupil_labs is imported inside the reader, not by importing this.
from nav.sources.neon_recording import NativeNeonRecordingReader, NeonRecordingReader, NeonRecordingRgbSource
from nav.sources.rgb import RgbSource
from nav.sources.scene_video import SceneVideoFeed
from nav.sources.video_file import URL_MARKER, VideoFileRgbSource
from nav.types import DepthFrameSource, PathSink
from nav.usermodel.config import UserModelConfig
from nav.walker import WalkerConfig

log = logging.getLogger(__name__)


class SourceKind(Enum):
    """Where frames come from. Values are the spellings the command line accepts."""

    VIDEO_FILE = "video_file"
    NEON_LIVE = "neon_live"
    ARCORE_TCP = "arcore_tcp"
    NEON_PLUGIN = "neon_plugin"
    NEON_RECORDING = "neon_recording"
    LOGGED = "logged"


class SinkKind(Enum):
    """Where the planned path goes. Values are the spellings the command line accepts."""

    DEBUG_WINDOW = "debug_window"
    WEB = "web"
    PHONE_APP = "phone_app"
    NONE = "none"


# These sources carry RGB only, so they need the depth estimator composed in behind them. The
# rest already deliver depth, so asking them for a model name would be meaningless.
ESTIMATOR_BACKED_SOURCES = frozenset({SourceKind.VIDEO_FILE, SourceKind.NEON_LIVE, SourceKind.NEON_RECORDING})
# Of those, the ones a person walks with while the estimator runs, where a CPU estimator is worth a
# warning. A recording on the CPU is only slow to process.
LIVE_ESTIMATOR_SOURCES = frozenset({SourceKind.NEON_LIVE})


@dataclass(frozen=True)
class RunConfig:
    """One whole run. Everything the factories and the loop need, and nothing they have to find."""

    source_kind: SourceKind
    sink_kinds: tuple[SinkKind, ...]
    goal_mode: GoalMode
    scene: SceneConfig = field(default_factory=SceneConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    usermodel: UserModelConfig = field(default_factory=UserModelConfig)
    walker: WalkerConfig = field(default_factory=WalkerConfig)
    tap: TapConfig = field(default_factory=TapConfig)
    # Where the timing log goes when it is not beside a recording. None with a recording means
    # timing.jsonl in the record directory, and None without one means no timing log at all.
    timing_log: str | None = None
    video: VideoConfig | None = None
    neon: NeonConfig | None = None
    arcore: ArCoreConfig | None = None
    neon_plugin: NeonPluginConfig | None = None
    neon_recording: NeonRecordingConfig | None = None
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
    recording_reader_factory: Callable[[Path], NeonRecordingReader] | None = None
    """Composition-root hook, set by tests to read a fake recording instead of a native one.

    No command line flag, for the same reason as estimator_factory. The native format can't be
    built by a test, so without this the neon_recording source could only be tested in pieces.
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


def _depth_checkpoint(text: str) -> DepthCheckpoint:
    """A checkpoint name, refused at parse time with the accepted names when it is unknown or relative."""
    try:
        return depth_checkpoint_from_name(text)
    except ValueError as refused:
        # argparse prints an ArgumentTypeError's own message. A plain ValueError becomes "invalid value".
        raise argparse.ArgumentTypeError(str(refused)) from refused


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
    pipeline.add_argument(
        "--sink",
        required=True,
        action="append",
        choices=[kind.value for kind in SinkKind],
        help="where the path goes. Repeat it for more than one display, as in --sink phone_app --sink web",
    )
    pipeline.add_argument("--goal", choices=[mode.value for mode in GoalMode], default=GoalMode.AHEAD.value)

    video = parser.add_argument_group("video_file source")
    video.add_argument("--path", help="recording on disk, or an IP camera stream URL")

    neon = parser.add_argument_group("neon_live source")
    neon.add_argument("--neon-address", help="the Neon's address, or omit it to discover the device")
    neon.add_argument("--neon-port", type=_port_number, default=NeonConfig.port)
    neon.add_argument(
        "--neon-replay",
        help="a folder written by examples/capture_neon_stream.py, played back at its recorded pace in place of the glasses",
    )

    arcore = parser.add_argument_group("arcore_tcp source")
    arcore.add_argument("--arcore-port", type=_port_number, default=ArCoreConfig.port)
    arcore.add_argument("--reconnect", action="store_true", help="keep listening after the phone disconnects")
    arcore.add_argument(
        "--arcore-accept-timeout",
        type=_positive_float,
        default=ArCoreConfig.accept_timeout_seconds,
        help="seconds to wait for the phone to connect. Launching the app by hand takes longer than the default",
    )

    recordings = parser.add_argument_group("neon_recording and neon_plugin sources")
    recordings.add_argument("--recording-dir", help="a native Neon recording folder. For neon_plugin, one the depth plugin has run over")
    # None rather than the dataclass default, so a rate given to another source can be told apart
    # from no rate given at all. The default is filled in from NeonRecordingConfig below.
    recordings.add_argument(
        "--recording-rate",
        type=_positive_float,
        default=None,
        help=f"neon_recording frames per second of recording to replay, {NeonRecordingConfig.frames_per_second:g} by default",
    )
    recordings.add_argument(
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
    estimator.add_argument(
        "--model",
        type=_depth_checkpoint,
        default=EstimatorConfig.model_name,
        help="Depth Anything 3 checkpoint, by its hub name. Only the ones that give meters are accepted",
    )
    estimator.add_argument("--process-resolution", type=_positive_int, default=EstimatorConfig.process_resolution)
    estimator.add_argument("--confidence-drop-percentile", type=_percentile, default=EstimatorConfig.confidence_drop_percentile)
    estimator.add_argument(
        "--fallback-fov",
        type=_positive_float,
        default=2 * EstimatorConfig.fallback_half_field_of_view_degrees,
        help=(
            "horizontal field of view in degrees, used when the camera has no calibration and the model "
            "returns no intrinsics. A phone is about 75. Ignored by neon_live and neon_recording, which use the glasses' own calibration"
        ),
    )

    tap = parser.add_argument_group("recording tap")
    tap.add_argument("--record-to", help="write every frame to this directory as it passes")
    tap.add_argument(
        "--timing-log",
        help="write the per-frame timing log to this file without recording frames. A recording run writes timing.jsonl beside its frames instead",
    )

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
    web.add_argument(
        "--demo-recording",
        help="a video file the page's Recording mode plays, served at /recording. Without it the page says no recording is configured",
    )

    phone = parser.add_argument_group("phone_app sink")
    phone.add_argument(
        "--phone-port",
        type=_port_number,
        default=PhoneAppConfig.port,
        help="the port the Pixel app connects to for paths. The phone uses the laptop address it already has",
    )

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
    sink_kinds = tuple(SinkKind(name) for name in arguments.sink)
    goal_mode = GoalMode(arguments.goal)

    # Two of the same display is two servers on one port, or two windows with one name. The
    # second would fail at start with a bind error that reads as another program holding it.
    repeated = [kind.value for kind in SinkKind if sink_kinds.count(kind) > 1]
    if repeated:
        parser.error(f"--sink {' and '.join(repeated)} given more than once, name each display at most once")

    # The loop and the replay act on these flags alone and never name a kind, so the kind checks
    # live here, before anything that opens a path.
    if arguments.reconnect and source_kind is not SourceKind.ARCORE_TCP:
        parser.error(f"--reconnect only applies to {SourceKind.ARCORE_TCP.value}, not {source_kind.value}")
    if arguments.realtime and source_kind is not SourceKind.LOGGED:
        parser.error(f"--realtime only applies to {SourceKind.LOGGED.value}, not {source_kind.value}")
    if arguments.neon_replay is not None and source_kind is not SourceKind.NEON_LIVE:
        parser.error(f"--neon-replay only applies to {SourceKind.NEON_LIVE.value}, not {source_kind.value}")
    if arguments.recording_rate is not None and source_kind is not SourceKind.NEON_RECORDING:
        parser.error(f"--recording-rate only applies to {SourceKind.NEON_RECORDING.value}, not {source_kind.value}")
    if arguments.neon_replay is not None and arguments.neon_address is not None:
        parser.error("--neon-address and --neon-replay cannot be used together, a replay plays a capture in place of the glasses")
    # One log, one place. With both, a reader would have to guess which file the run wrote.
    if arguments.timing_log is not None and arguments.record_to is not None:
        parser.error("--timing-log and --record-to cannot be used together, a recording run writes timing.jsonl beside its frames")
    # Two runs appended to one log read back as one run that is not one, so a used file is refused.
    if arguments.timing_log is not None and Path(arguments.timing_log).is_file() and Path(arguments.timing_log).stat().st_size > 0:
        parser.error(f"--timing-log {arguments.timing_log} already holds a timing log, pick a new file")
    # Either one would let the walk run and fail on its first line, leaving it untimed.
    if arguments.timing_log is not None and Path(arguments.timing_log).is_dir():
        parser.error(f"--timing-log {arguments.timing_log} is a directory. It names a file, such as frame_logs/replays/run_1.jsonl")
    if arguments.timing_log is not None and not Path(arguments.timing_log).parent.is_dir():
        parser.error(f"--timing-log {arguments.timing_log} is in a folder that doesn't exist: {Path(arguments.timing_log).parent}")
    # A recording nobody serves is a flag the run silently ignores, and a missing file would be
    # found by the first browser rather than before the model loads.
    if arguments.demo_recording is not None and SinkKind.WEB not in sink_kinds:
        parser.error(f"--demo-recording only applies with --sink {SinkKind.WEB.value}, which serves it")
    if arguments.demo_recording is not None and not Path(arguments.demo_recording).is_file():
        parser.error(f"--demo-recording {arguments.demo_recording} is not a file")

    video = None
    neon = None
    arcore = None
    neon_plugin = None
    neon_recording = None
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
            if arguments.neon_replay is not None and not (Path(arguments.neon_replay) / "meta.json").is_file():
                parser.error(f"--neon-replay {arguments.neon_replay} is not a capture folder, it has no meta.json")
            neon = NeonConfig(address=arguments.neon_address, port=arguments.neon_port, replay_dir=arguments.neon_replay)
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
        case SourceKind.NEON_RECORDING:
            _require(parser, arguments.recording_dir, "--recording-dir", source_kind)
            if not Path(arguments.recording_dir).is_dir():
                parser.error(f"--recording-dir {arguments.recording_dir} is not a directory")
            rate = NeonRecordingConfig.frames_per_second if arguments.recording_rate is None else arguments.recording_rate
            neon_recording = NeonRecordingConfig(recording_dir=arguments.recording_dir, frames_per_second=rate)
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

    for sink_kind in sink_kinds:
        match sink_kind:
            case SinkKind.WEB:
                web = WebConfig(port=arguments.web_port, recording_path=arguments.demo_recording)
            case SinkKind.PHONE_APP:
                phone_app = PhoneAppConfig(port=arguments.phone_port)
            case SinkKind.DEBUG_WINDOW | SinkKind.NONE:
                pass
            case _:
                raise ValueError(f"no argument handling for {sink_kind}")

    return RunConfig(
        source_kind=source_kind,
        sink_kinds=sink_kinds,
        goal_mode=goal_mode,
        scene=SceneConfig(floor_max_tilt_degrees=arguments.floor_max_tilt, floor_max_offset_meters=arguments.floor_max_height),
        walker=WalkerConfig(radius_meters=arguments.walker_radius),
        tap=TapConfig(log_dir=arguments.record_to),
        timing_log=arguments.timing_log,
        video=video,
        neon=neon,
        arcore=arcore,
        neon_plugin=neon_plugin,
        neon_recording=neon_recording,
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
    estimator = build(config.estimator)
    if config.source_kind in LIVE_ESTIMATOR_SOURCES and estimator.device == "cpu":
        # Warned rather than refused. The run still works, and on a laptop without an NVIDIA GPU
        # it is the only way to check the glasses connect at all.
        log.warning(
            "the depth estimator is on the CPU, so a live %s walk will run at a fraction of a frame a second. "
            "Install the CUDA build of torch, see server/README.md",
            config.source_kind.value,
        )
    return EstimatedDepthSource(rgb_source, estimator, config.estimator)


# The sources whose device hands over its compressed video for a display that decodes on its own.
# The glasses' live route, and its capture replay, which goes through the same receiver. A
# source that reads a Companion recording through a path of its own is not here until someone
# gives it a feed.
SOURCES_WITH_SCENE_VIDEO = frozenset({SourceKind.NEON_LIVE})


def build_source_and_sink(config: RunConfig, on_path_sent: Callable[[float], None] | None = None) -> tuple[DepthFrameSource, PathSink]:
    """
    Build the run's source and its sink together, sharing what only both can use.

    The scene video feed is the one such thing: the source's device offers compressed video into
    it, and the web sink reads it out. Built here, once, so neither end has to find the other,
    and only for a source that has video to offer. The loop calls this rather than the two
    factories, and a caller that needs one end alone still has them.

    :param config: The run configuration.
    :param on_path_sent: As for build_sink.
    :return: The source and the sink, in that order.
    :rtype: tuple[DepthFrameSource, PathSink]
    """
    video_feed = SceneVideoFeed() if config.source_kind in SOURCES_WITH_SCENE_VIDEO else None
    return build_source(config, video_feed=video_feed), build_sink(config, on_path_sent=on_path_sent, video_feed=video_feed)


def build_source(config: RunConfig, video_feed: SceneVideoFeed | None = None) -> DepthFrameSource:
    """
    Turn the chosen source kind into a source, wrapped in the recording tap when asked for.

    The tap wraps whatever was built, so every source records the same way and the replay reads
    one format back.

    :param config: The run configuration.
    :param video_feed: Where a source with compressed video offers it. None when no display wants it.
    :return: A source yielding DepthFrame objects.
    :rtype: DepthFrameSource
    """
    source = _build_inner_source(config, video_feed)
    if config.tap.log_dir is None:
        return source
    return RecordingTap(source, Path(config.tap.log_dir))


def _build_inner_source(config: RunConfig, video_feed: SceneVideoFeed | None) -> DepthFrameSource:
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

            return build_estimated_depth_source(NeonLiveRgbSource(config.neon, video_feed=video_feed), config)
        case SourceKind.LOGGED:
            if config.logged is None:
                raise ValueError("logged needs a logged config and none was built")
            return LoggedDepthFrameSource(Path(config.logged.log_dir), realtime=config.realtime_replay)
        case SourceKind.ARCORE_TCP:
            if config.arcore is None:
                raise ValueError("arcore_tcp needs an arcore config and none was built")
            return ArCoreTcpSource(config.arcore)
        case SourceKind.NEON_RECORDING:
            if config.neon_recording is None:
                raise ValueError("neon_recording needs a neon_recording config and none was built")
            return build_estimated_depth_source(
                NeonRecordingRgbSource(
                    config.neon_recording,
                    reader_factory=config.recording_reader_factory if config.recording_reader_factory is not None else NativeNeonRecordingReader,
                ),
                config,
            )
        case SourceKind.NEON_PLUGIN:
            if config.neon_plugin is None:
                raise ValueError("neon_plugin needs a neon_plugin config and none was built")
            return NeonPluginDepthFrameSource(config.neon_plugin, reader_factory=NativeNeonRecordingReader)
        case _:
            # Every member must be handled above. A missing case is a bug, not a reason to guess
            # at a source, and guessing is what the old script's hardware detection did.
            raise ValueError(f"no source constructor for {config.source_kind}")


def build_sink(
    config: RunConfig,
    on_path_sent: Callable[[float], None] | None = None,
    video_feed: SceneVideoFeed | None = None,
) -> PathSink:
    """
    Turn the chosen sink kinds into one sink.

    A run names one display or several. With several the loop still receives one sink, a fan-out
    that hands every path to each of them, so nothing upstream counts displays.

    :param config: The run configuration.
    :param on_path_sent: Given to the phone sink, the one display on the walker's path, and called
        with a path's `timestamp_seconds` once it is on the phone's socket. The timing log's hook.
    :param video_feed: Given to the web sink, the one display that decodes video on its own. None
        when the source has no video to offer, and the page then says so.
    :return: A sink accepting PlannedPath objects.
    :rtype: PathSink
    """
    sinks = [_build_one_sink(kind, config, on_path_sent, video_feed) for kind in config.sink_kinds]
    if len(sinks) == 1:
        return sinks[0]
    return FanOutSink(sinks)


def _build_one_sink(
    sink_kind: SinkKind,
    config: RunConfig,
    on_path_sent: Callable[[float], None] | None,
    video_feed: SceneVideoFeed | None,
) -> PathSink:
    match sink_kind:
        case SinkKind.DEBUG_WINDOW:
            return DebugWindowSink(config.debug_window)
        case SinkKind.WEB:
            if config.web is None:
                raise ValueError("web needs a web config and none was built")
            return WebSink(config.web, config.scene, video_feed=video_feed)
        case SinkKind.PHONE_APP:
            if config.phone_app is None:
                raise ValueError("phone_app needs a phone_app config and none was built")
            return PhoneAppSink(config.phone_app, on_sent=on_path_sent)
        case SinkKind.NONE:
            return NullSink()
        case _:
            raise ValueError(f"no sink constructor for {sink_kind}")


def _require(
    parser: argparse.ArgumentParser,
    value: object,
    argument_name: str,
    kind: SourceKind | SinkKind,
) -> None:
    """Exit with a parser error naming the missing argument and what needed it."""
    if value is None:
        parser.error(f"{argument_name} is required for {kind.value}")
