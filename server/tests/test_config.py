"""
Covers the command line, the two factories, and the validation the shared types do on themselves.

The factory tests are completeness checks rather than behavior checks. Every kind must reach an
arm, so that adding a member to either enum and forgetting the factory fails here instead of at
the start of a walk.
"""

# Standard library imports
import dataclasses

import pytest

# Third party imports
import numpy as np

# Local package imports
from nav.config import (
    GoalMode,
    RunConfig,
    SinkKind,
    SourceKind,
    build_run_config,
    build_sink,
    build_source,
    build_source_and_sink,
)
from nav.sources.config import (
    ArCoreConfig,
    DepthCheckpoint,
    DepthScale,
    EstimatorConfig,
    LoggedConfig,
    NeonConfig,
    NeonRecordingConfig,
    SourceMode,
    TapConfig,
    VideoConfig,
)
from nav.runtime.tap import RecordingTap
from nav.scene.config import SceneConfig
from nav.sinks.config import PhoneAppConfig, WebConfig
from nav.sources.scene_video import SceneVideoFeed
from nav.sources.switching import CurrentStretchOnly, SwitchableRgbSource
from nav.types import DepthFrame, PlannedPath, Pose
from nav.walker import WalkerConfig

from stubs import StubDepthEstimator

ORIENTATION_ONLY_POSE = Pose(orientation=np.array([1.0, 0.0, 0.0, 0.0]), position=None, has_position=False)
PINHOLE_INTRINSICS = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


def _depth_frame(**overrides: object) -> DepthFrame:
    """A valid frame, with one field swapped out, so each test names only what it is breaking."""
    fields = {
        "timestamp_seconds": 0.0,
        "depth_meters": np.ones((4, 4), dtype=np.float32),
        "intrinsics": PINHOLE_INTRINSICS,
        "pose": ORIENTATION_ONLY_POSE,
        "ground_plane": None,
        "gaze_pixel": None,
    }
    fields.update(overrides)
    return DepthFrame(**fields)

# A stream URL rather than a file name, because the parser now checks that a file path exists and
# these tests are about the parser, not about a file on disk.
STREAM_URL = "rtsp://camera.local/scene"
MINIMAL_VIDEO_ARGV = ["--source", "video_file", "--path", STREAM_URL, "--sink", "none"]


def test_minimal_command_line_builds_a_config() -> None:
    config = build_run_config(MINIMAL_VIDEO_ARGV)

    assert config.source_kind is SourceKind.VIDEO_FILE
    assert config.sink_kinds == (SinkKind.NONE,)
    assert config.goal_mode is GoalMode.AHEAD
    assert config.video is not None
    assert config.video.path == STREAM_URL
    assert config.estimator is not None
    assert config.estimator.model_name == EstimatorConfig.model_name
    # A config for a source that was not chosen stays None, so a later layer reading one is
    # reading a mistake rather than a stale default.
    assert config.neon is None
    assert config.arcore is None


def test_defaults_land_where_they_belong() -> None:
    config = build_run_config(MINIMAL_VIDEO_ARGV)

    # The footprint radius has exactly one home. A copy in SceneConfig or PlannerConfig is the
    # defect this asserts the absence of.
    assert config.walker.radius_meters == pytest.approx(0.35)
    assert not hasattr(config.scene, "radius_meters")
    assert not hasattr(config.planner, "radius_meters")
    assert config.tap.log_dir is None
    assert config.estimator_factory is None


@pytest.mark.parametrize(
    ("argv", "expected_in_message"),
    [
        (["--source", "video_file", "--sink", "none"], "--path"),
        (["--source", "neon_recording", "--sink", "none"], "--recording-dir"),
        (["--source", "logged", "--sink", "none"], "--log-dir"),
    ],
)
def test_missing_required_argument_names_it(argv: list[str], expected_in_message: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        build_run_config(argv)

    assert expected_in_message in capsys.readouterr().err


def test_a_video_path_that_is_not_a_file_is_refused_before_anything_loads(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    # The estimator loads 1.3 GB of weights before the first frame is asked for. A typo in the
    # path has to be refused by the parser, where it costs nothing, and the message names the path.
    missing = tmp_path / "nothing_here.mp4"

    with pytest.raises(SystemExit):
        build_run_config(["--source", "video_file", "--path", str(missing), "--sink", "none"])

    assert "nothing_here.mp4" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flag", "argv"),
    [
        ("--log-dir", ["--source", "logged", "--log-dir", "no_such_log", "--sink", "none"]),
        ("--recording-dir", ["--source", "neon_recording", "--recording-dir", "no_such_recording", "--sink", "none"]),
    ],
)
def test_a_directory_argument_that_is_not_a_directory_is_refused(flag: str, argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        build_run_config(argv)

    assert flag in capsys.readouterr().err


def test_a_video_path_that_exists_is_accepted(tmp_path) -> None:
    video = tmp_path / "walk.avi"
    video.write_bytes(b"not really a video, the parser only checks that it exists")

    config = build_run_config(["--source", "video_file", "--path", str(video), "--sink", "none"])

    assert config.video is not None and config.video.path == str(video)


def test_realtime_is_refused_for_a_source_that_is_not_the_replay(capsys: pytest.CaptureFixture[str]) -> None:
    # The same rule --reconnect already had. A flag the chosen source silently ignores is a run
    # that does something other than what was typed.
    with pytest.raises(SystemExit):
        build_run_config([*MINIMAL_VIDEO_ARGV, "--realtime"])

    assert "--realtime" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["--source", "bogus", "--sink", "none"],
        ["--source", "video_file", "--path", STREAM_URL, "--sink", "bogus"],
        ["--source", "video_file", "--path", STREAM_URL, "--sink", "none", "--goal", "bogus"],
    ],
)
def test_unknown_kind_is_refused(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        build_run_config(argv)


# Every source kind and the config it needs. Having the full set in one table is what makes the
# completeness check below possible.
BUILT_SOURCE_KINDS = {
    SourceKind.VIDEO_FILE: {"video": VideoConfig(path="scene.mp4")},
    SourceKind.NEON_LIVE: {"neon": NeonConfig()},
    SourceKind.LOGGED: {"logged": LoggedConfig(log_dir="a_log")},
    SourceKind.ARCORE_TCP: {"arcore": ArCoreConfig(port=0)},
    SourceKind.NEON_RECORDING: {"neon_recording": NeonRecordingConfig(recording_dir="a_recording")},
}


def _run_config_for(source_kind: SourceKind) -> RunConfig:
    return RunConfig(
        source_kind=source_kind,
        sink_kinds=(SinkKind.NONE,),
        goal_mode=GoalMode.AHEAD,
        estimator=EstimatorConfig(),
        # Injected at the composition root, so no test ever loads 1.3 GB of weights.
        estimator_factory=lambda estimator_config: StubDepthEstimator(),
        **BUILT_SOURCE_KINDS[source_kind],
    )


def test_every_source_kind_is_accounted_for() -> None:
    # A new member that nobody wired fails here rather than slipping past the test below by
    # not being in the table.
    assert set(BUILT_SOURCE_KINDS) == set(SourceKind)


@pytest.mark.parametrize("source_kind", sorted(BUILT_SOURCE_KINDS, key=lambda kind: kind.value))
def test_built_source_kinds_return_a_source(source_kind: SourceKind) -> None:
    source = build_source(_run_config_for(source_kind))

    # Constructing must not touch the device or the file. Both open lazily inside frames().
    assert hasattr(source, "frames")
    assert hasattr(source, "close")


def test_the_tap_wraps_whatever_source_was_built(tmp_path) -> None:
    # The tap wraps the built source rather than being a source of its own, so every source
    # records the same way and the replay reads one format back.
    config = RunConfig(
        source_kind=SourceKind.LOGGED,
        sink_kinds=(SinkKind.NONE,),
        goal_mode=GoalMode.AHEAD,
        logged=LoggedConfig(log_dir="a_log"),
        tap=TapConfig(log_dir=str(tmp_path / "recorded")),
    )

    assert isinstance(build_source(config), RecordingTap)


def test_without_record_to_the_source_is_not_wrapped() -> None:
    config = RunConfig(
        source_kind=SourceKind.LOGGED,
        sink_kinds=(SinkKind.NONE,),
        goal_mode=GoalMode.AHEAD,
        logged=LoggedConfig(log_dir="a_log"),
    )

    assert not isinstance(build_source(config), RecordingTap)

def test_a_built_source_refuses_a_missing_config() -> None:
    # RunConfig does not validate across its own fields, so a kind whose config was never built is
    # reachable. The factory must say which one rather than construct a source around a None.
    config = RunConfig(source_kind=SourceKind.VIDEO_FILE, sink_kinds=(SinkKind.NONE,), goal_mode=GoalMode.AHEAD)

    with pytest.raises(ValueError, match="video_file needs a video config"):
        build_source(config)


def test_estimator_backed_source_refuses_a_missing_estimator_config() -> None:
    config = RunConfig(
        source_kind=SourceKind.VIDEO_FILE,
        sink_kinds=(SinkKind.NONE,),
        goal_mode=GoalMode.AHEAD,
        video=VideoConfig(path="scene.mp4"),
    )

    with pytest.raises(ValueError, match="needs an estimator config"):
        build_source(config)


BUILT_SINK_KINDS = {
    SinkKind.NONE: {},
    SinkKind.DEBUG_WINDOW: {},
    SinkKind.WEB: {"web": WebConfig(port=0)},
    SinkKind.PHONE_APP: {"phone_app": PhoneAppConfig(port=1)},
}


def test_every_sink_kind_is_accounted_for() -> None:
    assert set(BUILT_SINK_KINDS) == set(SinkKind)


@pytest.mark.parametrize("sink_kind", sorted(BUILT_SINK_KINDS, key=lambda kind: kind.value))
def test_built_sink_kinds_return_a_sink_without_opening_anything(sink_kind: SinkKind) -> None:
    config = RunConfig(source_kind=SourceKind.VIDEO_FILE, sink_kinds=(sink_kind,), goal_mode=GoalMode.AHEAD, **BUILT_SINK_KINDS[sink_kind])

    sink = build_sink(config)

    # Construction must not open a window, bind a port or connect to a phone. All of that happens
    # on the first publish, so this test runs with no display and nothing listening on port 1.
    assert hasattr(sink, "publish")
    assert hasattr(sink, "close")
    sink.close()


def test_several_sinks_build_one_fan_out_over_them_all() -> None:
    # A walk puts the arrow on the phone and the depth view in a browser at the same time, and
    # the loop still receives one sink.
    from nav.sinks.fan_out import FanOutSink
    from nav.sinks.phone_app import PhoneAppSink
    from nav.sinks.web import WebSink

    config = build_run_config([*MINIMAL_VIDEO_ARGV[:-2], "--sink", "phone_app", "--sink", "web"])

    assert config.sink_kinds == (SinkKind.PHONE_APP, SinkKind.WEB)
    assert config.phone_app is not None and config.web is not None
    sink = build_sink(config)
    try:
        assert isinstance(sink, FanOutSink)
        assert [type(each) for each in sink.sinks] == [PhoneAppSink, WebSink]
    finally:
        sink.close()


def test_all_three_displays_can_run_together() -> None:
    config = build_run_config(
        [*MINIMAL_VIDEO_ARGV[:-2], "--sink", "phone_app", "--sink", "web", "--sink", "debug_window"]
    )

    sink = build_sink(config)
    try:
        assert len(sink.sinks) == 3
    finally:
        sink.close()


def test_one_sink_is_not_wrapped_in_a_fan_out() -> None:
    # The common case stays exactly what it was, so a single-display run has nothing extra in it.
    from nav.sinks.none import NullSink

    sink = build_sink(build_run_config(MINIMAL_VIDEO_ARGV))

    assert isinstance(sink, NullSink)


def test_naming_the_same_display_twice_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    # Two web sinks is two servers on one port. The second would fail at start with a bind error
    # that reads as another program holding it, which is the wrong thing to make someone debug.
    with pytest.raises(SystemExit):
        build_run_config([*MINIMAL_VIDEO_ARGV[:-2], "--sink", "web", "--sink", "web"])

    assert "given more than once" in capsys.readouterr().err


def test_a_built_sink_refuses_a_missing_config() -> None:
    config = RunConfig(source_kind=SourceKind.VIDEO_FILE, sink_kinds=(SinkKind.PHONE_APP,), goal_mode=GoalMode.AHEAD)

    with pytest.raises(ValueError, match="phone_app needs"):
        build_sink(config)



def test_default_checkpoint_is_a_metric_one() -> None:
    # The old script defaulted to DA3NESTED-GIANT-LARGE, which returns relative depth, so every
    # clearance in meters was wrong by an unknown scale and the file carried a cam_height rescale
    # to paper over it. Asserts the property rather than the exact name, so a version bump passes
    # and a swap back to a relative checkpoint does not.
    assert EstimatorConfig.model_name.depth_scale is not DepthScale.RELATIVE


def test_a_model_named_on_the_command_line_reaches_the_estimator_as_a_checkpoint() -> None:
    """The estimator config refuses a bare string, so the parser is where a name becomes a checkpoint."""
    config = build_run_config([*MINIMAL_VIDEO_ARGV, "--model", DepthCheckpoint.NESTED_GIANT_LARGE.value])

    assert config.estimator.model_name is DepthCheckpoint.NESTED_GIANT_LARGE


@pytest.mark.parametrize(
    ("name", "expected_in_message"),
    [
        ("depth-anything/DA3-NOT-A-MODEL", "unknown Depth Anything 3 checkpoint"),
        (DepthCheckpoint.SMALL.value, "gives relative depth"),
    ],
)
def test_a_model_that_is_unknown_or_relative_is_refused_at_parse_time(
    name: str, expected_in_message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exited:
        build_run_config([*MINIMAL_VIDEO_ARGV, "--model", name])

    assert exited.value.code == 2
    error = capsys.readouterr().err
    assert expected_in_message in error
    assert DepthCheckpoint.METRIC_LARGE.value in error, "the refusal names a checkpoint that would work"


def test_the_floor_tilt_and_fallback_fov_flags_reach_their_layers() -> None:
    # The first real recording was refused frame after frame by the floor gate the glasses
    # script shipped with, and the cloud was stretched by its 100 degree fallback field of view.
    # A tuning run has to reach both without editing code.
    config = build_run_config([*MINIMAL_VIDEO_ARGV, "--floor-max-tilt", "65", "--fallback-fov", "75"])

    assert config.scene.floor_max_tilt_degrees == pytest.approx(65.0)
    assert config.estimator is not None
    assert config.estimator.fallback_half_field_of_view_degrees == pytest.approx(37.5)


def test_the_floor_max_height_flag_reaches_the_scene() -> None:
    # The first Pixel walk's false plane put the camera 2.3 m up, and the ceiling that refuses it
    # is the one gate that has bitten on real data, so it has to be tunable without editing code.
    config = build_run_config([*MINIMAL_VIDEO_ARGV, "--floor-max-height", "1.9"])

    assert config.scene.floor_max_offset_meters == pytest.approx(1.9)


def test_the_floor_tilt_and_fallback_fov_defaults_match_their_configs() -> None:
    config = build_run_config(MINIMAL_VIDEO_ARGV)

    assert config.scene.floor_max_tilt_degrees == SceneConfig.floor_max_tilt_degrees
    assert config.scene.floor_max_offset_meters == SceneConfig.floor_max_offset_meters
    assert config.estimator is not None
    assert config.estimator.fallback_half_field_of_view_degrees == EstimatorConfig.fallback_half_field_of_view_degrees


def test_every_parsed_default_is_the_dataclass_default(tmp_path) -> None:
    # Each knob has one home, its layer's dataclass. The parser reads from there rather than
    # restating the number, so this is what catches a copy that drifted.
    config = build_run_config(MINIMAL_VIDEO_ARGV)
    assert config.estimator is not None
    assert config.walker.radius_meters == WalkerConfig.radius_meters
    assert config.scene.floor_max_offset_meters == SceneConfig.floor_max_offset_meters
    assert config.estimator.process_resolution == EstimatorConfig.process_resolution
    assert config.estimator.confidence_drop_percentile == EstimatorConfig.confidence_drop_percentile
    assert config.estimator.model_name == EstimatorConfig.model_name

    neon = build_run_config(["--source", "neon_live", "--sink", "none"]).neon
    assert neon is not None and neon.port == NeonConfig.port

    arcore = build_run_config(["--source", "arcore_tcp", "--sink", "none"]).arcore
    assert arcore is not None and arcore.port == ArCoreConfig.port
    assert arcore.accept_timeout_seconds == ArCoreConfig.accept_timeout_seconds

    recording = build_run_config(["--source", "neon_recording", "--recording-dir", _recording_dir(tmp_path), "--sink", "none"]).neon_recording
    assert recording is not None and recording.frames_per_second == NeonRecordingConfig.frames_per_second


def test_the_accept_timeout_flag_reaches_the_phone_source() -> None:
    # Launching the app by hand on a phone took longer than the default wait, and the laptop
    # ended the run before the phone connected. The wait has to be settable from the command line.
    arcore = build_run_config(["--source", "arcore_tcp", "--sink", "none", "--arcore-accept-timeout", "300"]).arcore

    assert arcore is not None and arcore.accept_timeout_seconds == pytest.approx(300.0)

    web = build_run_config([*MINIMAL_VIDEO_ARGV[:-2], "--sink", "web"]).web
    assert web is not None and web.port == WebConfig.port

    phone = build_run_config([*MINIMAL_VIDEO_ARGV[:-2], "--sink", "phone_app"]).phone_app
    assert phone is not None and phone.port == PhoneAppConfig.port


def test_phone_address_is_no_longer_an_argument(capsys: pytest.CaptureFixture[str]) -> None:
    # The phone connects to the laptop now, with the laptop address it already has. A flag for
    # the phone's address would be a flag nothing reads, and the parser must say so.
    with pytest.raises(SystemExit):
        build_run_config([*MINIMAL_VIDEO_ARGV[:-2], "--sink", "phone_app", "--phone-address", "10.0.0.2"])

    assert "unrecognized arguments: --phone-address" in capsys.readouterr().err


def test_neon_without_an_address_is_left_to_discovery() -> None:
    # Omitting the address is the normal case, not a missing argument. The source discovers the
    # device over mDNS, and an explicit address is the fallback for a network that blocks it.
    config = build_run_config(["--source", "neon_live", "--sink", "none"])

    assert config.neon is not None
    assert config.neon.address is None
    assert config.neon.port == 8080


def test_neon_address_is_carried_through_when_given() -> None:
    config = build_run_config(
        ["--source", "neon_live", "--sink", "none", "--neon-address", "10.0.0.5"]
    )

    assert config.neon is not None
    assert config.neon.address == "10.0.0.5"


def _capture_folder(tmp_path, with_meta: bool):
    folder = tmp_path / "walk_1"
    folder.mkdir()
    if with_meta:
        (folder / "meta.json").write_text("{}", encoding="utf-8")
    return folder


def test_a_neon_replay_folder_with_meta_reaches_the_neon_config(tmp_path) -> None:
    capture = _capture_folder(tmp_path, with_meta=True)

    config = build_run_config(["--source", "neon_live", "--sink", "none", "--neon-replay", str(capture)])

    assert config.neon is not None
    assert config.neon.replay_dir == str(capture)
    assert config.neon.address is None


def test_a_neon_replay_folder_without_meta_is_refused_naming_it(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    """An interrupted capture once had packets and no meta.json. Refused before the estimator loads its model."""
    folder = _capture_folder(tmp_path, with_meta=False)

    with pytest.raises(SystemExit) as exited:
        build_run_config(["--source", "neon_live", "--sink", "none", "--neon-replay", str(folder)])

    assert exited.value.code == 2
    assert f"--neon-replay {folder} is not a capture folder, it has no meta.json" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("source", "source_argv"),
    [
        ("video_file", ["--path", STREAM_URL]),
        ("arcore_tcp", []),
    ],
)
def test_neon_replay_is_refused_for_a_source_that_is_not_neon_live(
    source: str, source_argv: list[str], tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The same rule --realtime and --reconnect follow. A flag the chosen source silently ignores
    # is a run that does something other than what was typed.
    capture = _capture_folder(tmp_path, with_meta=True)

    with pytest.raises(SystemExit):
        build_run_config(["--source", source, *source_argv, "--sink", "none", "--neon-replay", str(capture)])

    assert f"--neon-replay only applies to neon_live, not {source}" in capsys.readouterr().err


def test_a_demo_capture_and_the_side_to_start_on_reach_the_neon_config(tmp_path) -> None:
    capture = _capture_folder(tmp_path, with_meta=True)

    config = build_run_config(
        ["--source", "neon_live", "--sink", "none", "--neon-address", "10.0.0.5", "--demo-capture", str(capture), "--start-with", "demo"]
    )

    assert config.neon is not None
    assert config.neon.demo_capture_dir == str(capture)
    assert config.neon.start_with is SourceMode.DEMO
    assert config.neon.address == "10.0.0.5", "the glasses side still has its address"


def test_a_run_without_a_demo_starts_with_the_glasses(tmp_path) -> None:
    config = build_run_config(["--source", "neon_live", "--sink", "none"])

    assert config.neon is not None and config.neon.demo_capture_dir is None
    assert config.neon.start_with is SourceMode.GLASSES


@pytest.mark.parametrize(
    ("extra_argv", "fragment"),
    [
        (["--start-with", "demo"], "--start-with demo needs --demo-capture"),
        (["--demo-capture", "{folder_without_meta}"], "is not a capture folder, it has no meta.json"),
        (["--demo-capture", "{capture}", "--neon-replay", "{capture}"], "--demo-capture and --neon-replay cannot be used together"),
    ],
)
def test_a_demo_that_couldnt_work_is_refused_before_anything_loads(tmp_path, capsys: pytest.CaptureFixture[str], extra_argv: list[str], fragment: str) -> None:
    capture = _capture_folder(tmp_path, with_meta=True)
    folder_without_meta = tmp_path / "half_written"
    folder_without_meta.mkdir()
    argv = [argument.format(capture=capture, folder_without_meta=folder_without_meta) for argument in extra_argv]

    with pytest.raises(SystemExit):
        build_run_config(["--source", "neon_live", "--sink", "none", *argv])

    assert fragment in capsys.readouterr().err


def test_a_demo_capture_is_refused_for_a_source_that_is_not_neon_live(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    capture = _capture_folder(tmp_path, with_meta=True)

    with pytest.raises(SystemExit):
        build_run_config([*MINIMAL_VIDEO_ARGV, "--demo-capture", str(capture)])

    assert "--demo-capture only applies to neon_live, not video_file" in capsys.readouterr().err


def test_the_source_and_the_page_share_one_switch_and_one_depth_model(tmp_path) -> None:
    # The page turns the switch, and the source plans on whatever it says, so they must be one object.
    pytest.importorskip("aiohttp", reason="the web sink comes with the web extra")
    capture = _capture_folder(tmp_path, with_meta=True)
    config = dataclasses.replace(
        build_run_config(["--source", "neon_live", "--sink", "web", "--demo-capture", str(capture)]),
        estimator_factory=lambda estimator_config: StubDepthEstimator(),
    )

    source, sink = build_source_and_sink(config)
    try:
        assert isinstance(source, CurrentStretchOnly), "frames from a side already left never reach the planner"
        assert isinstance(source.switch, SwitchableRgbSource)
        assert sink._source_switch is source.switch
        assert sink.video_feed is source.switch.video_feed
    finally:
        # Built, never started, so nothing is listening and nothing needs more than this.
        sink.close()
        source.close()


def test_neon_address_and_neon_replay_together_are_refused(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    """A replay plays a capture in place of the glasses, so the address would be ignored without a word."""
    capture = _capture_folder(tmp_path, with_meta=True)

    with pytest.raises(SystemExit):
        build_run_config(["--source", "neon_live", "--sink", "none", "--neon-address", "10.0.0.5", "--neon-replay", str(capture)])

    assert "--neon-address and --neon-replay cannot be used together" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flag", "bad_value"),
    [
        ("--walker-radius", "0"),
        ("--walker-radius", "-0.35"),
        ("--floor-max-height", "0"),
        ("--floor-max-height", "-2.2"),
        ("--process-resolution", "0"),
        ("--process-resolution", "-504"),
        ("--confidence-drop-percentile", "-1"),
        ("--confidence-drop-percentile", "100"),
        ("--arcore-port", "0"),
        ("--arcore-port", "70000"),
        ("--neon-port", "-1"),
        ("--web-port", "99999"),
    ],
)
def test_out_of_range_number_is_refused(flag: str, bad_value: str, capsys: pytest.CaptureFixture[str]) -> None:
    # argparse happily accepts a negative radius or a port above 65535, and the damage shows up
    # several layers later as arithmetic nobody can trace back to the command line.
    with pytest.raises(SystemExit):
        build_run_config([*MINIMAL_VIDEO_ARGV, flag, bad_value])

    assert flag in capsys.readouterr().err


def test_factory_refuses_a_kind_it_does_not_handle() -> None:
    # RunConfig does not validate its own fields, so this is reachable the day someone adds an
    # enum member and forgets the factory. The catch-all must say so rather than return None.
    config = RunConfig(source_kind="not_a_kind", sink_kinds=(SinkKind.NONE,), goal_mode=GoalMode.AHEAD)

    with pytest.raises(ValueError, match="no source constructor"):
        build_source(config)

    sink_config = RunConfig(source_kind=SourceKind.VIDEO_FILE, sink_kinds=("not_a_kind",), goal_mode=GoalMode.AHEAD)
    with pytest.raises(ValueError, match="no sink constructor"):
        build_sink(sink_config)


def test_pose_claiming_a_position_must_carry_one() -> None:
    with pytest.raises(ValueError, match="has_position"):
        Pose(orientation=np.array([1.0, 0.0, 0.0, 0.0]), position=None, has_position=True)


def test_pose_carrying_a_position_must_claim_one() -> None:
    with pytest.raises(ValueError, match="has_position"):
        Pose(orientation=np.array([1.0, 0.0, 0.0, 0.0]), position=np.zeros(3), has_position=False)


def test_consistent_pose_is_accepted() -> None:
    pose = Pose(orientation=np.array([1.0, 0.0, 0.0, 0.0]), position=None, has_position=False)

    assert pose.has_position is False


def test_depth_frame_accepts_a_well_formed_frame() -> None:
    frame = _depth_frame()

    assert frame.depth_meters.shape == (4, 4)


def test_depth_frame_rejects_a_depth_image_that_is_not_two_dimensional() -> None:
    with pytest.raises(ValueError, match="depth_meters"):
        _depth_frame(depth_meters=np.ones((4, 4, 3), dtype=np.float32))


def test_depth_frame_rejects_wrongly_shaped_intrinsics() -> None:
    with pytest.raises(ValueError, match="intrinsics"):
        _depth_frame(intrinsics=np.eye(4))


def test_depth_frame_rejects_a_wrongly_shaped_gaze_pixel() -> None:
    with pytest.raises(ValueError, match="gaze_pixel"):
        _depth_frame(gaze_pixel=np.array([1.0, 2.0, 3.0]))


def test_planned_path_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same shape"):
        PlannedPath(
            timestamp_seconds=0.0,
            times_seconds=np.array([0.0, 0.1, 0.2]),
            lateral_offsets_meters=np.array([0.0, 0.1]),
            lookahead_heading_radians=0.0,
            alarm=False,
            cumulative_cost_bits=0.0,
            scene_information_bits=0.0,
            avoidance_surprise_bits=0.0,
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_planned_path_rejects_a_non_finite_offset(bad_value: float) -> None:
    # A sink serializes this straight to JSON, and json.dumps writes a bare NaN that no strict
    # parser on the other end will read back.
    with pytest.raises(ValueError, match="lateral_offsets_meters"):
        PlannedPath(
            timestamp_seconds=0.0,
            times_seconds=np.array([0.0, 0.1]),
            lateral_offsets_meters=np.array([0.0, bad_value]),
            lookahead_heading_radians=0.0,
            alarm=False,
            cumulative_cost_bits=0.0,
            scene_information_bits=0.0,
            avoidance_surprise_bits=0.0,
        )


def test_planned_path_rejects_a_non_finite_time() -> None:
    with pytest.raises(ValueError, match="times_seconds"):
        PlannedPath(
            timestamp_seconds=0.0,
            times_seconds=np.array([0.0, np.nan]),
            lateral_offsets_meters=np.array([0.0, 0.1]),
            lookahead_heading_radians=0.0,
            alarm=False,
            cumulative_cost_bits=0.0,
            scene_information_bits=0.0,
            avoidance_surprise_bits=0.0,
        )


@pytest.mark.parametrize(("device", "warned"), [("cpu", True), ("cuda", False)])
def test_neon_live_on_a_cpu_estimator_warns_and_still_builds(device: str, warned: bool, caplog: pytest.LogCaptureFixture) -> None:
    config = dataclasses.replace(
        _run_config_for(SourceKind.NEON_LIVE),
        estimator_factory=lambda estimator_config: StubDepthEstimator(device=device),
    )

    with caplog.at_level("WARNING", logger="nav.config"):
        source = build_source(config)

    assert hasattr(source, "frames"), "warned, but must still build"
    assert any("on the CPU" in record.message for record in caplog.records) is warned


def test_a_recording_on_a_cpu_estimator_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    # A recording is only slow to process on the CPU. Nobody is walking behind it.
    config = dataclasses.replace(
        _run_config_for(SourceKind.VIDEO_FILE),
        estimator_factory=lambda estimator_config: StubDepthEstimator(device="cpu"),
    )

    with caplog.at_level("WARNING", logger="nav.config"):
        build_source(config)

    assert not any("on the CPU" in record.message for record in caplog.records)


def test_timing_log_and_record_to_together_are_refused(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    # A recording run writes timing.jsonl beside its frames. Two destinations would leave a reader guessing.
    with pytest.raises(SystemExit):
        build_run_config([*MINIMAL_VIDEO_ARGV, "--timing-log", str(tmp_path / "t.jsonl"), "--record-to", str(tmp_path / "log")])

    assert "--timing-log and --record-to cannot be used together" in capsys.readouterr().err


def test_a_timing_log_that_already_holds_lines_is_refused(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    # Two runs appended to one log read back as one run that is not one.
    used = tmp_path / "t.jsonl"
    used.write_bytes(b'{"timestamp_seconds": 1.0}\n')

    with pytest.raises(SystemExit):
        build_run_config([*MINIMAL_VIDEO_ARGV, "--timing-log", str(used)])

    assert "already holds a timing log" in capsys.readouterr().err


def test_a_new_or_empty_timing_log_file_is_accepted(tmp_path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")

    assert build_run_config([*MINIMAL_VIDEO_ARGV, "--timing-log", str(empty)]).timing_log == str(empty)
    assert build_run_config([*MINIMAL_VIDEO_ARGV, "--timing-log", str(tmp_path / "new.jsonl")]).timing_log == str(tmp_path / "new.jsonl")


@pytest.mark.parametrize("where", ["a directory", "a missing folder"])
def test_a_timing_log_that_is_a_directory_or_in_a_missing_folder_is_refused(where: str, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    """Either one let the walk start and fail on its first line, running untimed."""
    target = tmp_path if where == "a directory" else tmp_path / "no_such_folder" / "t.jsonl"

    with pytest.raises(SystemExit):
        build_run_config([*MINIMAL_VIDEO_ARGV, "--timing-log", str(target)])

    message = capsys.readouterr().err
    assert ("is a directory" in message) if where == "a directory" else ("in a folder that doesn't exist" in message)


# *******************************************
# Building both ends together
# *******************************************


def test_build_source_and_sink_gives_the_neon_source_and_the_web_sink_one_feed() -> None:
    # The glasses' video goes from the source's device to the web sink without the loop seeing it,
    # which only works if both ends were handed the same feed by the one place that builds them.
    config = RunConfig(
        source_kind=SourceKind.NEON_LIVE,
        sink_kinds=(SinkKind.WEB,),
        goal_mode=GoalMode.AHEAD,
        neon=NeonConfig(),
        estimator=EstimatorConfig(),
        estimator_factory=lambda estimator_config: StubDepthEstimator(),
        web=WebConfig(port=0),
    )

    source, sink = build_source_and_sink(config)
    try:
        assert isinstance(sink.video_feed, SceneVideoFeed)
        assert source.rgb_source.video_feed is sink.video_feed
    finally:
        sink.close()


def test_a_source_without_scene_video_leaves_the_web_sink_without_a_feed() -> None:
    config = RunConfig(
        source_kind=SourceKind.VIDEO_FILE,
        sink_kinds=(SinkKind.WEB,),
        goal_mode=GoalMode.AHEAD,
        video=VideoConfig(path=STREAM_URL),
        estimator=EstimatorConfig(),
        estimator_factory=lambda estimator_config: StubDepthEstimator(),
        web=WebConfig(port=0),
    )

    source, sink = build_source_and_sink(config)
    try:
        assert sink.video_feed is None
        assert not hasattr(source.rgb_source, "video_feed"), "a plain camera has no feed to offer"
    finally:
        sink.close()


# *******************************************
# The demo recording
# *******************************************


def test_there_is_no_demo_recording_flag_any_more(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    # The page used to play a file of its own, out of step with the planning. A demo from a
    # recording now converts it to a capture and replays that, so the flag is refused outright.
    recording = tmp_path / "demo.mp4"
    recording.write_bytes(b"\x00" * 16)

    with pytest.raises(SystemExit):
        build_run_config([*MINIMAL_VIDEO_ARGV[:-2], "--sink", "web", "--demo-recording", str(recording)])

    assert "--demo-recording" in capsys.readouterr().err


# *******************************************
# The neon_recording source's flags
# *******************************************


def _recording_dir(folder) -> str:
    """A folder the parser takes for a recording. The Companion app writes info.json into every export."""
    (folder / "info.json").write_text("{}", encoding="utf-8")
    return str(folder)


def test_the_recording_rate_reaches_the_recording_source(tmp_path) -> None:
    config = build_run_config(["--source", "neon_recording", "--recording-dir", _recording_dir(tmp_path), "--recording-rate", "3.5", "--sink", "none"])

    assert config.neon_recording == NeonRecordingConfig(recording_dir=str(tmp_path), frames_per_second=3.5)
    # A recording goes through the estimator like the live glasses, so it gets an estimator config.
    assert config.estimator is not None


@pytest.mark.parametrize("source", ["neon_live", "video_file", "logged"])
def test_a_recording_rate_given_to_another_source_is_refused(source: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        build_run_config(["--source", source, "--recording-rate", "2", "--sink", "none"])

    assert "--recording-rate only applies to neon_recording" in capsys.readouterr().err


@pytest.mark.parametrize("rate", ["0", "-1"])
def test_a_recording_rate_at_or_below_zero_is_refused(rate: str, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        build_run_config(["--source", "neon_recording", "--recording-dir", _recording_dir(tmp_path), "--recording-rate", rate, "--sink", "none"])

    assert "--recording-rate" in capsys.readouterr().err


@pytest.mark.parametrize("rate", ["nan", "inf"])
def test_a_recording_rate_that_is_not_a_finite_number_is_refused(rate: str, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    # float() reads both, and nan passes a "<= 0" check. Without the parser's refusal, inf thins to
    # nothing and nan raises from inside the source, after the model has loaded.
    with pytest.raises(SystemExit):
        build_run_config(["--source", "neon_recording", "--recording-dir", _recording_dir(tmp_path), "--recording-rate", rate, "--sink", "none"])

    message = capsys.readouterr().err
    assert "--recording-rate" in message
    assert "must be a finite number" in message


def test_a_folder_that_is_not_a_recording_is_refused_before_the_model_loads(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    # A capture folder or a parent folder, given by mistake. The reader would only say so after the
    # depth model had loaded.
    with pytest.raises(SystemExit):
        build_run_config(["--source", "neon_recording", "--recording-dir", str(tmp_path), "--sink", "none"])

    message = capsys.readouterr().err
    assert str(tmp_path) in message
    assert "is not a Neon recording, it has no info.json" in message


def test_the_retired_plugin_source_is_refused_naming_the_recording_source(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        build_run_config(["--source", "neon_plugin", "--recording-dir", ".", "--sink", "none"])

    message = capsys.readouterr().err
    assert "invalid choice: 'neon_plugin'" in message
    assert "neon_recording" in message


def test_the_retired_plugin_model_flag_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        build_run_config(["--source", "neon_recording", "--recording-dir", ".", "--plugin-model", "DA3Metric-Large", "--sink", "none"])

    assert "unrecognized arguments: --plugin-model" in capsys.readouterr().err
