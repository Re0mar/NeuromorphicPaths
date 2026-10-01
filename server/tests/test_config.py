"""
Covers the command line, the two factories, and the validation the shared types do on themselves.

The factory tests are completeness checks rather than behavior checks. Every kind must reach an
arm, so that adding a member to either enum and forgetting the factory fails here instead of at
the start of a walk.
"""

# Standard library imports
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
)
from nav.sources.config import (
    ArCoreConfig,
    EstimatorConfig,
    LoggedConfig,
    NeonConfig,
    NeonPluginConfig,
    TapConfig,
    VideoConfig,
)
from nav.runtime.tap import RecordingTap
from nav.scene.config import SceneConfig
from nav.sinks.config import PhoneAppConfig, WebConfig
from nav.types import DepthFrame, PlannedPath, Pose

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

MINIMAL_VIDEO_ARGV = ["--source", "video_file", "--path", "scene.mp4", "--sink", "none"]


def test_minimal_command_line_builds_a_config() -> None:
    config = build_run_config(MINIMAL_VIDEO_ARGV)

    assert config.source_kind is SourceKind.VIDEO_FILE
    assert config.sink_kind is SinkKind.NONE
    assert config.goal_mode is GoalMode.AHEAD
    assert config.video is not None
    assert config.video.path == "scene.mp4"
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
        (["--source", "neon_plugin", "--sink", "none"], "--recording-dir"),
        (["--source", "logged", "--sink", "none"], "--log-dir"),
        (["--source", "video_file", "--path", "scene.mp4", "--sink", "phone_app"], "--phone-address"),
    ],
)
def test_missing_required_argument_names_it(argv: list[str], expected_in_message: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        build_run_config(argv)

    assert expected_in_message in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["--source", "bogus", "--sink", "none"],
        ["--source", "video_file", "--path", "scene.mp4", "--sink", "bogus"],
        ["--source", "video_file", "--path", "scene.mp4", "--sink", "none", "--goal", "bogus"],
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
    SourceKind.NEON_PLUGIN: {"neon_plugin": NeonPluginConfig(recording_dir="a_recording")},
}


def _run_config_for(source_kind: SourceKind) -> RunConfig:
    return RunConfig(
        source_kind=source_kind,
        sink_kind=SinkKind.NONE,
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
        sink_kind=SinkKind.NONE,
        goal_mode=GoalMode.AHEAD,
        logged=LoggedConfig(log_dir="a_log"),
        tap=TapConfig(log_dir=str(tmp_path / "recorded")),
    )

    assert isinstance(build_source(config), RecordingTap)


def test_without_record_to_the_source_is_not_wrapped() -> None:
    config = RunConfig(
        source_kind=SourceKind.LOGGED,
        sink_kind=SinkKind.NONE,
        goal_mode=GoalMode.AHEAD,
        logged=LoggedConfig(log_dir="a_log"),
    )

    assert not isinstance(build_source(config), RecordingTap)

def test_a_built_source_refuses_a_missing_config() -> None:
    # RunConfig does not validate across its own fields, so a kind whose config was never built is
    # reachable. The factory must say which one rather than construct a source around a None.
    config = RunConfig(source_kind=SourceKind.VIDEO_FILE, sink_kind=SinkKind.NONE, goal_mode=GoalMode.AHEAD)

    with pytest.raises(ValueError, match="video_file needs a video config"):
        build_source(config)


def test_estimator_backed_source_refuses_a_missing_estimator_config() -> None:
    config = RunConfig(
        source_kind=SourceKind.VIDEO_FILE,
        sink_kind=SinkKind.NONE,
        goal_mode=GoalMode.AHEAD,
        video=VideoConfig(path="scene.mp4"),
    )

    with pytest.raises(ValueError, match="needs an estimator config"):
        build_source(config)


BUILT_SINK_KINDS = {
    SinkKind.NONE: {},
    SinkKind.DEBUG_WINDOW: {},
    SinkKind.WEB: {"web": WebConfig(port=0)},
    SinkKind.PHONE_APP: {"phone_app": PhoneAppConfig(address="127.0.0.1", port=1)},
}


def test_every_sink_kind_is_accounted_for() -> None:
    assert set(BUILT_SINK_KINDS) == set(SinkKind)


@pytest.mark.parametrize("sink_kind", sorted(BUILT_SINK_KINDS, key=lambda kind: kind.value))
def test_built_sink_kinds_return_a_sink_without_opening_anything(sink_kind: SinkKind) -> None:
    config = RunConfig(source_kind=SourceKind.VIDEO_FILE, sink_kind=sink_kind, goal_mode=GoalMode.AHEAD, **BUILT_SINK_KINDS[sink_kind])

    sink = build_sink(config)

    # Construction must not open a window, bind a port or connect to a phone. All of that happens
    # on the first publish, so this test runs with no display and nothing listening on port 1.
    assert hasattr(sink, "publish")
    assert hasattr(sink, "close")
    sink.close()


def test_a_built_sink_refuses_a_missing_config() -> None:
    config = RunConfig(source_kind=SourceKind.VIDEO_FILE, sink_kind=SinkKind.PHONE_APP, goal_mode=GoalMode.AHEAD)

    with pytest.raises(ValueError, match="phone_app needs"):
        build_sink(config)



def test_default_checkpoint_is_a_metric_one() -> None:
    # The old script defaulted to DA3NESTED-GIANT-LARGE, which returns relative depth, so every
    # clearance in meters was wrong by an unknown scale and the file carried a cam_height rescale
    # to paper over it. Asserts the property rather than the exact name, so a version bump passes
    # and a swap back to a relative checkpoint does not.
    assert "METRIC" in EstimatorConfig.model_name.upper()

def test_the_floor_tilt_and_fallback_fov_flags_reach_their_layers() -> None:
    # The first real recording was refused frame after frame by the floor gate the glasses
    # script shipped with, and the cloud was stretched by its 100 degree fallback field of view.
    # A tuning run has to reach both without editing code.
    config = build_run_config([*MINIMAL_VIDEO_ARGV, "--floor-max-tilt", "65", "--fallback-fov", "75"])

    assert config.scene.floor_max_tilt_degrees == pytest.approx(65.0)
    assert config.estimator is not None
    assert config.estimator.fallback_half_field_of_view_degrees == pytest.approx(37.5)


def test_the_floor_tilt_and_fallback_fov_defaults_match_their_configs() -> None:
    config = build_run_config(MINIMAL_VIDEO_ARGV)

    assert config.scene.floor_max_tilt_degrees == SceneConfig.floor_max_tilt_degrees
    assert config.estimator is not None
    assert config.estimator.fallback_half_field_of_view_degrees == EstimatorConfig.fallback_half_field_of_view_degrees


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


@pytest.mark.parametrize(
    ("flag", "bad_value"),
    [
        ("--walker-radius", "0"),
        ("--walker-radius", "-0.35"),
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
    config = RunConfig(source_kind="not_a_kind", sink_kind=SinkKind.NONE, goal_mode=GoalMode.AHEAD)

    with pytest.raises(ValueError, match="no source constructor"):
        build_source(config)

    sink_config = RunConfig(source_kind=SourceKind.VIDEO_FILE, sink_kind="not_a_kind", goal_mode=GoalMode.AHEAD)
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
            first_heading_radians=0.0,
            alarm=False,
            cumulative_cost_bits=0.0,
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
            first_heading_radians=0.0,
            alarm=False,
            cumulative_cost_bits=0.0,
        )


def test_planned_path_rejects_a_non_finite_time() -> None:
    with pytest.raises(ValueError, match="times_seconds"):
        PlannedPath(
            timestamp_seconds=0.0,
            times_seconds=np.array([0.0, np.nan]),
            lateral_offsets_meters=np.array([0.0, 0.1]),
            first_heading_radians=0.0,
            alarm=False,
            cumulative_cost_bits=0.0,
        )
