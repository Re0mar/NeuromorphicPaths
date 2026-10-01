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
from nav.types import DepthFrame, PlannedPath, Pose

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

MINIMAL_VIDEO_ARGV = ["--source", "video_file", "--path", "scene.mp4", "--model", "a/model", "--sink", "none"]


def test_minimal_command_line_builds_a_config() -> None:
    config = build_run_config(MINIMAL_VIDEO_ARGV)

    assert config.source_kind is SourceKind.VIDEO_FILE
    assert config.sink_kind is SinkKind.NONE
    assert config.goal_mode is GoalMode.AHEAD
    assert config.video is not None
    assert config.video.path == "scene.mp4"
    assert config.estimator is not None
    assert config.estimator.model_name == "a/model"
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
        (["--source", "video_file", "--model", "a/model", "--sink", "none"], "--path"),
        (["--source", "neon_live", "--model", "a/model", "--sink", "none"], "--neon-address"),
        (["--source", "neon_plugin", "--sink", "none"], "--recording-dir"),
        (["--source", "logged", "--sink", "none"], "--log-dir"),
        (["--source", "video_file", "--path", "scene.mp4", "--sink", "none"], "--model"),
        (["--source", "video_file", "--path", "scene.mp4", "--model", "a/model", "--sink", "phone_app"], "--phone-address"),
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
        ["--source", "video_file", "--path", "scene.mp4", "--model", "a/model", "--sink", "bogus"],
        ["--source", "video_file", "--path", "scene.mp4", "--model", "a/model", "--sink", "none", "--goal", "bogus"],
    ],
)
def test_unknown_kind_is_refused(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        build_run_config(argv)


@pytest.mark.parametrize("source_kind", list(SourceKind))
def test_every_source_kind_reaches_an_arm(source_kind: SourceKind) -> None:
    config = RunConfig(source_kind=source_kind, sink_kind=SinkKind.NONE, goal_mode=GoalMode.AHEAD)

    # NotImplementedError means the arm exists and names the step that fills it. ValueError would
    # mean the member fell through to the catch-all, which is the failure this is watching for.
    with pytest.raises(NotImplementedError):
        build_source(config)


@pytest.mark.parametrize("sink_kind", list(SinkKind))
def test_every_sink_kind_reaches_an_arm(sink_kind: SinkKind) -> None:
    config = RunConfig(source_kind=SourceKind.VIDEO_FILE, sink_kind=sink_kind, goal_mode=GoalMode.AHEAD)

    with pytest.raises(NotImplementedError):
        build_sink(config)


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
