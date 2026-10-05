"""
Golden slices: written, read back, refused, and cut through the real command.

The hand-built slice here is the arithmetic beside the golden test's snapshot. Its alarm and band
figures follow from the alarm's rule and the hold, worked out in the test, so a defect that moves
every number the same way can't pass it the way it could pass a snapshot.
"""

# Standard library imports
import dataclasses
import gzip
import json
from pathlib import Path

# Third party imports
import numpy as np
import pytest

# Local package imports
import nav.evaluation.fixture as fixture_module
from nav.evaluation.config import PlannerNumbersConfig
from nav.evaluation.fixture import (
    FIXTURE_FORMAT,
    MAX_FIXTURE_SET_BYTES,
    FixtureRefused,
    check_fixture_set_size,
    check_rounding,
    main,
    read_fixture,
    rounded_input,
    shortest_windows,
    write_fixture,
)
from nav.evaluation.planner_numbers import ClearanceBand, PlannerInput, whole_walk_numbers
from nav.evaluation.replay import replayed_frames
from nav.planner.config import GoalMode, PlannerConfig
from nav.types import ObstaclePoint, ObstacleSet
from nav.walker import WalkerConfig
from synthetic_walks import write_recording

PLANNER = PlannerConfig()
FRAME_SECONDS = 0.125
ORIGIN = np.array([0.1, 1.3, -2.0])
LATERAL = np.array([1.0, 0.0, 0.0])
FORWARD = np.array([0.0, 0.0, -1.0])
PROVENANCE = {"walk": "hand_built", "segment": -1, "slice_seconds": [0.0, 1.375], "cut_from": {"goal_mode": "ahead"}, "walker": dataclasses.asdict(WalkerConfig())}


def point(lateral: float, forward: float, clearance: float, group: int, **changes) -> ObstaclePoint:
    built = ObstaclePoint(lateral, forward, group, clearance, 0.05, None, None, False, np.zeros(3))
    return dataclasses.replace(built, **changes)


def planner_input(time: float, points: tuple[ObstaclePoint, ...] = (), world: bool = True) -> PlannerInput:
    return PlannerInput(
        time,
        ObstacleSet(time, points, len(points)),
        ORIGIN if world else None,
        LATERAL if world else None,
        FORWARD if world else None,
        None,
    )


def wall_ahead(time: float) -> PlannerInput:
    """A wall across the corridor 0.5 m from the footprint. 0.5 m at 1.4 m/s is 0.36 s, under the 0.7 s alarm threshold."""
    return planner_input(time, tuple(point(lateral, 0.85, 0.5, group, is_wall=True) for group, lateral in enumerate((-0.3, -0.1, 0.1, 0.3))))


def hand_built_slice() -> list[PlannerInput]:
    """
    Twelve frames 0.125 s apart: three empty, three with the wall ahead, six empty.

    0.125 is exact in binary, so the hold's edge falls on a frame without rounding deciding which side.
    """
    return [wall_ahead(FRAME_SECONDS * index) if 3 <= index <= 5 else planner_input(FRAME_SECONDS * index) for index in range(12)]


def write_document(path: Path, document: dict) -> Path:
    with gzip.open(path, "wb") as packed:
        packed.write(json.dumps(document).encode("utf-8"))
    return path


def valid_document() -> dict:
    return {"format": FIXTURE_FORMAT, **PROVENANCE, "frames": [fixture_module._encode_frame(planner_input(0.0, (point(0.0, 2.0, 1.6, 1),)))]}


def run(arguments: list, capsys) -> tuple[int, str, str]:
    code = main([str(argument) for argument in arguments])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture(scope="module")
def recording(tmp_path_factory) -> Path:
    """A straight walk whose box is always 1 to 3 m ahead, so it has no band or clear frames at all."""
    return write_recording(tmp_path_factory.mktemp("walk") / "straight", lambda time: 0.0, 22.0)


# *******************************************
# Written and read back
# *******************************************


def test_a_fixture_round_trips_every_field_the_planner_reads(tmp_path: Path) -> None:
    moving = point(0.123456, 2.345678, 1.987654, 7, closing_rate_mps=-0.0896077, velocity_mps=np.array([0.339976, 0.662309]), noise_scale_meters=0.0614627)
    wall = point(-0.6, 1.2, 0.95, 8, is_wall=True)
    inputs = [planner_input(0.0, (moving, wall)), planner_input(0.1, (wall,), world=False)]
    write_fixture(tmp_path / "golden_round.json.gz", inputs, PROVENANCE)
    golden = read_fixture(tmp_path / "golden_round.json.gz")
    assert golden.provenance == PROVENANCE
    for original, read in zip(inputs, golden.inputs, strict=True):
        stored = rounded_input(original)
        assert read.timestamp_seconds == stored.timestamp_seconds
        assert read.obstacles.groups_in_view == stored.obstacles.groups_in_view
        for a, b in zip(read.obstacles.points, stored.obstacles.points, strict=True):
            assert (a.lateral_meters, a.forward_meters, a.group_id, a.clearance_meters, a.noise_scale_meters, a.is_wall, a.closing_rate_mps) == (
                b.lateral_meters, b.forward_meters, b.group_id, b.clearance_meters, b.noise_scale_meters, b.is_wall, b.closing_rate_mps
            )
            assert (a.velocity_mps is None) == (b.velocity_mps is None)
            if a.velocity_mps is not None:
                assert np.array_equal(a.velocity_mps, b.velocity_mps)
        for name in ("origin", "lateral_axis", "forward_axis", "gaze_ground_point"):
            assert (getattr(read, name) is None) == (getattr(stored, name) is None)
    # Absolute values too, so a defect shared by the writer and rounded_input can't pass by agreeing with itself.
    moving_read = golden.inputs[0].obstacles.points[0]
    assert moving_read.lateral_meters == 0.1235
    assert moving_read.noise_scale_meters == 0.06146
    assert moving_read.closing_rate_mps == -0.0896
    assert np.array_equal(moving_read.velocity_mps, [0.34, 0.6623])
    assert golden.inputs[0].obstacles.points[1].is_wall is True
    assert golden.inputs[0].obstacles.points[1].closing_rate_mps is None
    assert golden.inputs[1].origin is None


def test_writing_the_same_slice_twice_gives_the_same_bytes(tmp_path: Path) -> None:
    write_fixture(tmp_path / "a" / "golden_x.json.gz", hand_built_slice(), PROVENANCE)
    write_fixture(tmp_path / "b" / "golden_x.json.gz", hand_built_slice(), PROVENANCE)
    assert (tmp_path / "a" / "golden_x.json.gz").read_bytes() == (tmp_path / "b" / "golden_x.json.gz").read_bytes()
    # Two writes in the same second would match even with a clock in the header, so check it directly:
    # bytes 4 to 8 of a gzip header are its mtime, and the name flag (bit 3 of byte 3) is off.
    header = (tmp_path / "a" / "golden_x.json.gz").read_bytes()[:10]
    assert header[4:8] == b"\0\0\0\0"
    assert header[3] & 0x08 == 0


def test_a_hand_built_slice_gives_the_numbers_worked_out_by_hand(tmp_path: Path) -> None:
    write_fixture(tmp_path / "golden_hand.json.gz", hand_built_slice(), PROVENANCE)
    golden = read_fixture(tmp_path / "golden_hand.json.gz")
    frames = replayed_frames(golden.inputs, PLANNER, WalkerConfig(), GoalMode.AHEAD)
    numbers = whole_walk_numbers(frames, PLANNER, PlannerNumbersConfig())
    # Nine empty frames are clear, the three with the wall 0.5 m out are close.
    assert numbers.frames_by_band[ClearanceBand.CLEAR] == 9
    assert numbers.frames_by_band[ClearanceBand.CLOSE] == 3
    # With nothing in view the goal straight ahead decides, so the heading is exactly zero.
    assert all(frames[index].path.first_heading_radians == 0.0 for index in (0, 1, 2, 6, 7, 8, 9, 10, 11))
    assert numbers.pinned_by_band[ClearanceBand.CLEAR] == 0
    # Raised at 0.375, 0.5 and 0.625 s. The hold runs 0.5 s from when the alarm was first raised, so
    # it is still up at 0.75 s and down at 0.875 s: four frames on, one change up and one down. The
    # longest it stayed up after its raise decision was last true is 0.75 - 0.625 = 0.125 s.
    assert numbers.alarm_on_frames == 4
    assert numbers.alarm_changes == 2
    assert numbers.longest_hold_seconds == 0.125


def test_shortest_windows_find_the_tightest_stretch() -> None:
    band, clear, near = ClearanceBand.BAND, ClearanceBand.CLEAR, ClearanceBand.NEAR
    bands = [near, band, clear, near, band, clear, near]
    assert shortest_windows(bands, 2) == [(0, 6), (1, 6)]
    assert shortest_windows(bands, 3) == []


# *******************************************
# Refused on read
# *******************************************


def test_an_unknown_fixture_format_is_refused(tmp_path: Path) -> None:
    path = write_document(tmp_path / "golden_old.json.gz", {**valid_document(), "format": "planner-golden-0"})
    with pytest.raises(FixtureRefused, match=r"golden_old\.json\.gz has format 'planner-golden-0'"):
        read_fixture(path)


def test_a_point_row_of_the_wrong_length_is_refused(tmp_path: Path) -> None:
    document = valid_document()
    document["frames"][0]["points"][0] = document["frames"][0]["points"][0][:7]
    with pytest.raises(FixtureRefused, match="frame 0, point 0: a point row has 8 values"):
        read_fixture(write_document(tmp_path / "golden_short.json.gz", document))


def test_a_non_finite_coordinate_is_refused(tmp_path: Path) -> None:
    document = valid_document()
    document["frames"][0]["points"][0][0] = float("inf")
    with pytest.raises(FixtureRefused, match="frame 0, point 0: lateral_meters must be a finite number"):
        read_fixture(write_document(tmp_path / "golden_inf.json.gz", document))


def test_timestamps_going_backwards_are_refused(tmp_path: Path) -> None:
    document = valid_document()
    later = dict(document["frames"][0], t=1.0)
    document["frames"] = [later, document["frames"][0]]
    with pytest.raises(FixtureRefused, match="frame 1: t goes backwards"):
        read_fixture(write_document(tmp_path / "golden_back.json.gz", document))


def test_a_missing_frame_field_is_refused(tmp_path: Path) -> None:
    document = valid_document()
    del document["frames"][0]["groups_in_view"]
    with pytest.raises(FixtureRefused, match="frame 0: missing groups_in_view"):
        read_fixture(write_document(tmp_path / "golden_missing.json.gz", document))


def test_a_world_frame_with_only_some_parts_is_refused(tmp_path: Path) -> None:
    document = valid_document()
    document["frames"][0]["lateral_axis"] = None
    with pytest.raises(FixtureRefused, match="must all be present or all null"):
        read_fixture(write_document(tmp_path / "golden_half.json.gz", document))


def test_a_fixture_cut_with_another_walker_is_refused(tmp_path: Path) -> None:
    write_fixture(tmp_path / "golden_walker.json.gz", hand_built_slice(), {**PROVENANCE, "walker": {"radius_meters": 0.5}})
    with pytest.raises(FixtureRefused, match="Re-cut the slice"):
        read_fixture(tmp_path / "golden_walker.json.gz").require_walker(WalkerConfig())


def test_a_non_finite_value_is_refused_on_write(tmp_path: Path) -> None:
    broken = planner_input(0.0, (point(float("nan"), 2.0, 1.6, 1),))
    with pytest.raises(ValueError):
        write_fixture(tmp_path / "golden_nan.json.gz", [broken], PROVENANCE)


# *******************************************
# Refused by the cutter
# *******************************************


def test_the_cutter_refuses_rounding_that_changes_a_plan(monkeypatch) -> None:
    # Rounding far coarser than the module's, which moves the wall out of the way, stands in for a
    # slice whose 0.1 mm rounding would decide a plan.
    def too_coarse(row: PlannerInput) -> PlannerInput:
        points = tuple(dataclasses.replace(p, forward_meters=p.forward_meters + 5.0, clearance_meters=p.clearance_meters + 5.0) for p in row.obstacles.points)
        return dataclasses.replace(row, obstacles=ObstacleSet(row.timestamp_seconds, points, len(points)))

    monkeypatch.setattr(fixture_module, "rounded_input", too_coarse)
    with pytest.raises(FixtureRefused, match="rounding changes the plan on 4 frames, first at t = 0.375"):
        check_rounding(hand_built_slice(), PLANNER, WalkerConfig(), GoalMode.AHEAD)


def test_the_real_rounding_leaves_the_hand_built_slice_alone() -> None:
    check_rounding(hand_built_slice(), PLANNER, WalkerConfig(), GoalMode.AHEAD)


def test_the_cutter_refuses_a_fixture_set_over_the_size_cap(tmp_path: Path) -> None:
    (tmp_path / "golden_big.json.gz").write_bytes(b"\0" * (MAX_FIXTURE_SET_BYTES - 10))
    new = tmp_path / "golden_new.json.gz"
    new.write_bytes(b"\0" * 20)
    with pytest.raises(FixtureRefused, match="over the cap of 1000000. The slice was not kept"):
        check_fixture_set_size(new)
    assert not new.exists()
    assert (tmp_path / "golden_big.json.gz").exists()


def test_a_fixture_set_under_the_cap_is_kept(tmp_path: Path) -> None:
    new = tmp_path / "golden_new.json.gz"
    new.write_bytes(b"\0" * 20)
    (tmp_path / "not_a_slice.bin").write_bytes(b"\0" * MAX_FIXTURE_SET_BYTES)
    assert check_fixture_set_size(new) == 20


def test_the_cutter_refuses_a_slice_short_of_band_or_clear_frames(recording: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "golden_short.json.gz"
    code, _, err = run(["cut", recording, "--start", 0.0, "--end", 21.0, "--out", out], capsys)
    assert code == 1
    assert "the window holds 0 band and 0 clear frames, and a slice needs 100 of each" in err
    assert not out.exists()


def test_the_cut_command_refuses_a_window_outside_the_segment(recording: Path, tmp_path: Path, capsys) -> None:
    code, _, err = run(["cut", recording, "--start", 5.0, "--end", 99.0, "--out", tmp_path / "golden_x.json.gz"], capsys)
    assert code == 2
    assert "is not inside segment -1" in err


def test_find_with_no_qualifying_window_exits_3(recording: Path, capsys) -> None:
    code, _, err = run(["find", recording], capsys)
    assert code == 3
    assert "the segment has 0 and 0" in err
