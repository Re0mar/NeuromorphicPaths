"""
The replay's route goal, rebuilt from the sidecars a live run's tracker wrote, against the live tracker's own goals.

The live tracker writes the sidecars here, through the real code, with the file service standing in for
the network. The timing log is written by hand, one line per frame with only its arrival.
"""

# Standard library imports
import json
from pathlib import Path

# Third party imports
import pytest

# Local package imports
from nav.evaluation.route_replay import RouteSidecarRefused, recorded_route_goals
from nav.route.course import Fix
from nav.route.geodesy import LatLon
from nav.route.route import RouteConfig
from nav.route.service import FileRouteService
from nav.runtime.route_goal import LOCATION_LOG_FILENAME, RouteGoalTracker
from nav.runtime.timing import TIMING_FILENAME

FIXTURE = Path(__file__).parent / "fixtures" / "ors_walking_response.json"
START = LatLon(51.8215, 5.8650)
DESTINATION = LatLon(51.8245, 5.8685)
CONFIG = RouteConfig()
GOAL_DISTANCE_METERS = 4.0


def write_timing(log_dir: Path, arrivals: dict[float, float]) -> None:
    lines = [
        json.dumps({"timestamp_seconds": timestamp, "capture_seconds": None, "arrival_seconds": arrival, "depth_ready_seconds": None, "plan_done_seconds": None, "floor_source": None})
        for timestamp, arrival in arrivals.items()
    ]
    (log_dir / TIMING_FILENAME).write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def recorded_live_run(log_dir: Path, arrivals: list[float]) -> list:
    """A live tracker fed one destination and a walking fix, asked for each arrival's goal. Its sidecars land in log_dir."""
    clock = iter([99.0] * 10)
    tracker = RouteGoalTracker(FileRouteService(FIXTURE), CONFIG, GOAL_DISTANCE_METERS, log_dir=log_dir, clock=lambda: next(clock))
    tracker.on_destination(DESTINATION)
    tracker.on_fix(Fix(START, 6.0, 330.0, 1.3, 1_760_000_000_000, 99.5))
    tracker.close()
    goals = [tracker.frame(arrival).goal for arrival in arrivals]
    return goals


def test_a_replay_reproduces_the_route_goal_from_the_sidecars(tmp_path: Path) -> None:
    arrivals = [99.8, 100.1, 100.4]
    live = recorded_live_run(tmp_path, arrivals)
    assert all(goal is not None for goal in live), "the live run must have had a goal to reproduce"
    write_timing(tmp_path, {1.0: 99.8, 1.1: 100.1, 1.2: 100.4})

    replayed = recorded_route_goals(tmp_path, [1.0, 1.1, 1.2], CONFIG, GOAL_DISTANCE_METERS)

    assert replayed == live


def test_a_frame_without_an_arrival_gets_no_route_goal(tmp_path: Path) -> None:
    recorded_live_run(tmp_path, [100.0])
    write_timing(tmp_path, {1.0: 100.0})
    assert recorded_route_goals(tmp_path, [1.0, 7.0], CONFIG, GOAL_DISTANCE_METERS)[1] is None


def test_a_fix_received_after_the_frame_is_not_used(tmp_path: Path) -> None:
    # The fix reached the laptop at 99.5. A frame that arrived at 99.4 had nothing yet.
    recorded_live_run(tmp_path, [100.0])
    write_timing(tmp_path, {1.0: 99.4, 1.1: 99.6})
    goals = recorded_route_goals(tmp_path, [1.0, 1.1], CONFIG, GOAL_DISTANCE_METERS)
    assert goals[0] is None and goals[1] is not None


def test_a_walk_without_sidecars_gets_no_route_goals(tmp_path: Path) -> None:
    assert recorded_route_goals(tmp_path, [1.0, 2.0], CONFIG, GOAL_DISTANCE_METERS) == [None, None]


def test_a_broken_location_line_is_refused_naming_the_line(tmp_path: Path) -> None:
    recorded_live_run(tmp_path, [100.0])
    write_timing(tmp_path, {1.0: 100.0})
    with (tmp_path / LOCATION_LOG_FILENAME).open("ab") as handle:
        handle.write(b'{"latitude": "north"}\n')
    with pytest.raises(RouteSidecarRefused, match="line 2"):
        recorded_route_goals(tmp_path, [1.0], CONFIG, GOAL_DISTANCE_METERS)
