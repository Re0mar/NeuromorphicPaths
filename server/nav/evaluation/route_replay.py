"""
The route goal a recorded walk's frames had, rebuilt from the sidecars beside the frame log.

A run with the page writes every fix it received to location.jsonl and every route it fetched to
route.jsonl, and its timing.jsonl says when each frame reached the laptop. Those three are enough to
give each frame the route goal the live run gave it, without the network: a frame sees the fixes the
laptop had received by its arrival, and the newest route fetched by then.

A frame with no arrival in the timing log gets no route goal. Its own timestamp is on another clock,
the glasses' or the phone's, and pairing a fix on it would pair the wrong fix.
"""

# Standard library imports
import json
from collections.abc import Sequence
from pathlib import Path

# Local package imports
from nav.planner.goal import RouteGoal
from nav.route import route_goal_for
from nav.route.bearing_filter import CircularExponentialFilter
from nav.route.course import Fix, FixHistory, check_fix
from nav.route.geodesy import LatLon
from nav.route.route import Route, RouteConfig, RouteFollower
from nav.route.service import MalformedRouteResponse, RouteFetchOutcome, parse_walking_response
from nav.runtime.route_goal import LOCATION_LOG_FILENAME, ROUTE_LOG_FILENAME
from nav.runtime.timing import TIMING_FILENAME, read_timing_log


class RouteSidecarRefused(ValueError):
    """A sidecar line the runtime could not have written. The message names the file and the line."""


def read_location_log(path: Path) -> list[Fix]:
    """
    Every fix a run received, in the order it received them.

    :raises RouteSidecarRefused: On a line that isn't a fix.
    """
    fixes = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            fix = Fix(
                LatLon(float(raw["latitude"]), float(raw["longitude"])),
                float(raw["accuracy_meters"]),
                None if raw["heading_degrees"] is None else float(raw["heading_degrees"]),
                None if raw["speed_mps"] is None else float(raw["speed_mps"]),
                int(raw["timestamp_ms"]),
                float(raw["received_seconds"]),
            )
            check_fix(fix)
        except (ValueError, KeyError, TypeError) as broken:
            raise RouteSidecarRefused(f"{path} line {number}: not a fix ({type(broken).__name__}: {broken})") from broken
        fixes.append(fix)
    return fixes


def read_route_log(path: Path) -> list[tuple[float, Route]]:
    """
    Every route a run fetched, with the laptop time it arrived. Fetches that failed are left out.

    :raises RouteSidecarRefused: On a line that isn't a fetch, or an OK fetch whose route doesn't parse.
    """
    routes = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            outcome = RouteFetchOutcome(raw["outcome"])
            fetched_seconds = float(raw["fetched_seconds"])
            if outcome is not RouteFetchOutcome.OK:
                continue
            route = parse_walking_response(json.dumps(raw["route"]))
        except (ValueError, KeyError, TypeError, MalformedRouteResponse) as broken:
            raise RouteSidecarRefused(f"{path} line {number}: not a fetch ({type(broken).__name__}: {broken})") from broken
        if route is None:
            raise RouteSidecarRefused(f"{path} line {number}: an OK fetch with no route")
        routes.append((fetched_seconds, route))
    return sorted(routes, key=lambda pair: pair[0])


def recorded_route_goals(
    log_dir: Path,
    timestamps: Sequence[float],
    config: RouteConfig,
    goal_distance_meters: float,
) -> list[RouteGoal | None]:
    """
    The route goal each frame had, in the order given.

    :param log_dir: The recording, with its sidecars and timing log.
    :param timestamps: The frames' own timestamps, in the order they were planned.
    :param config: The route layer's constants.
    :param goal_distance_meters: The planner's goal distance, for the tolerance's widening.
    :return: One per frame. All None for a recording with no fixes or no routes.
    :rtype: list[RouteGoal | None]
    """
    log_dir = Path(log_dir)
    location_path, route_path, timing_path = log_dir / LOCATION_LOG_FILENAME, log_dir / ROUTE_LOG_FILENAME, log_dir / TIMING_FILENAME
    if not (location_path.is_file() and route_path.is_file() and timing_path.is_file()):
        return [None] * len(timestamps)
    fixes = sorted(read_location_log(location_path), key=lambda fix: fix.received_seconds)
    routes = read_route_log(route_path)
    arrivals = {record.timestamp_seconds: record.arrival_seconds for record in read_timing_log(timing_path)}

    history = FixHistory(config)
    bearing_filter = CircularExponentialFilter(config.bearing_time_constant_seconds)
    follower: RouteFollower | None = None
    route_in_use: Route | None = None
    next_fix = 0
    goals: list[RouteGoal | None] = []
    for timestamp in timestamps:
        arrival = arrivals.get(timestamp)
        if arrival is None:
            goals.append(None)
            continue
        # Only what the laptop had by then: the fixes received, and the newest route fetched.
        while next_fix < len(fixes) and fixes[next_fix].received_seconds <= arrival:
            history.add(fixes[next_fix])
            next_fix += 1
        fetched = [route for fetched_seconds, route in routes if fetched_seconds <= arrival]
        if fetched and fetched[-1] is not route_in_use:
            route_in_use = fetched[-1]
            follower = RouteFollower(route_in_use, config)
            bearing_filter.reset()
        fix = history.nearest(arrival, config.fix_max_age_seconds)
        goal, _ = route_goal_for(fix, follower, bearing_filter, config, goal_distance_meters)
        goals.append(goal)
    return goals
