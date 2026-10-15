"""The route, the follower, its off-route timer, and route_goal_for, on routes laid out by hand."""

# Standard library imports
import math
from dataclasses import replace

# Third party imports
import pytest

# Local package imports
from nav.route import route_goal_for
from nav.route.bearing_filter import CircularExponentialFilter
from nav.route.course import Fix
from nav.route.geodesy import LatLon, distance_meters
from nav.route.route import Route, RouteConfig, RouteFollower, RouteState, check_route_config

CONFIG = RouteConfig()
START = LatLon(51.8200, 5.8600)
# About 1112 m due north of START, by haversine.
NORTH_END = LatLon(51.8300, 5.8600)
GOAL_DISTANCE_METERS = 4.0


def east_of(point: LatLon, meters: float) -> LatLon:
    """A point roughly this far east, on a sphere of the mean radius. Tests compare against haversine, not this."""
    return LatLon(point.latitude_degrees, point.longitude_degrees + meters / (111_195.0 * math.cos(math.radians(point.latitude_degrees))))


def north_route() -> RouteFollower:
    return RouteFollower(Route((START, NORTH_END)), CONFIG)


def fix_at(point: LatLon, heading: float | None = 0.0, speed: float | None = 1.4, accuracy: float = 0.0, received: float = 10.0) -> Fix:
    return Fix(point, accuracy, heading, speed, 1_760_000_000_000, received)


# *******************************************
# The route and the follower
# *******************************************


def test_projection_onto_a_straight_route() -> None:
    middle = LatLon(51.8250, 5.8600)
    beside = east_of(middle, 10.0)
    along, off, segment = north_route().project(beside)
    assert segment == 0
    assert along == pytest.approx(distance_meters(START, middle), rel=1e-3)
    assert off == pytest.approx(distance_meters(beside, middle), rel=1e-3)


def test_lookahead_crosses_a_segment_boundary() -> None:
    corner = LatLon(51.8209, 5.8600)  # about 100 m north
    east_end = east_of(corner, 100.0)
    follower = RouteFollower(Route((START, corner, east_end)), CONFIG)
    first_leg = follower.route.cumulative_meters[1]
    # 10 m short of the corner, the 20 m look-ahead lands 10 m past it on the east leg.
    point = follower.lookahead(first_leg - 10.0)
    assert distance_meters(corner, point) == pytest.approx(10.0, abs=0.05)
    assert point.longitude_degrees > corner.longitude_degrees


def test_lookahead_past_the_end_is_the_end() -> None:
    follower = north_route()
    assert follower.lookahead(follower.route.length_meters - 5.0) == NORTH_END


def test_remaining_meters_at_the_start_is_the_length() -> None:
    follower = north_route()
    assert follower.remaining_meters(0.0) == pytest.approx(distance_meters(START, NORTH_END), rel=1e-3)
    assert follower.remaining_meters(follower.route.length_meters + 3.0) == 0.0


def test_off_route_needs_the_distance_for_the_duration() -> None:
    middle = LatLon(51.8250, 5.8600)
    near = east_of(middle, 24.0)
    far = east_of(middle, 26.0)
    follower = north_route()
    assert not any(follower.off_route_for(near, 100.0 + second) for second in range(10))
    follower = north_route()
    assert not follower.off_route_for(far, 200.0)
    assert not follower.off_route_for(far, 202.9)
    assert follower.off_route_for(far, 203.1)


def test_coming_back_onto_the_route_restarts_the_off_route_timer() -> None:
    middle = LatLon(51.8250, 5.8600)
    follower = north_route()
    follower.off_route_for(east_of(middle, 26.0), 0.0)
    follower.off_route_for(middle, 2.0)
    assert not follower.off_route_for(east_of(middle, 26.0), 4.0)


def test_the_default_route_config_is_accepted() -> None:
    check_route_config(CONFIG)


def test_a_one_point_route_is_refused() -> None:
    with pytest.raises(ValueError, match="at least two points"):
        Route((START,))


def test_a_route_that_does_not_move_on_is_refused() -> None:
    with pytest.raises(ValueError, match="strictly increase"):
        Route((START, NORTH_END, NORTH_END))


@pytest.mark.parametrize(
    "name, value",
    [("lookahead_meters", 0.0), ("off_route_meters", -1.0), ("off_route_seconds", -0.1), ("course_min_speed_mps", 0.0), ("max_plausible_speed_mps", 0.0)],
)
def test_a_bad_route_constant_is_refused_by_name(name: str, value: float) -> None:
    with pytest.raises(ValueError, match=name):
        check_route_config(replace(CONFIG, **{name: value}))


# *******************************************
# route_goal_for
# *******************************************


def goal_for(fix: Fix | None, follower: RouteFollower | None = None, config: RouteConfig = CONFIG):
    follower = north_route() if follower is None else follower
    return route_goal_for(fix, follower, CircularExponentialFilter(config.bearing_time_constant_seconds), config, GOAL_DISTANCE_METERS)


def test_a_route_30_degrees_right_of_the_course_gives_that_relative_bearing() -> None:
    # The route runs north and the walker heads 330, so the route is 30 degrees to their right.
    goal, state = goal_for(fix_at(START, heading=330.0))
    assert state is RouteState.FOLLOWING
    assert goal is not None
    assert goal.relative_bearing_radians == pytest.approx(0.5236, abs=1e-4)


def test_no_route_gives_no_goal_and_the_no_route_tier() -> None:
    goal, state = route_goal_for(fix_at(START), None, CircularExponentialFilter(2.0), CONFIG, GOAL_DISTANCE_METERS)
    assert (goal, state) == (None, RouteState.NO_ROUTE)


def test_no_fix_gives_no_goal_and_the_no_fix_tier() -> None:
    assert goal_for(None) == (None, RouteState.NO_FIX)


def test_no_course_gives_no_goal_and_its_tier() -> None:
    assert goal_for(fix_at(START, speed=0.2)) == (None, RouteState.NO_COURSE)
    assert goal_for(fix_at(START, heading=None)) == (None, RouteState.NO_COURSE)


def test_a_poor_fix_widens_the_tolerance() -> None:
    goal, _ = goal_for(fix_at(START, accuracy=10.0))
    assert goal is not None
    # 10 m off, aimed 20 m ahead, is 10 x 4 / 20 = 2.0 m sideways at the planner's 4 m goal.
    assert goal.tolerance_meters == pytest.approx(2.0)
    # A look-ahead no default holds: 10 x 4 / 16 = 2.5 m.
    shorter = replace(CONFIG, lookahead_meters=16.0)
    goal, _ = goal_for(fix_at(START, accuracy=10.0), RouteFollower(Route((START, NORTH_END)), shorter), shorter)
    assert goal is not None and goal.tolerance_meters == pytest.approx(2.5)


def test_an_exact_fix_keeps_the_tolerance_floor() -> None:
    goal, _ = goal_for(fix_at(START, accuracy=0.0))
    assert goal is not None and goal.tolerance_meters == pytest.approx(CONFIG.route_goal_tolerance_meters)


def test_an_off_route_walker_keeps_the_old_routes_goal_with_its_tier() -> None:
    middle = LatLon(51.8250, 5.8600)
    follower = north_route()
    bearing_filter = CircularExponentialFilter(2.0)
    route_goal_for(fix_at(east_of(middle, 30.0), received=0.0), follower, bearing_filter, CONFIG, GOAL_DISTANCE_METERS)
    goal, state = route_goal_for(fix_at(east_of(middle, 30.0), received=3.5), follower, bearing_filter, CONFIG, GOAL_DISTANCE_METERS)
    assert state is RouteState.OFF_ROUTE
    assert goal is not None
