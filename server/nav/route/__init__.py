"""
The route layer: where the walker is on a walking route, and which way the route says to go.

It parses and computes, and writes no file. The runtime owns the fix history, the fetcher and the
files, and calls route_goal_for once a frame.
"""

# Standard library imports
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Local package imports
from nav.planner.goal import RouteGoal
from nav.route.bearing_filter import CircularExponentialFilter
from nav.route.course import Fix, course_degrees
from nav.route.geodesy import radians_of_degrees, wrap_degrees
from nav.route.route import RouteConfig, RouteFollower, RouteState


@dataclass(frozen=True)
class RouteFrame:
    """The route layer's answer for one frame: the goal, its tier, and what the page shows beside it. Unknowns are None."""

    goal: RouteGoal | None
    state: RouteState
    along_meters: float | None
    remaining_meters: float | None
    bearing_degrees: float | None
    fix_age_seconds: float | None
    accuracy_meters: float | None
    reason: str


@runtime_checkable
class RouteStateSink(Protocol):
    """A display that shows the route layer's state. The loop and the fan-out check for it, so a sink without it is left out."""

    def publish_route_state(self, frame: RouteFrame) -> None: ...


def route_goal_for(
    fix: Fix | None,
    follower: RouteFollower | None,
    bearing_filter: CircularExponentialFilter,
    config: RouteConfig,
    goal_distance_meters: float,
) -> tuple[RouteGoal | None, RouteState]:
    """
    The route goal for one frame, and the tier that says why when there is none.

    :param fix: The fix paired with this frame on the laptop's clock, or None when none was recent.
    :param follower: The route being followed, or None when there is no route.
    :param bearing_filter: Smooths the relative bearing across frames. Fed here.
    :param config: The route layer's constants.
    :param goal_distance_meters: How far ahead the planner puts its goal, for the tolerance's widening.
    :return: (the goal or None, the tier). An off-route walker still gets the old route's goal, with
        the off_route tier, while a reroute is awaited.
    :rtype: tuple[RouteGoal | None, RouteState]
    """
    if follower is None:
        return None, RouteState.NO_ROUTE
    if fix is None:
        return None, RouteState.NO_FIX
    course = course_degrees(fix, config)
    if course is None:
        return None, RouteState.NO_COURSE
    off_route = follower.off_route_for(fix.point, fix.received_seconds)
    relative = wrap_degrees(follower.bearing_to_lookahead(fix.point) - course)
    bearing_filter.update(relative, fix.received_seconds)
    smoothed = bearing_filter.value()
    if smoothed is None:
        raise RuntimeError("the bearing filter was fed and still holds no value")
    # A fix off by its accuracy, aimed at a point a look-ahead away, moves the goal sideways by about
    # the accuracy times goal distance over look-ahead. Never under the route goal's own floor.
    widened = fix.accuracy_meters * goal_distance_meters / config.lookahead_meters
    tolerance = max(config.route_goal_tolerance_meters, widened)
    goal = RouteGoal(radians_of_degrees(wrap_degrees(smoothed)), tolerance)
    return goal, RouteState.OFF_ROUTE if off_route else RouteState.FOLLOWING
