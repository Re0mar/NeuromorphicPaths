"""
A walking route, where the walker is on it, and the point a look-ahead further along that the goal aims at.

The route is a line, not a point. A straight line to the destination points through buildings, and the
planner only sees about 5 m. So the goal aims at the route a look-ahead past the walker's projection
onto it, far enough to be past a phone fix's 5 to 10 m of error in a street.
"""

# Standard library imports
import math
from dataclasses import dataclass
from enum import Enum

# Local package imports
from nav.route.geodesy import LatLon, bearing_degrees, local_frame


class RouteState(Enum):
    """What the route layer can say about this frame. Values are the spellings the page shows."""

    NO_ROUTE = "no_route"  # No destination, or no route fetched for it yet.
    NO_FIX = "no_fix"  # No fix recent enough to pair with this frame.
    NO_COURSE = "no_course"  # A fix, but standing or without a heading, so no direction to turn the route into.
    FOLLOWING = "following"
    OFF_ROUTE = "off_route"  # Too far from the route for long enough. The old route is still followed.
    REROUTING = "rerouting"  # A new route from the fix has been asked for. The old one is followed meanwhile.
    SERVICE_FAILED = "service_failed"  # The route service failed by name. No route goal.


@dataclass(frozen=True)
class RouteConfig:
    """The route layer's constants. All ours, and the outdoor walk is where they get moved."""

    # How far along the route past the walker the goal aims. Past a street fix's 5 to 10 m of error.
    lookahead_meters: float = 20.0
    # How far from the route, for how long, before a reroute is asked for.
    off_route_meters: float = 25.0
    off_route_seconds: float = 3.0
    # The fewest seconds between two reroutes, so a walker standing off route doesn't ask every frame.
    reroute_min_interval_seconds: float = 20.0
    # How old a fix may be, on the laptop's clock, and still pair with a frame.
    fix_max_age_seconds: float = 3.0
    # The slowest walk at which the phone's heading is a course. Standing still, it points anywhere.
    course_min_speed_mps: float = 0.5
    # How fast the smoothed relative bearing follows a new one.
    bearing_time_constant_seconds: float = 2.0
    # The route goal's tolerance with an exact fix. A poor fix widens it.
    route_goal_tolerance_meters: float = 1.0
    # A fix implying a faster walk than this from the previous one is a GPS jump, not a walker.
    max_plausible_speed_mps: float = 5.0
    # A heading turning faster than this is a GPS glitch, not a walker.
    max_heading_rate_degrees_per_second: float = 180.0


def check_route_config(config: RouteConfig) -> None:
    """
    Refuse a route configuration that cannot work.

    The look-ahead and the off-route distance have no relation imposed. The design needs none, and the
    defaults, 20 and 25, would refuse themselves under the obvious one.

    :raises ValueError: Naming the field.
    """
    above_zero = (
        "lookahead_meters",
        "off_route_meters",
        "route_goal_tolerance_meters",
        "course_min_speed_mps",
        "max_plausible_speed_mps",
        "max_heading_rate_degrees_per_second",
    )
    zero_or_more = ("off_route_seconds", "reroute_min_interval_seconds", "fix_max_age_seconds", "bearing_time_constant_seconds")
    for name in above_zero:
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be above zero, got {value}")
    for name in zero_or_more:
        value = getattr(config, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be zero or more, got {value}")


class Route:
    """A walking route as lat/lon points, with how far along it each point is."""

    def __init__(self, points: tuple[LatLon, ...]) -> None:
        if len(points) < 2:
            raise ValueError(f"a route needs at least two points, got {len(points)}")
        self._to_meters = local_frame(points[0])
        self._points = points
        self._xy = [self._to_meters(point) for point in points]
        cumulative = [0.0]
        for (east_start, north_start), (east_end, north_end) in zip(self._xy, self._xy[1:]):
            cumulative.append(cumulative[-1] + math.hypot(east_end - east_start, north_end - north_start))
        for index, (before, after) in enumerate(zip(cumulative, cumulative[1:])):
            if not after > before:
                raise ValueError(f"route distances must strictly increase, point {index + 1} is not past point {index}")
        self._cumulative = cumulative

    @property
    def points(self) -> tuple[LatLon, ...]:
        return self._points

    @property
    def cumulative_meters(self) -> tuple[float, ...]:
        return tuple(self._cumulative)

    @property
    def points_meters(self) -> tuple[tuple[float, float], ...]:
        """Each point's (east, north) in the route's own flat frame, centered on its first point."""
        return tuple(self._xy)

    @property
    def length_meters(self) -> float:
        return self._cumulative[-1]

    @property
    def destination(self) -> LatLon:
        return self._points[-1]

    def to_meters(self, point: LatLon) -> tuple[float, float]:
        """A point's (east, north) in the route's own flat frame, centered on its first point."""
        return self._to_meters(point)

    def point_at(self, along_meters: float) -> LatLon:
        """The point this far along, clamped to the route's two ends."""
        along = min(max(along_meters, 0.0), self.length_meters)
        for index in range(len(self._cumulative) - 1):
            start, end = self._cumulative[index], self._cumulative[index + 1]
            if along <= end:
                share = (along - start) / (end - start)
                first, second = self._points[index], self._points[index + 1]
                # Linear in degrees over one segment, which over a segment of a walking route is
                # the same as linear in meters to well under a centimeter.
                return LatLon(
                    first.latitude_degrees + share * (second.latitude_degrees - first.latitude_degrees),
                    first.longitude_degrees + share * (second.longitude_degrees - first.longitude_degrees),
                )
        return self._points[-1]


class RouteFollower:
    """Where a fix sits on one route, and the off-route timer. Nothing else is remembered."""

    def __init__(self, route: Route, config: RouteConfig) -> None:
        check_route_config(config)
        self._route = route
        self._config = config
        self._off_since: float | None = None
        self._last_timestamp: float | None = None

    @property
    def route(self) -> Route:
        return self._route

    def project(self, point: LatLon) -> tuple[float, float, int]:
        """
        The nearest point of the route to a fix.

        :return: (meters along the route, meters off it, the segment it lies on, counted from 0).
        :rtype: tuple[float, float, int]
        """
        east, north = self._route.to_meters(point)
        xy = self._route.points_meters
        cumulative = self._route.cumulative_meters
        best = (math.inf, 0.0, 0)
        for segment in range(len(xy) - 1):
            (east_start, north_start), (east_end, north_end) = xy[segment], xy[segment + 1]
            along_east, along_north = east_end - east_start, north_end - north_start
            length = cumulative[segment + 1] - cumulative[segment]
            share = ((east - east_start) * along_east + (north - north_start) * along_north) / length**2
            share = min(max(share, 0.0), 1.0)
            off = math.hypot(east - (east_start + share * along_east), north - (north_start + share * along_north))
            if off < best[0]:
                best = (off, cumulative[segment] + share * length, segment)
        off, along, segment = best
        return along, off, segment

    def lookahead(self, along_meters: float) -> LatLon:
        """The point the goal aims at: the look-ahead past this far along, or the destination."""
        return self._route.point_at(along_meters + self._config.lookahead_meters)

    def bearing_to_lookahead(self, point: LatLon) -> float:
        """The bearing from a fix to its look-ahead point, degrees from north."""
        along, _, _ = self.project(point)
        return bearing_degrees(point, self.lookahead(along))

    def remaining_meters(self, along_meters: float) -> float:
        return max(self._route.length_meters - along_meters, 0.0)

    def off_route_for(self, point: LatLon, timestamp_seconds: float) -> bool:
        """
        Whether the walker has been farther from the route than the limit for at least the duration.

        Timed on the timestamps passed in, never the wall clock, so a replay agrees with the live run.
        A clock that went backward starts the timer over.
        """
        if self._last_timestamp is not None and timestamp_seconds < self._last_timestamp:
            self._off_since = None
        self._last_timestamp = timestamp_seconds
        _, off, _ = self.project(point)
        if off <= self._config.off_route_meters:
            self._off_since = None
            return False
        if self._off_since is None:
            self._off_since = timestamp_seconds
        return timestamp_seconds - self._off_since >= self._config.off_route_seconds
