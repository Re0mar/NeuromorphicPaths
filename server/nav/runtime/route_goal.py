"""
The route goal over a run: fixes in, a destination in, a route fetched and followed, one goal per frame out.

The route package computes and asks the service. This module holds what a run keeps between frames,
decides when to ask, and writes what a replay needs beside the frame log:

- `location.jsonl`, every fix as received, with the phone's stamp and the laptop's
- `destination.json`, the latest destination set
- `route.jsonl`, every fetch with its laptop time, outcome, reason, and the route as GeoJSON when it got one

A run can reroute several times, so routes are a line each with the time they arrived, not one file.

Called from two threads. The web sink's loop hands in fixes and destinations, and the planning loop
asks for a frame's goal. One lock covers the state both touch.
"""

# Standard library imports
import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path

# Local package imports
from nav.clock import laptop_time_seconds
from nav.route import RouteFrame, route_goal_for
from nav.route.bearing_filter import CircularExponentialFilter
from nav.route.course import Fix, FixHistory, FixVerdict
from nav.route.geodesy import LatLon
from nav.route.route import Route, RouteConfig, RouteFollower, RouteState, check_route_config
from nav.route.service import RouteFetch, RouteFetcher, RouteFetchOutcome, RouteService
from nav.runtime.textio import append_text_lf, write_text_lf

log = logging.getLogger(__name__)

LOCATION_LOG_FILENAME = "location.jsonl"
DESTINATION_FILENAME = "destination.json"
ROUTE_LOG_FILENAME = "route.jsonl"
# Failures a retry can't mend. Asking again with the same key gets the same answer.
UNRECOVERABLE = frozenset({RouteFetchOutcome.NO_KEY, RouteFetchOutcome.KEY_REFUSED})


def route_as_geojson(route: Route) -> dict:
    """A route in the service's own response shape, [longitude, latitude], so the same parser reads it back."""
    coordinates = [[point.longitude_degrees, point.latitude_degrees] for point in route.points]
    return {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": coordinates}}]}


class RouteGoalTracker:
    """Everything the route goal remembers across a run. Owned by the run, which closes it."""

    def __init__(
        self,
        service: RouteService,
        config: RouteConfig,
        goal_distance_meters: float,
        log_dir: Path | None = None,
        clock: Callable[[], float] = laptop_time_seconds,
    ) -> None:
        """
        :param service: Where routes come from.
        :param config: The route layer's constants.
        :param goal_distance_meters: The planner's goal distance, for the route goal's tolerance.
        :param log_dir: The frame log's directory, when the run records, for the sidecars.
        :param clock: The laptop's clock. Replaced only by tests.
        """
        check_route_config(config)
        self._config = config
        self._goal_distance_meters = goal_distance_meters
        self._log_dir = log_dir
        self._clock = clock
        self._lock = threading.Lock()
        self._fetcher = RouteFetcher(service)
        self._history = FixHistory(config)
        self._filter = CircularExponentialFilter(config.bearing_time_constant_seconds)
        self._destination: LatLon | None = None
        self._follower: RouteFollower | None = None
        self._last_fetch: RouteFetch | None = None
        self._failed: RouteFetch | None = None
        self._requested_at: float | None = None

    @property
    def fix_counts(self) -> dict[FixVerdict, int]:
        """How many fixes were kept and dropped, by verdict."""
        with self._lock:
            return dict(self._history.counts)

    def on_fix(self, fix: Fix) -> None:
        """A fix from the page. Logged as received, then judged, and the first one fetches a held destination."""
        if self._log_dir is not None:
            line = {
                "latitude": fix.point.latitude_degrees,
                "longitude": fix.point.longitude_degrees,
                "accuracy_meters": fix.accuracy_meters,
                "heading_degrees": fix.heading_degrees,
                "speed_mps": fix.speed_mps,
                "timestamp_ms": fix.timestamp_ms,
                "received_seconds": fix.received_seconds,
            }
            append_text_lf(self._log_dir / LOCATION_LOG_FILENAME, json.dumps(line, allow_nan=False) + "\n")
        with self._lock:
            verdict = self._history.add(fix)
            if verdict is not FixVerdict.KEPT:
                log.info("fix dropped: %s, %d so far", verdict.value, self._history.counts[verdict])
            if self._destination is not None and self._follower is None and self._failed is None:
                self._ask(fix.point)

    def on_destination(self, destination: LatLon) -> None:
        """A destination from the page. Fetched from the latest fix, or held until the first fix arrives."""
        if self._log_dir is not None:
            body = {"latitude": destination.latitude_degrees, "longitude": destination.longitude_degrees, "set_seconds": self._clock()}
            write_text_lf(self._log_dir / DESTINATION_FILENAME, json.dumps(body, allow_nan=False) + "\n")
        with self._lock:
            self._destination = destination
            self._follower = None
            self._failed = None
            self._filter.reset()
            latest = self._history.latest
            if latest is not None:
                self._ask(latest.point)
            else:
                log.info("destination held until the first fix")

    def frame(self, arrival_seconds: float) -> RouteFrame:
        """
        The route goal for a frame that reached the laptop at this moment, on the laptop's clock.

        :raises Exception: Whatever a route request raised that the service could not name, since that
            is a defect and not a failed service.
        """
        fetch = self._fetcher.latest()
        with self._lock:
            if fetch is not None and fetch is not self._last_fetch:
                self._take(fetch)
            fix = self._history.nearest(arrival_seconds, self._config.fix_max_age_seconds)
            latest = self._history.latest
            fix_age = None if latest is None else arrival_seconds - latest.received_seconds
            accuracy = None if latest is None else latest.accuracy_meters
            if self._destination is None:
                return RouteFrame(None, RouteState.NO_ROUTE, None, None, None, fix_age, accuracy, "no destination set")
            if self._failed is not None:
                reason = self._failed.reason
                if self._maybe_retry(fix):
                    return RouteFrame(None, RouteState.NO_ROUTE, None, None, None, fix_age, accuracy, f"asking the route service again after: {reason}")
                return RouteFrame(None, RouteState.SERVICE_FAILED, None, None, None, fix_age, accuracy, reason)
            if self._follower is None:
                reason = "fetching a route" if self._fetcher.in_flight else "waiting for a fix to fetch a route from"
                return RouteFrame(None, RouteState.NO_ROUTE, None, None, None, fix_age, accuracy, reason)
            goal, state = route_goal_for(fix, self._follower, self._filter, self._config, self._goal_distance_meters)
            along = remaining = bearing = None
            if fix is not None:
                along, _, _ = self._follower.project(fix.point)
                remaining = self._follower.remaining_meters(along)
                bearing = self._follower.bearing_to_lookahead(fix.point)
            if state is RouteState.OFF_ROUTE and fix is not None:
                if self._fetcher.in_flight or self._ask_again(fix.point):
                    state = RouteState.REROUTING
            elif self._fetcher.in_flight and state is RouteState.FOLLOWING:
                state = RouteState.REROUTING
            return RouteFrame(goal, state, along, remaining, bearing, fix_age, accuracy, "")

    def close(self) -> None:
        self._fetcher.close()

    def _ask(self, origin: LatLon) -> bool:
        """Ask for a route from here to the destination. Called holding the lock."""
        if self._destination is None:
            return False
        started = self._fetcher.request(origin, self._destination)
        if started:
            self._requested_at = self._clock()
        return started

    def _ask_again(self, origin: LatLon) -> bool:
        """Reroute from the fix, at most once per minimum interval. Called holding the lock."""
        now = self._clock()
        if self._requested_at is not None and now - self._requested_at < self._config.reroute_min_interval_seconds:
            return False
        log.info("off route, asking for a new route from the fix")
        return self._ask(origin)

    def _maybe_retry(self, fix: Fix | None) -> bool:
        """
        After a failure a retry might mend, ask again once the minimum interval has passed. Called holding the lock.

        :return: Whether a request started. The failure is cleared when one did.
        """
        if self._failed is None or self._failed.outcome in UNRECOVERABLE or fix is None or self._fetcher.in_flight:
            return False
        if not self._ask_again(fix.point):
            return False
        self._failed = None
        return True

    def _take(self, fetch: RouteFetch) -> None:
        """Act on a finished request. Called holding the lock."""
        self._last_fetch = fetch
        if self._log_dir is not None:
            line = {
                "fetched_seconds": self._clock(),
                "outcome": fetch.outcome.value,
                "reason": fetch.reason,
                "route": None if fetch.route is None else route_as_geojson(fetch.route),
            }
            append_text_lf(self._log_dir / ROUTE_LOG_FILENAME, json.dumps(line, allow_nan=False) + "\n")
        if fetch.outcome is RouteFetchOutcome.OK and fetch.route is not None:
            self._follower = RouteFollower(fetch.route, self._config)
            self._failed = None
            self._filter.reset()
            log.info("route: %s", fetch.reason)
            return
        # Logged once here, as the fetch is taken, and shown on every frame until it is mended.
        log.warning("route service failed: %s, %s", fetch.outcome.value, fetch.reason)
        self._failed = fetch
