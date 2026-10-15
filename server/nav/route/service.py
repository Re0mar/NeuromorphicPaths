"""
Walking routes from a map service, and a file that stands in for one.

OpenRouteService's foot-walking profile is asked over the standard library's urllib. One call at the
start and one per reroute don't earn a dependency. Every way a call can fail is its own outcome, so a
refused key, a rate limit and "no route exists" never read alike, and none of them reads as an empty
route.

The key goes into the client's constructor, from there into the request header, and nowhere else. It
never appears in a reason, a log line or a repr.
"""

# Standard library imports
import json
import logging
import socket
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

# Local package imports
from nav.route.geodesy import LatLon
from nav.route.route import Route

log = logging.getLogger(__name__)

ORS_WALKING_URL = "https://api.openrouteservice.org/v2/directions/foot-walking/geojson"
# Short, so a dead network costs one planner-independent thread a few seconds and nothing else.
DEFAULT_TIMEOUT_SECONDS = 5.0


class RouteFetchOutcome(Enum):
    """How one route request ended. Values are the spellings the page and the logs show."""

    OK = "ok"
    NO_KEY = "no_key"  # No key was configured, so nothing was asked.
    KEY_REFUSED = "key_refused"  # 401 or 403.
    RATE_LIMITED = "rate_limited"  # 429.
    NO_ROUTE = "no_route"  # The service answered and found no route, as a 404 or as an empty result.
    SERVER_ERROR = "server_error"  # 5xx: the service is up but failing.
    TIMEOUT = "timeout"
    OFFLINE = "offline"  # No connection to the service at all.
    MALFORMED = "malformed"  # An answer that isn't a route in the shape the service documents.


@dataclass(frozen=True)
class RouteFetch:
    """One request's result: the outcome, the route when it is OK, and a reason without the key."""

    outcome: RouteFetchOutcome
    route: Route | None
    reason: str


class RouteService(Protocol):
    def walking_route(self, origin: LatLon, destination: LatLon) -> RouteFetch:
        """A walking route from one point to another. Never raises for a failure it can name."""
        ...


class MalformedRouteResponse(ValueError):
    """The service's answer is not a route in its documented GeoJSON shape."""


def parse_walking_response(body: str) -> Route | None:
    """
    A route from OpenRouteService's GeoJSON directions response.

    The coordinates are [longitude, latitude], and this is the one place they are swapped.

    :param body: The response body.
    :return: The route, or None when the answer is well formed and holds no route.
    :rtype: Route | None
    :raises MalformedRouteResponse: When the body is not that shape.
    """
    try:
        document = json.loads(body)
    except json.JSONDecodeError as decode_error:
        raise MalformedRouteResponse(f"not JSON: {decode_error.msg} at character {decode_error.pos}") from decode_error
    if not isinstance(document, dict) or not isinstance(document.get("features"), list):
        raise MalformedRouteResponse("no 'features' list")
    if not document["features"]:
        return None
    geometry = document["features"][0].get("geometry") if isinstance(document["features"][0], dict) else None
    if not isinstance(geometry, dict) or geometry.get("type") != "LineString" or not isinstance(geometry.get("coordinates"), list):
        raise MalformedRouteResponse("the first feature has no LineString geometry")
    points = []
    for index, pair in enumerate(geometry["coordinates"]):
        if not isinstance(pair, list) or len(pair) < 2 or not all(type(value) in (int, float) for value in pair[:2]):
            raise MalformedRouteResponse(f"coordinate {index} is not [longitude, latitude]")
        longitude, latitude = pair[0], pair[1]
        try:
            point = LatLon(float(latitude), float(longitude))
        except ValueError as range_error:
            raise MalformedRouteResponse(f"coordinate {index}: {range_error}") from range_error
        # A service repeats a vertex where two steps meet. The route needs every point to move on.
        if points and point == points[-1]:
            continue
        points.append(point)
    try:
        return Route(tuple(points))
    except ValueError as route_error:
        raise MalformedRouteResponse(str(route_error)) from route_error


class OpenRouteServiceClient:
    """OpenRouteService's foot-walking directions over urllib."""

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        opener: Callable[..., object] = urllib.request.urlopen,
    ) -> None:
        """
        :param api_key: The key, or "" when none is configured, which every request reports as no_key.
        :param timeout_seconds: Per request.
        :param opener: urllib.request.urlopen. Replaced only by tests.
        """
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._opener = opener

    def __repr__(self) -> str:
        return f"OpenRouteServiceClient(api_key={'<set>' if self._api_key else '<empty>'}, timeout_seconds={self._timeout_seconds})"

    def walking_route(self, origin: LatLon, destination: LatLon) -> RouteFetch:
        if not self._api_key:
            return RouteFetch(RouteFetchOutcome.NO_KEY, None, "no route service key is configured")
        body = json.dumps(
            {"coordinates": [[origin.longitude_degrees, origin.latitude_degrees], [destination.longitude_degrees, destination.latitude_degrees]]}
        ).encode("utf-8")
        request = urllib.request.Request(
            ORS_WALKING_URL,
            data=body,
            method="POST",
            headers={"Authorization": self._api_key, "Content-Type": "application/json", "Accept": "application/json, application/geo+json"},
        )
        try:
            with self._opener(request, timeout=self._timeout_seconds) as response:
                text = response.read().decode("utf-8")
        # HTTPError is a URLError, so it is caught first.
        except urllib.error.HTTPError as http_error:
            return _outcome_for_status(http_error.code)
        except (socket.timeout, TimeoutError) as timeout_error:
            return RouteFetch(RouteFetchOutcome.TIMEOUT, None, f"no answer within {self._timeout_seconds} s ({type(timeout_error).__name__})")
        except urllib.error.URLError as url_error:
            # urllib wraps a connect timeout in a URLError whose reason is the timeout.
            if isinstance(url_error.reason, (socket.timeout, TimeoutError)):
                return RouteFetch(RouteFetchOutcome.TIMEOUT, None, f"no answer within {self._timeout_seconds} s")
            return RouteFetch(RouteFetchOutcome.OFFLINE, None, f"could not reach the route service: {url_error.reason}")
        try:
            route = parse_walking_response(text)
        except MalformedRouteResponse as malformed:
            return RouteFetch(RouteFetchOutcome.MALFORMED, None, f"the route service's answer was not a route: {malformed}")
        if route is None:
            return RouteFetch(RouteFetchOutcome.NO_ROUTE, None, "the route service found no route")
        return RouteFetch(RouteFetchOutcome.OK, route, f"{route.length_meters:.0f} m, {len(route.points)} points")


def _outcome_for_status(status: int) -> RouteFetch:
    """A named outcome for an HTTP error status. A status with no meaning here is raised, not guessed at."""
    if status in (401, 403):
        return RouteFetch(RouteFetchOutcome.KEY_REFUSED, None, f"the route service refused the key (HTTP {status})")
    if status == 429:
        return RouteFetch(RouteFetchOutcome.RATE_LIMITED, None, "the route service's rate limit was reached (HTTP 429)")
    # OpenRouteService answers 404 when either point has no road or path near it, or no route joins them.
    if status == 404:
        return RouteFetch(RouteFetchOutcome.NO_ROUTE, None, "the route service found no route (HTTP 404)")
    if 500 <= status < 600:
        return RouteFetch(RouteFetchOutcome.SERVER_ERROR, None, f"the route service failed (HTTP {status})")
    raise RuntimeError(f"the route service answered HTTP {status}, which no outcome covers")


class FileRouteService:
    """Serves a stored service response through the same parser, or fails by name. For tests and replays."""

    def __init__(self, path: Path, fail_with: RouteFetchOutcome | None = None) -> None:
        if fail_with is RouteFetchOutcome.OK:
            raise ValueError("fail_with names a failure, and OK is not one")
        self._path = path
        self._fail_with = fail_with

    def walking_route(self, origin: LatLon, destination: LatLon) -> RouteFetch:
        if self._fail_with is not None:
            return RouteFetch(self._fail_with, None, f"stored service set to fail with {self._fail_with.value}")
        try:
            route = parse_walking_response(self._path.read_text(encoding="utf-8"))
        except MalformedRouteResponse as malformed:
            return RouteFetch(RouteFetchOutcome.MALFORMED, None, f"{self._path.name} is not a route: {malformed}")
        if route is None:
            return RouteFetch(RouteFetchOutcome.NO_ROUTE, None, f"{self._path.name} holds no route")
        return RouteFetch(RouteFetchOutcome.OK, route, f"{route.length_meters:.0f} m from {self._path.name}")


class RouteFetcher:
    """
    Runs route requests on a worker thread, one at a time, so a slow service never holds up planning.

    Owned by whoever builds it, which closes it. A failure the service names comes back as a RouteFetch.
    Anything else raised on the thread is a defect, and latest() raises it again rather than letting it
    read as a failed service.
    """

    def __init__(self, service: RouteService) -> None:
        self._service = service
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._latest: RouteFetch | None = None
        self._error: BaseException | None = None

    @property
    def in_flight(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def request(self, origin: LatLon, destination: LatLon) -> bool:
        """Start a request unless one is already running. Returns whether it started."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._thread = threading.Thread(target=self._fetch, args=(origin, destination), name="route-fetch", daemon=True)
            self._thread.start()
            return True

    def latest(self) -> RouteFetch | None:
        """The most recent finished request's result, or None before the first finishes."""
        with self._lock:
            if self._error is not None:
                error, self._error = self._error, None
                raise error
            return self._latest

    def close(self, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS + 1.0) -> None:
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout_seconds)

    def _fetch(self, origin: LatLon, destination: LatLon) -> None:
        try:
            fetch = self._service.walking_route(origin, destination)
        except Exception as unexpected_error:
            # Not a failure the service could name, so it's a defect. Kept for latest() to raise on the
            # planning thread, and logged here in case nobody calls latest() again.
            log.error("UNEXPECTED %s in a route request, may need a handler", type(unexpected_error).__name__, exc_info=True)
            with self._lock:
                self._error = unexpected_error
            return
        log.info("route request: %s, %s", fetch.outcome.value, fetch.reason)
        with self._lock:
            self._latest = fetch
