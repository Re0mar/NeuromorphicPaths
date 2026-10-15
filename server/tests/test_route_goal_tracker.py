"""
The route goal tracker over a run: fixes and a destination in, a route fetched, one goal per frame out, and the sidecars.

Routes come from the file service over the recorded fixture, so nothing here reaches the network.
"""

# Standard library imports
import json
import math
from pathlib import Path

# Third party imports
import pytest

# Local package imports
from nav.route.course import Fix
from nav.route.geodesy import LatLon
from nav.route.route import RouteConfig, RouteState
from nav.route.service import FileRouteService, RouteFetch, RouteFetchOutcome, parse_walking_response
from nav.runtime.route_goal import DESTINATION_FILENAME, LOCATION_LOG_FILENAME, ROUTE_LOG_FILENAME, RouteGoalTracker

FIXTURE = Path(__file__).parent / "fixtures" / "ors_walking_response.json"
START = LatLon(51.8215, 5.8650)
DESTINATION = LatLon(51.8245, 5.8685)
CONFIG = RouteConfig()
GOAL_DISTANCE_METERS = 4.0


class Clock:
    """The laptop's clock, moved by hand."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class CountingService:
    """A service that counts its requests and answers as told."""

    def __init__(self, answer: RouteFetch | BaseException | None = None) -> None:
        self.calls = 0
        self._answer = answer
        self._file = FileRouteService(FIXTURE)

    def walking_route(self, origin: LatLon, destination: LatLon) -> RouteFetch:
        self.calls += 1
        if isinstance(self._answer, BaseException):
            raise self._answer
        return self._file.walking_route(origin, destination) if self._answer is None else self._answer


def fix_at(point: LatLon, received: float, heading: float = 0.0, speed: float = 1.2, accuracy: float = 5.0) -> Fix:
    return Fix(point, accuracy, heading, speed, int(received * 1000) + 1_760_000_000_000, received)


def east_of(point: LatLon, meters: float) -> LatLon:
    return LatLon(point.latitude_degrees, point.longitude_degrees + meters / (111_195.0 * math.cos(math.radians(point.latitude_degrees))))


def settled(tracker: RouteGoalTracker) -> RouteGoalTracker:
    """Wait for any request in flight to finish. close() only joins the fetch thread, so the tracker carries on."""
    tracker.close()
    return tracker


def test_a_destination_and_a_fix_give_a_goal_on_the_route() -> None:
    tracker = RouteGoalTracker(CountingService(), CONFIG, GOAL_DISTANCE_METERS)
    tracker.on_fix(fix_at(START, 100.0))
    tracker.on_destination(DESTINATION)
    frame = settled(tracker).frame(100.2)
    assert frame.state is RouteState.FOLLOWING
    assert frame.goal is not None
    assert frame.along_meters == pytest.approx(0.0, abs=0.5)
    assert frame.remaining_meters == pytest.approx(469.7, abs=0.5)


def test_a_destination_before_any_fix_is_held_and_fetched_on_the_first_fix() -> None:
    service = CountingService()
    tracker = RouteGoalTracker(service, CONFIG, GOAL_DISTANCE_METERS)
    tracker.on_destination(DESTINATION)
    frame = tracker.frame(100.0)
    assert frame.state is RouteState.NO_ROUTE and "waiting for a fix" in frame.reason
    assert service.calls == 0
    tracker.on_fix(fix_at(START, 100.5))
    assert settled(tracker).frame(100.6).state is RouteState.FOLLOWING
    assert service.calls == 1


def test_no_destination_is_no_route() -> None:
    tracker = RouteGoalTracker(CountingService(), CONFIG, GOAL_DISTANCE_METERS)
    tracker.on_fix(fix_at(START, 100.0))
    frame = tracker.frame(100.1)
    assert (frame.goal, frame.state) == (None, RouteState.NO_ROUTE)


def test_a_phone_that_stops_sending_gives_no_fix_after_max_age() -> None:
    tracker = RouteGoalTracker(CountingService(), CONFIG, GOAL_DISTANCE_METERS)
    tracker.on_fix(fix_at(START, 100.0))
    tracker.on_destination(DESTINATION)
    settled(tracker)
    assert tracker.frame(101.0).state is RouteState.FOLLOWING
    stale = tracker.frame(103.5)
    assert (stale.goal, stale.state) == (None, RouteState.NO_FIX)
    # Told apart from a phone that never sent by the last fix's age.
    assert stale.fix_age_seconds == pytest.approx(3.5)


def test_a_failed_service_gives_its_tier_and_reason_and_no_goal() -> None:
    service = CountingService(RouteFetch(RouteFetchOutcome.RATE_LIMITED, None, "the route service's rate limit was reached (HTTP 429)"))
    tracker = RouteGoalTracker(service, CONFIG, GOAL_DISTANCE_METERS)
    tracker.on_fix(fix_at(START, 100.0))
    tracker.on_destination(DESTINATION)
    frame = settled(tracker).frame(100.1)
    assert (frame.goal, frame.state) == (None, RouteState.SERVICE_FAILED)
    assert "rate limit" in frame.reason


def test_a_refused_key_is_never_asked_again() -> None:
    clock = Clock()
    service = CountingService(RouteFetch(RouteFetchOutcome.KEY_REFUSED, None, "the route service refused the key (HTTP 401)"))
    tracker = RouteGoalTracker(service, CONFIG, GOAL_DISTANCE_METERS, clock=clock)
    tracker.on_fix(fix_at(START, 100.0))
    tracker.on_destination(DESTINATION)
    settled(tracker).frame(100.1)
    clock.now += CONFIG.reroute_min_interval_seconds + 1.0
    tracker.on_fix(fix_at(START, 101.0))
    settled(tracker).frame(101.1)
    assert service.calls == 1


def test_a_rate_limit_is_asked_again_after_the_interval_and_not_before() -> None:
    clock = Clock()
    service = CountingService(RouteFetch(RouteFetchOutcome.RATE_LIMITED, None, "rate limited"))
    tracker = RouteGoalTracker(service, CONFIG, GOAL_DISTANCE_METERS, clock=clock)
    tracker.on_fix(fix_at(START, 100.0))
    tracker.on_destination(DESTINATION)
    settled(tracker).frame(100.1)
    clock.now += CONFIG.reroute_min_interval_seconds - 1.0
    settled(tracker).frame(100.2)
    assert service.calls == 1
    clock.now += 2.0
    settled(tracker).frame(100.3)
    assert service.calls == 2


def test_an_off_route_walker_asks_for_a_new_route_and_says_rerouting() -> None:
    clock = Clock()
    service = CountingService()
    tracker = RouteGoalTracker(service, CONFIG, GOAL_DISTANCE_METERS, clock=clock)
    tracker.on_fix(fix_at(START, 100.0))
    tracker.on_destination(DESTINATION)
    settled(tracker).frame(100.1)
    clock.now += CONFIG.reroute_min_interval_seconds + 1.0
    away = east_of(START, 40.0)
    states = []
    for second in range(5):
        tracker.on_fix(fix_at(away, 110.0 + second, heading=0.0))
        states.append(tracker.frame(110.1 + second).state)
    assert RouteState.REROUTING in states
    assert service.calls == 2


def test_a_parser_bug_in_a_request_is_raised_by_frame() -> None:
    tracker = RouteGoalTracker(CountingService(KeyError("features")), CONFIG, GOAL_DISTANCE_METERS)
    tracker.on_fix(fix_at(START, 100.0))
    tracker.on_destination(DESTINATION)
    with pytest.raises(KeyError):
        settled(tracker).frame(100.1)


def test_the_sidecars_are_written_through_textio_with_lf(tmp_path: Path) -> None:
    clock = Clock(5000.0)
    tracker = RouteGoalTracker(CountingService(), CONFIG, GOAL_DISTANCE_METERS, log_dir=tmp_path, clock=clock)
    tracker.on_destination(DESTINATION)
    tracker.on_fix(fix_at(START, 100.0, accuracy=7.5))
    settled(tracker).frame(100.1)

    for name in (LOCATION_LOG_FILENAME, DESTINATION_FILENAME, ROUTE_LOG_FILENAME):
        assert b"\r\n" not in (tmp_path / name).read_bytes(), name
    location = json.loads((tmp_path / LOCATION_LOG_FILENAME).read_text(encoding="utf-8").splitlines()[0])
    # Both stamps, the phone's and the laptop's, so a replay pairs on the laptop's.
    assert location["received_seconds"] == 100.0 and location["timestamp_ms"] == 1_760_000_100_000
    assert location["accuracy_meters"] == 7.5
    destination = json.loads((tmp_path / DESTINATION_FILENAME).read_text(encoding="utf-8"))
    assert (destination["latitude"], destination["longitude"]) == (51.8245, 5.8685)
    fetched = json.loads((tmp_path / ROUTE_LOG_FILENAME).read_text(encoding="utf-8").splitlines()[0])
    assert fetched["outcome"] == "ok" and fetched["fetched_seconds"] == 5000.0
    # The logged route reads back through the service's own parser, longitude first.
    route = parse_walking_response(json.dumps(fetched["route"]))
    assert route is not None and route.points[0] == START


def test_a_bad_route_constant_stops_the_tracker_at_construction() -> None:
    with pytest.raises(ValueError, match="lookahead_meters"):
        RouteGoalTracker(CountingService(), RouteConfig(lookahead_meters=0.0), GOAL_DISTANCE_METERS)
