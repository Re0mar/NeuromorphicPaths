"""
The route service: the recorded response, the client's outcome for every named failure, the file fake and the fetcher.

The client is driven through a stubbed urlopen, so nothing here reaches the network.
"""

# Standard library imports
import io
import json
import socket
import threading
import urllib.error
from pathlib import Path

# Third party imports
import pytest

# Local package imports
from nav.route.geodesy import LatLon
from nav.route.service import (
    FileRouteService,
    OpenRouteServiceClient,
    RouteFetch,
    RouteFetcher,
    RouteFetchOutcome,
    parse_walking_response,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ors_walking_response.json"
ORIGIN = LatLon(51.8215, 5.8650)
DESTINATION = LatLon(51.8245, 5.8685)
# Made up for these tests. A real key never appears in a test.
MADE_UP_KEY = "test-key-5f3a9c"


class StubResponse(io.BytesIO):
    """What urlopen returns, as a context manager over a body."""

    def __enter__(self) -> "StubResponse":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def answering(body: str, seen: list | None = None):
    def opener(request, timeout: float) -> StubResponse:
        if seen is not None:
            seen.append(request)
        return StubResponse(body.encode("utf-8"))

    return opener


def raising(error: BaseException):
    def opener(request, timeout: float) -> StubResponse:
        raise error

    return opener


def http_error(status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.invalid", status, "stubbed", {}, None)


# *******************************************
# The recorded response
# *******************************************


def test_the_fixture_has_the_real_responses_keys_and_lon_lat_order() -> None:
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert document["type"] == "FeatureCollection"
    feature = document["features"][0]
    assert feature["geometry"]["type"] == "LineString"
    assert {"segments", "summary", "way_points"} <= set(feature["properties"])
    # Nijmegen is at about 5.86 east and 51.82 north, so a pair holding them the other way round is lat-lon.
    longitude, latitude = feature["geometry"]["coordinates"][0]
    assert 3.0 < longitude < 8.0 and 50.0 < latitude < 54.0


def test_a_recorded_response_parses_to_a_route_in_lat_lon() -> None:
    route = parse_walking_response(FIXTURE.read_text(encoding="utf-8"))
    assert route is not None
    assert route.points[0] == LatLon(51.8215, 5.8650)
    assert route.destination == LatLon(51.8245, 5.8685)
    # The response's own summary distance, from its geometry.
    assert route.length_meters == pytest.approx(469.7, abs=0.5)


def test_a_repeated_vertex_is_kept_once() -> None:
    body = json.dumps({"type": "FeatureCollection", "features": [{"geometry": {"type": "LineString", "coordinates": [[5.865, 51.8215], [5.865, 51.8215], [5.866, 51.8225]]}}]})
    route = parse_walking_response(body)
    assert route is not None and len(route.points) == 2


# *******************************************
# The client, one outcome per named failure
# *******************************************


def test_a_good_answer_is_ok_with_the_route() -> None:
    seen: list = []
    client = OpenRouteServiceClient(MADE_UP_KEY, opener=answering(FIXTURE.read_text(encoding="utf-8"), seen))
    fetch = client.walking_route(ORIGIN, DESTINATION)
    assert fetch.outcome is RouteFetchOutcome.OK
    assert fetch.route is not None
    # The key travels in the header, and the request's coordinates are longitude first.
    assert seen[0].get_header("Authorization") == MADE_UP_KEY
    assert json.loads(seen[0].data)["coordinates"] == [[5.8650, 51.8215], [5.8685, 51.8245]]


@pytest.mark.parametrize(
    "status, outcome",
    [
        (401, RouteFetchOutcome.KEY_REFUSED),
        (403, RouteFetchOutcome.KEY_REFUSED),
        (429, RouteFetchOutcome.RATE_LIMITED),
        (404, RouteFetchOutcome.NO_ROUTE),
        (500, RouteFetchOutcome.SERVER_ERROR),
        (503, RouteFetchOutcome.SERVER_ERROR),
    ],
)
def test_each_http_failure_is_its_own_outcome(status: int, outcome: RouteFetchOutcome) -> None:
    fetch = OpenRouteServiceClient(MADE_UP_KEY, opener=raising(http_error(status))).walking_route(ORIGIN, DESTINATION)
    assert fetch.outcome is outcome
    assert fetch.route is None


def test_a_status_no_outcome_covers_is_raised() -> None:
    with pytest.raises(RuntimeError, match="HTTP 400"):
        OpenRouteServiceClient(MADE_UP_KEY, opener=raising(http_error(400))).walking_route(ORIGIN, DESTINATION)


def test_an_answer_with_no_route_is_no_route() -> None:
    body = json.dumps({"type": "FeatureCollection", "features": []})
    assert OpenRouteServiceClient(MADE_UP_KEY, opener=answering(body)).walking_route(ORIGIN, DESTINATION).outcome is RouteFetchOutcome.NO_ROUTE


def test_a_timeout_is_timeout() -> None:
    for error in (socket.timeout("timed out"), urllib.error.URLError(socket.timeout("timed out"))):
        assert OpenRouteServiceClient(MADE_UP_KEY, opener=raising(error)).walking_route(ORIGIN, DESTINATION).outcome is RouteFetchOutcome.TIMEOUT


def test_no_connection_is_offline() -> None:
    error = urllib.error.URLError(OSError("Name or service not known"))
    assert OpenRouteServiceClient(MADE_UP_KEY, opener=raising(error)).walking_route(ORIGIN, DESTINATION).outcome is RouteFetchOutcome.OFFLINE


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        json.dumps({"type": "FeatureCollection"}),
        json.dumps({"features": [{"geometry": {"type": "Point", "coordinates": [5.86, 51.82]}}]}),
        json.dumps({"features": [{"geometry": {"type": "LineString", "coordinates": [[51.82, 95.0], [51.83, 95.0]]}}]}),
        json.dumps({"features": [{"geometry": {"type": "LineString", "coordinates": [[5.86, 51.82]]}}]}),
    ],
    ids=["not json", "no features", "not a line", "latitude out of range", "one point"],
)
def test_a_malformed_body_is_malformed(body: str) -> None:
    assert OpenRouteServiceClient(MADE_UP_KEY, opener=answering(body)).walking_route(ORIGIN, DESTINATION).outcome is RouteFetchOutcome.MALFORMED


def test_an_empty_key_is_no_key_before_any_request() -> None:
    calls: list = []
    fetch = OpenRouteServiceClient("", opener=answering("{}", calls)).walking_route(ORIGIN, DESTINATION)
    assert fetch.outcome is RouteFetchOutcome.NO_KEY
    assert calls == []


def test_the_key_never_appears_in_repr_or_reason() -> None:
    reasons = [repr(OpenRouteServiceClient(MADE_UP_KEY))]
    for error in (http_error(401), http_error(429), http_error(404), http_error(502), socket.timeout(), urllib.error.URLError("down")):
        reasons.append(OpenRouteServiceClient(MADE_UP_KEY, opener=raising(error)).walking_route(ORIGIN, DESTINATION).reason)
    reasons.append(OpenRouteServiceClient(MADE_UP_KEY, opener=answering("not json")).walking_route(ORIGIN, DESTINATION).reason)
    reasons.append(OpenRouteServiceClient(MADE_UP_KEY, opener=answering(FIXTURE.read_text(encoding="utf-8"))).walking_route(ORIGIN, DESTINATION).reason)
    assert all(MADE_UP_KEY not in reason for reason in reasons), reasons


# *******************************************
# The file service and the fetcher
# *******************************************


def test_the_file_service_serves_the_stored_route() -> None:
    fetch = FileRouteService(FIXTURE).walking_route(ORIGIN, DESTINATION)
    assert fetch.outcome is RouteFetchOutcome.OK and fetch.route is not None


@pytest.mark.parametrize("outcome", [outcome for outcome in RouteFetchOutcome if outcome is not RouteFetchOutcome.OK])
def test_the_file_service_fails_by_name(outcome: RouteFetchOutcome) -> None:
    fetch = FileRouteService(FIXTURE, fail_with=outcome).walking_route(ORIGIN, DESTINATION)
    assert fetch.outcome is outcome and fetch.route is None


def test_the_file_service_refuses_to_fail_with_ok() -> None:
    with pytest.raises(ValueError, match="OK"):
        FileRouteService(FIXTURE, fail_with=RouteFetchOutcome.OK)


class GatedService:
    """Answers only once released, so a test can hold a request in flight."""

    def __init__(self, answer: RouteFetch | BaseException) -> None:
        self.release = threading.Event()
        self.calls = 0
        self._answer = answer

    def walking_route(self, origin: LatLon, destination: LatLon) -> RouteFetch:
        self.calls += 1
        self.release.wait(5.0)
        if isinstance(self._answer, BaseException):
            raise self._answer
        return self._answer


def test_the_fetcher_runs_one_request_at_a_time() -> None:
    service = GatedService(RouteFetch(RouteFetchOutcome.NO_ROUTE, None, "stubbed"))
    fetcher = RouteFetcher(service)
    assert fetcher.request(ORIGIN, DESTINATION) is True
    assert fetcher.request(ORIGIN, DESTINATION) is False
    assert fetcher.latest() is None
    service.release.set()
    fetcher.close()
    assert service.calls == 1
    latest = fetcher.latest()
    assert latest is not None and latest.outcome is RouteFetchOutcome.NO_ROUTE
    assert fetcher.request(ORIGIN, DESTINATION) is True
    fetcher.close()


def test_a_parser_bug_is_raised_not_reported_as_failed() -> None:
    service = GatedService(KeyError("features"))
    service.release.set()
    fetcher = RouteFetcher(service)
    fetcher.request(ORIGIN, DESTINATION)
    fetcher.close()
    with pytest.raises(KeyError):
        fetcher.latest()
    # Raised once, then cleared, so the next frame isn't stopped by the same error again.
    assert fetcher.latest() is None
