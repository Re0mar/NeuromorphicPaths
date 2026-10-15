"""The route layer's geodesy, against values known from outside the code, never against a round trip through it."""

# Standard library imports
import math
import re

# Third party imports
import pytest

# Local package imports
from conftest import NAV_ROOT
from nav.route.geodesy import LatLon, bearing_degrees, distance_meters, local_frame, wrap_degrees

CAMPUS = LatLon(51.8225, 5.8670)


def test_due_north_is_bearing_zero() -> None:
    assert bearing_degrees(CAMPUS, LatLon(51.8325, 5.8670)) == pytest.approx(0.0, abs=1e-9)


def test_due_east_is_bearing_ninety() -> None:
    # On the equator a parallel is a great circle, so due east is exactly 90 all the way along.
    assert bearing_degrees(LatLon(0.0, 10.0), LatLon(0.0, 10.01)) == pytest.approx(90.0, abs=1e-9)


def test_due_south_and_west_complete_the_compass() -> None:
    assert bearing_degrees(CAMPUS, LatLon(51.8125, 5.8670)) == pytest.approx(180.0, abs=1e-9)
    assert bearing_degrees(LatLon(0.0, 10.0), LatLon(0.0, 9.99)) == pytest.approx(270.0, abs=1e-9)


def test_one_degree_of_latitude_is_111_km() -> None:
    # A degree along a meridian runs from 110.57 km at the equator to 111.69 km at a pole.
    assert 110_570.0 < distance_meters(LatLon(51.0, 5.0), LatLon(52.0, 5.0)) < 111_700.0


def test_the_local_frame_matches_haversine_per_point() -> None:
    to_meters = local_frame(CAMPUS)
    for index in range(10):
        point = LatLon(CAMPUS.latitude_degrees + 0.0007 * (index - 5), CAMPUS.longitude_degrees + 0.0011 * (index - 3))
        east, north = to_meters(point)
        assert math.hypot(east, north) == pytest.approx(distance_meters(CAMPUS, point), rel=1e-3)


def test_wrap_puts_190_at_minus_170() -> None:
    assert wrap_degrees(190.0) == pytest.approx(-170.0)
    assert wrap_degrees(-180.0) == pytest.approx(180.0)
    assert wrap_degrees(540.0) == pytest.approx(180.0)
    assert wrap_degrees(-370.0) == pytest.approx(-10.0)


@pytest.mark.parametrize("latitude", [91.0, -90.5, float("nan")])
def test_a_latitude_out_of_range_is_refused(latitude: float) -> None:
    with pytest.raises(ValueError, match="latitude"):
        LatLon(latitude, 5.0)


@pytest.mark.parametrize("longitude", [181.0, -180.5, float("inf")])
def test_a_longitude_out_of_range_is_refused(longitude: float) -> None:
    with pytest.raises(ValueError, match="longitude"):
        LatLon(51.0, longitude)


def test_every_angle_conversion_is_in_geodesy() -> None:
    # One file holds the conversions, so a missing one can't hide in a second copy with a different constant.
    pattern = re.compile(r"\bradians\(|\bdegrees\(|pi\s*/\s*180|180\s*/\s*(math\.)?pi|111[_,]?\d{3}")
    others = [path for path in sorted((NAV_ROOT / "route").glob("*.py")) if path.name != "geodesy.py"]
    # The package's other six files, so an empty glob can't pass for a clean one.
    assert len(others) >= 6, [path.name for path in others]
    offenders = []
    for path in others:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert offenders == [], "\n".join(offenders)
