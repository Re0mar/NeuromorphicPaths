"""
Every conversion between degrees, radians and meters on the ground, and nowhere else in the route package.

The earth is a sphere here, at its mean radius. Routes on foot are a few hundred meters, and over that
a sphere and the real ellipsoid disagree by centimeters. One radius serves both the flat local frame
and the haversine distance, so the two agree with each other to well under a tenth of a percent.

The local frame is equirectangular: east is longitude scaled by the cosine of the origin's latitude,
north is latitude. Fine for a few hundred meters. A route spanning kilometers would want a real
projection, and that swap stays inside this file.
"""

# Standard library imports
import math
from collections.abc import Callable
from dataclasses import dataclass

# Mean earth radius, the IUGG value.
EARTH_RADIUS_METERS = 6_371_008.8
# One degree of latitude on that sphere, 111.195 km.
METERS_PER_DEGREE_LATITUDE = EARTH_RADIUS_METERS * math.pi / 180.0


@dataclass(frozen=True)
class LatLon:
    """A point on the earth in decimal degrees, latitude first."""

    latitude_degrees: float
    longitude_degrees: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.latitude_degrees) or not -90.0 <= self.latitude_degrees <= 90.0:
            raise ValueError(f"latitude must be from -90 to 90 degrees, got {self.latitude_degrees}")
        if not math.isfinite(self.longitude_degrees) or not -180.0 <= self.longitude_degrees <= 180.0:
            raise ValueError(f"longitude must be from -180 to 180 degrees, got {self.longitude_degrees}")


def local_frame(origin: LatLon) -> Callable[[LatLon], tuple[float, float]]:
    """
    A flat frame around one point, in meters.

    :param origin: The point at (0, 0).
    :return: A function giving any point's (east, north) meters from the origin.
    :rtype: Callable[[LatLon], tuple[float, float]]
    """
    meters_per_degree_east = METERS_PER_DEGREE_LATITUDE * math.cos(math.radians(origin.latitude_degrees))

    def to_meters(point: LatLon) -> tuple[float, float]:
        east = (point.longitude_degrees - origin.longitude_degrees) * meters_per_degree_east
        north = (point.latitude_degrees - origin.latitude_degrees) * METERS_PER_DEGREE_LATITUDE
        return east, north

    return to_meters


def distance_meters(start: LatLon, end: LatLon) -> float:
    """The great-circle distance between two points, by haversine."""
    latitude_start = math.radians(start.latitude_degrees)
    latitude_end = math.radians(end.latitude_degrees)
    latitude_change = latitude_end - latitude_start
    longitude_change = math.radians(end.longitude_degrees - start.longitude_degrees)
    half_chord = math.sin(latitude_change / 2.0) ** 2 + math.cos(latitude_start) * math.cos(latitude_end) * math.sin(longitude_change / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_METERS * math.asin(min(1.0, math.sqrt(half_chord)))


def bearing_degrees(start: LatLon, end: LatLon) -> float:
    """
    The direction from one point to another, clockwise from north.

    example: due east -> 90.0

    :return: Degrees in [0, 360).
    :rtype: float
    """
    latitude_start = math.radians(start.latitude_degrees)
    latitude_end = math.radians(end.latitude_degrees)
    longitude_change = math.radians(end.longitude_degrees - start.longitude_degrees)
    east = math.sin(longitude_change) * math.cos(latitude_end)
    north = math.cos(latitude_start) * math.sin(latitude_end) - math.sin(latitude_start) * math.cos(latitude_end) * math.cos(longitude_change)
    return bearing_of_vector(east, north)


def wrap_degrees(angle_degrees: float) -> float:
    """
    An angle brought into (-180, 180].

    example: 190 -> -170, -180 -> 180
    """
    wrapped = math.fmod(angle_degrees, 360.0)
    if wrapped > 180.0:
        wrapped -= 360.0
    elif wrapped <= -180.0:
        wrapped += 360.0
    return wrapped


def vector_of_bearing(bearing: float) -> tuple[float, float]:
    """The unit (east, north) vector pointing along a bearing in degrees from north."""
    angle = math.radians(bearing)
    return math.sin(angle), math.cos(angle)


def bearing_of_vector(east: float, north: float) -> float:
    """The bearing in degrees from north, in [0, 360), that an (east, north) vector points along."""
    bearing = math.degrees(math.atan2(east, north)) % 360.0
    # A bearing a hair under 0 lands on exactly 360.0 after the modulo.
    return 0.0 if bearing >= 360.0 else bearing


def radians_of_degrees(angle_degrees: float) -> float:
    """Degrees to radians, for a bearing handed to the planner, which works in radians."""
    return math.radians(angle_degrees)
