"""The course from a fix, the fix history's drops, and pairing a frame with a fix on the laptop's clock."""

# Standard library imports
from dataclasses import replace

# Third party imports
import pytest

# Local package imports
from nav.route.course import Fix, FixHistory, FixVerdict, check_fix, course_degrees
from nav.route.geodesy import LatLon
from nav.route.route import RouteConfig

CONFIG = RouteConfig()
START = LatLon(51.8200, 5.8600)
# About 11 m north of START.
NEARBY = LatLon(51.8201, 5.8600)
# About 111 m north of START.
FAR = LatLon(51.8210, 5.8600)


def fix(point: LatLon = START, heading: float | None = 90.0, speed: float | None = 1.4, received: float = 100.0, phone_ms: int = 1_760_000_000_000) -> Fix:
    return Fix(point, 4.0, heading, speed, phone_ms, received)


def test_a_fix_at_speed_gives_its_heading() -> None:
    assert course_degrees(fix(heading=123.0, speed=1.0), CONFIG) == pytest.approx(123.0)


def test_a_slow_fix_gives_no_course() -> None:
    assert course_degrees(fix(speed=0.3), CONFIG) is None


def test_a_null_heading_gives_no_course() -> None:
    assert course_degrees(fix(heading=None), CONFIG) is None


def test_a_null_speed_gives_no_course() -> None:
    # Not a speed of 0, and not a course of 0, which would read as walking due north.
    assert course_degrees(fix(speed=None), CONFIG) is None


def test_the_minimum_speed_is_read_from_the_config() -> None:
    assert course_degrees(fix(speed=0.7), replace(CONFIG, course_min_speed_mps=0.8)) is None
    assert course_degrees(fix(speed=0.7), replace(CONFIG, course_min_speed_mps=0.6)) == pytest.approx(90.0)


def test_an_impossible_jump_is_dropped_and_counted() -> None:
    history = FixHistory(CONFIG)
    history.add(fix(START, received=100.0))
    # 111 m in a second.
    assert history.add(fix(FAR, received=101.0)) is FixVerdict.TOO_FAST
    assert history.counts[FixVerdict.TOO_FAST] == 1
    assert history.latest is not None and history.latest.point == START


def test_a_walking_step_is_kept() -> None:
    history = FixHistory(CONFIG)
    history.add(fix(START, received=100.0))
    assert history.add(fix(NEARBY, received=108.0)) is FixVerdict.KEPT


def test_a_bearing_jump_over_180_in_a_second_is_dropped() -> None:
    # A wrapped heading change can't exceed 180, so the rule is read as a turn rate: 170 degrees in
    # half a second is 340 a second.
    history = FixHistory(CONFIG)
    history.add(fix(START, heading=10.0, received=100.0))
    assert history.add(fix(START, heading=180.0, received=100.5)) is FixVerdict.HEADING_JUMP
    # 90 degrees over a second is a walker turning a corner.
    assert history.add(fix(START, heading=100.0, received=101.0)) is FixVerdict.KEPT


def test_an_equal_timestamp_step_is_its_own_branch() -> None:
    history = FixHistory(CONFIG)
    history.add(fix(START, received=100.0))
    assert history.add(fix(NEARBY, received=100.0)) is FixVerdict.SAME_TIME
    assert history.counts[FixVerdict.SAME_TIME] == 1


def test_a_laptop_clock_going_back_starts_the_history_over() -> None:
    history = FixHistory(CONFIG)
    history.add(fix(START, received=100.0))
    assert history.add(fix(FAR, received=50.0)) is FixVerdict.CLOCK_BACK
    assert history.nearest(100.0, 3.0) is None
    assert history.nearest(50.0, 3.0) is not None


def test_nearest_fix_within_max_age_on_the_laptop_clock() -> None:
    history = FixHistory(CONFIG)
    for second in range(5):
        history.add(fix(START, received=100.0 + second))
    nearest = history.nearest(102.4, 3.0)
    assert nearest is not None and nearest.received_seconds == pytest.approx(102.0)


def test_a_fix_is_paired_on_the_laptop_clock_not_the_phones() -> None:
    # The phone's clock runs 10 s ahead of the laptop's. Pairing on it would find nothing.
    history = FixHistory(CONFIG)
    history.add(fix(START, received=100.0, phone_ms=110_000))
    paired = history.nearest(100.2, 3.0)
    assert paired is not None and paired.timestamp_ms == 110_000


def test_no_fix_within_max_age_is_none() -> None:
    history = FixHistory(CONFIG)
    history.add(fix(START, received=100.0))
    assert history.nearest(103.5, 3.0) is None
    assert FixHistory(CONFIG).nearest(100.0, 3.0) is None


@pytest.mark.parametrize(
    "change, field",
    [
        ({"accuracy_meters": -1.0}, "accuracy_meters"),
        ({"heading_degrees": 360.0}, "heading_degrees"),
        ({"speed_mps": -0.1}, "speed_mps"),
        ({"timestamp_ms": 0}, "timestamp_ms"),
        ({"received_seconds": float("nan")}, "received_seconds"),
    ],
)
def test_a_fix_with_impossible_numbers_is_refused_by_name(change: dict, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        check_fix(replace(fix(), **change))
