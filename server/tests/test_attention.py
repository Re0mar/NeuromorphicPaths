"""The attention test's geometry and its credit: a look is on the danger's outline, and counts for a while."""

# Standard library imports
import math

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.planner.attention import (
    AttentionCredit,
    angle_between_radians,
    angle_to_outline_radians,
    danger_object,
    gaze_on_outlines,
    longest_dwell_seconds,
    pixel_ray,
    tolerance_radians,
)
from nav.types import CameraOutline, ObstaclePoint

FOCAL = 500.0
PRINCIPAL_X = 320.0
PRINCIPAL_Y = 240.0
INTRINSICS = np.array([[FOCAL, 0.0, PRINCIPAL_X], [0.0, FOCAL, PRINCIPAL_Y], [0.0, 0.0, 1.0]])
# A single point straight ahead of the camera, as an outline of no size.
AHEAD = CameraOutline(0.0, 0.0, 0.0, 0.0)
FIVE_DEGREES = tolerance_radians(5.0)
WALKING_SPEED = 1.4


def _pixel_at_degrees(right: float, down: float = 0.0) -> list[float]:
    """A pixel the given angles right of and below the optical axis."""
    return [PRINCIPAL_X + FOCAL * math.tan(math.radians(right)), PRINCIPAL_Y + FOCAL * math.tan(math.radians(down))]


def _outline_at_degrees(right: float, down: float = 0.0) -> CameraOutline:
    """An outline of no size at the given angles, the same as one point there."""
    across, below = math.tan(math.radians(right)), math.tan(math.radians(down))
    return CameraOutline(across, across, below, below)


def _noticed(pixels: list, offsets: list | None = None, window: float = 0.1, outline: CameraOutline | None = AHEAD, tolerance: float = FIVE_DEGREES) -> bool:
    pixel_array = np.array(pixels, dtype=np.float64).reshape(-1, 2)
    offset_array = np.zeros(len(pixel_array)) if offsets is None else np.array(offsets, dtype=np.float64)
    return gaze_on_outlines(pixel_array, offset_array, window, INTRINSICS, [outline], tolerance)


def _danger(lateral: float, forward: float) -> ObstaclePoint:
    return ObstaclePoint(lateral, forward, 1, 0.2, 0.05, None, None, False, np.zeros(3))


# *******************************************
# The angle
# *******************************************


def test_the_angle_at_the_principal_point_is_zero() -> None:
    ray = pixel_ray(np.array([PRINCIPAL_X, PRINCIPAL_Y]), INTRINSICS)

    assert float(angle_between_radians(ray, np.array([0.0, 0.0, 2.0]))) == 0.0


def test_a_known_pixel_offset_gives_the_hand_computed_angle() -> None:
    # 100 px right at a 500 px focal is atan(0.2), and 500 px right is 45 degrees.
    hundred_right = pixel_ray(np.array([420.0, 240.0]), INTRINSICS)
    five_hundred_right = pixel_ray(np.array([820.0, 240.0]), INTRINSICS)
    ahead = np.array([0.0, 0.0, 1.0])

    assert math.degrees(float(angle_between_radians(hundred_right, ahead))) == pytest.approx(11.309932474020215, abs=1e-9)
    assert math.degrees(float(angle_between_radians(five_hundred_right, ahead))) == pytest.approx(45.0, abs=1e-9)


def test_the_pixel_ray_matches_gaze_on_the_ground() -> None:
    # The arithmetic gaze_on_the_ground used before it called pixel_ray, written out by hand.
    pixel = np.array([123.0, 456.0])
    expected = np.array([(123.0 - PRINCIPAL_X) / FOCAL, (456.0 - PRINCIPAL_Y) / FOCAL, 1.0])

    assert np.array_equal(pixel_ray(pixel, INTRINSICS), expected)
    assert np.array_equal(pixel_ray(np.stack([pixel, pixel]), INTRINSICS), np.stack([expected, expected]))


def test_the_angle_to_an_outline_is_zero_inside_and_to_its_nearest_edge_outside() -> None:
    # An outline from 10 to 20 degrees right and 0 to 30 degrees down.
    outline = CameraOutline(math.tan(math.radians(10.0)), math.tan(math.radians(20.0)), 0.0, math.tan(math.radians(30.0)))
    rays = pixel_ray(np.array([_pixel_at_degrees(15.0, 15.0), _pixel_at_degrees(5.0, 0.0)]), INTRINSICS)

    inside, left_of_it = np.degrees(angle_to_outline_radians(rays, outline))

    assert inside == 0.0
    assert left_of_it == pytest.approx(5.0, abs=1e-9)


# *******************************************
# A look on the outline
# *******************************************


def test_a_sample_on_the_danger_is_noticed() -> None:
    assert _noticed([[PRINCIPAL_X, PRINCIPAL_Y]])


def test_a_sample_just_outside_the_tolerance_is_not() -> None:
    assert _noticed([_pixel_at_degrees(4.9)])
    assert not _noticed([_pixel_at_degrees(5.1)])


def test_five_degrees_accepts_4_9_degrees_and_rejects_0_1_radians() -> None:
    # 0.1 rad is 5.73 degrees. A tolerance read as radians would accept it.
    assert _noticed([_pixel_at_degrees(4.9)])
    assert not _noticed([_pixel_at_degrees(math.degrees(0.1))])


def test_the_test_is_symmetric_about_the_danger() -> None:
    # The danger sits 20 degrees right. Gaze 4 degrees either side of it counts, 6 either side does not.
    danger = _outline_at_degrees(20.0)

    assert _noticed([_pixel_at_degrees(16.0)], outline=danger)
    assert _noticed([_pixel_at_degrees(24.0)], outline=danger)
    assert not _noticed([_pixel_at_degrees(14.0)], outline=danger)
    assert not _noticed([_pixel_at_degrees(26.0)], outline=danger)


def test_a_look_at_the_body_of_an_obstacle_counts_though_its_nearest_point_is_far_below() -> None:
    # A chair: nearest point 35 degrees down by the floor, its back up to 5 degrees down. The gaze
    # rests 10 degrees down on the seat, 25 degrees from the nearest point.
    chair = CameraOutline(math.tan(math.radians(-5.0)), math.tan(math.radians(5.0)), math.tan(math.radians(5.0)), math.tan(math.radians(35.0)))
    on_the_seat = _pixel_at_degrees(0.0, 10.0)

    assert _noticed([on_the_seat], outline=chair)
    assert not _noticed([on_the_seat], outline=_outline_at_degrees(0.0, 35.0))


def test_no_samples_is_not_noticed() -> None:
    assert not gaze_on_outlines(None, None, 0.1, INTRINSICS, [AHEAD], FIVE_DEGREES)
    assert not gaze_on_outlines(np.empty((0, 2)), np.empty(0), 0.1, INTRINSICS, [AHEAD], FIVE_DEGREES)


def test_a_sample_outside_the_window_does_not_count() -> None:
    on_the_danger = [PRINCIPAL_X, PRINCIPAL_Y]

    assert not _noticed([on_the_danger], offsets=[0.15], window=0.1)
    assert not _noticed([on_the_danger], offsets=[-0.15], window=0.1)
    # The window's edge is inside it.
    assert _noticed([on_the_danger], offsets=[-0.1], window=0.1)


def test_one_sample_in_the_window_on_the_danger_is_enough() -> None:
    off = _pixel_at_degrees(30.0)

    assert _noticed([off, [PRINCIPAL_X, PRINCIPAL_Y], off], offsets=[-0.05, 0.0, 0.05])


def test_a_danger_with_no_outline_is_never_noticed() -> None:
    assert not _noticed([[PRINCIPAL_X, PRINCIPAL_Y]], outline=None)


def test_gaze_clearly_off_the_danger_never_notices_it_even_at_a_wide_tolerance() -> None:
    # Positive control first: 40 degrees accepts a look 30 degrees off.
    wide = tolerance_radians(40.0)
    assert _noticed([_pixel_at_degrees(30.0)], tolerance=wide)
    # Looking the other way, 45 degrees left, is 45 degrees off and never counts.
    assert not _noticed([_pixel_at_degrees(-45.0)], tolerance=wide)


@pytest.mark.parametrize("degrees", [0.0, -1.0, float("nan")])
def test_a_tolerance_at_or_under_zero_is_refused(degrees: float) -> None:
    with pytest.raises(ValueError, match="attention tolerance"):
        tolerance_radians(degrees)


@pytest.mark.parametrize("degrees", [45.0, 90.0, float("inf")])
def test_a_tolerance_of_45_or_more_is_refused(degrees: float) -> None:
    with pytest.raises(ValueError, match="attention tolerance"):
        tolerance_radians(degrees)


@pytest.mark.parametrize(
    "edges",
    [(0.2, 0.1, 0.0, 0.0), (0.0, 0.0, 0.3, 0.1), (float("nan"), 0.0, 0.0, 0.0)],
)
def test_an_outline_inside_out_or_not_finite_is_refused(edges: tuple) -> None:
    with pytest.raises(ValueError, match="outline"):
        CameraOutline(*edges)


# *******************************************
# The credit
# *******************************************


def test_a_look_is_noticed_on_its_own_frame() -> None:
    credit = AttentionCredit(2.0, 0.5, WALKING_SPEED)

    assert credit.update(_danger(0.0, 3.0), True, 10.0)


def test_the_credit_follows_the_danger_toward_the_walker() -> None:
    # Looked at 3 m ahead. One second later at walking pace it should be 1.6 m ahead.
    credit = AttentionCredit(2.0, 0.5, WALKING_SPEED)
    credit.update(_danger(0.0, 3.0), True, 10.0)

    assert credit.update(_danger(0.0, 1.6), False, 11.0)


def test_a_danger_where_the_looked_at_one_would_not_be_is_not_noticed() -> None:
    # Positive control in the test above. Here the danger is still at 3 m a second later, 1.4 m
    # from where the looked-at one would be. A cyclist cutting in is the case this guards.
    credit = AttentionCredit(2.0, 0.5, WALKING_SPEED)
    credit.update(_danger(0.0, 3.0), True, 10.0)

    assert not credit.update(_danger(0.0, 3.0), False, 11.0)
    assert not credit.update(_danger(1.0, 1.6), False, 11.0)


def test_the_credit_expires() -> None:
    credit = AttentionCredit(2.0, 0.5, WALKING_SPEED)
    credit.update(_danger(0.0, 4.0), True, 10.0)

    assert credit.update(_danger(0.0, 4.0 - 2.0 * WALKING_SPEED), False, 12.0)
    assert not credit.update(_danger(0.0, 4.0 - 2.1 * WALKING_SPEED), False, 12.1)
    # Gone for good, not only for that frame.
    assert not credit.update(_danger(0.0, 4.0 - 2.0 * WALKING_SPEED), False, 12.0 + 1e-9)


def test_a_zero_credit_counts_the_looking_frame_only() -> None:
    credit = AttentionCredit(0.0, 0.5, WALKING_SPEED)
    credit.update(_danger(0.0, 3.0), True, 10.0)

    assert not credit.update(_danger(0.0, 3.0), False, 10.01)


def test_an_empty_corridor_keeps_the_credit() -> None:
    credit = AttentionCredit(2.0, 0.5, WALKING_SPEED)
    credit.update(_danger(0.0, 3.0), True, 10.0)

    assert not credit.update(None, False, 10.5)
    assert credit.update(_danger(0.0, 3.0 - WALKING_SPEED), False, 11.0)


def test_a_new_danger_with_no_look_starts_unnoticed() -> None:
    assert not AttentionCredit(2.0, 0.5, WALKING_SPEED).update(_danger(0.0, 3.0), False, 10.0)


def test_a_backward_stamp_forgets_even_after_the_credit_expired() -> None:
    credit = AttentionCredit(2.0, 0.5, WALKING_SPEED)
    credit.update(_danger(0.0, 3.0), True, 10.0)
    credit.update(_danger(0.0, 3.0), False, 13.0)
    credit.update(_danger(0.0, 3.0), True, 13.5)

    # A stamp before the last one is a new run. The look at 13.5 s isn't carried into it. The danger
    # sits exactly where that look, run backward 8.5 s, would put it, so only the reset refuses it.
    assert not credit.update(_danger(0.0, 3.0 + WALKING_SPEED * 8.5), False, 5.0)


@pytest.mark.parametrize(
    "arguments, named",
    [((-1.0, 0.5, 1.4), "credit_seconds"), ((2.0, 0.0, 1.4), "match_meters"), ((2.0, 0.5, 0.0), "walking_speed_mps"), ((float("nan"), 0.5, 1.4), "credit_seconds")],
)
def test_a_credit_out_of_range_is_refused_by_name(arguments: tuple, named: str) -> None:
    with pytest.raises(ValueError, match=named):
        AttentionCredit(*arguments)


# *******************************************
# The danger's whole object
# *******************************************


def test_pieces_chained_within_the_link_are_one_object_and_a_gap_wider_than_it_is_not() -> None:
    # A row of pieces 0.3 m apart from the danger outward, then a gap of 0.5 m to another.
    danger = _danger(0.0, 1.0)
    row = [_danger(0.3, 1.0), _danger(0.6, 1.0), _danger(0.9, 1.0)]
    beyond = _danger(1.4, 1.0)

    pieces = danger_object(danger, [danger, *row, beyond], 0.4, 5.0)

    assert pieces[0] is danger
    assert set(map(id, pieces)) == set(map(id, [danger, *row]))


def test_a_link_exactly_at_the_distance_joins() -> None:
    danger = _danger(0.0, 1.0)
    neighbor = _danger(0.0, 1.25)

    assert len(danger_object(danger, [danger, neighbor], 0.25, 5.0)) == 2


def test_a_danger_alone_is_its_own_object_whether_or_not_it_is_in_the_list() -> None:
    danger = _danger(0.0, 1.0)

    assert danger_object(danger, [], 0.4, 0.6) == (danger,)
    assert danger_object(danger, [_danger(3.0, 3.0)], 0.4, 0.6) == (danger,)


def test_a_look_at_any_piece_of_the_object_is_a_look_at_it() -> None:
    base = _outline_at_degrees(0.0, 30.0)
    back = _outline_at_degrees(0.0, 5.0)
    on_the_back = np.array([_pixel_at_degrees(0.0, 5.0)])

    assert gaze_on_outlines(on_the_back, np.zeros(1), 0.1, INTRINSICS, [base, back], FIVE_DEGREES)
    assert not gaze_on_outlines(on_the_back, np.zeros(1), 0.1, INTRINSICS, [base], FIVE_DEGREES)


def test_an_object_whose_pieces_have_no_outlines_is_never_noticed() -> None:
    on_the_axis = np.array([[PRINCIPAL_X, PRINCIPAL_Y]])

    assert not gaze_on_outlines(on_the_axis, np.zeros(1), 0.1, INTRINSICS, [None, None], FIVE_DEGREES)
    assert not gaze_on_outlines(on_the_axis, np.zeros(1), 0.1, INTRINSICS, [], FIVE_DEGREES)


# *******************************************
# A steady look
# *******************************************


def test_the_longest_dwell_is_the_longest_unbroken_stretch_on_the_object() -> None:
    offsets = np.array([0.0, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030])
    on_object = np.array([True, True, False, True, True, True, False])

    assert longest_dwell_seconds(offsets, on_object, 0.02) == pytest.approx(0.010)


def test_a_dropout_longer_than_the_gap_breaks_a_steady_look() -> None:
    offsets = np.array([0.0, 0.005, 0.100, 0.105])
    on_object = np.ones(4, dtype=bool)

    assert longest_dwell_seconds(offsets, on_object, 0.02) == pytest.approx(0.005)
    assert longest_dwell_seconds(offsets, on_object, 0.2) == pytest.approx(0.105)


def test_offsets_out_of_order_give_the_same_dwell_as_sorted() -> None:
    offsets = np.array([0.015, 0.010, 0.005, 0.0])

    assert longest_dwell_seconds(offsets, np.ones(4, dtype=bool), 0.02) == pytest.approx(0.015)


def test_one_sample_on_the_object_is_a_dwell_of_zero_and_none_on_it_is_none() -> None:
    offsets = np.array([0.0, 0.005])

    assert longest_dwell_seconds(offsets, np.array([False, True]), 0.02) == 0.0
    assert longest_dwell_seconds(offsets, np.array([False, False]), 0.02) is None


def test_a_glance_passes_with_no_dwell_and_fails_a_dwell_a_steady_look_passes() -> None:
    on_the_axis = [PRINCIPAL_X, PRINCIPAL_Y]
    off = _pixel_at_degrees(30.0)
    offsets = np.arange(11) * 0.01
    glance = np.array([off] * 5 + [on_the_axis] + [off] * 5)
    steady = np.array([on_the_axis] * 11)

    assert gaze_on_outlines(glance, offsets, 0.2, INTRINSICS, [AHEAD], FIVE_DEGREES)
    assert not gaze_on_outlines(glance, offsets, 0.2, INTRINSICS, [AHEAD], FIVE_DEGREES, dwell_seconds=0.05)
    assert gaze_on_outlines(steady, offsets, 0.2, INTRINSICS, [AHEAD], FIVE_DEGREES, dwell_seconds=0.1)


def test_a_steady_look_sliding_across_two_pieces_of_one_object_stays_one_look() -> None:
    offsets = np.arange(11) * 0.01
    seat, back = _outline_at_degrees(0.0, 10.0), _outline_at_degrees(0.0, 5.0)
    sliding = np.array([_pixel_at_degrees(0.0, 10.0)] * 6 + [_pixel_at_degrees(0.0, 5.0)] * 5)

    assert gaze_on_outlines(sliding, offsets, 0.2, INTRINSICS, [seat, back], tolerance_radians(1.0), dwell_seconds=0.1)


def test_the_cap_stops_a_chain_running_along_a_row() -> None:
    # A row of seats 0.3 m apart running 1.5 m out. Capped at 0.6 m, only the first two join.
    danger = _danger(0.0, 1.0)
    row = [_danger(0.3 * step, 1.0) for step in range(1, 6)]

    pieces = danger_object(danger, [danger, *row], 0.4, 0.6)

    assert [piece.lateral_meters for piece in pieces] == pytest.approx([0.0, 0.3, 0.6])
