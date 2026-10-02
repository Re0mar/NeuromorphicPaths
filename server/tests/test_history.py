"""Covers N, the closing rate and the velocity, each against numbers the test computed itself."""

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.scene.history import ClearanceHistory

WINDOW = 0.5
MIN_SAMPLES = 3
FLOOR = 0.01


def _history() -> ClearanceHistory:
    return ClearanceHistory(window_seconds=WINDOW, min_samples=MIN_SAMPLES, noise_floor_meters=FLOOR)


def _feed(history: ClearanceHistory, clearances: list[float], group_id: int = 7, step: float = 0.1, centroids=None) -> None:
    for index, value in enumerate(clearances):
        centroid = np.array([0.0, value]) if centroids is None else centroids[index]
        history.update(index * step, [group_id], [value], [centroid])


def test_n_is_the_sample_standard_deviation_of_the_clearances() -> None:
    history = _history()
    clearances = [2.0, 2.1, 1.9, 2.05]
    _feed(history, clearances)

    assert history.noise_scale(7) == pytest.approx(np.std(clearances, ddof=1))


def test_n_grows_as_a_box_approaches_faster_than_the_noise() -> None:
    steady = _history()
    _feed(steady, [2.0, 2.0, 2.0, 2.0, 2.0])
    approaching = _history()
    _feed(approaching, [2.0, 1.8, 1.6, 1.4, 1.2])

    assert approaching.noise_scale(7) > steady.noise_scale(7)
    assert steady.noise_scale(7) == FLOOR


def test_too_few_samples_report_the_floor() -> None:
    history = _history()
    _feed(history, [2.0, 1.0])

    assert history.noise_scale(7) == FLOOR


def test_an_unknown_group_reports_the_floor() -> None:
    assert _history().noise_scale(99) == FLOOR


def test_n_never_goes_below_the_floor() -> None:
    history = _history()
    _feed(history, [2.0, 2.0, 2.0, 2.0])

    assert history.noise_scale(7) == FLOOR


def test_samples_older_than_the_window_are_evicted() -> None:
    history = _history()
    # Three wild early samples, then a long gap, then three identical ones. If the early ones
    # were still in the window, N would be large.
    _feed(history, [5.0, 0.5, 5.0])
    for index, value in enumerate([2.0, 2.0, 2.0]):
        history.update(10.0 + index * 0.1, [7], [value], [np.array([0.0, value])])

    assert history.noise_scale(7) == FLOOR


def test_closing_rate_is_the_clearance_lost_per_second() -> None:
    history = _history()
    _feed(history, [2.0, 1.8], step=0.1)

    # Lost 0.2 m in 0.1 s. Positive, because the group is approaching.
    assert history.closing_rate(7) == pytest.approx(2.0)


def test_closing_rate_is_negative_for_a_receding_group() -> None:
    history = _history()
    _feed(history, [1.8, 2.0], step=0.1)

    assert history.closing_rate(7) == pytest.approx(-2.0)


def test_closing_rate_is_none_after_one_sample() -> None:
    history = _history()
    _feed(history, [2.0])

    assert history.closing_rate(7) is None


def test_closing_rate_is_none_when_two_samples_share_a_timestamp() -> None:
    history = _history()
    history.update(1.0, [7], [2.0], [np.zeros(2)])
    history.update(1.0, [7], [1.0], [np.zeros(2)])

    assert history.closing_rate(7) is None


def test_velocity_is_the_centroid_displacement_per_second() -> None:
    history = _history()
    _feed(history, [3.0, 3.0], step=0.5, centroids=[np.array([0.0, 3.0]), np.array([0.5, 3.0])])

    assert history.velocity(7) == pytest.approx(np.array([1.0, 0.0]))


def test_velocity_is_none_after_one_sample() -> None:
    history = _history()
    _feed(history, [3.0])

    assert history.velocity(7) is None


def test_groups_unseen_past_the_window_are_forgotten() -> None:
    history = _history()
    _feed(history, [2.0, 2.0, 2.0], group_id=1)
    history.update(0.3, [2], [1.0], [np.zeros(2)])

    history.forget_unseen(5.0)

    assert history.tracked_group_ids() == []


def test_a_group_seen_inside_the_window_is_kept() -> None:
    history = _history()
    history.update(1.0, [1], [2.0], [np.zeros(2)])

    history.forget_unseen(1.2)

    assert history.tracked_group_ids() == [1]


def test_mismatched_update_lists_are_refused() -> None:
    with pytest.raises(ValueError, match="line up"):
        _history().update(0.0, [1, 2], [1.0], [np.zeros(2)])


@pytest.mark.parametrize(
    ("window", "min_samples", "floor"),
    [(0.0, 3, 0.01), (0.5, 1, 0.01), (0.5, 3, 0.0)],
)
def test_degenerate_configuration_is_refused(window: float, min_samples: int, floor: float) -> None:
    # min_samples of 1 would ask for a standard deviation with ddof=1 over one value, which is
    # undefined, and a zero floor would let surprise divide by zero.
    with pytest.raises(ValueError):
        ClearanceHistory(window_seconds=window, min_samples=min_samples, noise_floor_meters=floor)
