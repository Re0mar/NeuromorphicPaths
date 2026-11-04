"""Covers the position settings' refusals. The file is named for its package, since test_config.py is nav/config.py's."""

# Third party imports
import pytest

# Local package imports
from nav.pose.config import PositionConfig


@pytest.mark.parametrize(
    "field",
    ["depth_max_meters", "dense_slam_depth_difference_meters", "odometry_depth_difference_meters", "voxel_size_meters", "max_step_speed_mps", "stall_seconds"],
)
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_a_length_speed_or_time_must_be_finite_and_positive(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=f"{field} must be a finite positive number"):
        PositionConfig(**{field: value})


def test_the_restart_count_may_be_zero_but_not_negative() -> None:
    assert PositionConfig(max_restarts=0).max_restarts == 0
    with pytest.raises(ValueError, match="max_restarts must be zero or more"):
        PositionConfig(max_restarts=-1)


def test_the_block_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="voxel_block_count must be positive"):
        PositionConfig(voxel_block_count=0)


@pytest.mark.parametrize("value", [-0.1, 1.0])
def test_the_fitness_floor_is_a_share_below_one(value: float) -> None:
    with pytest.raises(ValueError, match=r"min_fitness must be in \[0, 1\)"):
        PositionConfig(min_fitness=value)
