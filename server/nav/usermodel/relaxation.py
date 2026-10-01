"""
How a heading error relaxes once the walker starts correcting it.

The paper models a correction as critically damped relaxation: the error s obeys
s'' + (2 / b) s' + (1 / b^2) s = 0, with b in seconds per bit. Critically damped means it settles
as fast as it can without overshooting, which is what a practised turn looks like. b is the one
number that says how quick this walker is, and it is a placeholder until a pointing test measures
it. That test is out of scope here.
"""

# Third party imports
import numpy as np

# Local package imports
from nav.usermodel.config import UserModelConfig

# Euler steps of a millisecond. The equation is stiff in the sense that b is a quarter second,
# so a coarser step would overshoot numerically what the equation itself never overshoots.
INTEGRATION_STEP_SECONDS = 0.001
# Past this the error is not relaxing, it is stuck, and inf says so better than a large number.
MAXIMUM_TURN_SECONDS = 10.0


def relax_step(error: float, velocity: float, dt: float, seconds_per_bit: float) -> tuple[float, float]:
    """
    One Euler step of the critically damped equation.

    Exposed so a display could animate the predicted relaxation. No sink in this task does.

    :param error: Current heading error, radians.
    :param velocity: Current rate of change of the error, radians per second.
    :param dt: Step length in seconds.
    :param seconds_per_bit: b.
    :return: (error, velocity) after the step.
    :rtype: tuple[float, float]
    """
    if seconds_per_bit <= 0:
        raise ValueError(f"seconds_per_bit must be positive, got {seconds_per_bit}")
    acceleration = -(2.0 / seconds_per_bit) * velocity - error / (seconds_per_bit**2)
    velocity = velocity + acceleration * dt
    error = error + velocity * dt
    return error, velocity


def predicted_turn_time(heading_error_radians: float, config: UserModelConfig) -> float:
    """
    Seconds the model predicts the walker needs to bring the error inside tolerance.

    Starts from rest at the given error and integrates until the error is within tolerance.

    :param heading_error_radians: The error to correct. Sign does not matter.
    :param config: b and the tolerance.
    :return: Seconds, or inf when the error has not settled within the maximum.
    :rtype: float
    """
    if config.seconds_per_bit <= 0:
        raise ValueError(f"seconds_per_bit must be positive, got {config.seconds_per_bit}")
    if config.heading_tolerance_radians <= 0:
        raise ValueError(f"heading tolerance must be positive, got {config.heading_tolerance_radians}")

    error = abs(float(heading_error_radians))
    if error <= config.heading_tolerance_radians:
        return 0.0

    velocity = 0.0
    elapsed = 0.0
    while elapsed < MAXIMUM_TURN_SECONDS:
        error, velocity = relax_step(error, velocity, INTEGRATION_STEP_SECONDS, config.seconds_per_bit)
        elapsed += INTEGRATION_STEP_SECONDS
        if abs(error) <= config.heading_tolerance_radians:
            return elapsed
    return float(np.inf)
