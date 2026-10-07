"""
Where the planner's natural-log values become the bits the course quotes.

His surprise is half of a squared normalized error, which is minus the log of a bell curve up to a
constant. That log is a natural log, so the value is in nats. The contact surprise is minus a
natural log of a probability, the sideways effort and the goal and previous-plan priors are half
squared ratios like his term, so every term the dynamic program sums is in nats, and so is the sum.

The course quotes surprise in bits. One bit is ln 2 nats, so a value in nats divided by ln 2 is the
same quantity in bits. Every place that labels a figure bits converts through here, and this is the
only file that spells out ln 2. Two copies of the factor are two places for the units to drift.
"""

# Third party imports
import numpy as np

# How many nats one bit is. Dividing a nats value by this gives bits.
NATS_PER_BIT = float(np.log(2.0))


def bits_from_nats(nats: np.ndarray | float) -> np.ndarray | float:
    """
    A natural-log value as bits. Works on scalars and on arrays alike.

    No checks. A negative input gives a negative output, and the caller's own checks are what refuse
    a value that must not be negative.

    :param nats: A cost or surprise in natural-log units.
    :return: The same quantity divided by ln 2.
    :rtype: np.ndarray | float
    """
    return nats / NATS_PER_BIT
