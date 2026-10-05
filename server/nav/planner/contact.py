"""
Contact surprise: how surprising it would be for the walker's body to touch a point.

This is a different question from the one his collision surprise in surprise.py answers. His
term, half of (N over S) squared, says how unsure the reading of something near is. A post
measured steadily has a small N, so it barely costs anything even at contact. This term says how
likely the body is to overlap the point at all, whatever the noise, so a steady post in the way
costs something to walk into. The two are evidence about different things, and the planner adds
them.

The probability first. S is the clearance between the body and the point: the center distance
minus the body half-width, negative when they would overlap. The walker's real position is
uncertain for two reasons, the reading's noise N and how far a person drifts from the line the
arrow asks for, the sway. Together they spread S by sigma = hypot(N, sway). The chance of contact
is p = Phi(-S / sigma), where Phi is the standard normal cumulative distribution.

Then the surprise. The surprise of getting past cleanly is -ln(1 - p), which is -ln Phi(S / sigma),
computed with scipy's log_ndtr so it stays accurate when p is tiny and when it is nearly one. It is
a natural log, the same log as his term, which is the negative log of a bell curve. That is what
lets the two add as independent evidence. The course quotes both in bits by convention.

Then the rate. The dynamic program multiplies every field value by the time step, so the term is
a rate per second: the surprise divided by tau, the time it takes to walk past an obstacle, twice
the half-width over walking speed. Halving the time step then leaves the total unchanged, and a
straight walk through an obstacle adds up to about one contact's worth.

Three limits. A center distance is never negative, so S never drops below minus the half-width and
one point can never cost more than -ln Phi(-half-width / sway), 6.61 at the shipped values. The
cap only matters for a much smaller sway. Consecutive slices that overlap the same obstacle are
summed as if each were a separate contact, which overcounts a long overlap. That is harmless when
the plan's aim is not to go there at all. And groups add, the way his term's do, while the scene
makes one group per 0.25 m cell, so a wall is many groups. Walking beside a wall therefore costs
about 1.4 to 1.5 times what one cell at the same clearance costs. His term overcounts a wall more,
2 to 4 times, under the same rule.
"""

# Third party imports
import numpy as np
from scipy.special import log_ndtr

# Local package imports
from nav.planner.config import PlannerConfig
from nav.walker import WalkerConfig


def _check_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be above zero for the contact term, got {value}")


def check_contact_config(config: PlannerConfig) -> None:
    """
    Refuse constants the contact term cannot use, before the first frame.

    Nothing is checked while the term is off, so a config that switches it off never fails on it.

    :param config: The sway, the cap, the body half-width and the walking speed.
    :raises ValueError: Naming the first field that is out of range.
    """
    if not config.contact_term_enabled:
        return
    # A zero sway can make sigma zero, and S over sigma undefined at contact.
    _check_positive("walker_sway_meters", config.walker_sway_meters)
    _check_positive("contact_surprise_cap", config.contact_surprise_cap)
    # Tau divides by the walking speed and multiplies by the half-width, so neither may be zero.
    _check_positive("body_half_width_meters", config.body_half_width_meters)
    _check_positive("walking_speed_mps", config.walking_speed_mps)


def contact_time_scale(config: PlannerConfig) -> float:
    """Tau, the seconds it takes to walk past an obstacle: twice the body half-width over walking speed."""
    return 2.0 * config.body_half_width_meters / config.walking_speed_mps


class ContactSurprise:
    """The contact surprise as a cost term, a rate per second."""

    def point_cost(
        self,
        center_distance_meters: np.ndarray,
        noise_meters: np.ndarray,
        config: PlannerConfig,
        walker: WalkerConfig,
    ) -> np.ndarray:
        """
        Minus ln Phi(S over sigma), capped, over tau, for every point from every candidate position.

        :param center_distance_meters: (cells, points) distance from each candidate position to each point.
        :param noise_meters: (points,) each point's N, walls already multiplied.
        :param config: The body half-width, the sway, the cap and the walking speed.
        :param walker: Unused. Contact is measured against the body, not the steering footprint.
        :return: (cells, points) contact surprise per second.
        :rtype: np.ndarray
        """
        clearance = center_distance_meters - config.body_half_width_meters
        spread = np.hypot(noise_meters[None, :], config.walker_sway_meters)
        surprise = np.minimum(-log_ndtr(clearance / spread), config.contact_surprise_cap)
        return surprise / contact_time_scale(config)
