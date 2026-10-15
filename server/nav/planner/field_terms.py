"""
Everything that adds to the field the dynamic program plans through, summed one way.

A term is a field over (step, lateral cell) with a declaration of how it is meant:

- how it is charged: a rate on every row, which the dynamic program multiplies by the time step, or a
  cost paid once on the last row, which has to be divided by the time step first so that
  multiplication gives it back in full
- whether the plan's cost counts it: the scene's terms are work the walker had to do, the walker's own
  priors (where they want to go, the side they were holding) are not
- what the information measure's prior keeps of it: the planner with nothing in view

The pipeline hands its terms here and gets back the field it keeps, the field it plans through and the
prior, with no code of its own per term. A new term, a preference for pavement over grass say, is one
more entry with its declaration. The professor's surprise is one term among them and is not touched.
"""

# Standard library imports
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

# Third party imports
import numpy as np

# Local package imports
from nav.planner.config import PlannerConfig
from nav.planner.goal import terminal_cost_as_rate


class Charging(Enum):
    """How the dynamic program should charge a term."""

    RATE = "rate"  # A cost per second on every row.
    TERMINAL_COST = "terminal cost"  # A cost paid once, held on the last row.


class WorkFigure(Enum):
    """Whether the plan's cost, which the user model's work figure reads, counts a term."""

    COUNTED = "counted"  # Something the scene imposed.
    LEFT_OUT = "left out"  # The walker's own: where they want to go, the side they were holding.


@dataclass(frozen=True)
class TermDeclaration:
    """What a term is, said once where it is defined."""

    name: str
    charging: Charging
    work_figure: WorkFigure


SCENE = TermDeclaration("scene", Charging.RATE, WorkFigure.COUNTED)
GOAL = TermDeclaration("goal", Charging.TERMINAL_COST, WorkFigure.LEFT_OUT)
PREVIOUS_PLAN = TermDeclaration("previous plan", Charging.RATE, WorkFigure.LEFT_OUT)


@dataclass(frozen=True)
class TermField:
    """
    One term's field this frame, and what the planner with nothing in view would have of it.

    field holds a terminal cost's values on its last row and zeros elsewhere, in the same units as a
    rate's row, so the kept field reads one unit throughout. prior is None for a term the planner with
    nothing in view would not have.
    """

    declaration: TermDeclaration
    field: np.ndarray
    prior: np.ndarray | None


@dataclass(frozen=True)
class SummedFields:
    """The terms added up three ways."""

    kept: np.ndarray  # Every term as declared, for the drawings and the band breakdown.
    planned: np.ndarray  # What the dynamic program plans through, terminal costs ready to be multiplied back.
    prior: np.ndarray  # The planned form of what the planner with nothing in view would have.


def as_planned(field: np.ndarray, declaration: TermDeclaration, config: PlannerConfig) -> np.ndarray:
    """
    A term's field in the form the dynamic program charges correctly.

    A rate is already that. A terminal cost's last row is divided by the time step, because the
    dynamic program multiplies every row by it. The rest of its rows are zero and stay so.
    """
    if declaration.charging is Charging.RATE:
        return field
    planned = field.copy()
    planned[-1] = terminal_cost_as_rate(field[-1], config)
    return planned


def sum_terms(terms: Sequence[TermField], config: PlannerConfig) -> SummedFields:
    """
    Add the terms up for keeping, for planning, and for the information measure's prior.

    In the order given, from zero, so the sums are the same to the bit as adding the terms by hand in
    that order. The pipeline's golden values rest on that.

    :raises ValueError: When there are no terms, or their shapes differ.
    """
    if not terms:
        raise ValueError("the field needs at least one term")
    shape = terms[0].field.shape
    for term in terms:
        if term.field.shape != shape or (term.prior is not None and term.prior.shape != shape):
            raise ValueError(f"the {term.declaration.name} term is not {shape}")
    kept = np.zeros(shape)
    planned = np.zeros(shape)
    prior = np.zeros(shape)
    for term in terms:
        kept += term.field
        planned += as_planned(term.field, term.declaration, config)
        if term.prior is not None:
            prior += as_planned(term.prior, term.declaration, config)
    return SummedFields(kept=kept, planned=planned, prior=prior)


def charges_left_out(terms: Sequence[TermField], offsets: np.ndarray, grid: np.ndarray, config: PlannerConfig) -> list[float]:
    """
    What each term the work figure leaves out charged along the chosen path, in the dynamic program's natural-log units.

    :param offsets: The chosen path's lateral offset per step.
    :return: One charge per left-out term, in the order given, for the caller to take off one by one.
    :rtype: list[float]
    """
    cells = np.argmin(np.abs(grid[None, :] - offsets[:, None]), axis=1)
    rows = np.arange(len(offsets))
    return [
        float(np.sum(as_planned(term.field, term.declaration, config)[rows, cells]) * config.time_step_seconds)
        for term in terms
        if term.declaration.work_figure is WorkFigure.LEFT_OUT
    ]


def last_row_only(row: np.ndarray, steps: int) -> np.ndarray:
    """A field that is this row on its last step and zero everywhere else, for a terminal cost."""
    field = np.zeros((steps, len(row)))
    field[-1] = row
    return field
