"""
The report's formatting of figures that vary between passes, or that are unknown.

A figure that is unknown on one pass and known on another must say so, not quietly show the known one.
"""

# Local package imports
from nav.evaluation.report import _span


def test_span_shows_one_value_when_passes_agree() -> None:
    assert _span([3, 3, 3], "{}") == "3"


def test_span_shows_min_to_max_when_passes_differ() -> None:
    assert _span([2.0, 1.0, 1.5], "{:.2f}") == "1.00 to 2.00"


def test_span_counts_unknown_passes() -> None:
    assert _span([1.0, None, 2.0], "{:.2f}") == "1.00 to 2.00 (1 of 3 passes unknown)"


def test_span_with_nothing_known_is_unknown() -> None:
    assert _span([None, None], "{:.2f}") == "unknown"
