"""
The walker's footprint, which the scene and the planner both need and neither owns.

This has a module of its own rather than a home in either layer, because the radius must be
defined exactly once. The scene subtracts it to get clearance S, and the planner's field measures
from the same edge. A second copy in SceneConfig or PlannerConfig would drift, and the symptom
would be a planner that steers to a gap the scene thought was too narrow.
"""

# Standard library imports
from dataclasses import dataclass


@dataclass(frozen=True)
class WalkerConfig:
    """How much room the walker needs."""

    radius_meters: float = 0.35  # Shoulder half-width plus a margin.
