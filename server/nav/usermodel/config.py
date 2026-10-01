"""What the user model needs. It measures what an avoidance cost the walker and never steers."""

# Standard library imports
from dataclasses import dataclass


@dataclass(frozen=True)
class UserModelConfig:
    """How the walker's turning is modeled."""

    seconds_per_bit: float = 0.25  # Placeholder until a walker is actually measured. See BUG-004.
    heading_tolerance_radians: float = 0.05  # About three degrees. Inside this, a turn is finished.
