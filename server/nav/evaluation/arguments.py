"""
The flags every replay command shares: which scene settings to replay with, planner overrides, and the cache.

One definition, so `python -m nav.evaluation` and `python -m nav.evaluation.check_planner` can't drift
apart on what the same flag means.
"""

# Standard library imports
import argparse
from pathlib import Path

# Local package imports
from nav.evaluation.replay import DEFAULT_CACHE_DIR


def add_replay_arguments(parser: argparse.ArgumentParser) -> None:
    """Add --cached, --cache-dir, --scene-defaults, --scene-set and --set to a parser."""
    parser.add_argument("--cached", action="store_true", help="one scene pass through the cache, for iterating, not for a verdict")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="where cached scene passes live, server/.replay_cache by default")
    parser.add_argument("--scene-defaults", action="store_true", help="today's scene defaults, for a recording with no run_config.json")
    parser.add_argument("--scene-set", action="append", default=[], metavar="FIELD=VALUE", help="override a SceneConfig field")
    parser.add_argument("--set", dest="planner_set", action="append", default=[], metavar="FIELD=VALUE", help="override a PlannerConfig field")
