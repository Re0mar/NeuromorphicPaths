"""
python -m nav.evaluation: score the planner's arrow against the walker's turns on recorded walks.

Run from server/ with its venv:

    python -m nav.evaluation frame_logs/wifi_run_2 --out results.md
    python -m nav.evaluation frame_logs/pixel_display_run --scene-defaults --scene-set floor_max_tilt_degrees=50
    python -m nav.evaluation --spread-only frame_logs/pixel_walk_3 frame_logs/wifi_run_2

A plain run takes three cold scene passes. The floor fit is seeded, so they should agree, and their
agreeing is the evidence. --cached takes one pass through the cache, for iterating.

Exit codes: 0 scored, 1 a recording or an override refused, 2 a usage error, 3 nothing to score,
which includes turns with no arrow to read before any of them.
"""

# Standard library imports
import argparse
import sys
from pathlib import Path

# Local package imports
from nav.evaluation.config import EvaluationConfig
from nav.evaluation.overrides import OverrideRefused, apply_overrides
from nav.evaluation.replay import (
    DEFAULT_CACHE_DIR,
    RecordingRefused,
    clone_state,
    evaluate_walk,
    scene_config_for,
    straight_spread_only,
)
from nav.evaluation.report import format_report, format_spread
from nav.planner.config import PlannerConfig
from nav.runtime.textio import write_text_lf
from nav.sources.framecodec import FrameDecodeError

EXIT_SCORED = 0
EXIT_REFUSED = 1
EXIT_NOTHING_SCORED = 3
DEFAULT_PASSES = 3


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m nav.evaluation", description="Score the planner's arrow against the walker's real turns.")
    parser.add_argument("log_dirs", type=Path, nargs="+", help="frame logs written by --record-to")
    parser.add_argument("--passes", type=int, default=None, help=f"cold scene passes per walk, {DEFAULT_PASSES} by default")
    parser.add_argument("--cached", action="store_true", help="one scene pass through the cache, for iterating, not for a verdict")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="where cached scene passes live, server/.replay_cache by default")
    parser.add_argument("--scene-defaults", action="store_true", help="today's scene defaults, for a recording with no run_config.json")
    parser.add_argument("--scene-set", action="append", default=[], metavar="FIELD=VALUE", help="override a SceneConfig field")
    parser.add_argument("--set", dest="planner_set", action="append", default=[], metavar="FIELD=VALUE", help="override a PlannerConfig field")
    parser.add_argument("--eval-set", action="append", default=[], metavar="FIELD=VALUE", help="override an EvaluationConfig field")
    parser.add_argument("--out", type=Path, help="also write the report here")
    parser.add_argument("--spread-only", action="store_true", help="read only the poses and print the straight-walking spread and threshold")
    arguments = parser.parse_args(argv)
    if arguments.passes is not None and arguments.passes < 1:
        parser.error("--passes must be at least 1")
    if arguments.cached and arguments.passes not in (None, 1):
        parser.error("--cached takes one pass. Several passes through one cache entry are one realization repeated")
    if arguments.passes is None:
        arguments.passes = 1 if arguments.cached else DEFAULT_PASSES
    return arguments


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        return _run(arguments)
    except (RecordingRefused, OverrideRefused, FileNotFoundError, FrameDecodeError) as refused:
        # A recording or an override this code can't use. Known and named, so it ends with the
        # message and no traceback. Anything else is a defect and keeps its traceback.
        print(f"{type(refused).__name__}: {refused}", file=sys.stderr)
        return EXIT_REFUSED


def _run(arguments: argparse.Namespace) -> int:
    config, evaluation_overrides = apply_overrides(EvaluationConfig(), arguments.eval_set)
    if arguments.spread_only:
        pooled = straight_spread_only(arguments.log_dirs, config)
        report = format_spread(pooled, [log_dir.name for log_dir in arguments.log_dirs], config)
        _emit(report, arguments.out)
        return EXIT_NOTHING_SCORED if pooled.failures else EXIT_SCORED

    planner_config, planner_overrides = apply_overrides(PlannerConfig(), arguments.planner_set)
    cache_dir = arguments.cache_dir if arguments.cached else None
    results = []
    for log_dir in arguments.log_dirs:
        scene_config, scene_source = scene_config_for(log_dir, arguments.scene_defaults, arguments.scene_set)
        results.append(evaluate_walk(log_dir, scene_config, scene_source, planner_config, config, arguments.passes, cache_dir))

    report = format_report(results, clone_state(), planner_config, planner_overrides, config, evaluation_overrides)
    _emit(report, arguments.out)

    if not any(result.segments for result in results):
        print(f"nothing scored: no walk had a segment of {config.min_segment_seconds:g} s or more", file=sys.stderr)
        return EXIT_NOTHING_SCORED
    if not any(segment.turns for result in results for segment in result.segments):
        print("nothing scored: no turn was found in any kept segment", file=sys.stderr)
        return EXIT_NOTHING_SCORED
    if not any(score.arrow_read for result in results for result_pass in result.passes for score in result_pass.scores):
        # Turns, but no arrow before any of them, say a scene that refused every frame. Reporting
        # that as a scored walk would turn "no arrow" into "the arrow never sidestepped".
        print("nothing scored: no turn had an arrow to read before it", file=sys.stderr)
        return EXIT_NOTHING_SCORED
    return EXIT_SCORED


def _emit(report: str, out: Path | None) -> None:
    print(report, end="")
    if out is not None:
        write_text_lf(out, report)


if __name__ == "__main__":
    sys.exit(main())
