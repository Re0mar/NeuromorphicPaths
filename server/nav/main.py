"""The entry point named by pyproject's console script and by python -m nav."""

# Standard library imports
import logging

# Local package imports
from nav.config import build_run_config
from nav.runtime.loop import run


def main(argv: list[str] | None = None) -> int:
    """
    Build everything from the command line and run the frame loop.

    :param argv: Argument list, or None to read sys.argv.
    :return: Process exit code.
    :rtype: int
    """
    config = build_run_config(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # --verbose is about this pipeline's per-stage timings and floor fits. Setting the root to
    # DEBUG instead drowns them in every HTTP library's request headers during the model download.
    logging.getLogger("nav").setLevel(logging.DEBUG if config.verbose else logging.INFO)
    return run(config)
