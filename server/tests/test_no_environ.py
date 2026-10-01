"""
Proves no module in nav reads an environment variable.

A run is meant to be fully described by the command line that started it. One os.environ.get in a
layer undoes that, and the symptom is a run that behaves differently on two machines with no
difference in the command anyone typed.

estimator.py is the single exception and only for a write: it sets a Hugging Face download flag
before importing the model. Writing a variable is a decision the run makes, not an input it takes.
"""

# Standard library imports
from pathlib import Path

# Local package imports
from conftest import MINIMUM_FILE_COUNT, NAV_ROOT

ENVIRONMENT_WRITER = Path("sources/estimator.py")


def _python_files() -> list[Path]:
    return sorted(NAV_ROOT.rglob("*.py"))


def test_scan_covers_the_package() -> None:
    # Without this, the test below passes against a glob that matched nothing at all.
    assert len(_python_files()) >= MINIMUM_FILE_COUNT


def test_nav_never_reads_the_environment() -> None:
    offenders: list[str] = []
    for path in _python_files():
        relative = path.relative_to(NAV_ROOT)
        if relative == ENVIRONMENT_WRITER:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "os.environ" in line or "getenv" in line:
                offenders.append(f"{relative}:{number} {line.strip()}")
    assert offenders == [], "\n".join(offenders)
