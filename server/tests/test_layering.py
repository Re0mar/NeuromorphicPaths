"""
Proves the shared layers do not need any of the device packages to import.

This is the one rule the whole refactor exists to buy: scene, planner and user model know nothing
about a Neon, a Pixel or a browser. It runs in a subprocess because blocking a module in this
process would poison every other test in the session.

While the device packages are empty this passes trivially. That is intended: the constraint is
in place before there is anything to break it.
"""

# Standard library imports
import subprocess
import sys

# The four packages a shared layer must never reach for. torch and depth_anything_3 belong to the
# estimator, pupil_labs to the Neon source, aiohttp to the web sink.
BLOCKED_PACKAGES = ("torch", "depth_anything_3", "pupil_labs", "aiohttp")

SHARED_LAYERS = ("nav.types", "nav.config", "nav.scene", "nav.planner", "nav.usermodel")


def test_shared_layers_import_without_device_packages() -> None:
    script = (
        "import sys\n"
        f"for name in {BLOCKED_PACKAGES!r}:\n"
        # Binding a name to None in sys.modules makes importing it raise ImportError, which is
        # what an environment without the optional extras would do.
        "    sys.modules[name] = None\n"
        f"import {', '.join(SHARED_LAYERS)}\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_blocking_actually_blocks() -> None:
    """Confirms the blocking trick works, so the test above cannot pass by not blocking anything."""
    script = (
        "import sys\n"
        "sys.modules['json'] = None\n"
        "import json\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    # ModuleNotFoundError, not a bare ImportError. Asserting on the concrete class is what proves
    # the import failed because the name was blocked, rather than for some unrelated reason.
    assert "ModuleNotFoundError" in completed.stderr
