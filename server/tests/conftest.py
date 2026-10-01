"""
Shared constants for the guard tests.

There is no sys.path manipulation here. The package is installed with pip install -e, so the tests
import nav the same way anything else does. A test suite that patches its own import path can pass
against a layout that would not install.
"""

# Standard library imports
from pathlib import Path

# Local package imports
import nav

NAV_ROOT = Path(nav.__file__).parent

# Floor count of .py files under nav/. The grep tests assert they scanned at least this many,
# because a glob that silently matched nothing reports no violations and reads exactly like a
# clean package. Raise this when files are added.
MINIMUM_FILE_COUNT = 12
