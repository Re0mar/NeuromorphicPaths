"""Makes python -m nav work, which is how the pipeline is documented to be run."""

# Standard library imports
import sys

# Local package imports
from nav.main import main

if __name__ == "__main__":
    sys.exit(main())
