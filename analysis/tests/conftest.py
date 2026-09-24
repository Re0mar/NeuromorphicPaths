# Standard library imports
from pathlib import Path

# Third party imports
import pytest

# Local package imports
from neuromorphicpaths_analysis.detector.path_detector import DEFAULT_MODEL_PATH, REPOSITORY_ROOT

TEST_PHOTOS_DIR = REPOSITORY_ROOT / "OldAppEnvrionmentStuff" / "artifacts" / "images"


# These fail rather than skip. A skipped test reports green, so a moved model file or photo
# folder would leave the suite passing without having run the model at all.

@pytest.fixture
def model_path() -> Path:
    assert DEFAULT_MODEL_PATH.exists(), f"model file missing at {DEFAULT_MODEL_PATH}"
    return DEFAULT_MODEL_PATH


@pytest.fixture
def test_photos_dir() -> Path:
    assert TEST_PHOTOS_DIR.is_dir(), f"test photo folder missing at {TEST_PHOTOS_DIR}"
    return TEST_PHOTOS_DIR
