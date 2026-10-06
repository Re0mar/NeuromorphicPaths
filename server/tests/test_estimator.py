"""
Covers the estimator module without the model: the checkpoint set, the scale each checkpoint's depth
is in, and the estimator's own wrapping of Depth Anything 3.

DepthEstimator is built with object.__new__ and handed a small fake model, so its constructor, the
one place torch and the weights are loaded, never runs. Everything after loading is ordinary Python.
"""

# Standard library imports
import contextlib
from dataclasses import dataclass
from types import SimpleNamespace

# Third party imports
import numpy as np
import pytest

# Local package imports
from nav.sources.config import DepthCheckpoint, DepthScale, EstimatorConfig, depth_checkpoint_from_name
from nav.sources.estimator import METRIC_MODEL_CANONICAL_FOCAL_PIXELS, DepthEstimator, canonical_focal_for

# The hub names of the checkpoints whose depth is meters, or converts to meters. Written out rather
# than read from the enum, so a member moved to the wrong scale shows up here.
METRIC_NAMES = (
    "depth-anything/DA3METRIC-LARGE",
    "depth-anything/DA3NESTED-GIANT-LARGE",
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
)
# The any-view and monocular checkpoints, right only up to an unknown scale.
RELATIVE_NAMES = (
    "depth-anything/DA3-SMALL",
    "depth-anything/DA3-BASE",
    "depth-anything/DA3-LARGE",
    "depth-anything/DA3-GIANT",
    "depth-anything/DA3MONO-LARGE",
)


@pytest.mark.parametrize(
    "name",
    [
        "depth-anything/DA3-HUGE",
        # A local copy of the nested model. The old substring test read "METRIC" in a path like this
        # as the standalone checkpoint and converted its meters a second time.
        "C:/models/DA3HF-VITG-METRIC_VITL",
        "",
    ],
)
def test_an_unknown_checkpoint_name_is_refused_naming_the_accepted_ones(name: str) -> None:
    """A name the pipeline cannot place has an unknown depth scale, so it is refused with what would work."""
    with pytest.raises(ValueError, match="unknown Depth Anything 3 checkpoint") as refusal:
        depth_checkpoint_from_name(name)

    for accepted in METRIC_NAMES:
        assert accepted in str(refusal.value)


@pytest.mark.parametrize("name", RELATIVE_NAMES)
def test_a_relative_checkpoint_is_refused_because_the_planner_needs_meters(name: str) -> None:
    """Relative depth would put every clearance off by an unknown scale, so it never reaches the scene."""
    with pytest.raises(ValueError, match="relative depth, and the planner's clearance needs meters") as refusal:
        depth_checkpoint_from_name(name)

    assert name in str(refusal.value)
    assert "depth-anything/DA3METRIC-LARGE" in str(refusal.value)


@pytest.mark.parametrize("name", RELATIVE_NAMES)
def test_a_config_holding_a_relative_checkpoint_is_refused(name: str) -> None:
    """A config built in code, not from the command line, gets the same refusal."""
    with pytest.raises(ValueError, match="relative depth"):
        EstimatorConfig(model_name=DepthCheckpoint(name))


def test_a_config_holding_a_name_instead_of_a_checkpoint_is_refused() -> None:
    """A string past the parser is a name nobody checked, which is how the substring bug got in."""
    with pytest.raises(TypeError, match="depth_checkpoint_from_name"):
        EstimatorConfig(model_name="depth-anything/DA3METRIC-LARGE")


def test_every_checkpoint_has_a_depth_scale() -> None:
    """A checkpoint added without a scale would otherwise fall through and be read as meters."""
    assert {checkpoint.depth_scale for checkpoint in DepthCheckpoint} <= set(DepthScale)
    assert {checkpoint.value for checkpoint in DepthCheckpoint} == set(METRIC_NAMES) | set(RELATIVE_NAMES)


def test_every_checkpoint_either_has_a_canonical_focal_answer_or_is_refused() -> None:
    """canonical_focal_for has to cover the whole set, and refuse exactly the relative ones."""
    for checkpoint in DepthCheckpoint:
        if checkpoint.value in RELATIVE_NAMES:
            with pytest.raises(ValueError, match="relative depth"):
                canonical_focal_for(checkpoint)
        else:
            assert canonical_focal_for(checkpoint) in (METRIC_MODEL_CANONICAL_FOCAL_PIXELS, None)


@pytest.mark.parametrize(
    ("name", "expected_focal"),
    [
        # The standalone metric model answers for a 300 px focal, so the source converts it.
        ("depth-anything/DA3METRIC-LARGE", 300.0),
        # The nested model scales its depth to its metric branch inside the library. Converting it
        # again would shrink every distance by the focal over 300.
        ("depth-anything/DA3NESTED-GIANT-LARGE", None),
        ("depth-anything/DA3NESTED-GIANT-LARGE-1.1", None),
    ],
)
def test_only_the_standalone_metric_checkpoint_needs_converting(name: str, expected_focal: float | None) -> None:
    """Which checkpoints need the focal conversion decides whether distances come out in meters."""
    checkpoint = depth_checkpoint_from_name(name)

    assert checkpoint.value == name
    assert canonical_focal_for(checkpoint) == expected_focal


def test_the_default_checkpoint_is_the_standalone_metric_one() -> None:
    """The default fits the 4 GB card and gives meters, and its value is what from_pretrained loads."""
    assert EstimatorConfig().model_name is DepthCheckpoint.METRIC_LARGE
    assert EstimatorConfig().model_name.value == "depth-anything/DA3METRIC-LARGE"


class RecordingInputProcessor:
    """Stands in for Depth Anything 3's InputProcessor, remembering how each call asked to run."""

    def __init__(self) -> None:
        self.sequential_flags: list[bool] = []

    def __call__(self, images: list, *, sequential: bool = False, **options) -> list:
        self.sequential_flags.append(sequential)
        return images


def _estimator_without_loading(model: object, config: EstimatorConfig | None = None) -> DepthEstimator:
    """A DepthEstimator whose constructor never ran, so no torch and no weights."""
    estimator = object.__new__(DepthEstimator)
    estimator._model = model
    estimator._config = config if config is not None else EstimatorConfig()
    # inference_mode is the only part of torch the estimate path touches.
    estimator._torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
    estimator.device = "cpu"
    return estimator


def test_a_model_with_no_input_processor_is_warned_about_and_still_runs(caplog: pytest.LogCaptureFixture) -> None:
    """A later library that renamed the processor should leak threads again, not stop the run."""
    model = SimpleNamespace()
    estimator = _estimator_without_loading(model)

    with caplog.at_level("WARNING", logger="nav.sources.estimator"):
        estimator._preprocess_on_the_calling_thread()

    assert any("no input_processor" in record.message for record in caplog.records)
    assert not hasattr(model, "input_processor")


def test_preprocessing_is_made_sequential_so_no_thread_pool_is_built_per_frame() -> None:
    """Each pool left about 7 threads behind, and on the walk that put the glasses' decoder seconds behind."""
    processor = RecordingInputProcessor()
    model = SimpleNamespace(input_processor=processor)
    estimator = _estimator_without_loading(model)

    estimator._preprocess_on_the_calling_thread()
    model.input_processor(["frame"])

    assert processor.sequential_flags == [True]


@dataclass
class FakePrediction:
    """The fields of Depth Anything 3's Prediction the estimator reads, each batched by one image."""

    depth: np.ndarray
    conf: np.ndarray | None
    intrinsics: np.ndarray | None


class FakeInferenceModel:
    """Returns a fixed prediction, and remembers the process resolution it was asked for."""

    def __init__(self, prediction: FakePrediction) -> None:
        self._prediction = prediction
        self.process_resolutions: list[int] = []

    def inference(self, images: list, process_res: int) -> FakePrediction:
        self.process_resolutions.append(process_res)
        return self._prediction


def test_an_estimate_without_model_intrinsics_carries_none_and_the_metric_focal() -> None:
    """Choosing a stand-in camera belongs to the composed source, which knows if the camera has a calibration."""
    raw_depth = np.full((1, 4, 6), 2.0, dtype=np.float32)
    model = FakeInferenceModel(FakePrediction(depth=raw_depth, conf=None, intrinsics=None))
    estimator = _estimator_without_loading(model, EstimatorConfig(model_name=DepthCheckpoint.METRIC_LARGE, process_resolution=280))

    estimate = estimator.estimate(np.zeros((8, 12, 3), dtype=np.uint8))

    assert estimate.intrinsics is None
    assert estimate.confidence is None
    assert estimate.canonical_focal_pixels == METRIC_MODEL_CANONICAL_FOCAL_PIXELS
    assert estimate.depth == pytest.approx(raw_depth[0])
    assert model.process_resolutions == [280]


def test_an_estimate_from_the_nested_checkpoint_is_already_meters() -> None:
    """The nested checkpoint's depth must not be converted again by the composed source."""
    model_intrinsics = np.array([[[5.0, 0.0, 3.0], [0.0, 5.0, 2.0], [0.0, 0.0, 1.0]]])
    prediction = FakePrediction(
        depth=np.ones((1, 4, 6), dtype=np.float32),
        conf=np.ones((1, 4, 6), dtype=np.float32),
        intrinsics=model_intrinsics,
    )
    estimator = _estimator_without_loading(
        FakeInferenceModel(prediction),
        EstimatorConfig(model_name=DepthCheckpoint.NESTED_GIANT_LARGE),
    )

    estimate = estimator.estimate(np.zeros((8, 12, 3), dtype=np.uint8))

    assert estimate.canonical_focal_pixels is None
    assert estimate.intrinsics == pytest.approx(model_intrinsics[0])
    assert estimate.confidence is not None
