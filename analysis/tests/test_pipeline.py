"""
End-to-end: a real photo through the app's model, the outline trace and the point-of-view
estimate, both as library calls and through the geometry command.

Uses IMG_2319.JPG from OldAppEnvrionmentStuff/artifacts/images, an iPhone 12 ultra-wide shot of a straight footpath
held at roughly chest height. The ranges are loose on purpose. They catch a broken pipeline,
not a small change in the model.
"""

# Standard library imports
import csv
import subprocess
import sys
from pathlib import Path

# Third party imports
import cv2
import numpy
import pytest

# Local package imports
from neuromorphicpaths_analysis.detector import PathDetector
from neuromorphicpaths_analysis.geometry import CAPTURE_PROFILES, CaptureProfileName, PovStatus, estimate_pov
from neuromorphicpaths_analysis.recordings import load_image

PROFILE_NAME = CaptureProfileName.IPHONE_12_ULTRA_WIDE


@pytest.fixture
def photo_path(test_photos_dir: Path) -> Path:
    photo = test_photos_dir / "IMG_2319.JPG"
    assert photo.exists(), f"test photo missing at {photo}"
    return photo


def run_geometry_command(model_path: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, "-m", "neuromorphicpaths_analysis.geometry", *arguments,
            "--profile", PROFILE_NAME.value, "--model", str(model_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def test_real_photo_gives_a_reliable_point_of_view(model_path: Path, photo_path: Path) -> None:
    result = PathDetector(model_path).detect_path(load_image(photo_path))
    estimate = estimate_pov(result, CAPTURE_PROFILES[PROFILE_NAME])

    assert estimate.status is PovStatus.OK
    assert estimate.reliable
    assert 10 < estimate.pitch_deg < 35  # tilted down at the path, not level and not at the feet
    assert abs(estimate.heading_deg) < 10  # pointed along the path
    assert 0.4 < estimate.position_fraction < 0.8  # standing on the path, a little right of center
    assert 0.8 < estimate.camera_height_m < 2.0  # hand-held, with the profile's 1.5 m path width


def test_command_prints_and_saves_a_single_image(tmp_path: Path, model_path: Path, photo_path: Path) -> None:
    csv_path = tmp_path / "pov.csv"

    completed = run_geometry_command(model_path, str(photo_path), "--csv", str(csv_path))

    assert "IMG_2319.JPG" in completed.stdout
    rows = read_csv(csv_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "OK" and rows[0]["reliable"] == "True"
    assert rows[0]["time_s"] == ""


def test_command_samples_a_video(tmp_path: Path, model_path: Path, photo_path: Path) -> None:
    # Five frames of the same photo at 5 fps. Every second frame keeps frames 0, 2 and 4.
    video_path = tmp_path / "walk.mp4"
    frame = cv2.resize(cv2.cvtColor(numpy.asarray(load_image(photo_path)), cv2.COLOR_RGB2BGR), (1008, 756))
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (1008, 756))
    for _ in range(5):
        writer.write(frame)
    writer.release()
    csv_path = tmp_path / "pov.csv"

    run_geometry_command(model_path, str(video_path), "--every", "2", "--csv", str(csv_path))

    rows = read_csv(csv_path)
    assert [row["frame"] for row in rows] == ["walk@000000", "walk@000002", "walk@000004"]
    assert [float(row["time_s"]) for row in rows] == pytest.approx([0.0, 0.4, 0.8])
    assert all(row["status"] == "OK" for row in rows)
