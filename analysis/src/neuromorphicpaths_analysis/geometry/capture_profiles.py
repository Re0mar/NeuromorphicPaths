"""
Known capture setups: the camera, where it sits and what path it usually sees.

A profile is picked by name, and any field can be overridden for one run with
dataclasses.replace, for example a participant's measured eye height on the Neon profile.
Adding a setup means adding a member to CaptureProfileName and an entry to CAPTURE_PROFILES.
"""

# Standard library imports
import math
from dataclasses import dataclass
from enum import Enum


class CaptureProfileName(Enum):
    BELGIAN_DATASET = "belgian-dataset"
    NEON = "neon"
    META_GLASSES = "meta-glasses"
    IPHONE_12_ULTRA_WIDE = "iphone-12-ultra-wide"


@dataclass(frozen=True)
class CaptureProfile:
    """
    What is known about one capture setup. None means unknown for this setup.

    With camera_height_m known, the estimate reports the path's width. With path_width_m known,
    it reports the camera's height. With both, it reports both, which checks one against the
    other.
    """

    name: CaptureProfileName
    # Focal length in pixels, measured on frames reference_width_px wide. Scaled to other frame
    # widths, which holds as long as the frame is a resize of the same sensor crop.
    focal_length_px: float | None
    reference_width_px: int
    camera_height_m: float | None
    path_width_m: float | None
    note: str

    def focal_length_for(self, frame_width_px: int) -> float | None:
        if self.focal_length_px is None:
            return None
        return self.focal_length_px * frame_width_px / self.reference_width_px


# A typical Belgian sidewalk. The dataset's streets vary, so this is the value to override
# first when a frame's height estimate looks off.
DEFAULT_PATH_WIDTH_M = 1.5

CAPTURE_PROFILES: dict[CaptureProfileName, CaptureProfile] = {
    CaptureProfileName.BELGIAN_DATASET: CaptureProfile(
        name=CaptureProfileName.BELGIAN_DATASET,
        # No EXIF and mixed cameras. Assumed to be a phone main camera, roughly 67 degrees across.
        focal_length_px=1450.0,
        reference_width_px=1920,
        camera_height_m=None,
        path_width_m=DEFAULT_PATH_WIDTH_M,
        note="Assumed focal length. Height varies by frame, about 0.2 to 1 m.",
    ),
    CaptureProfileName.NEON: CaptureProfile(
        name=CaptureProfileName.NEON,
        # Each headset ships its own calibration in the recording's scene_camera.json. Fill this
        # from that file, and undistort frames first, since the wide lens bends straight edges.
        focal_length_px=None,
        reference_width_px=1600,
        # Eye height differs per participant. Override with the measured value.
        camera_height_m=None,
        path_width_m=DEFAULT_PATH_WIDTH_M,
        note="Set focal length from the device calibration and eye height per participant.",
    ),
    CaptureProfileName.META_GLASSES: CaptureProfile(
        name=CaptureProfileName.META_GLASSES,
        # Not published for the Wearables Device Access Toolkit stream. Needs a checkerboard calibration.
        focal_length_px=None,
        reference_width_px=504,
        camera_height_m=None,
        path_width_m=DEFAULT_PATH_WIDTH_M,
        note="Portrait 504x896 stream. Focal length unknown until calibrated.",
    ),
    CaptureProfileName.IPHONE_12_ULTRA_WIDE: CaptureProfile(
        name=CaptureProfileName.IPHONE_12_ULTRA_WIDE,
        # EXIF gives a 14 mm equivalent, measured against the 43.27 mm diagonal of a 35 mm film
        # frame, so focal length is 14 times the 5040 px image diagonal over 43.27. Dividing by the
        # 36 mm film width instead is only right for 3:2 images, and about 4% short for this 4:3 one.
        # The phone corrects this lens's distortion in the JPEG, so straight edges stay straight.
        focal_length_px=14 * math.hypot(4032, 3024) / 43.27,
        reference_width_px=4032,
        camera_height_m=None,
        path_width_m=DEFAULT_PATH_WIDTH_M,
        note="The 0.5x lens, as in the repository's test photos. Hand-held, so height varies.",
    ),
}
