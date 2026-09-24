"""Command line options for choosing a capture profile and overriding its fields for one run."""

# Standard library imports
import argparse
import dataclasses

# Local package imports
from neuromorphicpaths_analysis.geometry.capture_profiles import (
    CAPTURE_PROFILES,
    CaptureProfile,
    CaptureProfileName,
)


def add_profile_arguments(parser: argparse.ArgumentParser, default: CaptureProfileName) -> None:
    parser.add_argument("--profile", choices=[name.value for name in CaptureProfileName], default=default.value)
    parser.add_argument("--focal-length", type=float, help="focal length in pixels at the profile's reference width")
    parser.add_argument("--camera-height", type=float, help="known camera height in meters")
    parser.add_argument("--path-width", type=float, help="assumed path width in meters")


def profile_from_arguments(arguments: argparse.Namespace) -> CaptureProfile:
    """
    The chosen profile with any command line overrides applied.

    :rtype: CaptureProfile
    """
    # The command line is a serialization boundary, so the name becomes the enum here and only here.
    profile = CAPTURE_PROFILES[CaptureProfileName(arguments.profile)]
    overrides = {
        field: value
        for field, value in (
            ("focal_length_px", arguments.focal_length),
            ("camera_height_m", arguments.camera_height),
            ("path_width_m", arguments.path_width),
        )
        if value is not None
    }
    return dataclasses.replace(profile, **overrides)
