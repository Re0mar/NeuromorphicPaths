"""Command line option for choosing which app's decoding the detector copies."""

# Standard library imports
import argparse

# Local package imports
from neuromorphicpaths_analysis.detector.path_detector import DecoderPreset


def add_decoder_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--decoder",
        choices=[preset.value for preset in DecoderPreset],
        default=DecoderPreset.FIRST_APP.value,
        help="which app's way of turning the model output into a mask and edges",
    )


def decoder_from_arguments(arguments: argparse.Namespace) -> DecoderPreset:
    # The command line is a serialization boundary, so the name becomes the enum here and only here.
    return DecoderPreset(arguments.decoder)
