"""
Turns a Neon Companion recording into a capture, so `--neon-replay` plays it like a live walk.

    python examples/recording_to_capture.py "<Companion recording folder>" frame_logs/captures/<name>
    python -m nav --source neon_live --neon-replay frame_logs/captures/<name> --sink web

The first line runs once per recording, in seconds rather than minutes, since nothing is decoded.
The second is the demo. The page's video is then the recording's own, in step with the arrow, the
view from above and the depth picture, because the laptop plans on the frames it shows.

To move the demo to another recording, convert that one and point `--neon-replay` at it. Depth is
estimated from the video on every run, so there is no depth to regenerate. The output folder must
not exist yet, so an earlier capture is never overwritten.
"""

# Standard library imports
import argparse
import sys
from pathlib import Path

# Local package imports
from nav.sources.neon_device import apply_opencv_pyav_import_workaround


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert a Neon Companion recording into a capture for --neon-replay.")
    parser.add_argument("recording_dir", help="the recording as the Companion app exports it, with its info.json")
    parser.add_argument("out_dir", help="a new folder for the capture")
    arguments = parser.parse_args(argv)

    # Before the recording reader loads the Pupil Labs package, which brings PyAV with it.
    apply_opencv_pyav_import_workaround()
    # Imported after the workaround, for the same reason.
    from nav.sources.recording_capture import convert_recording

    counts = convert_recording(Path(arguments.recording_dir), Path(arguments.out_dir))
    print(f"wrote {arguments.out_dir}: {counts['scene']} scene packets, {counts['imu']} IMU readings, {counts['gaze']} gaze samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
