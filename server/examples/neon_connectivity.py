"""
Shows the Neon's scene camera with the gaze point and the eye images drawn on it.

The old ObserverTest.py, with the device address from the command line instead of a constant.
Run it to see that the glasses are streaming before starting a real run.

    python examples/neon_connectivity.py                    # discover the device
    python examples/neon_connectivity.py --address 10.0.0.5  # when discovery is blocked

Escape closes the window.
"""

# Standard library imports
import argparse
import sys

# Third party imports
import cv2

# Local package imports
from nav.sources.neon_live import apply_opencv_pyav_import_workaround

GAZE_RADIUS_PIXELS = 80
GAZE_COLOR = (0, 0, 255)
ESCAPE = 27


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show the Neon's scene camera with gaze and eyes.")
    parser.add_argument("--address", help="the Neon's address, or omit it to discover the device")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--timeout", type=float, default=10.0, help="seconds to search for the device")
    arguments = parser.parse_args(argv)

    apply_opencv_pyav_import_workaround()
    from pupil_labs.realtime_api.simple import Device, discover_one_device

    if arguments.address is None:
        print(f"discovering a Neon, up to {arguments.timeout:.0f} s")
        device = discover_one_device(max_search_duration_seconds=arguments.timeout)
        if device is None:
            print("no device found. If the network blocks mDNS, pass --address from the Companion app's streaming screen")
            return 1
    else:
        device = Device(address=arguments.address, port=arguments.port)
    print(f"connected to {device}")

    try:
        while True:
            matched = device.receive_matched_scene_and_eyes_video_frames_and_gaze()
            if not matched:
                print("no matched frame this round")
                continue

            scene = matched.scene.bgr_pixels
            cv2.circle(scene, (int(matched.gaze.x), int(matched.gaze.y)), GAZE_RADIUS_PIXELS, GAZE_COLOR, 15)
            eyes = matched.eyes.bgr_pixels
            height, width, _ = eyes.shape
            scene[:height, :width, :] = eyes

            cv2.imshow("Neon scene camera with eyes and gaze", scene)
            if cv2.waitKey(1) & 0xFF == ESCAPE:
                break
    except KeyboardInterrupt:
        pass
    finally:
        device.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
