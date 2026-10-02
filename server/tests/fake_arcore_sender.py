"""
Stands in for the Pixel app, sending synthetic depth frames in the real wire format.

Every frame goes through the same encode_frame the laptop decodes with. Nothing in here builds
bytes by hand except send_raw, which exists so the negative tests can put garbage on the wire,
and which no shipping code may call.

Run it from a second terminal to drive the live source without a phone:

    python tests/fake_arcore_sender.py --port 9000 --count 100 --gap 0.1
"""

# Standard library imports
import argparse
import socket
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

# Third party imports
import numpy as np

# The tests directory is not a package, so running this as a script needs the stubs beside it.
sys.path.insert(0, str(Path(__file__).parent))

# Local package imports
from nav.sources.estimator import fallback_intrinsics
from nav.sources.framecodec import encode_frame, write_message
from nav.types import DepthFrame, Plane, Pose

DEFAULT_DEPTH_SHAPE = (120, 160)
CAMERA_HEIGHT_METERS = 1.6
BOX_DISTANCE_METERS = 3.0
FORWARD_STEP_METERS_PER_FRAME = 0.05
FRAME_INTERVAL_SECONDS = 1.0 / 30.0


def synthetic_frames(count: int, depth_shape: tuple[int, int] = DEFAULT_DEPTH_SHAPE) -> Iterator[DepthFrame]:
    """
    A level camera over a flat floor, a box standing ahead, and a pose walking forward.

    The floor comes from the pinhole model rather than a painted gradient, so the scene's plane
    fit recovers a real plane from it and the ground plane sent alongside is the same one.

    :param count: How many frames to produce.
    :param depth_shape: (rows, columns) of the depth image.
    :return: Frames with has_position True and a known ground plane.
    :rtype: Iterator[DepthFrame]
    """
    height, width = depth_shape
    intrinsics = fallback_intrinsics(height, width)
    focal_y = intrinsics[1, 1]
    principal_y = intrinsics[1, 2]

    rows = np.arange(height, dtype=np.float64)[:, None]
    below_horizon = rows - principal_y
    with np.errstate(divide="ignore", invalid="ignore"):
        floor = np.where(below_horizon > 0, CAMERA_HEIGHT_METERS * focal_y / below_horizon, 0.0)
    floor_image = np.broadcast_to(floor, depth_shape).astype(np.float32)

    # Camera axes: y points down, so the floor is at +y and its normal points at -y. The plane
    # satisfies normal . p + offset == 0, which puts the offset at +CAMERA_HEIGHT.
    ground_plane = Plane(normal=np.array([0.0, -1.0, 0.0]), offset_meters=CAMERA_HEIGHT_METERS)

    box_rows = slice(height // 3, height // 2)
    box_columns = slice(width * 2 // 5, width * 3 // 5)

    for index in range(count):
        depth = floor_image.copy()
        # The box gets nearer as the walker advances, which is what gives the scene's history
        # something to measure.
        depth[box_rows, box_columns] = max(0.5, BOX_DISTANCE_METERS - index * FORWARD_STEP_METERS_PER_FRAME)
        yield DepthFrame(
            timestamp_seconds=index * FRAME_INTERVAL_SECONDS,
            depth_meters=depth,
            intrinsics=intrinsics,
            # The app's convention, which the wire format states: the world has y up, so a level
            # camera's orientation is a half turn about x, and walking forward is along world -z.
            # A fake that sent an identity orientation would describe a world with y down, and
            # the floor fit, which reads gravity from this pose, would look for the floor overhead.
            pose=Pose(
                orientation=np.array([0.0, 1.0, 0.0, 0.0]),
                position=np.array([0.0, 0.0, -index * FORWARD_STEP_METERS_PER_FRAME]),
                has_position=True,
                orientation_is_gravity_aligned=True,
            ),
            ground_plane=ground_plane,
            gaze_pixel=None,
        )


class FakeArCoreSender:
    """One connection to the laptop, over which frames and, in tests, raw bytes can be sent."""

    def __init__(self, address: str, port: int, connect_timeout_seconds: float = 5.0) -> None:
        self._address = address
        self._port = port
        self._connect_timeout_seconds = connect_timeout_seconds
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        self._sock = socket.create_connection((self._address, self._port), timeout=self._connect_timeout_seconds)

    def send_frame(self, frame: DepthFrame) -> None:
        # A FrameEncodeError here means this script built a bad frame. It propagates, because that
        # is our bug and not something the wire did.
        write_message(self._require_socket(), encode_frame(frame))

    def send_raw(self, payload: bytes) -> None:
        """Put arbitrary bytes on the wire. For negative tests only. Nothing shipping calls this."""
        self._require_socket().sendall(payload)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _require_socket(self) -> socket.socket:
        if self._sock is None:
            raise ConnectionError("sender is not connected, call connect() first")
        return self._sock


def send_frames(address: str, port: int, frames: Iterable[DepthFrame], frame_gap_seconds: float = 0.0) -> int:
    """
    Connect, send every frame, disconnect. Returns how many were sent.

    :param address: The laptop's address.
    :param port: The laptop's listening port.
    :param frames: The frames to send, encoded one by one.
    :param frame_gap_seconds: Sleep between frames, to look like a camera rather than a burst.
    :return: Frames sent.
    :rtype: int
    """
    sender = FakeArCoreSender(address, port)
    sender.connect()
    sent = 0
    try:
        for frame in frames:
            sender.send_frame(frame)
            sent += 1
            if frame_gap_seconds > 0:
                time.sleep(frame_gap_seconds)
    finally:
        sender.close()
    return sent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send synthetic ARCore depth frames to the laptop pipeline.")
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--gap", type=float, default=FRAME_INTERVAL_SECONDS, help="seconds between frames")
    arguments = parser.parse_args(argv)

    sent = send_frames(arguments.address, arguments.port, synthetic_frames(arguments.count), arguments.gap)
    print(f"sent {sent} frames to {arguments.address}:{arguments.port}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
