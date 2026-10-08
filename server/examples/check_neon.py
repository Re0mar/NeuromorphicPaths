"""
Checks that a Neon is reachable and streaming, and exits non-zero when it is not.

Run by hand when the glasses are present. Not a test, because it can never run unattended on
the laptop, and a test that always skips reports success while asserting nothing.

    python examples/check_neon.py                      # discover the device over the network
    python examples/check_neon.py --neon-address 10.0.0.5   # when the network blocks discovery
    python examples/check_neon.py --neon-replay frame_logs/captures/neon_walk_2   # a recorded capture

The address and port are read through the same command line parser the pipeline uses, so what
works here works for `python -m nav --source neon_live`.

After the first frame it watches the IMU for a few seconds. A walk on an IMU that sends only empty
orientations has no pose on any frame, so every floor is fitted against a level head. That happened
on 2026-10-05 for minutes at a time, and a session found out from one warning line mid-walk.
"""

# Standard library imports
import argparse
import logging
import sys
import time

# Local package imports
from nav.config import build_run_config
from nav.pose.imu_orientation import IMU_MATCH_TOLERANCE_SECONDS
from nav.sources.neon_device import DeviceImuStatus
from nav.sources.neon_live import NeonCalibrationError, NeonLiveRgbSource
from nav.sources.rgb import RgbFrame

EXIT_READY = 0
EXIT_NOT_READY = 1

# How long to watch the IMU after the first frame. About 300 readings at the Neon's 110 Hz, enough to
# tell a stream of empty readings from a few dropped ones, and short enough to run before every walk.
IMU_WATCH_SECONDS = 3.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check that a Neon is reachable and streaming.")
    parser.add_argument("--neon-address", help="the Neon's address, or omit it to discover the device")
    parser.add_argument("--neon-port", type=int, default=8080)
    parser.add_argument("--neon-replay", help="check a capture folder written by capture_neon_stream.py instead of the glasses")
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="seconds to wait for discovery. Waiting for the first frame warns every few seconds, and Ctrl+C stops it",
    )
    arguments = parser.parse_args(argv)
    # The source logs what it measured on the way in, the clock round trip and the undistortion
    # crop among it, and those lines are half of what this check is for.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    pipeline_argv = ["--source", "neon_live", "--sink", "none", "--neon-port", str(arguments.neon_port)]
    if arguments.neon_address is not None:
        pipeline_argv += ["--neon-address", arguments.neon_address]
    if arguments.neon_replay is not None:
        pipeline_argv += ["--neon-replay", arguments.neon_replay]
    config = build_run_config(pipeline_argv)
    assert config.neon is not None
    neon_config = type(config.neon)(
        address=config.neon.address,
        port=config.neon.port,
        discovery_timeout_seconds=arguments.timeout,
        replay_dir=config.neon.replay_dir,
    )

    source = NeonLiveRgbSource(neon_config)
    started = time.monotonic()
    try:
        try:
            frame = next(source.frames())
        except ConnectionError as unreachable:
            print(f"no device: {unreachable}")
            if arguments.neon_replay is not None:
                print(f"the capture in {arguments.neon_replay} was tried")
            elif arguments.neon_address is None:
                print("discovery was tried. If this network blocks mDNS, read the address off the Companion app's streaming screen and pass --neon-address")
            else:
                print(f"a direct connection to {arguments.neon_address}:{arguments.neon_port} was tried")
            return EXIT_NOT_READY
        except NeonCalibrationError as no_calibration:
            # Connected, but without the calibration the pipeline would be guessing where things are.
            print(f"connected, but {no_calibration}")
            return EXIT_NOT_READY
        except StopIteration:
            print(f"connected but no frame arrived within {time.monotonic() - started:.1f} s")
            return EXIT_NOT_READY

        _print_frame(frame, source, time.monotonic() - started)
        # The device process counts every IMU reading as it arrives, frames pulled or not, so the
        # watch is a wait and one question at the end.
        time.sleep(IMU_WATCH_SECONDS)
        try:
            status = source.imu_status()
        except ConnectionError as stopped:
            print(f"  IMU            the device stopped answering while the IMU was watched: {stopped}")
            return EXIT_NOT_READY
        return _judge_imu(status)
    finally:
        source.close()


def _print_frame(frame: RgbFrame, source: NeonLiveRgbSource, elapsed_seconds: float) -> None:
    print(f"frame after {elapsed_seconds:.1f} s")
    print(f"  timestamp      {frame.timestamp_seconds:.3f}")
    print(f"  image          {frame.image_rgb.shape[1]}x{frame.image_rgb.shape[0]}")
    print(f"  gaze           {'yes' if frame.gaze_pixel is not None else 'no, or cropped off by undistortion'}")
    # The scene and IMU streams start together, so a first frame can be captured before the IMU's
    # first reading. The line below says whether the IMU is sending anything usable at all.
    no_pose = (
        f"no, no usable IMU reading within {IMU_MATCH_TOLERANCE_SECONDS * 1000:.0f} ms of the frame's capture. "
        "A first frame can precede the IMU's first reading. The IMU line below says whether the stream is usable"
    )
    print(f"  gravity pose   {'yes' if frame.pose is not None else no_pose}")

    undistorter = source.undistorter
    if undistorter is not None and frame.camera_matrix is not None:
        matrix = frame.camera_matrix
        print(f"  camera matrix  fx {matrix[0, 0]:.1f}  fy {matrix[1, 1]:.1f}  cx {matrix[0, 2]:.1f}  cy {matrix[1, 2]:.1f}, after undistortion")
        lost = undistorter.field_of_view_before_degrees - undistorter.field_of_view_after_degrees
        print(
            f"  field of view  {undistorter.field_of_view_before_degrees:.1f} deg wide as delivered, "
            f"{undistorter.field_of_view_after_degrees:.1f} deg after undistortion, {lost:.1f} deg cropped"
        )
    if source.clock_offset_seconds is None:
        print("  clock offset   not measured, so capture times and the capture share will be missing. See the log for why")
    else:
        print(f"  clock offset   {source.clock_offset_seconds * 1000:.1f} ms, laptop minus Neon. The round trip is in the log")


def _judge_imu(status: DeviceImuStatus) -> int:
    """Print what the IMU sent, and say whether a walk could have a pose on its frames."""
    readings, empty = status.readings, status.empty_readings
    if readings == 0:
        print("  IMU            no readings arrived since connect. The IMU stream is silent, so no frame would have a pose")
        print("IMU NOT READY: restart streaming in the Companion app and run this again before walking")
        return EXIT_NOT_READY
    if empty == readings:
        print(f"  IMU            all {readings} readings since connect were empty orientations")
        if status.unstamped_empty_readings == empty:
            # The 2026-10-05 signature: the packets decode to nothing at all, timestamp included.
            print("                 none carried a timestamp either, so every packet decoded to nothing at all")
        print("IMU NOT READY: every frame would go without a pose, and every floor would be fitted against a level head.")
        # Reproduced on 2026-10-08: the phone serves the IMU stream to one client, and a second one
        # connected beside it gets packets that decode to nothing. It clears the moment the first
        # client's connection is gone. A crashed run takes its socket with it, so the culprit is a
        # run that is still alive, such as one hung in another terminal. Pupil Labs' answer to the
        # same report, pl-realtime-api issue 71, force-stopping the Companion app, drops that client
        # when it can't be found.
        print("Another program is still connected to the glasses' IMU stream, and the phone sends a second client nothing.")
        print("Close the other run (a hung terminal, a second script), or force-stop and restart the Companion app, then run this check again")
        return EXIT_NOT_READY
    if empty:
        print(f"  IMU            {readings} readings since connect, {empty} empty ({empty / readings:.1%}). Frames near those have no pose")
    else:
        print(f"  IMU            {readings} readings since connect, none empty")
    return EXIT_READY


if __name__ == "__main__":
    sys.exit(main())
