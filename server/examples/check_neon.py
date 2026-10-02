"""
Checks that a Neon is reachable and streaming, and exits non-zero when it is not.

Run by hand when the glasses are present. Not a test, because it can never run unattended on
the laptop, and a test that always skips reports success while asserting nothing.

    python examples/check_neon.py                      # discover the device over the network
    python examples/check_neon.py --neon-address 10.0.0.5   # when the network blocks discovery

The address and port are read through the same command line parser the pipeline uses, so what
works here works for `python -m nav --source neon_live`.
"""

# Standard library imports
import argparse
import sys
import time

# Local package imports
from nav.config import build_run_config
from nav.sources.neon_live import NeonLiveRgbSource


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check that a Neon is reachable and streaming.")
    parser.add_argument("--neon-address", help="the Neon's address, or omit it to discover the device")
    parser.add_argument("--neon-port", type=int, default=8080)
    parser.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for discovery and for a frame")
    arguments = parser.parse_args(argv)

    pipeline_argv = ["--source", "neon_live", "--sink", "none", "--neon-port", str(arguments.neon_port)]
    if arguments.neon_address is not None:
        pipeline_argv += ["--neon-address", arguments.neon_address]
    config = build_run_config(pipeline_argv)
    assert config.neon is not None
    neon_config = type(config.neon)(address=config.neon.address, port=config.neon.port, discovery_timeout_seconds=arguments.timeout)

    source = NeonLiveRgbSource(neon_config)
    started = time.monotonic()
    try:
        frames = source.frames()
        frame = next(frames)
    except ConnectionError as unreachable:
        print(f"no device: {unreachable}")
        if arguments.neon_address is None:
            print("discovery was tried. If this network blocks mDNS, read the address off the Companion app's streaming screen and pass --neon-address")
        else:
            print(f"a direct connection to {arguments.neon_address}:{arguments.neon_port} was tried")
        return 1
    except StopIteration:
        print(f"connected but no frame arrived within {time.monotonic() - started:.1f} s")
        return 1
    finally:
        source.close()

    print(f"frame after {time.monotonic() - started:.1f} s")
    print(f"  timestamp      {frame.timestamp_seconds:.3f}")
    print(f"  image          {frame.image_rgb.shape[1]}x{frame.image_rgb.shape[0]}")
    print(f"  gaze           {'yes' if frame.gaze_pixel is not None else 'no'}")
    print(f"  imu orientation {'yes' if frame.imu_orientation_wxyz is not None else 'not yet, the IMU stream lags the first scene frame'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
