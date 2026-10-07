"""
Stands in for the Pixel app on the path port, reading the paths the laptop sends back.

The twin of fake_arcore_sender.py. The timing log stamps a path as sent only once the phone
sink has written it to a phone's socket, so a run with no phone attached has no send times. With
this reading the path port, a replay measures the laptop's whole share through the real sink.

Start it after the laptop, or give it --wait to keep retrying while the laptop starts:

    python tests/fake_path_reader.py --port 9100 --wait 30
"""

# Standard library imports
import argparse
import socket
import sys
import time

# Local package imports
from nav.sources.framecodec import StreamClosedError, decode_path, read_message

CONNECT_RETRY_SECONDS = 0.1


def read_paths(address: str, port: int, wait_seconds: float = 0.0) -> int:
    """
    Connect, read and decode paths until the laptop closes the connection. Returns how many were read.

    :param address: The laptop's address.
    :param port: The laptop's path port.
    :param wait_seconds: Keep retrying a refused connection this long, so the reader can start first.
    :return: Paths read.
    :rtype: int
    """
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            connection = socket.create_connection((address, port))
            break
        except ConnectionRefusedError:
            # The laptop isn't listening yet. Expected when the two are started together.
            if time.monotonic() >= deadline:
                raise
            time.sleep(CONNECT_RETRY_SECONDS)
    read = 0
    try:
        while True:
            try:
                decode_path(read_message(connection))
            except StreamClosedError:
                # The laptop closed the connection at the end of its run. That's the normal end.
                return read
            except (ConnectionResetError, ConnectionAbortedError):
                # Windows resets rather than closes when the laptop process exits. Also an end.
                return read
            read += 1
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read the paths the laptop sends to the phone, standing in for the app.")
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--wait", type=float, default=0.0, help="seconds to keep retrying the connection while the laptop starts")
    arguments = parser.parse_args(argv)

    read = read_paths(arguments.address, arguments.port, arguments.wait)
    print(f"read {read} paths from {arguments.address}:{arguments.port}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
