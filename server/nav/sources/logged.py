"""
Replays a frame log written by the recording tap.

Nothing is skipped. A frame that will not decode stops the replay, because a replay that quietly
drops frames gives a different answer from the run it is meant to reproduce, and the difference is
invisible.
"""

# Standard library imports
import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path

# Local package imports
from nav.sources.framecodec import (
    INDEX_FILENAME,
    FrameDecodeError,
    decode_frame,
    read_message_from_file,
)
from nav.types import DepthFrame

log = logging.getLogger(__name__)


class LoggedDepthFrameSource:
    """Yields the frames of a recorded log, in sequence order."""

    def __init__(self, log_dir: Path, realtime: bool = False) -> None:
        self._log_dir = Path(log_dir)
        self._realtime = realtime

    def _read_index(self) -> list[dict]:
        index_path = self._log_dir / INDEX_FILENAME
        if not index_path.is_file():
            raise FileNotFoundError(f"no {INDEX_FILENAME} in {self._log_dir}, this is not a frame log")

        entries: list[dict] = []
        for line_number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as json_error:
                raise FrameDecodeError(f"{INDEX_FILENAME} line {line_number} is not valid JSON: {json_error}") from json_error
            for key in ("sequence", "timestamp_seconds", "file"):
                if key not in entry:
                    raise FrameDecodeError(f"{INDEX_FILENAME} line {line_number} is missing {key!r}")
            entries.append(entry)

        if not entries:
            raise FrameDecodeError(f"{index_path} is empty, there is nothing to replay")

        # Written in order, but sorted anyway so a hand-edited log replays in the order it claims.
        return sorted(entries, key=lambda entry: entry["sequence"])

    def frames(self) -> Iterator[DepthFrame]:
        entries = self._read_index()
        log.info("replaying %d frames from %s", len(entries), self._log_dir)

        previous_timestamp: float | None = None
        for entry in entries:
            frame_path = self._log_dir / entry["file"]
            if not frame_path.is_file():
                raise FrameDecodeError(f"{INDEX_FILENAME} lists {entry['file']} and it is not in {self._log_dir}")

            try:
                frame = decode_frame(read_message_from_file(frame_path))
            except FrameDecodeError as decode_error:
                # Named rather than wrapped in a bare raise, so the failing file is in the message.
                raise FrameDecodeError(f"{entry['file']} did not decode: {decode_error}") from decode_error

            if self._realtime and previous_timestamp is not None:
                gap_seconds = frame.timestamp_seconds - previous_timestamp
                if gap_seconds > 0:
                    time.sleep(gap_seconds)
            previous_timestamp = frame.timestamp_seconds

            yield frame

    def close(self) -> None:
        """Nothing to release. Present because the loop closes every source it was given."""
