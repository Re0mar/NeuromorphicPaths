"""
Writes every frame to disk as it passes, without changing what the pipeline sees.

Wraps any depth source, so a live walk can be recorded and then replayed through the same scene
and planner code. The tap and the replay source share one codec, which is what makes the replay
worth anything.
"""

# Standard library imports
import json
import logging
from collections.abc import Iterator
from pathlib import Path

# Local package imports
from nav.runtime.textio import append_text_lf
from nav.sources.framecodec import (
    FRAME_FILENAME_TEMPLATE,
    INDEX_FILENAME,
    encode_frame,
    write_message_to_file,
)
from nav.types import DepthFrame, DepthFrameSource

log = logging.getLogger(__name__)


class RecordingTap:
    """A depth source that writes what it yields, and yields it unchanged."""

    def __init__(self, inner: DepthFrameSource, log_dir: Path) -> None:
        self._inner = inner
        self._log_dir = Path(log_dir)
        self._index_path = self._log_dir / INDEX_FILENAME

    def _prepare_directory(self) -> None:
        if self._log_dir.exists() and any(self._log_dir.iterdir()):
            # Two runs interleaved in one log produce a sequence that reads back as one recording
            # and is not one. Refusing is cheaper than discovering that during analysis.
            raise FileExistsError(f"{self._log_dir} already has files in it, pick an empty directory")
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def frames(self) -> Iterator[DepthFrame]:
        self._prepare_directory()
        log.info("recording frames to %s", self._log_dir)

        for sequence, frame in enumerate(self._inner.frames()):
            filename = FRAME_FILENAME_TEMPLATE.format(sequence=sequence)
            write_message_to_file(self._log_dir / filename, encode_frame(frame))
            append_text_lf(
                self._index_path,
                json.dumps(
                    {
                        "sequence": sequence,
                        "timestamp_seconds": frame.timestamp_seconds,
                        "file": filename,
                    },
                    allow_nan=False,
                )
                + "\n",
            )
            yield frame

    def close(self) -> None:
        self._inner.close()
