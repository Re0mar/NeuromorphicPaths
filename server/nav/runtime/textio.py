"""
The only module in the package that opens a text file for writing.

Everything goes through here so the line ending is decided once. Python's text mode on Windows
converts every newline to CRLF, and .gitattributes cannot reach a file the program writes at
runtime, so a frame log written on this laptop would not be byte-identical to one written on
anything else.
"""

# Standard library imports
from pathlib import Path


def write_text_lf(path: Path, text: str) -> None:
    """Replace the file's contents with text, LF endings regardless of platform."""
    path.write_bytes(text.encode("utf-8"))


def append_text_lf(path: Path, text: str) -> None:
    """Append text to the file, LF endings regardless of platform."""
    with path.open("ab") as handle:
        handle.write(text.encode("utf-8"))
