"""
Every relative link in the repository's Markdown resolves to a file or folder that exists.

A doc renamed or moved leaves links pointing nowhere, and nothing else in the suite reads the docs.
Web and file:// links aren't checked, see `UNCHECKED_LINK_PREFIXES`. Recordings, virtual
environments and build output are left out, because they're local and never committed.
"""

# Standard library imports
import re
from pathlib import Path

# Third party imports
import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
# Folders that hold no committed docs: local recordings, environments, build output, tool caches.
SKIPPED_FOLDERS = {".git", ".venv", "frame_logs", "build", ".gradle", "node_modules", ".pytest_cache", "__pycache__", ".idea"}
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")
# Web links need the network. A file:// link names a path on whoever's machine wrote it, as the
# generated notes under OldAppEnvrionmentStuff/.artifacts do, so it can't resolve anywhere else.
UNCHECKED_LINK_PREFIXES = ("http://", "https://", "mailto:", "file://")


def _markdown_files() -> list[Path]:
    return sorted(
        path
        for path in REPOSITORY.rglob("*.md")
        if not SKIPPED_FOLDERS.intersection(path.relative_to(REPOSITORY).parts)
    )


def _dead_links(document: Path) -> list[str]:
    dead = []
    for line_number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), start=1):
        for match in LINK.finditer(line):
            target = match.group(1)
            if target.startswith(UNCHECKED_LINK_PREFIXES):
                continue
            if not (document.parent / target).exists():
                dead.append(f"{document.relative_to(REPOSITORY)}:{line_number}: {target}")
    return dead


def test_the_repository_has_markdown_to_check() -> None:
    """A wrong root would find nothing, and an empty check passes."""
    names = {path.relative_to(REPOSITORY).as_posix() for path in _markdown_files()}
    assert "README.md" in names
    assert "docs/guides/frame_to_arrow_delay.md" in names


@pytest.mark.parametrize("document", _markdown_files(), ids=lambda path: path.relative_to(REPOSITORY).as_posix())
def test_every_relative_link_resolves(document: Path) -> None:
    assert _dead_links(document) == []
