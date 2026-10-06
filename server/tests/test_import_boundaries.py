"""
Proves each optional package is imported in exactly one file, and that kind enums stay in config.

The rule these enforce is in PIPELINE_DESIGN.md: everything device-specific lives in a source or a
sink. An import is how that rule gets broken, usually by someone reaching for torch in the scene
layer because it was convenient.

Scoped to all of nav/, not to the modules where each concept is defined. A guard that only looks
where a thing belongs cannot see where it got written by mistake.
"""

# Standard library imports
import ast
from pathlib import Path

# Local package imports
from conftest import MINIMUM_FILE_COUNT, NAV_ROOT

# Top-level package name to the files in nav/ allowed to import it, as paths relative to nav/.
# Anything else importing one of these is the defect.
ALLOWED_IMPORTERS = {
    "torch": {Path("sources/estimator.py")},
    "depth_anything_3": {Path("sources/estimator.py")},
    # The live client and the recording reader are one distribution with two entry points. The
    # live client runs in a child process, and neon_stream.py is the receiver inside it. The file
    # that starts the process, neon_device.py, reaches the client only through neon_stream.py.
    # neon_plugin.py reads recordings.
    "pupil_labs": {Path("sources/neon_stream.py"), Path("sources/neon_plugin.py")},
    "aiohttp": {Path("sinks/web.py")},
}

# The kind enums are the command line's vocabulary. Past config.py the pipeline holds built
# objects, so a module naming a kind is a module branching on what device it got.
KIND_ENUM_NAMES = ("SourceKind", "SinkKind")
KIND_ENUM_HOME = Path("config.py")


def _python_files() -> list[Path]:
    return sorted(NAV_ROOT.rglob("*.py"))


def _top_level_imports(tree: ast.AST) -> list[tuple[str, int]]:
    """Every absolute import in the file as (top-level package name, line number)."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name.split(".")[0], node.lineno))
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import, which can only ever reach inside nav.
            if node.level == 0 and node.module is not None:
                found.append((node.module.split(".")[0], node.lineno))
    return found


def test_scan_covers_the_package() -> None:
    # Fewer files than at scaffold time means the glob moved, not that the package got cleaner.
    # Without this, both tests below pass perfectly against an empty list.
    assert len(_python_files()) >= MINIMUM_FILE_COUNT


def test_optional_packages_are_imported_in_one_file_each() -> None:
    offenders: list[str] = []
    for path in _python_files():
        relative = path.relative_to(NAV_ROOT)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for package, line in _top_level_imports(tree):
            allowed = ALLOWED_IMPORTERS.get(package)
            if allowed is not None and relative not in allowed:
                names = ", ".join(str(path) for path in sorted(allowed))
                offenders.append(f"{relative}:{line} imports {package}, allowed only in {names}")
    assert offenders == [], "\n".join(offenders)


def test_kind_enums_stay_in_config() -> None:
    offenders: list[str] = []
    for path in _python_files():
        relative = path.relative_to(NAV_ROOT)
        if relative == KIND_ENUM_HOME:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.alias):
                name = node.asname or node.name.split(".")[-1]
            if name in KIND_ENUM_NAMES:
                line = getattr(node, "lineno", 0)
                offenders.append(f"{relative}:{line} names {name}, which belongs to {KIND_ENUM_HOME}")
    assert offenders == [], "\n".join(offenders)
