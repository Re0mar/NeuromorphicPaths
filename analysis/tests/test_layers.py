"""
The layer rules from analysis/README.md, checked on the import statements themselves.

Every "must not" is paired with a "must", because a module that imports nothing passes every
ban while being unable to do its job.
"""

# Standard library imports
import ast
from pathlib import Path

PACKAGE_NAME = "neuromorphicpaths_analysis"
PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / PACKAGE_NAME

# Which sibling layers each layer may import. A new layer fails test_every_layer_has_a_rule
# until it gets a row here, so the rules cannot be skipped by accident.
ALLOWED_LAYERS = {
    "detector": set(),
    "scoring": {"detector"},
    "labeling": set(),
}
REQUIRED_LAYERS = {
    "scoring": {"detector"},
}
# The labeling extra is a multi-gigabyte install. Nothing outside labeling may need it.
HEAVY_PACKAGES = {"torch", "transformers"}


def imported_modules(layer: str) -> set[str]:
    found: set[str] = set()
    for source_path in (PACKAGE_ROOT / layer).rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module)
    return found


def imported_layers(layer: str) -> set[str]:
    prefix = f"{PACKAGE_NAME}."
    return {
        module.removeprefix(prefix).split(".")[0]
        for module in imported_modules(layer)
        if module.startswith(prefix)
    } - {layer}


def test_every_layer_has_a_rule() -> None:
    layers = {path.name for path in PACKAGE_ROOT.iterdir() if (path / "__init__.py").exists()}
    assert layers == set(ALLOWED_LAYERS)


def test_layers_only_import_allowed_layers() -> None:
    for layer, allowed in ALLOWED_LAYERS.items():
        assert imported_layers(layer) <= allowed, f"{layer} imports {imported_layers(layer) - allowed}"


def test_layers_import_what_they_depend_on() -> None:
    for layer, required in REQUIRED_LAYERS.items():
        assert required <= imported_layers(layer), f"{layer} no longer imports {required}"


def test_heavy_packages_stay_in_labeling() -> None:
    for layer in set(ALLOWED_LAYERS) - {"labeling"}:
        heavy = {module.split(".")[0] for module in imported_modules(layer)} & HEAVY_PACKAGES
        assert not heavy, f"{layer} imports {heavy}"
    assert HEAVY_PACKAGES <= {module.split(".")[0] for module in imported_modules("labeling")}
