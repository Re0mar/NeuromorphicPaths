"""
Where the route key is read from, and the gitignore proof before it is used.

Every test builds its own pair of files in a temporary directory with a made-up key. No test reads a
real secrets file.
"""

# Standard library imports
from pathlib import Path

# Third party imports
import pytest

# Local package imports
from nav.route.secrets import (
    PROTECTING_GITIGNORE_PATTERNS,
    RouteSecretsError,
    check_secrets_location,
    find_settings_file,
    read_route_key,
    repository_root,
    secrets_file_beside,
)

MADE_UP_KEY = "test-key-7d21e0"


def repository(tmp_path: Path, gitignore: str | None) -> Path:
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    if gitignore is not None:
        (root / ".gitignore").write_text(gitignore, encoding="utf-8")
    return root


def test_the_settings_file_is_found_by_walking_up(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    settings = tmp_path / "a" / ".env_route"
    settings.write_text("", encoding="utf-8")
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    assert find_settings_file(deep) == settings.resolve()


def test_a_settings_file_below_the_start_is_not_found(tmp_path: Path) -> None:
    # The walk goes up, never down. Whatever sits above the temporary directory, it isn't this one.
    (tmp_path / "below").mkdir()
    (tmp_path / "below" / ".env_route").write_text("", encoding="utf-8")
    found = find_settings_file(tmp_path)
    assert found != (tmp_path / "below" / ".env_route").resolve()


def test_the_key_is_read_from_the_secrets_file_beside_it(tmp_path: Path) -> None:
    settings = tmp_path / ".env_route"
    settings.write_text("", encoding="utf-8")
    secrets_file = secrets_file_beside(settings)
    assert secrets_file == tmp_path / ".env_route_secrets"
    secrets_file.write_text(f"# the route service\nOTHER=1\nORS_API_KEY = {MADE_UP_KEY}\n", encoding="utf-8")
    assert read_route_key(secrets_file) == MADE_UP_KEY


def test_a_missing_secrets_file_is_a_named_error(tmp_path: Path) -> None:
    with pytest.raises(RouteSecretsError, match="no secrets file"):
        read_route_key(tmp_path / ".env_route_secrets")


def test_a_missing_key_line_is_a_named_error(tmp_path: Path) -> None:
    secrets_file = tmp_path / ".env_route_secrets"
    secrets_file.write_text("OTHER=1\n", encoding="utf-8")
    with pytest.raises(RouteSecretsError, match="no ORS_API_KEY line"):
        read_route_key(secrets_file)


def test_an_empty_key_line_is_a_named_error(tmp_path: Path) -> None:
    secrets_file = tmp_path / ".env_route_secrets"
    secrets_file.write_text("ORS_API_KEY=\n", encoding="utf-8")
    with pytest.raises(RouteSecretsError, match="empty"):
        read_route_key(secrets_file)


def test_an_unprotected_secrets_file_is_refused_with_the_reason(tmp_path: Path) -> None:
    # Looks like it covers the file and doesn't: `.env.*` needs a dot after `.env`.
    root = repository(tmp_path, ".env\n.env.*\n")
    reason = check_secrets_location(root, root / "server" / ".env_route_secrets")
    assert reason is not None and ".env.*" in reason and MADE_UP_KEY not in reason


def test_a_repository_with_no_gitignore_is_refused(tmp_path: Path) -> None:
    root = repository(tmp_path, None)
    assert check_secrets_location(root, root / ".env_route_secrets") is not None


@pytest.mark.parametrize("pattern", PROTECTING_GITIGNORE_PATTERNS)
def test_each_protecting_pattern_is_accepted(tmp_path: Path, pattern: str) -> None:
    root = repository(tmp_path, f".venv/\n{pattern}\n")
    assert check_secrets_location(root, root / "server" / ".env_route_secrets") is None


def test_a_secrets_file_outside_the_repository_needs_no_pattern(tmp_path: Path) -> None:
    root = repository(tmp_path, ".env.*\n")
    assert check_secrets_location(root, tmp_path / "elsewhere" / ".env_route_secrets") is None


def test_the_repository_root_is_found_from_inside_it(tmp_path: Path) -> None:
    root = repository(tmp_path, None)
    (root / "server").mkdir()
    assert repository_root(root / "server") == root.resolve()


def test_this_repositorys_gitignore_protects_the_secrets_file() -> None:
    # The real .gitignore, read, never the secrets file itself.
    root = repository_root(Path(__file__).parent)
    assert root is not None
    assert check_secrets_location(root, root / "server" / ".env_route_secrets") is None
