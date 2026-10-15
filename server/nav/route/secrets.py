"""
Where the route service's key comes from: a secrets file beside a settings file, never the environment.

`.env_route` marks the place, found by walking up from the working directory. `.env_route_secrets`
beside it holds `ORS_API_KEY=...`. Before the key is used, a gitignore pattern must be shown to cover
the secrets file, because a `.gitignore` reading `.env` and `.env.*` looks like it does and doesn't:
`.env.*` needs a dot after `.env`.
"""

# Standard library imports
from pathlib import Path

SETTINGS_FILENAME = ".env_route"
SECRETS_FILENAME = ".env_route_secrets"
KEY_NAME = "ORS_API_KEY"
# Any one of these lines in the repository's .gitignore keeps the secrets file out of git.
PROTECTING_GITIGNORE_PATTERNS = (".env_*", ".env_route*", ".env_route_secrets", "*.env_route_secrets")


class RouteSecretsError(Exception):
    """The secrets file or its key line is missing. The message names which, never the key."""


def find_settings_file(start: Path) -> Path | None:
    """The nearest `.env_route` at or above a directory, or None."""
    for directory in (start.resolve(), *start.resolve().parents):
        candidate = directory / SETTINGS_FILENAME
        if candidate.is_file():
            return candidate
    return None


def secrets_file_beside(settings_file: Path) -> Path:
    return settings_file.parent / SECRETS_FILENAME


def read_route_key(secrets_file: Path) -> str:
    """
    The key from the secrets file's `ORS_API_KEY=` line.

    :raises RouteSecretsError: When the file or the line is missing, or the line has no value.
    """
    if not secrets_file.is_file():
        raise RouteSecretsError(f"no secrets file at {secrets_file}")
    for raw_line in secrets_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == KEY_NAME:
            key = value.strip().strip('"').strip("'")
            if not key:
                raise RouteSecretsError(f"{secrets_file} has an empty {KEY_NAME} line")
            return key
    raise RouteSecretsError(f"{secrets_file} has no {KEY_NAME} line")


def repository_root(path: Path) -> Path | None:
    """The nearest directory at or above a path that holds a `.git`, or None outside any repository."""
    for directory in (path.resolve(), *path.resolve().parents):
        if (directory / ".git").exists():
            return directory
    return None


def check_secrets_location(root: Path, secrets_file: Path) -> str | None:
    """
    Whether git could commit the secrets file. The Jira tooling's check, in the same shape.

    :param root: The repository's root.
    :param secrets_file: Where the key is read from.
    :return: Why the file isn't protected, or None when it is, or when it sits outside the repository.
    :rtype: str | None
    """
    try:
        secrets_file.resolve().relative_to(root.resolve())
    except ValueError:
        # Outside the tree, so git here can't commit it.
        return None
    gitignore = root / ".gitignore"
    if not gitignore.is_file():
        return f"no .gitignore at {root}. Add a line reading `.env_route*` before putting a key in {SECRETS_FILENAME}"
    lines = {line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()}
    if any(pattern in lines for pattern in PROTECTING_GITIGNORE_PATTERNS):
        return None
    return f"{gitignore} does not ignore {SECRETS_FILENAME}. `.env.*` does not match it. Add `.env_route*`"
