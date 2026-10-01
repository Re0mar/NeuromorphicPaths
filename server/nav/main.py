"""The entry point named by pyproject's console script and by python -m nav."""


def main(argv: list[str] | None = None) -> int:
    """
    Build everything from the command line and run the frame loop.

    :param argv: Argument list, or None to read sys.argv.
    :return: Process exit code.
    :rtype: int
    """
    # Stubbed the same way the factory arms in config.py are, so the entry points pyproject
    # declares resolve to something that says where the body is rather than to nothing at all.
    raise NotImplementedError("the frame loop and entry point land in STEP_09")
