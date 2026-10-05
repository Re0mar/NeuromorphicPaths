"""
FIELD=VALUE overrides for a frozen config dataclass, typed from the field's own declaration.

Used for the planner, the scene and the evaluation alike. A bad override raises `OverrideRefused`, its
own type, so the command can refuse it by name without also swallowing every other ValueError.
"""

# Standard library imports
import dataclasses
import math
from typing import TypeVar

ConfigType = TypeVar("ConfigType")

TRUE_SPELLINGS = ("true",)
FALSE_SPELLINGS = ("false",)


class OverrideRefused(ValueError):
    """An override that names no field, or whose value doesn't parse as that field's type."""


def apply_overrides(config: ConfigType, overrides: list[str]) -> tuple[ConfigType, frozenset[str]]:
    """
    Apply FIELD=VALUE overrides to a frozen dataclass.

    :param config: The config to start from.
    :param overrides: Each one FIELD=VALUE. bool takes true or false, int a whole number, float any
        number including inf.
    :return: The new config and the names that were overridden.
    :rtype: tuple[ConfigType, frozenset[str]]
    :raises OverrideRefused: On a missing =, an unknown field, a field of another type, a value
        that doesn't parse, or a value the config's own checks refuse.
    """
    fields = {field.name: field for field in dataclasses.fields(config)}
    changes: dict[str, object] = {}
    for override in overrides:
        name, separator, text = override.partition("=")
        if not separator:
            raise OverrideRefused(f"an override is FIELD=VALUE, got {override!r}")
        if name not in fields:
            raise OverrideRefused(f"{type(config).__name__} has no field {name!r}. Valid fields: {', '.join(sorted(fields))}")
        changes[name] = _parse(name, _type_name(fields[name].type), text)
    try:
        return dataclasses.replace(config, **changes), frozenset(changes)
    except ValueError as refused_by_config:
        # The value parsed, and the config's own checks refused it. Still the user's input, so it is
        # refused by name like any other bad override, with the config's reason.
        raise OverrideRefused(f"{type(config).__name__} refused {', '.join(overrides)}: {refused_by_config}") from refused_by_config


def _type_name(declared: object) -> str:
    # A module with postponed annotations declares types as strings. Either way the name is what matters.
    return declared if isinstance(declared, str) else getattr(declared, "__name__", str(declared))


def _parse(name: str, type_name: str, text: str) -> object:
    if type_name == "bool":
        if text.lower() in TRUE_SPELLINGS:
            return True
        if text.lower() in FALSE_SPELLINGS:
            return False
        raise OverrideRefused(f"{name} wants true or false, got {text!r}")
    if type_name == "int":
        try:
            return int(text)
        except ValueError:
            raise OverrideRefused(f"{name} wants a whole number, got {text!r}") from None
    if type_name == "float":
        try:
            value = float(text)
        except ValueError:
            raise OverrideRefused(f"{name} wants a number, got {text!r}") from None
        if math.isnan(value):
            raise OverrideRefused(f"{name} wants a number, got NaN")
        return value
    raise OverrideRefused(f"{name} is a {type_name}, which an override can't set")
