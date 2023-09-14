"""String formatting utilities for general use in the SnowML Reposiory.

This file contains a collection of utilities that help with formatting strings. Functionality is not limited to tests
only. Anything that is re-usable across different modules and related to string formatting should go here.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

_WHITESPACE_COMPACT_RE = re.compile(r"\s+")
_WHITESPACE_NO_NEWLINE_RE = re.compile(r"[ \t\r\f\v]+")
_WHITESPACE_COMPACT_KEEP_NEWLINE_RE = re.compile(r"([ \t\r\f\v]*\n[ \t\r\f\v]*)")


@dataclass
class SqlStr:
    """Class to represent SQL strings for formatting.

    This is mainly to distinguish SQL from regular strings."""

    sql: str

    def __init__(self, sql: str) -> None:
        self.sql = sql

    def __repr__(self) -> str:
        return self.sql


def format_value_for_select(value: Any) -> str:
    """Format a value for inclusion in a SELECT query. returns a string with the correct formatting.

    Currently supported types:
        str: Enclose in single quotes and escape single quotes inside the string.
        datetime: Convert isoformat string and instruct the backend to interpret the string as a timestamp.
        dict: Convert keys to strings and recursively handle values. Instruct the backend to interpret the list of keys
            and values as an object.

        Everything else we attempt to convert to string but will not enclose in quotes.

    Args:
        value: Value to be formatted.

    Returns:
        String with the formatted value.
    """

    # Have to use an explicit comparison with None as "The truth value of an array with more than one element is
    # ambiguous." for numpy arrays.
    if value is None:
        return "null"

    if isinstance(value, str):
        return "'" + value.replace("'", "\\'") + "'"
    elif isinstance(value, SqlStr):
        return repr(value)
    elif isinstance(value, datetime):
        return "TO_TIMESTAMP('" + value.isoformat() + "')"
    elif isinstance(value, dict):
        # Converting all keys to strings and recursively format the values. When iterating over dictionaries, items
        # appear in random order due to the hashing of the keys. We ensure reproducibility by sorting the items during
        # formatting.

        return (
            "OBJECT_CONSTRUCT("
            + ",".join([f"'{k}',{format_value_for_select(v)}" for k, v in sorted(value.items())])
            + ")"
        )
    elif hasattr(value, "__iter__"):
        # If value is iterable (e.g. list or array) convert it to an array recursively.
        return "ARRAY_CONSTRUCT(" + ",".join([format_value_for_select(x) for x in value]) + ")"
    else:
        return str(value)


def unwrap(text: str, keep_newlines: bool = False) -> str:
    """Unwraps a string into a single line while preserving word boundaries. Leading and trailing spaces are removed.

    Args:
        text: Text to unwrap.
        keep_newlines: Keep newlines when formatting.

    Returns:
        Unwrapped text.
    """

    # Normalize whitespace:
    if keep_newlines:
        # Three stages:
        # 1. (innermost) Normalize all non-newline whitespace to spaces.
        # 2. (middle)    Collapse any sequence of newline plus any non-newline whitespace into just newline.
        # 3. (outer)    Strip leading and trailing whitespace.
        return _WHITESPACE_COMPACT_KEEP_NEWLINE_RE.sub("\n", _WHITESPACE_NO_NEWLINE_RE.sub(" ", text)).strip()
    else:
        # Normalize all whitespace into single spaces and strip leading and trailing whitespace.
        return _WHITESPACE_COMPACT_RE.sub(" ", text).strip()
