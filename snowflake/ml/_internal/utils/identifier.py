import re
from typing import Any, List, Optional, Tuple, Union, overload

from snowflake.snowpark._internal.analyzer import analyzer_utils

# Snowflake Identifier Regex. See https://docs.snowflake.com/en/sql-reference/identifiers-syntax.html.
_SF_UNQUOTED_CASE_INSENSITIVE_IDENTIFIER = "[A-Za-z_][A-Za-z0-9_$]*"
_SF_UNQUOTED_CASE_SENSITIVE_IDENTIFIER = "[A-Z_][A-Z0-9_$]*"
SF_QUOTED_IDENTIFIER = '"(?:[^"]|"")*"'
_SF_IDENTIFIER = f"({_SF_UNQUOTED_CASE_INSENSITIVE_IDENTIFIER}|{SF_QUOTED_IDENTIFIER})"
SF_IDENTIFIER_RE = re.compile(_SF_IDENTIFIER)
_SF_SCHEMA_LEVEL_OBJECT = (
    rf"(?:(?:(?P<db>{_SF_IDENTIFIER})\.)?(?P<schema>{_SF_IDENTIFIER})\.)?(?P<object>{_SF_IDENTIFIER})(?P<others>.*)"
)
_SF_SCHEMA_LEVEL_OBJECT_RE = re.compile(_SF_SCHEMA_LEVEL_OBJECT)

UNQUOTED_CASE_INSENSITIVE_RE = re.compile(f"^({_SF_UNQUOTED_CASE_INSENSITIVE_IDENTIFIER})$")
UNQUOTED_CASE_SENSITIVE_RE = re.compile(f"^({_SF_UNQUOTED_CASE_SENSITIVE_IDENTIFIER})$")
QUOTED_IDENTIFIER_RE = re.compile(f"^({SF_QUOTED_IDENTIFIER})$")
DOUBLE_QUOTE = '"'

quote_name_without_upper_casing = analyzer_utils.quote_name_without_upper_casing


def _is_quoted(id: str) -> bool:
    """Checks if input is quoted.

    NOTE: Snowflake treats all identifiers as UPPERCASE by default. That is 'Hello' would become 'HELLO'. To preserve
    case, one needs to use quoted identifiers, e.g. "Hello" (note the double quote). Callers must take care of that
    quoting themselves. This library assumes that if there is double-quote both sides, it is escaped, otherwise does not
    require.

    Args:
        id: The string to be checked

    Returns:
        True if the `id` is quoted with double-quote to preserve case. Returns False otherwise.

    Raises:
        ValueError: If the id is invalid.
    """
    if not id:
        raise ValueError("Invalid id passed.")
    if len(id) < 2:
        return False
    if id[0] == '"' and id[-1] == '"':
        if len(id) == 2:
            raise ValueError("Invalid id passed.")
        if not QUOTED_IDENTIFIER_RE.match(id):
            raise ValueError("Invalid id passed.")
        return True
    if not UNQUOTED_CASE_INSENSITIVE_RE.match(id):
        raise ValueError("Invalid id passed.")
    return False  # To keep mypy happy


def _get_unescaped_name(id: str) -> str:
    """Remove double quotes and unescape quotes between them from id if quoted.
        Uppercase if not quoted.

    NOTE: See note in :meth:`_is_quoted`.

    Args:
        id: The string to be checked & treated.

    Returns:
        String with quotes removed if quoted; original string otherwise.
    """
    if not _is_quoted(id):
        return id.upper()
    unquoted_id = id[1:-1]
    return unquoted_id.replace(DOUBLE_QUOTE + DOUBLE_QUOTE, DOUBLE_QUOTE)


def _get_escaped_name(id: str) -> str:
    """Add double quotes to escape quotes.
        Replace double quotes with double double quotes if there is existing double quotes

    NOTE: See note in :meth:`_is_quoted`.

    Args:
        id: The string to be checked & treated.

    Returns:
        String with quotes would doubled; original string would add double quotes.
    """
    escape_quotes = id.replace(DOUBLE_QUOTE, DOUBLE_QUOTE + DOUBLE_QUOTE)
    return DOUBLE_QUOTE + escape_quotes + DOUBLE_QUOTE


def get_inferred_name(id: str) -> str:
    """Double quote id when it is case-sensitive and can start with and
    contain any valid characters; unquote otherwise.

    Examples:
        COL1 -> COL1
        1COL -> "1COL"
        Col -> "Col"
        "COL" -> \"""COL""\"  (ignore '\')
        COL 1 -> "COL 1"

    Args:
        id: The string to be checked & treated.

    Returns:
        Double quoted identifier if necessary; unquoted string otherwise.
    """
    if UNQUOTED_CASE_SENSITIVE_RE.match(id):
        return id
    escaped_id = get_escaped_names(id)
    assert isinstance(escaped_id, str)
    return escaped_id


def concat_names(ids: List[str]) -> str:
    """Concatenates `ids` to form one valid id.

    NOTE: See note in :meth:`_is_quoted`.

    Args:
        ids: List of identifiers to be concatenated.

    Returns:
        Concatenated identifier.
    """
    quotes_needed = False
    parts = []
    for id in ids:
        if _is_quoted(id):
            # If any part is quoted, the user cares about case.
            quotes_needed = True
            # Remove quotes before using it.
            id = _get_unescaped_name(id)
        parts.append(id)
    final_id = "".join(parts)
    if quotes_needed:
        return _get_escaped_name(final_id)
    return final_id


def parse_schema_level_object_identifier(
    path: str,
) -> Tuple[Union[str, Any], Union[str, Any], Union[str, Any], Union[str, Any]]:
    """Parse a string which starts with schema level object.

    Args:
        path: A string starts with a schema level object path, which is in the format '<db>.<schema>.<object_name>'.
            Here, '<db>', '<schema>' and '<object_name>' are all snowflake identifiers.

    Returns:
        A tuple of 4 strings in the form of (db, schema, object_name, others). 'db', 'schema', 'object_name' are parsed
            from the schema level object and 'others' are all the content post to the object.

    Raises:
        ValueError: If the id is invalid.
    """
    res = _SF_SCHEMA_LEVEL_OBJECT_RE.fullmatch(path)
    if not res:
        raise ValueError(f"Invalid identifier. It should start with database.schema.stage. Getting {path}")
    return res.group("db"), res.group("schema"), res.group("object"), res.group("others")


def get_schema_level_object_identifier(
    db: Optional[str], schema: Optional[str], object_name: str, others: Optional[str] = None
) -> str:
    """The reverse operation of parse_schema_level_object_identifier

    Args:
        db: Database level object name.
        schema: Schema level object name.
        object_name: stage/table level object name. Must be not None.
        others: All other part attached.

    Returns:
        A string in format '<db>.<schema>.<object_name><others>'

    Raises:
        ValueError: If the identifiers is invalid.
    """

    for identifier in (db, schema, object_name):
        if identifier is not None and SF_IDENTIFIER_RE.match(identifier) is None:
            raise ValueError(f"Invalid identifier {identifier}")

    if others is None:
        others = ""

    return ".".join(filter(None, (db, schema, object_name))) + others


@overload
def get_unescaped_names(ids: None) -> None:
    ...


@overload
def get_unescaped_names(ids: str) -> str:
    ...


@overload
def get_unescaped_names(ids: List[str]) -> List[str]:
    ...


def get_unescaped_names(ids: Optional[Union[str, List[str]]]) -> Optional[Union[str, List[str]]]:
    """Given a user provided identifier(s), this method will compute the equivalent column name identifier(s) in the
    response pandas dataframe(i.e., in the response of snowpark_df.to_pandas()) using the rules defined here
    https://docs.snowflake.com/en/sql-reference/identifiers-syntax.

    Args:
        ids: User provided column name identifier(s).

    Returns:
        Equivalent column name identifier(s) in the response pandas dataframe.

    Raises:
        ValueError: if input types is unsupported or column name identifiers are invalid.
    """

    if ids is None:
        return None
    elif type(ids) is list:
        return [_get_unescaped_name(id) for id in ids]
    elif type(ids) is str:
        return _get_unescaped_name(ids)
    else:
        raise ValueError("Unsupported type. Only string or list of string are supported for selecting columns.")


@overload
def get_escaped_names(ids: None) -> None:
    ...


@overload
def get_escaped_names(ids: str) -> str:
    ...


@overload
def get_escaped_names(ids: List[str]) -> List[str]:
    ...


def get_escaped_names(ids: Optional[Union[str, List[str]]]) -> Optional[Union[str, List[str]]]:
    """Given a user provided identifier(s), this method will compute the equivalent column name identifier(s)
    in case of column name contains special characters, and maintains case-sensitivity
    https://docs.snowflake.com/en/sql-reference/identifiers-syntax.

    Args:
        ids: User provided column name identifier(s).

    Returns:
        Double-quoted Identifiers for column names, to make sure that column names are case sensitive

    Raises:
        ValueError: if input types is unsupported or column name identifiers are invalid.
    """

    if ids is None:
        return None
    elif type(ids) is list:
        return [_get_escaped_name(id) for id in ids]
    elif type(ids) is str:
        return _get_escaped_name(ids)
    else:
        raise ValueError("Unsupported type. Only string or list of string are supported for selecting columns.")


def remove_prefix(s: str, prefix: str) -> str:
    """Remove prefix from a string.

    Args:
        s: string to remove prefix from.
        prefix: prefix to match.

    Returns:
        string with the prefix removed.
    """
    if s.startswith(prefix):
        return s[len(prefix) :]
    return s
