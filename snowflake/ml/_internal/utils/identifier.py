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
    """Checks if input *identifier* is quoted.

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
        raise ValueError(f"Invalid id {id} passed. ID is empty.")
    if len(id) >= 2 and id[0] == '"' and id[-1] == '"':
        if len(id) == 2:
            raise ValueError(f"Invalid id {id} passed. ID is empty.")
        if not QUOTED_IDENTIFIER_RE.match(id):
            raise ValueError(f"Invalid id {id} passed. ID is quoted but does not match the quoted rule.")
        return True
    if not UNQUOTED_CASE_SENSITIVE_RE.match(id):
        raise ValueError(f"Invalid id {id} passed. ID is unquoted but does not match the unquoted rule.")
    return False


def _get_unescaped_name(id: str) -> str:
    """Remove double quotes and unescape quotes between them from id if quoted.
        Return as it is otherwise

    NOTE: See note in :meth:`_is_quoted`.

    Args:
        id: The string to be checked & treated.

    Returns:
        String with quotes removed if quoted; original string otherwise.
    """
    if not _is_quoted(id):
        return id
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


def get_inferred_name(name: str) -> str:
    """Double quote name when it is case-sensitive and can start with and
    contain any valid characters; otherwise, keep it as it is.

    Examples:
        COL1 -> COL1
        1COL -> "1COL"
        Col -> "Col"
        "COL" -> \"""COL""\"  (ignore '\')
        COL 1 -> "COL 1"

    Args:
        name: The string to be checked & treated.

    Returns:
        Double quoted identifier if necessary; unquoted string otherwise.
    """
    if UNQUOTED_CASE_SENSITIVE_RE.match(name):
        return name
    escaped_id = _get_escaped_name(name)
    assert isinstance(escaped_id, str)
    return escaped_id


def concat_names(names: List[str]) -> str:
    """Concatenates `names` to form one valid id.


    Args:
        names: List of identifiers to be concatenated.

    Returns:
        Concatenated identifier.
    """
    parts = []
    for name in names:
        if QUOTED_IDENTIFIER_RE.match(name):
            # If any part is quoted identifier, we need to remove the quotes
            unescaped_name: str = _get_unescaped_name(name)
            parts.append(unescaped_name)
        else:
            parts.append(name)
    final_id = "".join(parts)
    return get_inferred_name(final_id)


def rename_to_valid_snowflake_identifier(name: str) -> str:
    if QUOTED_IDENTIFIER_RE.match(name) is None and UNQUOTED_CASE_SENSITIVE_RE.match(name) is None:
        name = get_inferred_name(name)
    return name


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
    return (
        res.group("db"),
        res.group("schema"),
        res.group("object"),
        res.group("others"),
    )


def get_schema_level_object_identifier(
    db: Optional[str],
    schema: Optional[str],
    object_name: str,
    others: Optional[str] = None,
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

    This function will mimic the behavior of Snowpark's `to_pandas()` from Snowpark DataFrame.

    Examples:
        COL1 -> COL1
        "Col" -> Col
        \"""COL""\" -> "COL"  (ignore '\')
        "COL 1" -> COL 1

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
def get_inferred_names(names: None) -> None:
    ...


@overload
def get_inferred_names(names: str) -> str:
    ...


@overload
def get_inferred_names(names: List[str]) -> List[str]:
    ...


def get_inferred_names(names: Optional[Union[str, List[str]]]) -> Optional[Union[str, List[str]]]:
    """Given a user provided *string(s)*, this method will compute the equivalent column name identifier(s)
    in case of column name contains special characters, and maintains case-sensitivity
    https://docs.snowflake.com/en/sql-reference/identifiers-syntax.

    This function will mimic the behavior of Snowpark's `create_dataframe` from pandas DataFrame.

    Examples:
        COL1 -> COL1
        1COL -> "1COL"
        Col -> "Col"
        "COL" -> \"""COL""\"  (ignore '\')
        COL 1 -> "COL 1"

    Args:
        names: User provided column name identifier(s).

    Returns:
        Double-quoted Identifiers for column names, to make sure that column names are case sensitive

    Raises:
        ValueError: if input types is unsupported or column name identifiers are invalid.
    """

    if names is None:
        return None
    elif type(names) is list:
        return [get_inferred_name(id) for id in names]
    elif type(names) is str:
        return get_inferred_name(names)
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


def resolve_identifier(name: str) -> str:
    """Given a user provided *string*, resolve following Snowflake identifier resolution strategies:
        https://docs.snowflake.com/en/sql-reference/identifiers-syntax#label-identifier-casing

        This function will mimic the behavior of the SQL parser.

    Examples:
        COL1 -> COL1
        1COL -> Raise Error
        Col -> COL
        "COL" -> COL
        COL 1 -> Raise Error

    Args:
        name: the string to be resolved.

    Raises:
        ValueError: if input would not be accepted by SQL parser.

    Returns:
        Resolved identifier
    """
    if QUOTED_IDENTIFIER_RE.match(name):
        unescaped = _get_unescaped_name(name)
        if UNQUOTED_CASE_SENSITIVE_RE.match(unescaped):
            return unescaped
        return name
    elif UNQUOTED_CASE_INSENSITIVE_RE.match(name):
        return name.upper()
    else:
        raise ValueError(f"Invalid name {name} passed. ID is not quoted and cannot normalized.")
