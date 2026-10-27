def string_literal(value: str) -> str:
    """Format a string as a single-quoted SQL string literal.

    Snowflake treats a backslash inside a string literal as an escape character, so backslashes are doubled along with
    single quotes. Escaping only the quotes would let a value containing a backslash end the literal early.

    Args:
        value: String to be formatted.

    Returns:
        Single-quoted SQL string literal that evaluates to ``value``.
    """
    return "'" + value.replace("\\", "\\\\").replace("'", "''") + "'"
