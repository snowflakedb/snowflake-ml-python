from typing import List

from snowflake.ml._internal.utils import identifier


class SqlIdentifier(str):
    """Represents an identifier in SQL. An identifier has 3 states:
        1. User input: this is the raw input string to initializer.
        2. identifier(): this is the state that ready input to SQL.
        3. resolved(): this is the state how the identifier stored in database.

    For example:
        1. user input                           ->    2. identifier()     -> 3. resolved()
        SqlIdentifier('abc', case_sensitive=False)          ABC                 ABC
        SqlIdentifier('"abc"', case_sensitive=False)        "abc"               abc
        SqlIdentifier('abc', case_sensitive=True)           "abc"               abc
    """

    def __new__(cls, name: str, *, case_sensitive: bool = False) -> "SqlIdentifier":
        """Create new instance of sql identifier.
            Refer to here for more details: https://docs.snowflake.com/en/sql-reference/identifiers-syntax

        Args:
            name: A string name.
            case_sensitive: If False, then the input string is considered case insensitive and will follow SQL
                identifier parsing rule; if True, then the input string is considered case sensitive, so quotes are
                automatically added if necessary to make sure the original input's cases are preserved.
                Default to False.

        Raises:
            ValueError: input name is not a valid identifier.

        Returns:
            Returns new instance created.
        """
        # TODO (wezhou) add stronger validation to recognize a valid snowflake identifier.
        if not name:
            raise ValueError(f"name:`{name}` is not a valid identifier.")
        if case_sensitive:
            return super().__new__(cls, identifier.get_inferred_name(name))
        else:
            return super().__new__(cls, identifier.resolve_identifier(name))

    def __init__(self, name: str, case_sensitive: bool = False) -> None:
        """Initialize sql identifier.

        Args:
            name: A string name.
            case_sensitive: If False, then the input string is considered case insensitive and will follow SQL
                identifier parsing rule; if True, then the input string is considered case sensitive, so quotes are
                automatically added if necessary to make sure the original input's cases are preserved.
                Default to False.
        """
        super().__init__()

    def identifier(self) -> str:
        """Get the identifier value. This is how the string looks like input to SQL.

        Returns:
            An identifier string.
        """
        return str(self)

    def resolved(self) -> str:
        """Get a resolved string after applying identifier requirement rules. This is how the identifier stored
            in database.

        Returns:
            A resolved string.
        """
        return identifier.get_unescaped_names(str(self))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SqlIdentifier):
            return self.resolved() == other.resolved()
        if isinstance(other, str):
            return str(self) == other
        return False

    def __hash__(self) -> int:
        return super().__hash__()


def to_sql_identifiers(list_of_str: List[str], *, case_sensitive: bool = False) -> List[SqlIdentifier]:
    return [SqlIdentifier(val, case_sensitive=case_sensitive) for val in list_of_str]
