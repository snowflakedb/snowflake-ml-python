from typing import List

from snowflake.ml._internal.utils import identifier


class SqlIdentifier(str):
    """Represents an identifier in SQL. An identifier has 3 states:
        1. User input: this is the raw input string to initializer.
        2. identifier(): this is the state that ready input to SQL.
        3. resolved(): this is the state how the identifier stored in database.

    For example:
        1. user input               ->    2. identifier()     -> 3. resolved()
        SqlIdentifier('abc', True)          ABC               ABC
        SqlIdentifier('"abc"', True)       "abc"              abc
        SqlIdentifier('abc', False)        "abc"              abc
    """

    def __new__(cls, name: str, quotes_to_preserve_case: bool = True) -> "SqlIdentifier":
        """Create new instance of sql identifier.
            Refer to here for more details: https://docs.snowflake.com/en/sql-reference/identifiers-syntax

        Args:
            name: A string name.
            quotes_to_preserve_case: If true, then double quotes are needed to preserve case. This is the default
                mode. When it's false, case are preserved automatically. For instance, This happens when you trying
                to construct SqlIdentifier from result of SQL queries.

        Raises:
            ValueError: input name is not a valid identifier.

        Returns:
            Returns new instance created.
        """
        # TODO (wezhou) add stronger validation to recognize a valid snowflake identifier.
        if not name:
            raise ValueError(f"name:`{name}` is not a valid identifier.")
        if quotes_to_preserve_case:
            return super().__new__(cls, identifier.resolve_identifier(name))
        else:
            return super().__new__(cls, identifier.get_inferred_name(name))

    def __init__(self, name: str, quotes_to_preserve_case: bool = True) -> None:
        """Initialize sql identifier.

        Args:
            name: A string name.
            quotes_to_preserve_case: If true then double quotes are needed to preserve case-sensitivity.
                Otherwise, case-sensivitity are preserved automatically.
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


def to_sql_identifiers(list_of_str: List[str], quotes_to_preserve_case: bool = True) -> List[SqlIdentifier]:
    return [SqlIdentifier(val, quotes_to_preserve_case) for val in list_of_str]
