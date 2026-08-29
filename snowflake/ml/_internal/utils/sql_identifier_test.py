from absl.testing import absltest, parameterized

from snowflake.ml._internal.utils import sql_identifier

# Names that are not valid unquoted SQL identifiers and require quoting.
_QUOTED_IDENTIFIER_STORED_NAMES = (
    "APP-INF-FOO",
    "APP INF FOO",
    "123FOO",
)


class SqlIdentifierTest(parameterized.TestCase):
    def test_sql_identifier(self) -> None:
        id = sql_identifier.SqlIdentifier("abc", case_sensitive=False)
        self.assertEqual(id.identifier(), "ABC")
        self.assertEqual(id.resolved(), "ABC")

        id = sql_identifier.SqlIdentifier('"abc"', case_sensitive=False)
        self.assertEqual(id.identifier(), '"abc"')
        self.assertEqual(id.resolved(), "abc")

        id = sql_identifier.SqlIdentifier("abc", case_sensitive=True)
        self.assertEqual(id.identifier(), '"abc"')
        self.assertEqual(id.resolved(), "abc")

        id = sql_identifier.SqlIdentifier("ABC", case_sensitive=True)
        self.assertEqual(id.identifier(), "ABC")
        self.assertEqual(id.resolved(), "ABC")

    def test_sql_identifier_equality(self) -> None:
        id_1 = sql_identifier.SqlIdentifier("abc", case_sensitive=False)
        id_2 = sql_identifier.SqlIdentifier("ABC", case_sensitive=False)
        self.assertEqual(id_1, id_2)

        id_1 = sql_identifier.SqlIdentifier('"ABC"', case_sensitive=False)
        id_2 = sql_identifier.SqlIdentifier("ABC", case_sensitive=False)
        self.assertEqual(id_1, id_2)

        id_1 = sql_identifier.SqlIdentifier("abc", case_sensitive=True)
        id_2 = sql_identifier.SqlIdentifier('"abc"', case_sensitive=False)
        self.assertEqual(id_1, id_2)

        id_1 = sql_identifier.SqlIdentifier("abc", case_sensitive=True)
        id_2 = sql_identifier.SqlIdentifier("abc", case_sensitive=True)
        self.assertEqual(id_1, id_2)

        id_1 = sql_identifier.SqlIdentifier("ABC", case_sensitive=True)
        id_2 = sql_identifier.SqlIdentifier("abc", case_sensitive=True)
        self.assertNotEqual(id_1, id_2)

    @parameterized.parameters(*_QUOTED_IDENTIFIER_STORED_NAMES)  # type: ignore[misc]
    def test_quoted_session_identifier_equals_show_stored_name(self, stored_name: str) -> None:
        session_role = sql_identifier.SqlIdentifier(f'"{stored_name}"')
        show_owner = sql_identifier.SqlIdentifier(stored_name, case_sensitive=True)
        self.assertEqual(session_role, show_owner)

    def test_parse_fully_qualified_name(self) -> None:
        self.assertTupleEqual(
            sql_identifier.parse_fully_qualified_name("abc"), (None, None, sql_identifier.SqlIdentifier("abc"))
        )
        self.assertTupleEqual(
            sql_identifier.parse_fully_qualified_name('"schema".abc'),
            (None, sql_identifier.SqlIdentifier("schema", case_sensitive=True), sql_identifier.SqlIdentifier("abc")),
        )
        self.assertTupleEqual(
            sql_identifier.parse_fully_qualified_name('db."schema".abc'),
            (
                sql_identifier.SqlIdentifier("db"),
                sql_identifier.SqlIdentifier("schema", case_sensitive=True),
                sql_identifier.SqlIdentifier("abc"),
            ),
        )

        with self.assertRaises(ValueError):
            sql_identifier.parse_fully_qualified_name('db."schema".abc.def')

        with self.assertRaises(ValueError):
            sql_identifier.parse_fully_qualified_name("abc-def")

    def test_get_fully_qualified_name(self) -> None:
        self.assertEqual(
            "MYDB.MYSCHEMA.ABC",
            sql_identifier.get_fully_qualified_name(
                None, None, sql_identifier.SqlIdentifier("abc"), "mydb", "myschema"
            ),
        )
        self.assertEqual(
            "MYDB.MYSCHEMA.ABC",
            sql_identifier.get_fully_qualified_name(
                "mydb", "myschema", sql_identifier.SqlIdentifier("abc"), None, None
            ),
        )
        self.assertEqual(
            "ABC",
            sql_identifier.get_fully_qualified_name(None, None, sql_identifier.SqlIdentifier("abc"), None, None),
        )
        self.assertEqual(
            'MYDB.MYSCHEMA."abc"',
            sql_identifier.get_fully_qualified_name(
                "mydb", "myschema", sql_identifier.SqlIdentifier('"abc"'), None, None
            ),
        )
        self.assertEqual(
            '"mydb"."myschema".ABC',
            sql_identifier.get_fully_qualified_name(
                '"mydb"', '"myschema"', sql_identifier.SqlIdentifier("abc"), None, None
            ),
        )
        self.assertEqual(
            '"mydb"."myschema".ABC',
            sql_identifier.get_fully_qualified_name(
                None, None, sql_identifier.SqlIdentifier("abc"), '"mydb"', '"myschema"'
            ),
        )


if __name__ == "__main__":
    absltest.main()
