from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier


class SqlIdentifierTest(absltest.TestCase):
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


if __name__ == "__main__":
    absltest.main()
