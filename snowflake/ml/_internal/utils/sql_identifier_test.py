from absl.testing import absltest

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier


class SqlIdentifierTest(absltest.TestCase):
    def test_sql_identifier(self) -> None:
        id = SqlIdentifier("abc", case_sensitive=False)
        self.assertEqual(id.identifier(), "ABC")
        self.assertEqual(id.resolved(), "ABC")

        id = SqlIdentifier('"abc"', case_sensitive=False)
        self.assertEqual(id.identifier(), '"abc"')
        self.assertEqual(id.resolved(), "abc")

        id = SqlIdentifier("abc", case_sensitive=True)
        self.assertEqual(id.identifier(), '"abc"')
        self.assertEqual(id.resolved(), "abc")

        id = SqlIdentifier("ABC", case_sensitive=True)
        self.assertEqual(id.identifier(), "ABC")
        self.assertEqual(id.resolved(), "ABC")

    def test_sql_identifier_equality(self) -> None:
        id_1 = SqlIdentifier("abc", case_sensitive=False)
        id_2 = SqlIdentifier("ABC", case_sensitive=False)
        self.assertEqual(id_1, id_2)

        id_1 = SqlIdentifier('"ABC"', case_sensitive=False)
        id_2 = SqlIdentifier("ABC", case_sensitive=False)
        self.assertEqual(id_1, id_2)

        id_1 = SqlIdentifier("abc", case_sensitive=True)
        id_2 = SqlIdentifier('"abc"', case_sensitive=False)
        self.assertEqual(id_1, id_2)

        id_1 = SqlIdentifier("abc", case_sensitive=True)
        id_2 = SqlIdentifier("abc", case_sensitive=True)
        self.assertEqual(id_1, id_2)

        id_1 = SqlIdentifier("ABC", case_sensitive=True)
        id_2 = SqlIdentifier("abc", case_sensitive=True)
        self.assertNotEqual(id_1, id_2)


if __name__ == "__main__":
    absltest.main()
