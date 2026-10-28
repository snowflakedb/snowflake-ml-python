from absl.testing import absltest

from snowflake.ml._internal.utils import sql_literal


class SqlLiteralTest(absltest.TestCase):
    def test_string_literal(self) -> None:
        self.assertEqual(sql_literal.string_literal("plain"), "'plain'")
        self.assertEqual(sql_literal.string_literal("it's"), "'it''s'")
        self.assertEqual(sql_literal.string_literal(r"C:\path\new"), r"'C:\\path\\new'")
        self.assertEqual(
            sql_literal.string_literal("X\\'; DROP EXPERIMENT VICTIM; --"), r"'X\\''; DROP EXPERIMENT VICTIM; --'"
        )
        self.assertEqual(sql_literal.string_literal("trailing\\"), r"'trailing\\'")


if __name__ == "__main__":
    absltest.main()
