from absl.testing.absltest import TestCase, main

from snowflake.ml._internal.utils.string_matcher import (
    StringMatcherIgnoreWhitespace,
    StringMatcherSql,
)


class StringMatcherIgnoreWhitespaceTest(TestCase):
    """Testing StringMatcherIgnoreWhitespace function."""

    def test_with_spaces(self) -> None:
        """Test that spaces are ignored when matching strings."""
        self.assertEqual(StringMatcherIgnoreWhitespace("text    with  spaces  .   "), "text with spaces.")

    def test_with_tabs(self) -> None:
        """Test that tabs are ignored when matching strings."""
        self.assertEqual(StringMatcherIgnoreWhitespace("\ttext\twith\ttabs\t"), "text with\ttabs")

    def test_with_newlines_cr(self) -> None:
        """Test that newline and carriage return are ignored when matching strings."""
        self.assertEqual(StringMatcherIgnoreWhitespace("text\r\nwith\tcr\rnl\n"), "text\nwith\rcr\tnl")


class StringMatcherSqlTest(TestCase):
    """Testing StringMatcherSql."""

    def test_simple_statement_exact_match(self) -> None:
        """Verify that identical strings are matched."""
        self.assertEqual(StringMatcherSql("select * from my_table"), "select * from my_table")

    def test_simple_statement_ignore_case_match(self) -> None:
        """Verify that case of unquoted identifiers does not matter."""
        self.assertEqual(StringMatcherSql("SELECT * FROM my_table"), "select * from MY_TABLE")

    def test_ignore_case_except_quoted(self) -> None:
        """Verify that case of quoted identifiers DOES matter."""
        self.assertNotEqual(StringMatcherSql('SELECT "column" FROM my_table'), 'SELECT "COLUMN" FROM my_table')

    def test_simple_statement_ignore_whitespace_match(self) -> None:
        """Verify that whitespace is ignored when matching."""
        self.assertEqual(StringMatcherSql("select  *  from \n my_table"), "\n\nselect * from my_table")

    def test_detect_additional_tokens_mismatch(self) -> None:
        """Verify that additional tokens are detected in both directions."""
        self.assertNotEqual(StringMatcherSql("SELECT * FROM my_table;"), "SELECT * FROM my_table")
        self.assertNotEqual(StringMatcherSql("SELECT * FROM my_table"), "SELECT * FROM my_table;")

    def test_detect_token_type_mismatch(self) -> None:
        """Verify that token type mismatches are recognized."""
        self.assertNotEqual(StringMatcherSql("SELECT * FROM ( my_table ) "), "SELECT * FROM ( DATABASE )")
        self.assertNotEqual(StringMatcherSql("SELECT 10 FROM my_table "), 'SELECT "10" FROM my_table')

    def test_leading_trailing_whitespace_invariance(self) -> None:
        """Verify that strings who only differ in leading or trailing whitespace are matched."""
        self.assertEqual(StringMatcherSql("\nselect * from my_table"), "select * from my_table\n")


if __name__ == "__main__":
    main()
