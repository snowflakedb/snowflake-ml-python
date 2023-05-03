from datetime import datetime

import numpy as np
from absl.testing.absltest import TestCase, main
from formatting import SqlStr, format_value_for_select, unwrap


class FormattingTest(TestCase):
    def test_format_value_for_select_str(self) -> None:
        """Test string formatting for the value in a select clause."""
        self.assertEqual(format_value_for_select("string"), "'string'")

    def test_format_value_for_select_sql(self) -> None:
        """Test that SQL strings will not be quoted."""
        self.assertEqual(format_value_for_select(SqlStr("CURRENT_TIME()")), "CURRENT_TIME()")

    def test_format_value_for_select_datetime(self) -> None:
        """Test timestamp formatting for the value in a select clause."""
        dt = datetime.fromisoformat("2011-11-04T00:05:23+04:00")
        self.assertEqual(format_value_for_select(dt), "TO_TIMESTAMP('2011-11-04T00:05:23+04:00')")

    def test_format_value_for_select_dict(self) -> None:
        """Test dictionary formatting for the value in a select clause."""
        self.assertEqual(format_value_for_select({"key": "value_str"}), "OBJECT_CONSTRUCT('key','value_str')")
        self.assertEqual(format_value_for_select({"key": 2.71828}), "OBJECT_CONSTRUCT('key',2.71828)")
        self.assertEqual(
            format_value_for_select({"key": {"nested_key": "value_str"}}),
            "OBJECT_CONSTRUCT('key',OBJECT_CONSTRUCT('nested_key','value_str'))",
        )

    def test_format_value_for_select_dict_sorting(self) -> None:
        """Test that the output of dictionary formatting has stable order."""
        # When iterating over dictionaries, items appear in random order due to the hashing of the keys. We ensure
        # reproducibility by sorting the items during formatting.
        self.assertEqual(
            format_value_for_select({"A": "a", "B": "b", "C": "c"}), "OBJECT_CONSTRUCT('A','a','B','b','C','c')"
        )

    def test_format_value_for_select_numpy_array(self) -> None:
        """Test formatting of array-likes for the value in a select clause with."""
        self.assertEqual(format_value_for_select(np.array([0, 1])), "ARRAY_CONSTRUCT(0,1)")
        self.assertEqual(
            format_value_for_select(np.array([[0, 1], [2, 3]])),
            "ARRAY_CONSTRUCT(ARRAY_CONSTRUCT(0,1),ARRAY_CONSTRUCT(2,3))",
        )

    def test_format_value_for_select_other(self) -> None:
        """Test formatting of other values."""
        self.assertEqual(format_value_for_select(23), "23")

    def test_format_value_for_select_null(self) -> None:
        """Test that None/null is correctly handled when formatting values."""
        self.assertEqual(format_value_for_select(None), "null")

    def test_unwrap(self) -> None:
        """Test string unwrapping."""
        self.assertEqual(unwrap("  Test\t \n String\r\n"), "Test String")
        self.assertEqual(unwrap("nothing to do"), "nothing to do")

    def test_unwrap_keep_newlines(self) -> None:
        """Unwrap a string but keep newlines."""
        self.assertEqual(unwrap("  line1  \n  \t   line2 ", keep_newlines=True), "line1\nline2")
        self.assertEqual(
            unwrap(
                """Expected
                   Actual""",
                keep_newlines=True,
            ),
            "Expected\nActual",
        )


if __name__ == "__main__":
    main()
