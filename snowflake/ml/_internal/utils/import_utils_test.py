from absl.testing import absltest

from snowflake.ml._internal.utils import import_utils


class ConditionalImportTest(absltest.TestCase):
    def test_positive_import_or_get_dummy(self) -> None:
        """Test if import_or_get_dummy() can import package/module/symbol as expected."""
        # Test importing package
        snowpark, available = import_utils.import_or_get_dummy("snowflake.snowpark")
        self.assertTrue(available)
        self.assertTrue(hasattr(snowpark, "row"))

        # Test importing module
        snowpark_row, available = import_utils.import_or_get_dummy("snowflake.snowpark.row")
        self.assertTrue(available)
        self.assertTrue(hasattr(snowpark_row, "Row"))

        # Test importing class
        Row, available = import_utils.import_or_get_dummy("snowflake.snowpark.Row")
        self.assertTrue(available)
        self.assertTrue(hasattr(Row, "as_dict"))

    def test_negative_import_or_get_dummy(self) -> None:
        """Test if import_or_get_dummy() will return dummy object when the import target is not available."""

        snowpark, available = import_utils.import_or_get_dummy("snowflake.snowparks")
        self.assertFalse(available)
        with self.assertRaises(ImportError):
            hasattr(snowpark, "row")

        # Test importing trivial module
        snowpark_row, available = import_utils.import_or_get_dummy("snowflake.snowpark.rows")
        self.assertFalse(available)
        with self.assertRaises(ImportError):
            hasattr(snowpark_row, "Row")

        # Test importing trivial class
        Row, available = import_utils.import_or_get_dummy("snowflake.snowpark.Rows")
        self.assertFalse(available)
        with self.assertRaises(ImportError):
            self.assertTrue(hasattr(Row, "as_dict"))

    def test_positive_import_with_fallbacks(self) -> None:
        module = import_utils.import_with_fallbacks("snowflake.snowpark")
        self.assertIsNotNone(module)

        module = import_utils.import_with_fallbacks("snowflake.snowpark.Row")
        self.assertIsNotNone(module)

        module = import_utils.import_with_fallbacks("not.a.real.module", "snowflake.snowpark")
        self.assertIsNotNone(module)

    def test_negative_import_with_fallbacks(self) -> None:
        with self.assertRaises(ImportError):
            _ = import_utils.import_with_fallbacks("snowflake.snowpark.NotARealModule")

        with self.assertRaises(ImportError):
            _ = import_utils.import_with_fallbacks("NotARealModule")

        with self.assertRaises(ImportError):
            _ = import_utils.import_with_fallbacks("notamodule", "snowflake.snowpark.NotARealModule")


if __name__ == "__main__":
    absltest.main()
