from unittest.mock import MagicMock

from absl.testing import absltest, parameterized

from snowflake.ml.jobs._utils import object_naming


class ResolveDatabaseSchemaTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.session = MagicMock()
        self.session.get_current_database.return_value = "CONTEXT_DB"
        self.session.get_current_schema.return_value = "CONTEXT_SCHEMA"

    def test_falls_back_to_session_context(self) -> None:
        self.assertEqual(
            object_naming.resolve_database_schema(self.session, None, None), ("CONTEXT_DB", "CONTEXT_SCHEMA")
        )

    def test_explicit_values_take_precedence(self) -> None:
        self.assertEqual(
            object_naming.resolve_database_schema(self.session, "my_db", "my_schema"), ("MY_DB", "MY_SCHEMA")
        )
        self.session.get_current_database.assert_not_called()
        self.session.get_current_schema.assert_not_called()

    def test_quoted_identifiers_preserved(self) -> None:
        self.assertEqual(
            object_naming.resolve_database_schema(self.session, '"lower db"', '"lower schema"'),
            ('"lower db"', '"lower schema"'),
        )

    def test_missing_database_raises(self) -> None:
        self.session.get_current_database.return_value = None
        with self.assertRaisesRegex(ValueError, "Database must be specified"):
            object_naming.resolve_database_schema(self.session, None, "my_schema")

    def test_missing_schema_raises(self) -> None:
        self.session.get_current_schema.return_value = None
        with self.assertRaisesRegex(ValueError, "Schema must be specified"):
            object_naming.resolve_database_schema(self.session, "my_db", None)


class BuildObjectStagePathTest(parameterized.TestCase):
    @parameterized.parameters(  # type: ignore[misc]
        ("my_stage", "@my_stage/my_object"),
        ("@my_stage", "@my_stage/my_object"),
        ("@my_stage/", "@my_stage/my_object"),
        ("@my_stage/subdir", "@my_stage/subdir/my_object"),
        ("@my_stage/subdir/", "@my_stage/subdir/my_object"),
        ("@DB.SCHEMA.MY_STAGE", "@DB.SCHEMA.MY_STAGE/my_object"),
        ("DB.SCHEMA.MY_STAGE/subdir", "@DB.SCHEMA.MY_STAGE/subdir/my_object"),
    )
    def test_build_object_stage_path(self, stage_name: str, expected: str) -> None:
        self.assertEqual(object_naming.build_object_stage_path(stage_name, "my_object").as_posix(), expected)


if __name__ == "__main__":
    absltest.main()
