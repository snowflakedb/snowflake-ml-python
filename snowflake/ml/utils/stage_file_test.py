from typing import cast
from unittest import mock

from absl.testing import absltest, parameterized

from snowflake.ml.test_utils import mock_session
from snowflake.ml.utils import stage_file
from snowflake.snowpark import Row, Session
from snowflake.snowpark.types import StructType


class ListStageFilesTest(parameterized.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.session = cast(Session, self.m_session)

    def _mock_create_dataframe(self, data: list[tuple[str, ...]], schema: StructType) -> mock.MagicMock:
        mock_df = mock.MagicMock()
        column_name = schema.fields[0].name
        mock_df.collect.return_value = [Row(**{column_name: d[0]}) for d in data]
        return mock_df

    @parameterized.parameters(  # type: ignore[misc]
        ("@DB.SCHEMA.MY_STAGE/path",),
        ("DB.SCHEMA.MY_STAGE/path",),
    )
    def test_basic(self, stage_path: str) -> None:
        mock_rows = [
            Row(name="MY_STAGE/path/file1.jpg"),
            Row(name="MY_STAGE/path/file2.jpg"),
            Row(name="MY_STAGE/path/nested/file3.jpg"),
        ]
        with mock.patch.object(self.session, "sql") as mock_sql:
            mock_sql.return_value.collect.return_value = mock_rows
            with mock.patch.object(
                self.session, "create_dataframe", create=True, side_effect=self._mock_create_dataframe
            ):
                result = stage_file.list_stage_files(self.session, stage_path)
                mock_sql.assert_called_once_with("LIST @DB.SCHEMA.MY_STAGE/path")
                paths = [row["FILE_PATH"] for row in result.collect()]
                self.assertEqual(
                    paths,
                    [
                        "@DB.SCHEMA.MY_STAGE/path/file1.jpg",
                        "@DB.SCHEMA.MY_STAGE/path/file2.jpg",
                        "@DB.SCHEMA.MY_STAGE/path/nested/file3.jpg",
                    ],
                )

    def test_with_pattern(self) -> None:
        mock_rows = [Row(name="STAGE/img.jpg")]
        with mock.patch.object(self.session, "sql") as mock_sql:
            mock_sql.return_value.collect.return_value = mock_rows
            with mock.patch.object(
                self.session, "create_dataframe", create=True, side_effect=self._mock_create_dataframe
            ):
                result = stage_file.list_stage_files(self.session, "@DB.SCHEMA.STAGE", pattern=".*\\.jpg")
                mock_sql.assert_called_once_with("LIST @DB.SCHEMA.STAGE PATTERN = '.*\\.jpg'")
                paths = [row["FILE_PATH"] for row in result.collect()]
                self.assertEqual(paths, ["@DB.SCHEMA.STAGE/img.jpg"])

    def test_custom_column_name(self) -> None:
        mock_rows = [Row(name="STAGE/file.txt")]
        with mock.patch.object(self.session, "sql") as mock_sql:
            mock_sql.return_value.collect.return_value = mock_rows
            with mock.patch.object(
                self.session, "create_dataframe", create=True, side_effect=self._mock_create_dataframe
            ):
                result = stage_file.list_stage_files(self.session, "@DB.SCHEMA.STAGE", column_name="IMAGES")
                paths = [row["IMAGES"] for row in result.collect()]
                self.assertEqual(paths, ["@DB.SCHEMA.STAGE/file.txt"])

    def test_empty_results(self) -> None:
        with mock.patch.object(self.session, "sql") as mock_sql:
            mock_sql.return_value.collect.return_value = []
            with mock.patch.object(self.session, "create_dataframe", create=True) as mock_create_df:
                mock_create_df.return_value.collect.return_value = []
                result = stage_file.list_stage_files(self.session, "@DB.SCHEMA.STAGE")
                self.assertEqual(result.collect(), [])
                # Verify StructType schema is used for empty results
                mock_create_df.assert_called_once()
                _, kwargs = mock_create_df.call_args
                schema = kwargs["schema"]
                self.assertEqual(len(schema.fields), 1)
                self.assertEqual(schema.fields[0].name, "FILE_PATH")

    def test_sql_error(self) -> None:
        with mock.patch.object(self.session, "sql") as mock_sql:
            mock_sql.return_value.collect.side_effect = Exception("Stage not found")
            with self.assertRaisesRegex(RuntimeError, "Failed to list stage location"):
                stage_file.list_stage_files(self.session, "@DB.SCHEMA.STAGE")

    def test_missing_name_column(self) -> None:
        mock_rows = [Row(other_col="value")]
        with mock.patch.object(self.session, "sql") as mock_sql:
            mock_sql.return_value.collect.return_value = mock_rows
            with self.assertRaisesRegex(RuntimeError, "missing 'name' column"):
                stage_file.list_stage_files(self.session, "@DB.SCHEMA.STAGE")

    def test_invalid_name_format(self) -> None:
        mock_rows = [Row(name="no_slash")]
        with mock.patch.object(self.session, "sql") as mock_sql:
            mock_sql.return_value.collect.return_value = mock_rows
            with self.assertRaisesRegex(RuntimeError, "invalid 'name' value"):
                stage_file.list_stage_files(self.session, "@DB.SCHEMA.STAGE")

    def test_external_s3_stage(self) -> None:
        mock_rows = [
            Row(name="s3://my-bucket/kitty.jpeg"),
            Row(name="s3://my-bucket/subdir/data.parquet"),
        ]
        with mock.patch.object(self.session, "sql") as mock_sql:
            mock_sql.return_value.collect.return_value = mock_rows
            with mock.patch.object(
                self.session, "create_dataframe", create=True, side_effect=self._mock_create_dataframe
            ):
                result = stage_file.list_stage_files(self.session, "@DB.SCHEMA.MY_EXT_STAGE")
                paths = [row["FILE_PATH"] for row in result.collect()]
                self.assertEqual(
                    paths,
                    [
                        "@DB.SCHEMA.MY_EXT_STAGE/kitty.jpeg",
                        "@DB.SCHEMA.MY_EXT_STAGE/subdir/data.parquet",
                    ],
                )

    def test_external_s3_stage_with_subdir(self) -> None:
        mock_rows = [
            Row(name="s3://my-bucket/prefix/file1.jpg"),
        ]
        with mock.patch.object(self.session, "sql") as mock_sql:
            mock_sql.return_value.collect.return_value = mock_rows
            with mock.patch.object(
                self.session, "create_dataframe", create=True, side_effect=self._mock_create_dataframe
            ):
                result = stage_file.list_stage_files(self.session, "@DB.SCHEMA.MY_EXT_STAGE/subdir")
                paths = [row["FILE_PATH"] for row in result.collect()]
                self.assertEqual(paths, ["@DB.SCHEMA.MY_EXT_STAGE/prefix/file1.jpg"])

    def test_external_gcs_stage(self) -> None:
        mock_rows = [
            Row(name="gcs://my-gcs-bucket/file.csv"),
        ]
        with mock.patch.object(self.session, "sql") as mock_sql:
            mock_sql.return_value.collect.return_value = mock_rows
            with mock.patch.object(
                self.session, "create_dataframe", create=True, side_effect=self._mock_create_dataframe
            ):
                result = stage_file.list_stage_files(self.session, "@DB.SCHEMA.GCS_STAGE")
                paths = [row["FILE_PATH"] for row in result.collect()]
                self.assertEqual(paths, ["@DB.SCHEMA.GCS_STAGE/file.csv"])

    def test_external_azure_stage(self) -> None:
        mock_rows = [
            Row(name="azure://myaccount.blob.core.windows.net/container/file.txt"),
        ]
        with mock.patch.object(self.session, "sql") as mock_sql:
            mock_sql.return_value.collect.return_value = mock_rows
            with mock.patch.object(
                self.session, "create_dataframe", create=True, side_effect=self._mock_create_dataframe
            ):
                result = stage_file.list_stage_files(self.session, "@DB.SCHEMA.AZURE_STAGE")
                paths = [row["FILE_PATH"] for row in result.collect()]
                self.assertEqual(paths, ["@DB.SCHEMA.AZURE_STAGE/container/file.txt"])

    def test_external_stage_invalid_url(self) -> None:
        mock_rows = [Row(name="s3://bucket-only")]
        with mock.patch.object(self.session, "sql") as mock_sql:
            mock_sql.return_value.collect.return_value = mock_rows
            with self.assertRaisesRegex(RuntimeError, "invalid 'name' value"):
                stage_file.list_stage_files(self.session, "@DB.SCHEMA.STAGE")


class ListStageFilesFromDirectoryTablesTest(parameterized.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.session = cast(Session, self.m_session)

    @parameterized.parameters(  # type: ignore[misc]
        ("@DB.SCHEMA.MY_STAGE",),
        ("DB.SCHEMA.MY_STAGE",),
    )
    def test_composes_lazy_dataframe(self, stage_name: str) -> None:
        lazy_df = mock.MagicMock()
        final_df = mock.MagicMock()
        lazy_df.select.return_value = final_df

        with mock.patch.object(self.session, "sql", return_value=lazy_df) as mock_sql:
            result = stage_file.list_stage_files_from_directory_tables(self.session, stage_name)

        mock_sql.assert_called_once_with("SELECT RELATIVE_PATH FROM DIRECTORY(@DB.SCHEMA.MY_STAGE)")
        lazy_df.select.assert_called_once()
        self.assertIs(result, final_df)

        # The function must compose a plan only — it must not execute anything.
        lazy_df.collect.assert_not_called()
        final_df.collect.assert_not_called()

        (col_expr,), _ = lazy_df.select.call_args
        self.assertEqual(col_expr.getName(), '"FILE_PATH"')

    def test_custom_column_name(self) -> None:
        lazy_df = mock.MagicMock()
        with mock.patch.object(self.session, "sql", return_value=lazy_df):
            stage_file.list_stage_files_from_directory_tables(self.session, "@DB.SCHEMA.STAGE", column_name="IMAGES")

        (col_expr,), _ = lazy_df.select.call_args
        self.assertEqual(col_expr.getName(), '"IMAGES"')

    def test_rejects_subpath(self) -> None:
        with mock.patch.object(self.session, "sql") as mock_sql:
            with self.assertRaisesRegex(ValueError, "without a subpath"):
                stage_file.list_stage_files_from_directory_tables(self.session, "@DB.SCHEMA.STAGE/sub")
            mock_sql.assert_not_called()


if __name__ == "__main__":
    absltest.main()
