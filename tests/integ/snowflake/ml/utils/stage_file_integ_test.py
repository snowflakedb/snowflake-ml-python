import tempfile
import uuid
from pathlib import Path

from absl.testing import absltest

from snowflake.ml.utils.stage_file import (
    list_stage_files,
    list_stage_files_from_directory_tables,
)
from tests.integ.snowflake.ml.test_utils import common_test_base, db_manager


class ListStageFilesIntegTest(common_test_base.CommonTestBase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._run_id = uuid.uuid4().hex[:8]
        cls._test_db = f"SNOWML_TEST_STAGE_FILE_DB_{cls._run_id}"
        cls._test_stage = f"SNOWML_TEST_STAGE_{cls._run_id}"

    def setUp(self) -> None:
        super().setUp()
        self._db_manager = db_manager.DBManager(self.session)
        self._db_manager.create_database(self._test_db)
        self._stage_name = self._db_manager.create_stage(self._test_stage, schema_name="PUBLIC", db_name=self._test_db)

        # Upload test files to stage
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_files = ["image1.jpg", "image2.jpg", "document.pdf", "data.csv"]
            for filename in test_files:
                filepath = Path(tmpdir) / filename
                filepath.write_text(f"test content for {filename}")
                self.session.sql(f"PUT file://{filepath} @{self._stage_name} AUTO_COMPRESS=FALSE").collect()

            # Create files in a subdirectory
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            for filename in ["nested1.jpg", "nested2.txt"]:
                filepath = subdir / filename
                filepath.write_text(f"test content for {filename}")
                self.session.sql(f"PUT file://{filepath} @{self._stage_name}/subdir AUTO_COMPRESS=FALSE").collect()

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._test_db, if_exists=True)
        super().tearDown()

    def test_list_all_files(self) -> None:
        df = list_stage_files(self.session, f"@{self._stage_name}")
        paths = [row["FILE_PATH"] for row in df.collect()]

        self.assertLen(paths, 6)
        self.assertTrue(all(path.startswith(f"@{self._stage_name}") for path in paths))

    def test_list_files_with_pattern(self) -> None:
        df = list_stage_files(self.session, f"@{self._stage_name}", pattern=".*\\.jpg")
        paths = [row["FILE_PATH"] for row in df.collect()]

        self.assertLen(paths, 3)  # image1.jpg, image2.jpg, nested1.jpg
        self.assertTrue(all(path.endswith(".jpg") for path in paths))

    def test_list_files_in_subdirectory(self) -> None:
        df = list_stage_files(self.session, f"@{self._stage_name}/subdir")
        paths = [row["FILE_PATH"] for row in df.collect()]

        self.assertLen(paths, 2)
        self.assertTrue(all("subdir" in path for path in paths))

    def test_list_files_with_custom_column_name(self) -> None:
        df = list_stage_files(self.session, f"@{self._stage_name}", column_name="MY_COLUMN")
        rows = df.collect()

        self.assertLen(rows, 6)
        self.assertIn("MY_COLUMN", rows[0].as_dict())

    def test_list_files_with_pattern_in_subdirectory(self) -> None:
        df = list_stage_files(self.session, f"@{self._stage_name}/subdir", pattern=".*\\.jpg")
        paths = [row["FILE_PATH"] for row in df.collect()]

        self.assertLen(paths, 1)  # nested1.jpg
        self.assertTrue(paths[0].endswith("nested1.jpg"))

    def test_list_files_without_at_prefix(self) -> None:
        # Test that the function auto-prepends "@"
        df = list_stage_files(self.session, self._stage_name)
        paths = [row["FILE_PATH"] for row in df.collect()]

        self.assertLen(paths, 6)
        self.assertTrue(all(path.startswith(f"@{self._stage_name}") for path in paths))


class ListStageFilesFromDirectoryTablesIntegTest(common_test_base.CommonTestBase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._run_id = uuid.uuid4().hex[:8]
        cls._test_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(cls._run_id, "db").upper()
        cls._test_schema = "PUBLIC"
        cls._test_stage = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(cls._run_id, "stage").upper()

    def setUp(self) -> None:
        super().setUp()
        self._db_manager = db_manager.DBManager(self.session)
        self._db_manager.create_database(self._test_db)
        # db_manager.create_stage does not expose DIRECTORY, so create the stage directly.
        self._stage_name = f"{self._test_db}.{self._test_schema}.{self._test_stage}"
        self.session.sql(f"CREATE OR REPLACE STAGE {self._stage_name} DIRECTORY = (ENABLE = TRUE)").collect()

        # Upload a mix of filenames — including one with a space so we verify that RELATIVE_PATH
        # is returned raw (not URL-encoded) end-to-end.
        self._expected_filenames = ["image1.jpg", "image2.jpg", "data.csv", "hello world.txt"]
        with tempfile.TemporaryDirectory() as tmpdir:
            for filename in self._expected_filenames:
                filepath = Path(tmpdir) / filename
                filepath.write_text(f"test content for {filename}")
                # Quote the local path so spaces are handled by PUT.
                self.session.sql(f"PUT 'file://{filepath}' @{self._stage_name} AUTO_COMPRESS=FALSE").collect()

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._test_db, if_exists=True)
        super().tearDown()

    def test_list_all_files(self) -> None:
        df = list_stage_files_from_directory_tables(self.session, f"@{self._stage_name}")
        paths = [row["FILE_PATH"] for row in df.collect()]

        self.assertLen(paths, len(self._expected_filenames))
        self.assertTrue(all(path.startswith(f"@{self._stage_name}/") for path in paths))
        self.assertEqual(
            sorted(paths),
            sorted(f"@{self._stage_name}/{name}" for name in self._expected_filenames),
        )

    def test_filenames_with_spaces(self) -> None:
        df = list_stage_files_from_directory_tables(self.session, f"@{self._stage_name}")
        paths = [row["FILE_PATH"] for row in df.collect()]

        self.assertIn(f"@{self._stage_name}/hello world.txt", paths)
        self.assertFalse(any("%20" in path for path in paths))

    def test_custom_column_name(self) -> None:
        df = list_stage_files_from_directory_tables(self.session, f"@{self._stage_name}", column_name="MY_COLUMN")
        rows = df.collect()

        self.assertLen(rows, len(self._expected_filenames))
        self.assertIn("MY_COLUMN", rows[0].as_dict())

    def test_without_at_prefix(self) -> None:
        df = list_stage_files_from_directory_tables(self.session, self._stage_name)
        paths = [row["FILE_PATH"] for row in df.collect()]

        self.assertLen(paths, len(self._expected_filenames))
        self.assertTrue(all(path.startswith(f"@{self._stage_name}/") for path in paths))

    def test_rejects_subpath(self) -> None:
        with self.assertRaisesRegex(ValueError, "without a subpath"):
            list_stage_files_from_directory_tables(self.session, f"@{self._stage_name}/subdir")


if __name__ == "__main__":
    absltest.main()
