import os

from absl.testing.absltest import TestCase, main

from snowflake.ml.utils.temp_file_utils import cleanup_temp_files, get_temp_file_path


class TempFileUtilsTest(TestCase):
    """Testing temp file util functions."""

    def test_create_and_delete_temp_files(self) -> None:
        file_name = get_temp_file_path()
        self.assertFalse(os.path.exists(file_name))
        os.open(file_name, os.O_CREAT | os.O_RDWR)
        self.assertTrue(os.path.exists(file_name))
        cleanup_temp_files(file_name)
        self.assertFalse(os.path.exists(file_name))

    def test_delete_non_existant_file(self) -> None:
        file_name = get_temp_file_path()
        self.assertFalse(os.path.exists(file_name))
        cleanup_temp_files(file_name)


if __name__ == "__main__":
    main()
