import numpy as np
from absl.testing import absltest
from fsspec.implementations import local

from snowflake.ml.fileset import parquet_parser, parquet_test_util


class ParquetParserTest(absltest.TestCase):
    def setUp(self) -> None:
        self._file0, self._file1, self._file2 = parquet_test_util.write_parquet_file()

    def test_parquet_parser_batch_size_one(self) -> None:
        """Test if the parquet parser could yield expected result with default batch size."""
        expected_res = [
            {"col1": np.array([0]), "col2": np.array([10]), "col3": np.array(["a"], dtype="object")},
            {"col1": np.array([1]), "col2": np.array([11]), "col3": np.array(["ab"], dtype="object")},
            {"col1": np.array([2]), "col2": np.array([12]), "col3": np.array(["abc"], dtype="object")},
            {"col1": np.array([3]), "col2": np.array([13]), "col3": np.array(["m"], dtype="object")},
            {"col1": np.array([4]), "col2": np.array([14]), "col3": np.array(["mn"], dtype="object")},
            {"col1": np.array([5]), "col2": np.array([np.NaN]), "col3": np.array(["mnm"], dtype="object")},
            {"col1": np.array([6]), "col2": np.array([16]), "col3": np.array(["mnmn"], dtype="object")},
        ]
        files = [self._file0.name, self._file1.name, self._file2.name]
        fs = local.LocalFileSystem()
        pq_parser = parquet_parser.ParquetParser(files, fs, 1, False)
        count = 0
        for batch in pq_parser:
            np.testing.assert_equal(batch, expected_res[count])
            count += 1
        self.assertEqual(count, len(expected_res))

    def test_parquet_parser_batch_size_two(self) -> None:
        """Test if the parquet parser could yield expected result with batch size bigger than one."""
        expected_res = [
            {"col1": np.array([0, 1]), "col2": np.array([10, 11]), "col3": np.array(["a", "ab"], dtype="object")},
            {"col1": np.array([2, 3]), "col2": np.array([12, 13]), "col3": np.array(["abc", "m"], dtype="object")},
            {"col1": np.array([4, 5]), "col2": np.array([14, np.NaN]), "col3": np.array(["mn", "mnm"], dtype="object")},
        ]
        files = [self._file0.name, self._file1.name, self._file2.name]
        pq_parser = parquet_parser.ParquetParser(files, local.LocalFileSystem(), 2, False)
        count = 0
        for batch in pq_parser:
            np.testing.assert_array_equal(batch["col1"], expected_res[count]["col1"])
            np.testing.assert_array_equal(batch["col2"], expected_res[count]["col2"])
            np.testing.assert_array_equal(batch["col3"], expected_res[count]["col3"])
            count += 1
        self.assertEqual(count, len(expected_res))

    def test_parquet_parser_shuffle(self) -> None:
        """Test if the parquet parser could generate random ordered result with shuffle=True."""
        expected_res = [
            {"col1": np.array([6]), "col2": np.array([16]), "col3": np.array(["mnmn"], dtype="object")},
            {"col1": np.array([5]), "col2": np.array([np.NaN]), "col3": np.array(["mnm"], dtype="object")},
            {"col1": np.array([3]), "col2": np.array([13]), "col3": np.array(["m"], dtype="object")},
            {"col1": np.array([4]), "col2": np.array([14]), "col3": np.array(["mn"], dtype="object")},
            {"col1": np.array([1]), "col2": np.array([11]), "col3": np.array(["ab"], dtype="object")},
            {"col1": np.array([2]), "col2": np.array([12]), "col3": np.array(["abc"], dtype="object")},
            {"col1": np.array([0]), "col2": np.array([10]), "col3": np.array(["a"], dtype="object")},
        ]
        np.random.seed(2)
        files = [self._file0.name, self._file1.name, self._file2.name]
        fs = local.LocalFileSystem()
        pq_parser = parquet_parser.ParquetParser(files, fs, 1, True)
        count = 0
        for batch in pq_parser:
            np.testing.assert_equal(batch, expected_res[count])
            count += 1
        self.assertEqual(count, len(expected_res))

    def test_parquet_parser_not_drop_last_batch(self) -> None:
        """Test if the last batch of data could be generated when drop_last_batch is set to be False."""
        expected_res = [
            {"col1": np.array([0, 1]), "col2": np.array([10, 11]), "col3": np.array(["a", "ab"], dtype="object")},
            {"col1": np.array([2, 3]), "col2": np.array([12, 13]), "col3": np.array(["abc", "m"], dtype="object")},
            {"col1": np.array([4, 5]), "col2": np.array([14, np.NaN]), "col3": np.array(["mn", "mnm"], dtype="object")},
            {"col1": np.array([6]), "col2": np.array([16]), "col3": np.array(["mnmn"], dtype="object")},
        ]
        files = [self._file0.name, self._file1.name, self._file2.name]
        pq_parser = parquet_parser.ParquetParser(
            files, local.LocalFileSystem(), batch_size=2, shuffle=False, drop_last_batch=False
        )
        count = 0
        for batch in pq_parser:
            np.testing.assert_array_equal(batch["col1"], expected_res[count]["col1"])
            np.testing.assert_array_equal(batch["col2"], expected_res[count]["col2"])
            np.testing.assert_array_equal(batch["col3"], expected_res[count]["col3"])
            count += 1
        self.assertEqual(count, len(expected_res))


if __name__ == "__main__":
    absltest.main()
