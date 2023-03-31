import numpy as np
from absl.testing import absltest
from fsspec.implementations import local

from snowflake.ml.fileset import parquet_test_util, tf_dataset


class TfDataSetTest(absltest.TestCase):
    """Tests the tf.DataSet wrapper around the parquet parser.

    parquet_parser_test.py contains more comprehensive test cases.
    """

    def setUp(self) -> None:
        self._file0, self._file1, self._file2 = parquet_test_util.write_parquet_file()

    def testReadAndParseParquet(self) -> None:
        expected_res = [
            {"col1": np.array([0, 1]), "col2": np.array([10, 11]), "col3": np.array([b"a", b"ab"], dtype="object")},
            {"col1": np.array([2, 3]), "col2": np.array([12, 13]), "col3": np.array([b"abc", b"m"], dtype="object")},
            {
                "col1": np.array([4, 5]),
                "col2": np.array([14, np.NaN]),
                "col3": np.array([b"mn", b"mnm"], dtype="object"),
            },
        ]
        files = [self._file0.name, self._file1.name, self._file2.name]
        dp = tf_dataset.read_and_parse_parquet(
            files, local.LocalFileSystem(), batch_size=2, shuffle=False, drop_last_batch=True
        )
        count = 0
        for batch in dp:
            np.testing.assert_array_equal(batch["col1"].numpy(), expected_res[count]["col1"])
            np.testing.assert_array_equal(batch["col2"].numpy(), expected_res[count]["col2"])
            np.testing.assert_array_equal(batch["col3"].numpy(), expected_res[count]["col3"])
            count += 1
        self.assertEqual(count, len(expected_res))


if __name__ == "__main__":
    absltest.main()
