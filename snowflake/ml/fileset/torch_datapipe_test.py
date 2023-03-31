from typing import Dict, Iterable

import numpy as np
import torch
from absl.testing import absltest
from fsspec.implementations import local
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper

from snowflake.ml.fileset import parquet_test_util, torch_datapipe


class TorchDataPipeTest(absltest.TestCase):
    """Tests the torch DataPipe wrapper around the parquet parser.

    parquet_parser_test.py contains more comprehensive test cases.
    """

    def setUp(self) -> None:
        self._file0, self._file1, self._file2 = parquet_test_util.write_parquet_file()

    def testDataPipe(self) -> None:
        expected_res = [
            {"col1": np.array([0, 1]), "col2": np.array([10, 11]), "col3": np.array(["a", "ab"], dtype="object")},
            {"col1": np.array([2, 3]), "col2": np.array([12, 13]), "col3": np.array(["abc", "m"], dtype="object")},
            {"col1": np.array([4, 5]), "col2": np.array([14, np.NaN]), "col3": np.array(["mn", "mnm"], dtype="object")},
        ]
        files = IterableWrapper([self._file0.name, self._file1.name, self._file2.name])
        dp = torch_datapipe.ReadAndParseParquet(files, local.LocalFileSystem(), 2, shuffle=False, drop_last_batch=True)
        count = 0
        for batch in dp:
            np.testing.assert_array_equal(batch["col1"], expected_res[count]["col1"])
            np.testing.assert_array_equal(batch["col2"], expected_res[count]["col2"])
            np.testing.assert_array_equal(batch["col3"], expected_res[count]["col3"])
            count += 1
        self.assertEqual(count, len(expected_res))

        # also make sure that the datapipe can be a terminal datapipe for DataLoader
        dp = torch_datapipe.ReadAndParseParquet(files, local.LocalFileSystem(), 2, shuffle=False, drop_last_batch=True)
        dl: Iterable[Dict[str, torch.Tensor]] = DataLoader(dp, batch_size=None, num_workers=0)
        for tensor_batch in dl:
            for col, tensor in tensor_batch.items():
                if col != "col3":
                    self.assertIsInstance(tensor, torch.Tensor)


if __name__ == "__main__":
    absltest.main()
