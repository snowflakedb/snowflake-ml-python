from typing import Dict, Iterable

import numpy as np
import tensorflow  # noqa: F401 # SNOW-1502273 test fails if TensorFlow not imported globally
import torch
import torch.utils.data as torch_data
from absl.testing import absltest

from snowflake.ml.data import data_connector
from snowflake.ml.data._internal import arrow_ingestor
from snowflake.ml.fileset import parquet_test_util


class DataConnectorTest(absltest.TestCase):
    """Tests the DataConnector wrappers around the parquet parser.

    parquet_parser_test.py contains more comprehensive test cases.
    """

    def setUp(self) -> None:
        self._files = parquet_test_util.write_parquet_file()
        ingestor = arrow_ingestor.ArrowIngestor(None, [f.name for f in self._files])  # type: ignore[arg-type]
        self._sut = data_connector.DataConnector(ingestor)

    def test_to_torch_datapipe(self) -> None:
        expected_res = [
            {"col1": np.array([0, 1]), "col2": np.array([10, 11]), "col3": np.array(["a", "ab"], dtype="object")},
            {"col1": np.array([2, 3]), "col2": np.array([12, 13]), "col3": np.array(["abc", "m"], dtype="object")},
            {"col1": np.array([4, 5]), "col2": np.array([14, np.NaN]), "col3": np.array(["mn", "mnm"], dtype="object")},
        ]
        dp = self._sut.to_torch_datapipe(batch_size=2, shuffle=False, drop_last_batch=True)
        count = 0
        for batch in dp:
            np.testing.assert_array_equal(batch["col1"], expected_res[count]["col1"])
            np.testing.assert_array_equal(batch["col2"], expected_res[count]["col2"])
            np.testing.assert_array_equal(batch["col3"], expected_res[count]["col3"])
            count += 1
        self.assertEqual(count, len(expected_res))

        # also make sure that the datapipe can be a terminal datapipe for DataLoader
        dp = self._sut.to_torch_datapipe(batch_size=2, shuffle=False, drop_last_batch=True)
        dl: Iterable[Dict[str, torch.Tensor]] = torch_data.DataLoader(dp, batch_size=None, num_workers=0)
        for tensor_batch in dl:
            for col, tensor in tensor_batch.items():
                if col != "col3":
                    self.assertIsInstance(tensor, torch.Tensor)

    def test_to_torch_dataset(self) -> None:
        expected_res = [
            {"col1": np.array([0, 1]), "col2": np.array([10, 11]), "col3": ["a", "ab"]},
            {"col1": np.array([2, 3]), "col2": np.array([12, 13]), "col3": ["abc", "m"]},
            {"col1": np.array([4, 5]), "col2": np.array([14, np.NaN]), "col3": ["mn", "mnm"]},
        ]
        ds = self._sut.to_torch_dataset(shuffle=False)
        count = 0
        for batch in torch_data.DataLoader(ds, batch_size=2, shuffle=False, drop_last=True):
            np.testing.assert_array_equal(batch["col1"], expected_res[count]["col1"])  # type: ignore[arg-type]
            np.testing.assert_array_equal(batch["col2"], expected_res[count]["col2"])  # type: ignore[arg-type]
            np.testing.assert_array_equal(batch["col3"], expected_res[count]["col3"])  # type: ignore[arg-type]
            count += 1
        self.assertEqual(count, len(expected_res))

    def test_to_torch_dataset_multiprocessing(self) -> None:
        ds = self._sut.to_torch_dataset(shuffle=False)

        # FIXME: This test runs pretty slowly, probably due to multiprocessing overhead
        # Make sure dataset works with num_workers > 0 (and doesn't duplicate data)
        self.assertEqual(
            len(list(torch_data.DataLoader(ds, batch_size=2, shuffle=False, drop_last=True, num_workers=2))),
            3,
        )

    def test_to_tf_dataset(self) -> None:
        expected_res = [
            {"col1": np.array([0, 1]), "col2": np.array([10, 11]), "col3": np.array([b"a", b"ab"], dtype="object")},
            {"col1": np.array([2, 3]), "col2": np.array([12, 13]), "col3": np.array([b"abc", b"m"], dtype="object")},
            {
                "col1": np.array([4, 5]),
                "col2": np.array([14, np.NaN]),
                "col3": np.array([b"mn", b"mnm"], dtype="object"),
            },
        ]
        dp = self._sut.to_tf_dataset(batch_size=2, shuffle=False, drop_last_batch=True)
        count = 0
        for batch in dp:
            np.testing.assert_array_equal(batch["col1"].numpy(), expected_res[count]["col1"])
            np.testing.assert_array_equal(batch["col2"].numpy(), expected_res[count]["col2"])
            np.testing.assert_array_equal(batch["col3"].numpy(), expected_res[count]["col3"])
            count += 1
        self.assertEqual(count, len(expected_res))


if __name__ == "__main__":
    absltest.main()
