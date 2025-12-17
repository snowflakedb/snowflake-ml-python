from typing import Iterable, Optional
from unittest import mock

import cloudpickle as cp
import numpy as np
import torch
import torch.utils.data as torch_data
from absl.testing import absltest, parameterized

from snowflake import snowpark
from snowflake.ml.data import data_connector
from snowflake.ml.data._internal import arrow_ingestor
from snowflake.ml.fileset import parquet_test_util


class DataConnectorTest(parameterized.TestCase):
    """Tests the DataConnector class"""

    def setUp(self) -> None:
        self._files = parquet_test_util.write_parquet_file()
        ingestor = arrow_ingestor.ArrowIngestor(None, [f.name for f in self._files])  # type: ignore[arg-type]
        self._sut = data_connector.DataConnector(ingestor)

    def test_to_torch_datapipe(self) -> None:
        expected_res = [
            {"col1": np.array([[0], [1]]), "col2": np.array([[10], [11]]), "col3": ["a", "ab"]},
            {"col1": np.array([[2], [3]]), "col2": np.array([[12], [13]]), "col3": ["abc", "m"]},
            {"col1": np.array([[4], [5]]), "col2": np.array([[14], [np.NaN]]), "col3": ["mn", "mnm"]},
        ]
        dp = self._sut.to_torch_datapipe(batch_size=2, shuffle=False, drop_last_batch=True)
        count = 0
        for batch in dp:
            np.testing.assert_array_equal(batch["col1"], expected_res[count]["col1"])  # type: ignore[arg-type]
            np.testing.assert_array_equal(batch["col2"], expected_res[count]["col2"])  # type: ignore[arg-type]
            np.testing.assert_array_equal(batch["col3"], expected_res[count]["col3"])  # type: ignore[arg-type]
            count += 1
        self.assertEqual(count, len(expected_res))

        # also make sure that the datapipe can be a terminal datapipe for DataLoader
        dp = self._sut.to_torch_datapipe(batch_size=2, shuffle=False, drop_last_batch=True)
        dl: Iterable[dict[str, torch.Tensor]] = torch_data.DataLoader(dp, batch_size=None, num_workers=0)
        for tensor_batch in dl:
            for col, tensor in tensor_batch.items():
                if col != "col3":
                    self.assertIsInstance(tensor, torch.Tensor)

        # Ensure iterating through a second time (e.g. second epoch) works
        count2 = 0
        for batch in dl:
            np.testing.assert_array_equal(batch["col1"].numpy(), expected_res[count2]["col1"])  # type: ignore[arg-type]
            np.testing.assert_array_equal(batch["col2"].numpy(), expected_res[count2]["col2"])  # type: ignore[arg-type]
            np.testing.assert_array_equal(batch["col3"], expected_res[count2]["col3"])  # type: ignore[arg-type]
            count2 += 1
        self.assertEqual(count2, len(expected_res))

    def test_to_torch_datapipe_multiprocessing(self) -> None:
        dp = self._sut.to_torch_datapipe(batch_size=2, shuffle=False, drop_last_batch=True)

        # FIXME: This test runs pretty slowly, probably due to multiprocessing overhead
        # Make sure dataset works with num_workers > 0 (and doesn't duplicate data)
        self.assertEqual(
            len(list(torch_data.DataLoader(dp, batch_size=None, num_workers=2))),
            3,
        )

    @parameterized.parameters((1, 2), (2, None), (None, 2), (None, 7), (7, None))  # type: ignore[misc]
    def test_to_torch_dataset_batch_sizes(self, native_batch: Optional[int], data_loader_batch: Optional[int]) -> None:
        # The expected dimensions of each column will be (data_loader_batch, native_batch, sample_dim).
        # Column 1 - scalar data: (data_loader_batch, native_batch, 1)
        # Column 2 - 2D numerical data: (data_loader_batch, native_batch, 2)
        dims = () + (data_loader_batch,) if data_loader_batch is not None else ()
        dims = dims + (native_batch,) if native_batch is not None else dims

        expected_data_dims = {
            "col1": torch.Size(dims + (2,)),
            "col2": torch.Size(dims + (1,)),
        }
        files = parquet_test_util.write_parquet_file(multi_dim_cols=True)
        ingestor = arrow_ingestor.ArrowIngestor(None, [f.name for f in files])  # type: ignore[arg-type]
        connector = data_connector.DataConnector(ingestor)

        ds = connector.to_torch_dataset(shuffle=False, batch_size=native_batch)
        loader = (
            torch_data.DataLoader(ds, batch_size=data_loader_batch, shuffle=False)
            if data_loader_batch is None
            else torch_data.DataLoader(ds, batch_size=data_loader_batch, shuffle=False, drop_last=True)
        )

        for b in loader:
            for k, v in b.items():
                expected_size = expected_data_dims.get(k)
                if expected_size:
                    actual_size = v.size()
                    self.assertEqual(actual_size, expected_size)

    def test_to_torch_dataset_native_batch(self) -> None:
        expected_res = [
            {"col1": np.array([[0], [1]]), "col2": np.array([[10], [11]]), "col3": ["a", "ab"]},
            {"col1": np.array([[2], [3]]), "col2": np.array([[12], [13]]), "col3": ["abc", "m"]},
            {"col1": np.array([[4], [5]]), "col2": np.array([[14], [np.NaN]]), "col3": ["mn", "mnm"]},
        ]
        ds = self._sut.to_torch_dataset(batch_size=2, shuffle=False, drop_last_batch=True)
        count = 0
        loader = torch_data.DataLoader(ds, batch_size=None)
        for batch in loader:
            np.testing.assert_array_equal(batch["col1"], expected_res[count]["col1"])  # type: ignore[arg-type]
            np.testing.assert_array_equal(batch["col2"], expected_res[count]["col2"])  # type: ignore[arg-type]
            np.testing.assert_array_equal(batch["col3"], expected_res[count]["col3"])  # type: ignore[arg-type]
            count += 1
        self.assertEqual(count, len(expected_res))

        # Ensure iterating through a second time (e.g. second epoch) works
        count2 = 0
        for batch in loader:
            np.testing.assert_array_equal(batch["col1"].numpy(), expected_res[count2]["col1"])  # type: ignore[arg-type]
            np.testing.assert_array_equal(batch["col2"].numpy(), expected_res[count2]["col2"])  # type: ignore[arg-type]
            np.testing.assert_array_equal(batch["col3"], expected_res[count2]["col3"])  # type: ignore[arg-type]
            count2 += 1
        self.assertEqual(count2, len(expected_res))

    def test_to_torch_dataset_batch_size_none(self) -> None:
        expected_res = [
            {
                "col1": np.array([0]),
                "col2": np.array([10]),
                "col3": np.array(["a"], dtype="object"),
            },
            {
                "col1": np.array([1]),
                "col2": np.array([11]),
                "col3": np.array(["ab"], dtype="object"),
            },
            {
                "col1": np.array([2]),
                "col2": np.array([12]),
                "col3": np.array(["abc"], dtype="object"),
            },
            {
                "col1": np.array([3]),
                "col2": np.array([13]),
                "col3": np.array(["m"], dtype="object"),
            },
            {
                "col1": np.array([4]),
                "col2": np.array([14]),
                "col3": np.array(["mn"], dtype="object"),
            },
            {
                "col1": np.array([5]),
                "col2": np.array([np.NaN]),
                "col3": np.array(["mnm"], dtype="object"),
            },
            {
                "col1": np.array([6]),
                "col2": np.array([16]),
                "col3": np.array(["mnmn"], dtype="object"),
            },
        ]
        ds = self._sut.to_torch_dataset(batch_size=None, shuffle=False, drop_last_batch=True)
        count = 0
        for batch in ds:
            np.testing.assert_array_equal(batch["col1"], expected_res[count]["col1"])
            np.testing.assert_array_equal(batch["col2"], expected_res[count]["col2"])
            np.testing.assert_array_equal(batch["col3"], expected_res[count]["col3"])
            count += 1
        self.assertEqual(count, len(expected_res))

        # Ensure iterating through a second time (e.g. second epoch) works
        count2 = 0
        for batch in ds:
            np.testing.assert_array_equal(batch["col1"], expected_res[count2]["col1"])
            np.testing.assert_array_equal(batch["col2"], expected_res[count2]["col2"])
            np.testing.assert_array_equal(batch["col3"], expected_res[count2]["col3"])
            count2 += 1
        self.assertEqual(count2, len(expected_res))

    def test_to_torch_dataset_loader_batch(self) -> None:
        files = parquet_test_util.write_parquet_file(multi_dim_cols=True)
        ingestor = arrow_ingestor.ArrowIngestor(None, [f.name for f in files])  # type: ignore[arg-type]
        connector = data_connector.DataConnector(ingestor)

        expected_res = [
            {
                "col1": torch.tensor([[0, 100], [1, 110], [2, 200]]),
                "col2": torch.tensor([[10.0], [11.0], [12.0]], dtype=torch.float64),
                "col3": ["a", "ab", "abc"],
            },
            {
                "col1": torch.tensor([[3, 300], [4, 400], [5, 500]]),
                "col2": torch.tensor([[13.0], [14.0], [np.NaN]], dtype=torch.float64),
                "col3": ["m", "mn", "mnm"],
            },
        ]
        ds = connector.to_torch_dataset(batch_size=None, shuffle=False)

        count = 0
        loader = torch_data.DataLoader(ds, batch_size=3, shuffle=False, drop_last=True)
        for batch in loader:
            torch.testing.assert_close(batch["col1"], expected_res[count]["col1"])
            torch.testing.assert_close(batch["col2"], expected_res[count]["col2"], equal_nan=True)
            np.testing.assert_array_equal(batch["col3"], expected_res[count]["col3"])  # type: ignore[arg-type]
            count += 1
        self.assertEqual(count, len(expected_res))

        # Ensure iterating through a second time (e.g. second epoch) works
        count2 = 0
        for batch in loader:
            torch.testing.assert_close(batch["col1"], expected_res[count2]["col1"])
            torch.testing.assert_close(batch["col2"], expected_res[count2]["col2"], equal_nan=True)
            np.testing.assert_array_equal(batch["col3"], expected_res[count2]["col3"])  # type: ignore[arg-type]
            count2 += 1
        self.assertEqual(count2, len(expected_res))

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

        # Ensure iterating through a second time (e.g. second epoch) works
        count2 = 0
        for batch in dp:
            np.testing.assert_array_equal(batch["col1"].numpy(), expected_res[count2]["col1"])
            np.testing.assert_array_equal(batch["col2"].numpy(), expected_res[count2]["col2"])
            np.testing.assert_array_equal(batch["col3"].numpy(), expected_res[count2]["col3"])
            count2 += 1
        self.assertEqual(count2, len(expected_res))

    def test_to_ray_dataset(self) -> None:
        ray_data_ingestor = mock.Mock()
        ray_data_ingestor.to_ray_dataset = mock.Mock()
        self._sut._ingestor = ray_data_ingestor
        self._sut.to_ray_dataset()
        ray_data_ingestor.to_ray_dataset.assert_called_once()

    def test_to_ray_dataset_with_ingestor_method(self) -> None:
        """Test to_ray_dataset when the ingestor has a to_ray_dataset method."""
        ray_data_ingestor = mock.Mock()
        ray_data_ingestor.to_ray_dataset = mock.Mock(return_value="mock_ray_dataset")
        self._sut._ingestor = ray_data_ingestor

        result = self._sut.to_ray_dataset()
        ray_data_ingestor.to_ray_dataset.assert_called_once()
        self.assertEqual(result, "mock_ray_dataset")

    @mock.patch("builtins.__import__", side_effect=ImportError("Ray not installed"))
    def test_to_ray_dataset_ray_not_installed(self, mock_ray: mock.Mock) -> None:
        """Test to_ray_dataset when Ray is not installed."""
        with self.assertRaises(ImportError) as context:
            self._sut.to_ray_dataset()
        self.assertIn("Ray is not installed, please install ray in your local environment.", str(context.exception))

    def test_pickle_basic(self) -> None:
        """Test that DataConnector can be pickled and unpickled."""
        mock_session = mock.Mock(spec=snowpark.Session)
        mock_session.get_current_account.return_value = "ACCT"
        mock_session.get_current_role.return_value = "ROLE"
        mock_session.get_current_database.return_value = "DB"
        mock_session.get_current_schema.return_value = "SCHEMA"
        ingestor = arrow_ingestor.ArrowIngestor(mock_session, [f.name for f in self._files])
        original_data_sources = ingestor.data_sources

        # Test with a session in the DataConnector object
        sut = data_connector.DataConnector(ingestor)
        with mock.patch("snowflake.snowpark.session._active_sessions", {mock_session}):
            pickled_dc = cp.dumps(sut)
            unpickled_dc = cp.loads(pickled_dc)

        self.assertIsNotNone(unpickled_dc)
        self.assertIsInstance(unpickled_dc, data_connector.DataConnector)
        self.assertIsNotNone(unpickled_dc._ingestor)
        self.assertIsInstance(unpickled_dc._ingestor, arrow_ingestor.ArrowIngestor)
        self.assertEqual(unpickled_dc.data_sources, original_data_sources)

    def test_pickle_derived_class(self) -> None:
        """Test that a derived class of DataConnector can be pickled and unpickled."""
        mock_session = mock.Mock(spec=snowpark.Session)
        mock_session.get_current_account.return_value = "ACCT"
        mock_session.get_current_role.return_value = "ROLE"
        mock_session.get_current_database.return_value = "DB"
        mock_session.get_current_schema.return_value = "SCHEMA"
        ingestor = arrow_ingestor.ArrowIngestor(mock_session, [f.name for f in self._files])
        original_data_sources = ingestor.data_sources

        class DerivedDataConnector(data_connector.DataConnector):
            pass

        # Test with a session in the DataConnector object
        sut = DerivedDataConnector(ingestor)
        with mock.patch("snowflake.snowpark.session._active_sessions", {mock_session}):
            pickled_dc = cp.dumps(sut)
            unpickled_dc = cp.loads(pickled_dc)

        self.assertIsNotNone(unpickled_dc)
        self.assertIsInstance(unpickled_dc, DerivedDataConnector)
        self.assertIsNotNone(unpickled_dc._ingestor)
        self.assertIsInstance(unpickled_dc._ingestor, arrow_ingestor.ArrowIngestor)
        self.assertEqual(unpickled_dc.data_sources, original_data_sources)

        # Test pickling the same DataConnector object again
        with mock.patch("snowflake.snowpark.session._active_sessions", {mock_session}):
            pickled_dc = cp.dumps(sut)
            unpickled_dc = cp.loads(pickled_dc)

        self.assertIsNotNone(unpickled_dc)
        self.assertIsInstance(unpickled_dc, DerivedDataConnector)
        self.assertIsNotNone(unpickled_dc._ingestor)
        self.assertIsInstance(unpickled_dc._ingestor, arrow_ingestor.ArrowIngestor)
        self.assertEqual(unpickled_dc.data_sources, original_data_sources)


if __name__ == "__main__":
    absltest.main()
