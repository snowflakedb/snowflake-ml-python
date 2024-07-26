from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Type, TypeVar

import numpy.typing as npt

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml.data import data_ingestor, data_source
from snowflake.ml.data._internal.arrow_ingestor import ArrowIngestor as DefaultIngestor

if TYPE_CHECKING:
    import pandas as pd
    import tensorflow as tf
    from torch.utils import data as torch_data

    # This module can't actually depend on dataset to avoid a circular dependency
    # Dataset -> DatasetReader -> DataConnector -!-> Dataset
    from snowflake.ml import dataset

_PROJECT = "DataConnector"

DataConnectorType = TypeVar("DataConnectorType", bound="DataConnector")


class DataConnector:
    """Snowflake data reader which provides application integration connectors"""

    def __init__(
        self,
        ingestor: data_ingestor.DataIngestor,
    ) -> None:
        self._ingestor = ingestor

    @classmethod
    def from_dataframe(cls: Type[DataConnectorType], df: snowpark.DataFrame, **kwargs: Any) -> DataConnectorType:
        if len(df.queries["queries"]) != 1 or len(df.queries["post_actions"]) != 0:
            raise ValueError("DataFrames with multiple queries and/or post-actions not supported")
        source = data_source.DataFrameInfo(df.queries["queries"][0])
        assert df._session is not None
        ingestor = DefaultIngestor(df._session, [source])
        return cls(ingestor, **kwargs)

    @classmethod
    def from_dataset(cls: Type[DataConnectorType], ds: "dataset.Dataset", **kwargs: Any) -> DataConnectorType:
        dsv = ds.selected_version
        assert dsv is not None
        source = data_source.DatasetInfo(
            ds.fully_qualified_name, dsv.name, dsv.url(), exclude_cols=(dsv.label_cols + dsv.exclude_cols)
        )
        ingestor = DefaultIngestor(ds._session, [source])
        return cls(ingestor, **kwargs)

    @property
    def data_sources(self) -> List[data_source.DataSource]:
        return self._ingestor.data_sources

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject_extractor=lambda self: type(self).__name__,
        func_params_to_log=["batch_size", "shuffle", "drop_last_batch"],
    )
    def to_tf_dataset(
        self, *, batch_size: int, shuffle: bool = False, drop_last_batch: bool = True
    ) -> "tf.data.Dataset":
        """Transform the Snowflake data into a ready-to-use TensorFlow tf.data.Dataset.

        Args:
            batch_size: It specifies the size of each data batch which will be
                yield in the result datapipe
            shuffle: It specifies whether the data will be shuffled. If True, files will be shuffled, and
                rows in each file will also be shuffled.
            drop_last_batch: Whether the last batch of data should be dropped. If set to be true,
                then the last batch will get dropped if its size is smaller than the given batch_size.

        Returns:
            A tf.data.Dataset that yields batched tf.Tensors.
        """
        import tensorflow as tf

        def generator() -> Generator[Dict[str, npt.NDArray[Any]], None, None]:
            yield from self._ingestor.to_batches(batch_size, shuffle, drop_last_batch)

        # Derive TensorFlow signature
        first_batch = next(self._ingestor.to_batches(1, shuffle=False, drop_last_batch=False))
        tf_signature = {
            k: tf.TensorSpec(shape=(None,), dtype=tf.dtypes.as_dtype(v.dtype), name=k) for k, v in first_batch.items()
        }

        return tf.data.Dataset.from_generator(generator, output_signature=tf_signature)

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject_extractor=lambda self: type(self).__name__,
        func_params_to_log=["batch_size", "shuffle", "drop_last_batch"],
    )
    def to_torch_datapipe(
        self, *, batch_size: int, shuffle: bool = False, drop_last_batch: bool = True
    ) -> "torch_data.IterDataPipe":  # type: ignore[type-arg]
        """Transform the Snowflake data into a ready-to-use Pytorch datapipe.

        Return a Pytorch datapipe which iterates on rows of data.

        Args:
            batch_size: It specifies the size of each data batch which will be
                yield in the result datapipe
            shuffle: It specifies whether the data will be shuffled. If True, files will be shuffled, and
                rows in each file will also be shuffled.
            drop_last_batch: Whether the last batch of data should be dropped. If set to be true,
                then the last batch will get dropped if its size is smaller than the given batch_size.

        Returns:
            A Pytorch iterable datapipe that yield data.
        """
        from torch.utils.data.datapipes import iter as torch_iter

        return torch_iter.IterableWrapper(  # type: ignore[no-untyped-call]
            self._ingestor.to_batches(batch_size, shuffle, drop_last_batch)
        )

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject_extractor=lambda self: type(self).__name__,
        func_params_to_log=["limit"],
    )
    def to_pandas(self, limit: Optional[int] = None) -> "pd.DataFrame":
        """Retrieve the Snowflake data as a Pandas DataFrame.

        Args:
            limit: If specified, the maximum number of rows to load into the DataFrame.

        Returns:
            A Pandas DataFrame.
        """
        return self._ingestor.to_pandas(limit)
