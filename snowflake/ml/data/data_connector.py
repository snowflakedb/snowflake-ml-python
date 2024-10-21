import os
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Type, TypeVar

import numpy.typing as npt
from typing_extensions import deprecated

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml.data import data_ingestor, data_source
from snowflake.ml.data._internal.arrow_ingestor import ArrowIngestor
from snowflake.ml.modeling._internal.constants import (
    IN_ML_RUNTIME_ENV_VAR,
    USE_OPTIMIZED_DATA_INGESTOR,
)

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

    DEFAULT_INGESTOR_CLASS: Type[data_ingestor.DataIngestor] = ArrowIngestor

    def __init__(
        self,
        ingestor: data_ingestor.DataIngestor,
    ) -> None:
        self._ingestor = ingestor

    @classmethod
    @snowpark._internal.utils.private_preview(version="1.6.0")
    def from_dataframe(
        cls: Type[DataConnectorType],
        df: snowpark.DataFrame,
        ingestor_class: Optional[Type[data_ingestor.DataIngestor]] = None,
        **kwargs: Any
    ) -> DataConnectorType:
        if len(df.queries["queries"]) != 1 or len(df.queries["post_actions"]) != 0:
            raise ValueError("DataFrames with multiple queries and/or post-actions not supported")
        source = data_source.DataFrameInfo(df.queries["queries"][0])
        assert df._session is not None
        return cls.from_sources(df._session, [source], ingestor_class=ingestor_class, **kwargs)

    @classmethod
    def from_dataset(
        cls: Type[DataConnectorType],
        ds: "dataset.Dataset",
        ingestor_class: Optional[Type[data_ingestor.DataIngestor]] = None,
        **kwargs: Any
    ) -> DataConnectorType:
        dsv = ds.selected_version
        assert dsv is not None
        source = data_source.DatasetInfo(
            ds.fully_qualified_name, dsv.name, dsv.url(), exclude_cols=(dsv.label_cols + dsv.exclude_cols)
        )
        return cls.from_sources(ds._session, [source], ingestor_class=ingestor_class, **kwargs)

    @classmethod
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject_extractor=lambda cls: cls.__name__,
        func_params_to_log=["sources", "ingestor_class"],
    )
    def from_sources(
        cls: Type[DataConnectorType],
        session: snowpark.Session,
        sources: List[data_source.DataSource],
        ingestor_class: Optional[Type[data_ingestor.DataIngestor]] = None,
        **kwargs: Any
    ) -> DataConnectorType:
        ingestor_class = ingestor_class or cls.DEFAULT_INGESTOR_CLASS
        ingestor = ingestor_class.from_sources(session, sources)
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

    @deprecated(
        "to_torch_datapipe() is deprecated and will be removed in a future release. Use to_torch_dataset() instead"
    )
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
        from snowflake.ml.data import torch_utils

        return torch_utils.TorchDataPipeWrapper(
            self._ingestor, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last_batch
        )

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject_extractor=lambda self: type(self).__name__,
        func_params_to_log=["batch_size", "shuffle", "drop_last_batch"],
    )
    def to_torch_dataset(
        self, *, batch_size: Optional[int] = None, shuffle: bool = False, drop_last_batch: bool = True
    ) -> "torch_data.IterableDataset":  # type: ignore[type-arg]
        """Transform the Snowflake data into a PyTorch Iterable Dataset to be used with a DataLoader.

        Return a PyTorch Dataset which iterates on rows of data.

        Args:
            batch_size: It specifies the size of each data batch which will be yielded in the result dataset.
                Batching is pushed down to data ingestion level which may be more performant than DataLoader
                batching.
            shuffle: It specifies whether the data will be shuffled. If True, files will be shuffled, and
                rows in each file will also be shuffled.
            drop_last_batch: Whether the last batch of data should be dropped. If set to be true,
                then the last batch will get dropped if its size is smaller than the given batch_size.

        Returns:
            A PyTorch Iterable Dataset that yields data.
        """
        from snowflake.ml.data import torch_utils

        return torch_utils.TorchDatasetWrapper(
            self._ingestor, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last_batch
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


# Switch to use Runtime's Data Ingester if running in ML runtime
# Fail silently if the data ingester is not found
if os.getenv(IN_ML_RUNTIME_ENV_VAR) and os.getenv(USE_OPTIMIZED_DATA_INGESTOR):
    try:
        from runtime_external_entities import get_ingester_class

        DataConnector.DEFAULT_INGESTOR_CLASS = get_ingester_class()
    except ImportError:
        """Runtime Default Ingester not found, ignore"""
        pass
