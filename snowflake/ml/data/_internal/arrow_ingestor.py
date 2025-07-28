import collections
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Deque, Iterator, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds

if TYPE_CHECKING:
    import ray

from snowflake import snowpark
from snowflake.ml._internal.utils import mixins
from snowflake.ml.data import data_ingestor, data_source, ingestor_utils

_EMPTY_RECORD_BATCH = pa.RecordBatch.from_arrays([], [])

# The row count for batches read from PyArrow Dataset. This number should be large enough so that
# dataset.to_batches() would read in a very large portion of, if not entirely, a parquet file.
_DEFAULT_DATASET_BATCH_SIZE = 1000000


class _RecordBatchesBuffer:
    """A queue that stores record batches and tracks the total num of rows in it."""

    def __init__(self) -> None:
        self.buffer: Deque[pa.RecordBatch] = collections.deque()
        self.num_rows = 0

    def append(self, rb: pa.RecordBatch) -> None:
        self.buffer.append(rb)
        self.num_rows += rb.num_rows

    def appendleft(self, rb: pa.RecordBatch) -> None:
        self.buffer.appendleft(rb)
        self.num_rows += rb.num_rows

    def popleft(self) -> pa.RecordBatch:
        popped = self.buffer.popleft()
        self.num_rows -= popped.num_rows
        return popped


class ArrowIngestor(data_ingestor.DataIngestor, mixins.SerializableSessionMixin):
    """Read and parse the data sources into an Arrow Dataset and yield batched numpy array in dict."""

    def __init__(
        self,
        session: snowpark.Session,
        data_sources: Sequence[data_source.DataSource],
        format: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            session: The Snowpark Session to use.
            data_sources: List of data sources to ingest.
            format: Currently “parquet”, “ipc”/”arrow”/”feather”, “csv”, “json”, and “orc” are supported.
                Will be inferred if not specified.
            kwargs: Miscellaneous arguments passed to underlying PyArrow Dataset initializer.
        """
        self._session = session
        self._data_sources = list(data_sources)
        self._format = format
        self._kwargs = kwargs

        self._schema: Optional[pa.Schema] = None

    @classmethod
    def from_sources(cls, session: snowpark.Session, sources: Sequence[data_source.DataSource]) -> "ArrowIngestor":
        if session is None:
            raise ValueError("Session is required")
        return cls(session, sources)

    @classmethod
    def from_ray_dataset(
        cls,
        ray_ds: "ray.data.Dataset",
    ) -> "ArrowIngestor":
        raise NotImplementedError

    @property
    def data_sources(self) -> list[data_source.DataSource]:
        return self._data_sources

    def to_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last_batch: bool = True,
    ) -> Iterator[dict[str, npt.NDArray[Any]]]:
        """Iterate through PyArrow Dataset to generate batches whose length equals to expected batch size.

        As we are generating batches with the exactly same length, the last few rows in each file might get left as they
        are not long enough to form a batch. These rows will be put into a temporary buffer and combine with the first
        few rows of the next file to generate a new batch.

        Args:
            batch_size: Specifies the size of each batch that will be yield
            shuffle: Whether the data in the file will be shuffled. If set to be true, it will first randomly shuffle
                the order of files, and then shuflle the order of rows in each file.
            drop_last_batch: Whether the last batch of data should be dropped. If set to be true, then the last
                batch will get dropped if its size is smaller than the given batch_size.

        Yields:
            A dict mapping column names to the corresponding data fetch from that column.
        """
        self._rb_buffer = _RecordBatchesBuffer()

        # Extract schema if not already known
        dataset = self._get_dataset(shuffle)
        if self._schema is None:
            self._schema = dataset.schema

        for rb in _retryable_batches(dataset, batch_size=max(_DEFAULT_DATASET_BATCH_SIZE, batch_size)):
            if shuffle:
                rb = rb.take(np.random.permutation(rb.num_rows))
            self._rb_buffer.append(rb)
            while self._rb_buffer.num_rows >= batch_size:
                yield self._get_batches_from_buffer(batch_size)

        if self._rb_buffer.num_rows and not drop_last_batch:
            yield self._get_batches_from_buffer(batch_size)

    def to_pandas(self, limit: Optional[int] = None) -> pd.DataFrame:
        ds = self._get_dataset(shuffle=False)
        table = ds.to_table() if limit is None else ds.head(num_rows=limit)
        return table.to_pandas(split_blocks=True, self_destruct=True)

    def _get_dataset(self, shuffle: bool) -> pds.Dataset:
        format = self._format
        sources: list[Any] = []
        source_format = None
        for source in self._data_sources:
            if isinstance(source, str):
                sources.append(source)
                source_format = format or os.path.splitext(source)[-1]
            elif isinstance(source, data_source.DatasetInfo):
                if not self._kwargs.get("filesystem"):
                    self._kwargs["filesystem"] = ingestor_utils.get_dataset_filesystem(self._session, source)
                sources.extend(
                    ingestor_utils.get_dataset_files(self._session, source, filesystem=self._kwargs["filesystem"])
                )
                source_format = "parquet"
            elif isinstance(source, data_source.DataFrameInfo):
                # FIXME: This currently loads all result batches into memory so that it
                #        can be passed into pyarrow.dataset as a list/tuple of pa.RecordBatches
                #        We may be able to optimize this by splitting the result batches into
                #        in-memory (first batch) and file URLs (subsequent batches) and creating a
                #        union dataset.
                sources.append(_cast_if_needed(ingestor_utils.get_dataframe_arrow_table(self._session, source)))
                source_format = None  # Arrow Dataset expects "None" for in-memory datasets
            else:
                raise RuntimeError(f"Unsupported data source type: {type(source)}")

            # Make sure source types not mixed
            if format and format != source_format:
                raise RuntimeError(f"Unexpected data source format (expected {format}, found {source_format})")
            format = source_format

        # Re-shuffle input files on each iteration start
        if shuffle:
            np.random.shuffle(sources)
        pa_dataset: pds.Dataset = pds.dataset(sources, format=format, **self._kwargs)
        return pa_dataset

    def _get_batches_from_buffer(self, batch_size: int) -> dict[str, npt.NDArray[Any]]:
        """Generate new batches from the existing record batch buffer."""
        cnt_rbs_num_rows = 0
        candidates = []

        # Keep popping record batches in buffer until there are enough rows for a batch.
        while self._rb_buffer.num_rows and cnt_rbs_num_rows < batch_size:
            candidate = self._rb_buffer.popleft()
            cnt_rbs_num_rows += candidate.num_rows
            candidates.append(candidate)

        # When there are more rows than needed, slice the last popped batch to fit batch_size.
        if cnt_rbs_num_rows > batch_size:
            row_diff = cnt_rbs_num_rows - batch_size
            slice_target = candidates[-1]
            cut_off = slice_target.num_rows - row_diff
            to_merge = slice_target.slice(length=cut_off)
            left_over = slice_target.slice(offset=cut_off)
            candidates[-1] = to_merge
            self._rb_buffer.appendleft(left_over)

        res = _merge_record_batches(candidates)
        return _record_batch_to_arrays(res)


def _merge_record_batches(record_batches: list[pa.RecordBatch]) -> pa.RecordBatch:
    """Merge a list of arrow RecordBatches into one. Similar to MergeTables."""
    if not record_batches:
        return _EMPTY_RECORD_BATCH
    if len(record_batches) == 1:
        return record_batches[0]
    record_batches = list(filter(lambda rb: rb.num_rows > 0, record_batches))
    one_chunk_table = pa.Table.from_batches(record_batches).combine_chunks()
    batches = one_chunk_table.to_batches(max_chunksize=None)
    return batches[0]


def _record_batch_to_arrays(rb: pa.RecordBatch) -> dict[str, npt.NDArray[Any]]:
    """Transform the record batch to a (string, numpy array) dict."""
    batch_dict = {}
    for column, column_schema in zip(rb, rb.schema):
        # zero_copy_only=False because of nans. Ideally nans should have been imputed in feature engineering.
        array = column.to_numpy(zero_copy_only=False)
        # If this column is a list, use the underlying type from the list values. Since this is just one column,
        # there should only be one type within the list.
        # TODO: Refactor to reduce data copies.
        if isinstance(column_schema.type, pa.ListType):
            # Update dtype of outer array:
            array = np.array(array.tolist(), dtype=column_schema.type.value_type.to_pandas_dtype())

        batch_dict[column_schema.name] = array

    return batch_dict


def _retryable_batches(
    dataset: pds.Dataset, batch_size: int, max_retries: int = 3, delay: int = 0
) -> Iterator[pa.RecordBatch]:
    """Make the Dataset to_batches retryable."""
    retries = 0
    current_batch_index = 0

    while True:
        try:
            for batch_index, batch in enumerate(dataset.to_batches(batch_size=batch_size)):
                if batch_index < current_batch_index:
                    # Skip batches that have already been processed
                    continue

                yield batch
                current_batch_index = batch_index + 1
            # Exit the loop once all batches are processed
            break

        except Exception as e:
            if retries < max_retries:
                retries += 1
                logging.info(f"Error encountered: {e}. Retrying {retries}/{max_retries}...")
                time.sleep(delay)
            else:
                raise e


def _cast_if_needed(
    batch: Union[pa.Table, pa.RecordBatch], schema: Optional[pa.Schema] = None
) -> Union[pa.Table, pa.RecordBatch]:
    """
    Cast the batch to be compatible with downstream frameworks. Returns original batch if cast is not necessary.
    Besides casting types to match `schema` (if provided), this function also applies the following casting:
        - Decimal (fixed-point) types: Convert to float or integer types based on scale and byte length

    Args:
        batch: The PyArrow batch to cast if needed
        schema: Optional schema the batch should be casted to match. Note that compatibility type casting takes
            precedence over the provided schema, e.g. if the schema has decimal types the result will be further
            cast into integer/float types.

    Returns:
        The type-casted PyArrow batch, or the original batch if casting was not necessary
    """
    schema = schema or batch.schema
    assert len(batch.schema) == len(schema)
    fields = []
    cast_needed = False
    for field, target in zip(batch.schema, schema):
        # Need to convert decimal types to supported types. This behavior supersedes target schema data types
        if pa.types.is_decimal(target.type):
            byte_length = int(target.metadata.get(b"byteLength", 8))
            if int(target.metadata.get(b"scale", 0)) > 0:
                target = target.with_type(pa.float32() if byte_length == 4 else pa.float64())
            else:
                if byte_length == 2:
                    target = target.with_type(pa.int16())
                elif byte_length == 4:
                    target = target.with_type(pa.int32())
                else:  # Cap out at 64-bit
                    target = target.with_type(pa.int64())
        if not field.equals(target):
            cast_needed = True
            field = target
        fields.append(field)

    if cast_needed:
        return batch.cast(pa.schema(fields))
    return batch
