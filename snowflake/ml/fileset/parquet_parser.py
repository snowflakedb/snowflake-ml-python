import collections
from typing import Any, Deque, Dict, Iterator, List

import fsspec
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.dataset as ds

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


class ParquetParser:
    """Read and parse the given parquet files and yield batched numpy array in dict.

    Args:
        file_paths: A list of parquet file URIs to read and parse.
        filesystem: A fsspec/pyarrow file system that is used to open given file URIs.
        batch_size: Specifies the size of each batch that will be yield
        shuffle: Whether the data in the file will be shuffled. If set to be true, it will first randomly shuffle
            the order of files, and then shuflle the order of rows in each file.
        drop_last_batch: Whether the last batch of data should be dropped. If set to be true, then the last batch will
            get dropped if its size is smaller than the given batch_size.

    Returns:
        A PyTorch iterable datapipe that yields batched numpy array in dict. The keys will be the column names in
        the parquet files, and the value will be the column value as a list.
    """

    def __init__(
        self,
        file_paths: List[str],
        filesystem: fsspec.AbstractFileSystem,
        batch_size: int,
        shuffle: bool = True,
        drop_last_batch: bool = True,
    ) -> None:
        self._file_paths = file_paths
        self._fs = filesystem
        self._batch_size = batch_size
        self._dataset_batch_size = max(_DEFAULT_DATASET_BATCH_SIZE, self._batch_size)
        self._shuffle = shuffle
        self._drop_last_batch = drop_last_batch

    def __iter__(self) -> Iterator[Dict[str, npt.NDArray[Any]]]:
        """Iterate through PyArrow Dataset to generate batches whose length equals to expected batch size.

        As we are generating batches with the exactly same length, the last few rows in each file might get left as they
        are not long enough to form a batch. These rows will be put into a temporary buffer and combine with the first
        few rows of the next file to generate a new batch.

        Yields:
            A dict mapping column names to the corresponding data fetch from that column.
        """
        self._rb_buffer = _RecordBatchesBuffer()
        files = list(self._file_paths)
        if self._shuffle:
            np.random.shuffle(files)
        pa_dataset: ds.Dataset = ds.dataset(files, format="parquet", filesystem=self._fs)

        for rb in pa_dataset.to_batches(batch_size=self._dataset_batch_size):
            if self._shuffle:
                rb = rb.take(np.random.permutation(rb.num_rows))
            self._rb_buffer.append(rb)
            while self._rb_buffer.num_rows >= self._batch_size:
                yield self._get_batches_from_buffer()

        if self._rb_buffer.num_rows and not self._drop_last_batch:
            yield self._get_batches_from_buffer()

    def _get_batches_from_buffer(self) -> Dict[str, npt.NDArray[Any]]:
        """Generate new batches from the existing record batch buffer."""
        cnt_rbs_num_rows = 0
        candidates = []

        # Keep popping record batches in buffer until there are enough rows for a batch.
        while self._rb_buffer.num_rows and cnt_rbs_num_rows < self._batch_size:
            candidate = self._rb_buffer.popleft()
            cnt_rbs_num_rows += candidate.num_rows
            candidates.append(candidate)

        # When there are more rows than needed, slice the last popped batch to fit batch_size.
        if cnt_rbs_num_rows > self._batch_size:
            row_diff = cnt_rbs_num_rows - self._batch_size
            slice_target = candidates[-1]
            cut_off = slice_target.num_rows - row_diff
            to_merge = slice_target.slice(length=cut_off)
            left_over = slice_target.slice(offset=cut_off)
            candidates[-1] = to_merge
            self._rb_buffer.appendleft(left_over)

        res = _merge_record_batches(candidates)
        return _record_batch_to_arrays(res)


def _merge_record_batches(record_batches: List[pa.RecordBatch]) -> pa.RecordBatch:
    """Merge a list of arrow RecordBatches into one. Similar to MergeTables."""
    if not record_batches:
        return _EMPTY_RECORD_BATCH
    if len(record_batches) == 1:
        return record_batches[0]
    record_batches = list(filter(lambda rb: rb.num_rows > 0, record_batches))
    one_chunk_table = pa.Table.from_batches(record_batches).combine_chunks()
    batches = one_chunk_table.to_batches(max_chunksize=None)
    return batches[0]


def _record_batch_to_arrays(rb: pa.RecordBatch) -> Dict[str, npt.NDArray[Any]]:
    """Transform the record batch to a (string, numpy array) dict."""
    batch_dict = {}
    for column, column_schema in zip(rb, rb.schema):
        # zero_copy_only=False because of nans. Ideally nans should have been imputed in feature engineering.
        array = column.to_numpy(zero_copy_only=False)
        batch_dict[column_schema.name] = array
    return batch_dict
