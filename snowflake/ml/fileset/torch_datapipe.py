from typing import Any, Dict, Iterator

import fsspec
import numpy.typing as npt
from torchdata.datapipes.iter import IterDataPipe

from snowflake.ml.fileset import parquet_parser


class ReadAndParseParquet(IterDataPipe):
    """Read and parse the parquet files yield batched numpy array in dict.

    Args:
        input_datapipe: A datapipe of input parquet file URIs to read and parse.
            Note that the datapipe must be finite.
        filesystem: A fsspec/pyarrow file system that is used to open given file URIs.
        batch_size: Specifies the size of each batch that will be yield
        shuffle: Whether the data in the file will be shuffled. If set to be true, it will first randomly shuffle
            the order of files, and then shuflle the order of rows in each file.
        drop_last_batch: Whether the last batch of data should be dropped. If set to be true, then the last batch will
            get dropped if its size is smaller than the given batch_size.

    Returns:
        A PyTorch iterable datapipe that yields batched numpy array in dict. The keys will be the column names in
        the parquet files.

    Example:
        >>> from snowflake.ml.fileset import sfcfs, torch_datapipe
        >>> from torchdata.datapipes.iter import FSSpecFileLister
        >>> conn = snowflake.connector.connect(**connection_parameters)
        >>> fs = sfcfs.SFFileSystem(conn)
        >>> filedp = FSSpecFileLister(root=dir_path, masks="*.parquet", mode="rb", sf_connection=conn)
        >>> parquet_dp = torch_datapipe.ReadAndParseParquet(file_dp, fs, batch_size = 2)
        >>> for batch in parquet_dp:
        >>>     print(batch)
    ----
    {'_COL_1': [32.5000, 6.0000], '_COL_2': [-73.9542, -73.9875]}
    """

    def __init__(
        self,
        input_datapipe: IterDataPipe[str],
        filesystem: fsspec.AbstractFileSystem,
        batch_size: int,
        shuffle: bool,
        drop_last_batch: bool,
    ) -> None:
        self._input_datapipe = input_datapipe
        self._fs = filesystem
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last_batch = drop_last_batch

    def __iter__(self) -> Iterator[Dict[str, npt.NDArray[Any]]]:
        yield from parquet_parser.ParquetParser(
            list(self._input_datapipe), self._fs, self._batch_size, self._shuffle, self._drop_last_batch
        )
