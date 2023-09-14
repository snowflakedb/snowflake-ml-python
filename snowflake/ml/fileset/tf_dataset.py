from typing import Any, Dict, Generator, List

import fsspec
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml.fileset import parquet_parser


def read_and_parse_parquet(
    files: List[str],
    filesystem: fsspec.AbstractFileSystem,
    batch_size: int,
    shuffle: bool,
    drop_last_batch: bool,
) -> tf.data.Dataset:
    """Creates a tf.data.Dataset that reads given parquet files into batched Tensors.

    Args:
        files: A list of input parquet file URIs to read and parse. The parquet files should
            have the same schema.
        filesystem: A fsspec/pyarrow file system that is used to open given file URIs.
        batch_size: Specifies the size of each batch that will be yield. It is preferred to
            set it to your training batch size, and avoid using dataset.{batch(),rebatch()} later.
        shuffle: Whether the data in the file will be shuffled. If set to be true, it will first randomly shuffle
            the order of files, and then shuflle the order of rows in each file. It is preferred
            to shuffle the data this way than dataset.unbatch().shuffle().rebatch().
        drop_last_batch: Whether the last batch of data should be dropped. If set to be true, then the last batch will
            get dropped if its size is smaller than the given batch_size.

    Returns:
        A tf.data.Dataset generates batched Tensors in a dict. The keys will be the column names in
        the parquet files.

    Raises:
        SnowflakeMLException: if `files` is empty.

    Example:
        >>> from snowflake.ml.fileset import sfcfs, tf_dataset
        >>> conn = snowflake.connector.connect(**connection_parameters)
        >>> fs = sfcfs.SFFileSystem(conn)
        >>> files = fs.ls(dir_path)
        >>> ds = tf_dataset.parse_and_read_parquet(files, fs, batch_size = 2)
        >>> for batch in ds:
        >>>     print(batch)
    ----
    {'_COL_1': <tf.Tensor: shape=(2,), dtype=float32, numpy=[32.5000, 6.0000]>,
     '_COL_2': <tf.Tensor: shape=(2,), dtype=float32, numpy=[-73.9542, -73.9875]>}
    """
    if not files:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.SNOWML_READ_FAILED,
            original_exception=ValueError("At least one file is needed to create a TF dataset."),
        )

    def generator() -> Generator[Dict[str, npt.NDArray[Any]], None, None]:
        yield from parquet_parser.ParquetParser(list(files), filesystem, batch_size, shuffle, drop_last_batch)

    return tf.data.Dataset.from_generator(generator, output_signature=_derive_signature(files[0], filesystem))


def _arrow_type_to_tensor_spec(field: pa.Field) -> tf.TensorSpec:
    try:
        dtype = tf.dtypes.as_dtype(field.type.to_pandas_dtype())
    except TypeError:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_DATA_TYPE,
            original_exception=TypeError(f"Column {field.name} has unsupportd type {field.type}."),
        )
    # First dimension is batch dimension.
    return tf.TensorSpec(shape=(None,), dtype=dtype)


def _derive_signature(file: str, filesystem: fsspec.AbstractFileSystem) -> Dict[str, tf.TensorSpec]:
    """Derives the signature of the TF dataset from one parquet file."""
    # TODO(zpeng): pq.read_schema does not support `filesystem` until pyarrow>=10.
    # switch to pq.read_schema when we depend on that.
    schema = pq.read_table(file, filesystem=filesystem).schema
    # Signature:
    # The dataset yields dicts. Keys are column names; values are 1-D tensors (
    # the first dimension is batch dimension).
    return {field.name: _arrow_type_to_tensor_spec(field) for field in schema}
