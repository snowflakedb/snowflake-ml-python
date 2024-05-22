from typing import Any, List

import pandas as pd
from pyarrow import parquet as pq

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.lineage import data_source, lineage_utils
from snowflake.ml._internal.utils import import_utils
from snowflake.ml.fileset import snowfs

_PROJECT = "Dataset"
_SUBPROJECT = "DatasetReader"
TARGET_FILE_SIZE = 32 * 2**20  # The max file size for data loading.


class DatasetReader:
    """Snowflake Dataset abstraction which provides application integration connectors"""

    @telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
    def __init__(
        self,
        session: snowpark.Session,
        sources: List[data_source.DataSource],
    ) -> None:
        """Initialize a DatasetVersion object.

        Args:
            session: Snowpark Session to interact with Snowflake backend.
            sources: Data sources to read from.

        Raises:
            ValueError: `sources` arg was empty or null
        """
        if not sources:
            raise ValueError("Invalid input: empty `sources` list not allowed")
        self._session = session
        self._sources = sources
        self._fs: snowfs.SnowFileSystem = snowfs.SnowFileSystem(
            snowpark_session=self._session,
            cache_type="bytes",
            block_size=2 * TARGET_FILE_SIZE,
        )

        self._files: List[str] = []

    def _list_files(self) -> List[str]:
        """Private helper function that lists all files in this DatasetVersion and caches the results."""
        if self._files:
            return self._files

        files: List[str] = []
        for source in self._sources:
            # Sort within each source for consistent ordering
            files.extend(sorted(self._fs.ls(source.url)))  # type: ignore[arg-type]
        files.sort()

        self._files = files
        return self._files

    @property
    def data_sources(self) -> List[data_source.DataSource]:
        return self._sources

    @telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
    def files(self) -> List[str]:
        """Get the list of remote file paths for the current DatasetVersion.

        The file paths follows the snow protocol.

        Returns:
            A list of remote file paths

        Example:
        >>> dsv.files()
        ----
        ["snow://dataset/mydb.myschema.mydataset/versions/test/data_0_0_0.snappy.parquet",
         "snow://dataset/mydb.myschema.mydataset/versions/test/data_0_0_1.snappy.parquet"]
        """
        files = self._list_files()
        return [self._fs.unstrip_protocol(f) for f in files]

    @telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
    def filesystem(self) -> snowfs.SnowFileSystem:
        """Return an fsspec FileSystem which can be used to load the DatasetVersion's `files()`"""
        return self._fs

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        func_params_to_log=["batch_size", "shuffle", "drop_last_batch"],
    )
    def to_torch_datapipe(self, *, batch_size: int, shuffle: bool = False, drop_last_batch: bool = True) -> Any:
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

        Examples:
        >>> dp = dataset.to_torch_datapipe(batch_size=1)
        >>> for data in dp:
        >>>     print(data)
        ----
        {'_COL_1':[10]}
        """
        IterableWrapper, _ = import_utils.import_or_get_dummy("torchdata.datapipes.iter.IterableWrapper")
        torch_datapipe_module, _ = import_utils.import_or_get_dummy("snowflake.ml.fileset.torch_datapipe")

        self._fs.optimize_read(self._list_files())

        input_dp = IterableWrapper(self._list_files())
        return torch_datapipe_module.ReadAndParseParquet(input_dp, self._fs, batch_size, shuffle, drop_last_batch)

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        func_params_to_log=["batch_size", "shuffle", "drop_last_batch"],
    )
    def to_tf_dataset(self, *, batch_size: int, shuffle: bool = False, drop_last_batch: bool = True) -> Any:
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

        Examples:
        >>> dp = dataset.to_tf_dataset(batch_size=1)
        >>> for data in dp:
        >>>     print(data)
        ----
        {'_COL_1': <tf.Tensor: shape=(1,), dtype=int64, numpy=[10]>}
        """
        tf_dataset_module, _ = import_utils.import_or_get_dummy("snowflake.ml.fileset.tf_dataset")

        self._fs.optimize_read(self._list_files())

        return tf_dataset_module.read_and_parse_parquet(
            self._list_files(), self._fs, batch_size, shuffle, drop_last_batch
        )

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
        func_params_to_log=["only_feature_cols"],
    )
    def to_snowpark_dataframe(self, only_feature_cols: bool = False) -> snowpark.DataFrame:
        """Convert the DatasetVersion to a Snowpark DataFrame.

        Args:
            only_feature_cols: If True, drops exclude_cols and label_cols from returned DataFrame.
                The original DatasetVersion is unaffected.

        Returns:
            A Snowpark dataframe that contains the data of this DatasetVersion.

        Note: The dataframe generated by this method might not have the same schema as the original one. Specifically,
            - NUMBER type with scale != 0 will become float.
            - Unsupported types (see comments of :func:`Dataset.create_version`) will not have any guarantee.
                For example, an OBJECT column may be scanned back as a STRING column.
        """
        file_path_pattern = ".*data_.*[.]parquet"
        dfs: List[snowpark.DataFrame] = []
        for source in self._sources:
            df = self._session.read.option("pattern", file_path_pattern).parquet(source.url)
            if only_feature_cols and source.exclude_cols:
                df = df.drop(source.exclude_cols)
            dfs.append(df)

        combined_df = dfs[0]
        for df in dfs[1:]:
            combined_df = combined_df.union_all_by_name(df)
        return lineage_utils.patch_dataframe(combined_df, data_sources=self._sources, inplace=True)

    @telemetry.send_api_usage_telemetry(project=_PROJECT, subproject=_SUBPROJECT)
    def to_pandas(self) -> pd.DataFrame:
        """Retrieve the DatasetVersion contents as a Pandas Dataframe"""
        files = self._list_files()
        if not files:
            return pd.DataFrame()  # Return empty DataFrame
        self._fs.optimize_read(files)
        pd_ds = pq.ParquetDataset(files, filesystem=self._fs)
        return pd_ds.read_pandas().to_pandas()
