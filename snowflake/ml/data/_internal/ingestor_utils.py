from typing import List, Optional

import fsspec

from snowflake import snowpark
from snowflake.connector import result_batch
from snowflake.ml.data import data_source
from snowflake.ml.fileset import snowfs

_TARGET_FILE_SIZE = 32 * 2**20  # The max file size for data loading.


def get_dataframe_result_batches(
    session: snowpark.Session, df_info: data_source.DataFrameInfo
) -> List[result_batch.ResultBatch]:
    cursor = session._conn._cursor

    if df_info.query_id:
        query_id = df_info.query_id
    else:
        query_id = session.sql(df_info.sql).collect_nowait().query_id

    # TODO: Check if query result cache is still live
    cursor.get_results_from_sfqid(sfqid=query_id)

    # Prefetch hook should be set by `get_results_from_sfqid`
    # This call blocks until the query results are ready
    if cursor._prefetch_hook is None:
        raise RuntimeError("Loading data from result query failed unexpectedly. Please contact Snowflake support.")
    cursor._prefetch_hook()
    batches = cursor.get_result_batches()
    if batches is None:
        raise ValueError(
            "Failed to retrieve training data. Query status:" f" {session._conn._conn.get_query_status(query_id)}"
        )
    return batches


def get_dataset_filesystem(
    session: snowpark.Session, ds_info: Optional[data_source.DatasetInfo] = None
) -> fsspec.AbstractFileSystem:
    # We can't directly load the Dataset to avoid a circular dependency
    # Dataset -> DatasetReader -> DataConnector -> DataIngestor -> (?) ingestor_utils -> Dataset
    # TODO: Automatically pick appropriate fsspec implementation based on protocol in URL
    return snowfs.SnowFileSystem(
        snowpark_session=session,
        cache_type="bytes",
        block_size=2 * _TARGET_FILE_SIZE,
    )


def get_dataset_files(
    session: snowpark.Session, ds_info: data_source.DatasetInfo, filesystem: Optional[fsspec.AbstractFileSystem] = None
) -> List[str]:
    if filesystem is None:
        filesystem = get_dataset_filesystem(session, ds_info)
    assert bool(ds_info.url)  # Not null or empty
    return sorted(filesystem.ls(ds_info.url))
