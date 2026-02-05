from typing import Optional

from snowflake.snowpark import DataFrame, Session
from snowflake.snowpark._internal import utils as snowpark_utils
from snowflake.snowpark.types import StringType, StructField, StructType


def list_stage_files(
    session: Session,
    stage_path: str,
    *,
    pattern: Optional[str] = None,
    column_name: str = "FILE_PATH",
) -> DataFrame:
    """
    List files from a Snowflake stage and return a DataFrame with fully qualified paths.

    This utility simplifies preparing file data for batch inference by converting
    stage file listings into a format ready for ModelVersion.run_batch().

    Args:
        session: Active Snowpark session.
        stage_path: Stage path (e.g., "@DB.SCHEMA.STAGE" or "@DB.SCHEMA.STAGE/subdir").
        pattern: Optional regex pattern to filter files (e.g., ".*\\.jpg").
        column_name: Name of the output column. Defaults to "FILE_PATH".

    Returns:
        DataFrame with a single column containing fully qualified stage paths.

    Raises:
        RuntimeError: If the LIST command fails or returns unexpected results.

    Example:
        List all files in a stage directory::

            >>> df = list_stage_files(session, "@MY_DB.MY_SCHEMA.MY_STAGE/path")
            >>> df.show()
            -------------------------------------------------
            |"FILE_PATH"                                    |
            -------------------------------------------------
            |@MY_DB.MY_SCHEMA.MY_STAGE/path/file1.avi    |
            |@MY_DB.MY_SCHEMA.MY_STAGE/path/file2.mp3    |
            -------------------------------------------------

        Filter files using a regex pattern::

            >>> df = list_stage_files(session, "@MY_DB.MY_SCHEMA.MY_STAGE/path", pattern=".*\\.jpg")
            >>> df.show()
            -------------------------------------------------
            |"FILE_PATH"                                    |
            -------------------------------------------------
            |@MY_DB.MY_SCHEMA.MY_STAGE/path/image1.jpg    |
            |@MY_DB.MY_SCHEMA.MY_STAGE/path/image2.jpg    |
            -------------------------------------------------

        Use a custom column name::

            >>> df = list_stage_files(
                        session,
                        "@MY_DB.MY_SCHEMA.MY_STAGE/path",
                        pattern=".*\\.jpg",
                        column_name="IMAGES"
                     )
            >>> df.show()
            -------------------------------------------------
            |"IMAGES"                                       |
            -------------------------------------------------
            |@MY_DB.MY_SCHEMA.MY_STAGE/path/image1.jpg    |
            |@MY_DB.MY_SCHEMA.MY_STAGE/path/image2.jpg    |
            -------------------------------------------------
    """
    if not stage_path.startswith("@"):
        stage_path = "@" + stage_path

    sql = f"LIST {stage_path}"
    if pattern:
        escaped_pattern = snowpark_utils.escape_single_quotes(pattern)  # type: ignore[no-untyped-call]
        sql += f" PATTERN = '{escaped_pattern}'"

    try:
        list_results = session.sql(sql).collect()
    except Exception as e:
        raise RuntimeError(f"Failed to list stage location '{stage_path}': {e}")

    fully_qualified_stage = stage_path.split("/", 1)[0]

    file_paths = []
    for row in list_results:
        row_dict = row.as_dict()
        name = row_dict.get("name")
        if not name:
            raise RuntimeError(f"Unexpected LIST result format, missing 'name' column: {row_dict}")
        # name looks like "<stage>/<relative_path>"
        try:
            _, relative_path = name.split("/", 1)
        except ValueError:
            raise RuntimeError(f"Unexpected LIST result format, invalid 'name' value: {name}")
        fully_qualified = f"{fully_qualified_stage}/{relative_path}"
        file_paths.append((fully_qualified,))

    return session.create_dataframe(file_paths, schema=StructType([StructField(column_name, StringType())]))
