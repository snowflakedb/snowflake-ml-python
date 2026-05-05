from typing import Optional, cast

from snowflake.snowpark import DataFrame, Session, functions
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
        # For external stages, name is a cloud URL like "s3://bucket/path/file.txt".
        # For internal stages, name is like "stage_name/path/file.txt".
        # We need the relative path after the stage root in both cases.
        if "://" in name:
            # External stage: strip scheme + bucket, e.g. "s3://bucket/path/file" -> "path/file"
            after_scheme = name.split("://", 1)[1]
            slash_idx = after_scheme.find("/")
            if slash_idx == -1 or slash_idx == len(after_scheme) - 1:
                raise RuntimeError(f"Unexpected LIST result format, invalid 'name' value: {name}")
            relative_path = after_scheme[slash_idx + 1 :]
        else:
            # Internal stage: strip stage name prefix
            try:
                _, relative_path = name.split("/", 1)
            except ValueError:
                raise RuntimeError(f"Unexpected LIST result format, invalid 'name' value: {name}")
        fully_qualified = f"{fully_qualified_stage}/{relative_path}"
        file_paths.append((fully_qualified,))

    return session.create_dataframe(file_paths, schema=StructType([StructField(column_name, StringType())]))


def list_stage_files_from_directory_tables(
    session: Session,
    stage_name: str,
    *,
    column_name: str = "FILE_PATH",
) -> DataFrame:
    """
    List files from a Snowflake stage's directory table and return a DataFrame with fully qualified paths.

    Unlike :func:`list_stage_files`, this utility queries the stage's directory table
    (``SELECT RELATIVE_PATH FROM DIRECTORY(@stage)``) rather than issuing a ``LIST`` command.
    The stage must have its directory table enabled and refreshed for the listing to be populated.

    The returned DataFrame is lazy — no query is executed by this call. The caller materializes
    the plan when they need the rows (e.g. ``ModelVersion.run_batch``, ``collect``, ``to_pandas``).

    Args:
        session: Active Snowpark session.
        stage_name: Fully qualified stage reference, e.g. ``"@DB.SCHEMA.STAGE"``. A subpath is
            not supported — only a bare stage reference is accepted.
        column_name: Name of the output column. Defaults to ``"FILE_PATH"``.

    Returns:
        DataFrame with a single column containing fully qualified stage paths, suitable as the
        input DataFrame to ``ModelVersion.run_batch()``.

    Raises:
        ValueError: If ``stage_name`` contains a subpath.

    Example:
        List all files in a stage via its directory table::

            >>> df = list_stage_files_from_directory_tables(session, "@MY_DB.MY_SCHEMA.MY_STAGE")
            >>> df.show()
            -------------------------------------------------
            |"FILE_PATH"                                    |
            -------------------------------------------------
            |@MY_DB.MY_SCHEMA.MY_STAGE/file1.avi            |
            |@MY_DB.MY_SCHEMA.MY_STAGE/file2.mp3            |
            -------------------------------------------------
    """
    if not stage_name.startswith("@"):
        stage_name = "@" + stage_name

    if "/" in stage_name[1:]:
        raise ValueError(
            f"stage_name must be a bare stage reference like '@DB.SCHEMA.STAGE' "
            f"without a subpath, got: {stage_name}"
        )

    return cast(
        DataFrame,
        session.sql(f"SELECT RELATIVE_PATH FROM DIRECTORY({stage_name})").select(
            functions.concat(functions.lit(f"{stage_name}/"), functions.col("RELATIVE_PATH")).alias(column_name)
        ),
    )
