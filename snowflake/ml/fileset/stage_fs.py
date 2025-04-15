import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Union, cast

import fsspec
from fsspec.implementations import http as httpfs

from snowflake import snowpark
from snowflake.connector import connection, errorcode, errors as snowpark_errors
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
    fileset_error_messages,
    fileset_errors,
)
from snowflake.snowpark import exceptions as snowpark_exceptions
from snowflake.snowpark._internal import utils as snowpark_utils
from snowflake.snowpark._internal.analyzer import snowflake_plan

# The default length of how long a presigned url stays active in seconds.
# Presigned url here is used to fetch file objects from Snowflake when SFStageFileSystem.open() is called.
_PRESIGNED_URL_LIFETIME_SEC = 14400

# The threshold of when the presigned url should get refreshed before its expiration.
_PRESIGNED_URL_HEADROOM_SEC = 3600


_PROJECT = "FileSet"


@dataclass(frozen=True)
class _PresignedUrl:
    """File system metadata for each stage file."""

    __slots__ = ["url", "expire_at"]
    url: str
    expire_at: float

    def is_expiring(self, headroom_sec: float = _PRESIGNED_URL_HEADROOM_SEC) -> bool:
        """Check if a token is going to expire in <headroom_sec> seconds."""
        return not self.expire_at or time.time() > self.expire_at - headroom_sec


def _get_httpfs_kwargs(**kwargs: Any) -> dict[str, Any]:
    """Extract kwargs that are meaningful to HTTPFileSystem."""
    httpfs_related_keys = [
        "block_size",
        "cache_type",
        "cache_options",
    ]
    httpfs_related_kwargs = {}
    for key in httpfs_related_keys:
        v = kwargs.get(key, None)
        if v:
            httpfs_related_kwargs[key] = v
    return httpfs_related_kwargs


class SFStageFileSystem(fsspec.AbstractFileSystem):
    """ A Snowflake stage file system based on fsspec (https://filesystem-spec.readthedocs.io/).

    It grants user readonly access to Snowflake stage as if it were a file system. It exposes a
    filesystem-like API (ls, open) on top of Snowflake internal stage storage.

    Example: Create FS object and do file operation
    --------
    >>> conn = snowflake.connector.connect(**connection_parameters)
    >>> sffs = SFStageFileSystem(db="MYDB", schema="public", stage="FOO", sf_connection=conn)
    >>> sffs.ls("nytrain")
    ['nytrain/data_0_0_0.csv', 'nytrain/data_0_0_1.csv']
    >>> with sffs.open('nytrain/data_0_0_1.csv', mode='rb') as f:
    >>>     print(f.readline())
    b'2014-02-05 14:35:00.00000054,13,2014-02-05 14:35:00 UTC,\
        -74.00688,40.73049,-74.00563,40.70676,2\n'
    """

    # Cached state marking whether we should use Snowpark file download instead of pre-signed URLs:
    #   None -> Try pre-signed URL access, fall back to file download
    #   True -> Use file download path without trying pre-signed URL access
    #   False -> Use pre-signed URL access, skip download fallback on failure
    _USE_FALLBACK_FILE_ACCESS = (
        True if snowpark_utils.is_in_stored_procedure() else None  # type: ignore[no-untyped-call]
    )

    def __init__(
        self,
        *,
        db: str,
        schema: str,
        stage: str,
        snowpark_session: Optional[snowpark.Session] = None,
        sf_connection: Optional[connection.SnowflakeConnection] = None,
        **kwargs: Any,
    ) -> None:
        """Initiate the file system with stage information and a snowflake connection.

        Args:
            db: The database name of the target internal stage.
            schema: The schema name of the target internal stage.
            stage: The name of the target stage.
            snowpark_session: A Snowpark session object. Mutually exclusive to `sf_connection`.
            sf_connection: A Snowflake python connection object. Mutually exclusive to `snowpark_session`.
            **kwargs : Optional. Other parameters that can be passed on to fsspec. Currently supports:
                - skip_instance_cache: Int. Controls reuse of instances.
                - cache_type, cache_options, block_size: Configure file buffering.
                See more information in https://filesystem-spec.readthedocs.io/en/latest/features.html

        Raises:
            ValueError: An error occurred when not exactly one of sf_connection and snowpark_session is given.
        """
        if sf_connection and snowpark_session:
            raise ValueError(fileset_error_messages.BOTH_SF_CONNECTION_AND_SNOWPARK_SESSION_SPECIFIED)
        if not sf_connection and not snowpark_session:
            raise ValueError(fileset_error_messages.NO_SF_CONNECTION_OR_SNOWPARK_SESSION)
        if sf_connection:
            self._session = snowpark.Session.builder.config("connection", sf_connection).create()
        else:
            self._session = snowpark_session

        logging.debug(f"Creating new stage file system on @{db}.{schema}.{stage}")
        self._db = db
        self._schema = schema
        self._stage = stage
        self._url_cache: dict[str, _PresignedUrl] = {}

        httpfs_kwargs = _get_httpfs_kwargs(**kwargs)
        self._fs = httpfs.HTTPFileSystem(**httpfs_kwargs)

        super().__init__(**kwargs)

    @property
    def stage_name(self) -> str:
        """Get the Snowflake path to this stage.

        Returns:
            String in the format of "@{database}.{schema}.{stage}".
                Example: @mydb.myschema.mystage
        """
        return f"@{self._db}.{self._schema}.{self._stage}"

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        func_params_to_log=["detail"],
    )
    def ls(self, path: str, detail: bool = False) -> Union[list[str], list[dict[str, Any]]]:
        """Override fsspec `ls` method. List single "directory" with or without details.

        Args:
            path: Relative file paths in the stage.
                Example:
                    "": Empty string points to the stage root.
                    "mydir": Points to the "mydir" file/directory under the stage.
                    "dir1/dir2": Points to the "dir2" file/directory in the "dir1" directory.
            detail: Whether to present detailed information of results. If set to be False, a list of filenames will be
                returned. If set to be True, each list item in the result is a dict, whose keys are "name", "size",
                "type", "md5" and "last_modified".

        Returns:
            A list of filename if `detail` is false, or a list of dict if `detail` is true.

        Raises:
            SnowflakeMLException: An error occurred when the given path points to a stage that cannot be found.
            SnowflakeMLException: An error occurred when Snowflake cannot list files in the given stage path.
        """
        try:
            loc = self.stage_name
            path = path.lstrip("/")
            async_job: snowpark.AsyncJob = self._session.sql(f"LIST '{loc}/{path}'").collect(block=False)
            objects: list[snowpark.Row] = _resolve_async_job(async_job)
        except snowpark_exceptions.SnowparkSQLException as e:
            if e.sql_error_code == fileset_errors.ERRNO_DOMAIN_NOT_EXIST:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.SNOWML_NOT_FOUND,
                    original_exception=fileset_errors.StageNotFoundError(
                        f"Stage {loc} does not exist or is not authorized."
                    ),
                )
            else:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWML_ERROR,
                    original_exception=fileset_errors.FileSetError(str(e)),
                )
        files = self._parse_list_result(objects, path)
        if detail:
            return files
        else:
            return [f["name"] for f in files]

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
    )
    def optimize_read(self, files: Optional[list[str]] = None) -> None:
        """Prefetch and cache the presigned urls for all the given files to speed up the read performance.

        All the files introduced here will have their urls cached. Further open() on any of cached urls will lead to a
        batch refreshment of all the cached urls if that url is inactive.

        Args:
            files: A list of file paths. If not given, all the files that have urls cached will refresh their url cache.
        """
        if not files:
            files = list(self._url_cache.keys())
        url_lifetime = _PRESIGNED_URL_LIFETIME_SEC
        start_time = time.time()
        logging.info(f"Start batch fetching presigned urls for {self.stage_name}.")
        presigned_urls = self._fetch_presigned_urls(files, url_lifetime)
        expire_at = start_time + url_lifetime

        for presigned_url in presigned_urls:
            file_path, url = presigned_url
            self._url_cache[file_path] = _PresignedUrl(url, expire_at)
            logging.debug(f"Retrieved presigned url for {file_path}.")
        logging.info(f"Finished batch fetching presigned urls for {self.stage_name}.")

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
    )
    def _open(self, path: str, mode: str = "rb", **kwargs: Any) -> fsspec.spec.AbstractBufferedFile:
        """Override fsspec `_open` method. Open a file for reading.

        The opened file will be readable for 4 hours. After that, you need to reopen the file.

        Args:
            path: Path of file in Snowflake stage.
            mode: One of 'r', 'rb'. These have the same meaning as they do for the built-in `open` function.
            **kwargs: Extra options that supported by fsspec. See more in
                https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.open

        Returns:
            A fsspec file-like object.

        Raises:
            SnowflakeMLException: An error occurred when the given path points to a file that cannot be found.
            snowpark_exceptions.SnowparkClientException: File access failed with a Snowpark exception
        """
        path = path.lstrip("/")
        if self._USE_FALLBACK_FILE_ACCESS:
            return self._open_with_snowpark(path)
        cached_presigned_url = self._url_cache.get(path, None)
        try:
            if not cached_presigned_url:
                res = self._fetch_presigned_urls([path])
                url = res[0][1]
                expire_at = time.time() + _PRESIGNED_URL_LIFETIME_SEC
                cached_presigned_url = _PresignedUrl(url, expire_at)
                self._url_cache[path] = cached_presigned_url
                logging.debug(f"Retrieved presigned url for {path}.")
            elif cached_presigned_url.is_expiring():
                self.optimize_read()
                cached_presigned_url = self._url_cache[path]
        except snowpark_exceptions.SnowparkClientException as e:
            if self._USE_FALLBACK_FILE_ACCESS == False:  # noqa: E712 # Fallback disabled
                raise
            # This may be an intermittent failure, so don't set _USE_FALLBACK_FILE_ACCESS = True
            logging.warning(f"Pre-signed URL generation failed with {e.message}, trying fallback file access")
            return self._open_with_snowpark(path)
        url = cached_presigned_url.url
        try:
            return self._fs._open(url, mode=mode, **kwargs)
        except FileNotFoundError:
            # Enable fallback if _USE_FALLBACK_FILE_ACCESS is True or None; set to False to disable
            if self._USE_FALLBACK_FILE_ACCESS != False:  # noqa: E712
                content = self._open_with_snowpark(path)
                self._USE_FALLBACK_FILE_ACCESS = True
                return content
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.SNOWML_NOT_FOUND,
                original_exception=fileset_errors.StageFileNotFoundError(f"Stage file {path} doesn't exist."),
            )

    def _open_with_snowpark(self, path: str, **kwargs: dict[str, Any]) -> fsspec.spec.AbstractBufferedFile:
        """Open the a file for reading using snowflake.snowpark.file_operation

        Args:
            path: Path of file in Snowflake stage.
            **kwargs: Extra options to pass to snowflake.snowpark.file_operation.get_stream

        Returns:
            A fsspec file-like object.

        Raises:
            SnowflakeMLException: An error occurred when the given path points to a file that cannot be found.
            SnowflakeMLException: An unknown Snowpark error occurred during file read.
        """
        try:
            return self._session.file.get_stream(f"{self.stage_name}/{path}", **kwargs)
        except snowpark_exceptions.SnowparkSQLException as e:
            if _match_error_code(e, errorcode.ER_FILE_NOT_EXISTS):
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.SNOWML_NOT_FOUND,
                    original_exception=fileset_errors.StageFileNotFoundError(f"Stage file {path} doesn't exist."),
                )
            else:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                    original_exception=e,
                )

    def _parse_list_result(self, list_result: list[snowpark.Row], search_path: str) -> list[dict[str, Any]]:
        """Convert the result from LIST query to the expected format of fsspec ls() method.

        Note that Snowflake LIST query has different behavior with ls(). LIST query will return all the stage files
        whose path starts with the given search_path. However, ls() should only return files that either exactly
        match the given search_path or are under the given search_path directory.

        For example, both "mydir/hello_world.txt" and "mydir1/hello_world.txt" will appear as a result of "LIST mydir",
        but only the former one is in "mydir" directory. As a result, only "mydir/hello_world.txt" should be returned by
        ls("mydir").

        Args:
            list_result: The result of LIST query, a list where each item is a Snowpark Row with four items:
                name, size, md5 and last_modified.
            search_path: The path that was used by the List query to get the list_result.

        Returns:
            A list of dict, where each dict contains key-value pairs as the properties of a file.
        """
        files: dict[str, dict[str, Any]] = {}
        search_path = search_path.strip("/")
        for row in list_result:
            name, size, md5, last_modified = row["name"], row["size"], row["md5"], row["last_modified"]
            obj_path = self._stage_path_to_relative_path(name)
            if obj_path == search_path:
                # If there is a exact match, then the matched object will always be a file object.
                self._add_file_info_helper(files, obj_path, size, "file", md5, last_modified)
                continue
            elif search_path and not obj_path.startswith(search_path + "/"):
                # If the path doesn't start with "<search_path>/", the object is not under the <search_path> directory.
                continue

            # Now we want to distinguish whether the object is in a subdirecotry of the given search path.
            rel_file_path = obj_path[len(search_path) :].lstrip("/")
            slash_idx = rel_file_path.find("/")
            if slash_idx == -1:
                # There is no subdirectory.
                self._add_file_info_helper(files, obj_path, size, "file", md5, last_modified)
            else:
                # There is a subdirectory. We will only add the top level subdirectory to the result of ls().
                dir_path = "" if not search_path else f"{search_path}/"
                dir_path += f"{rel_file_path[:slash_idx + 1]}"
                self._add_file_info_helper(files, dir_path, 0, "directory", None, None)
        return list(files.values())

    def _stage_path_to_relative_path(self, stage_path: str) -> str:
        """Convert a stage file path which comes from the LIST query to a relative file path in that stage.

        The file path returned by LIST query always has the format "<stage_name>/<relative_file_path>".
                Only <relative_file_path> will be returned.

        Args:
            stage_path: A string started with the name of the stage.

        Returns:
            A string of the relative stage path.
        """
        return stage_path[len(self._stage) + 1 :]

    def _add_file_info_helper(
        self,
        files: dict[str, dict[str, Any]],
        object_path: str,
        file_size: int,
        file_type: str,
        md5: Optional[str],
        last_modified: Optional[str],
    ) -> None:
        files.setdefault(
            object_path,
            {
                "name": object_path,
                "size": file_size,
                "type": file_type,
                "md5": md5,
                "last_modified": last_modified,
            },
        )

    def _fetch_presigned_urls(
        self, files: list[str], url_lifetime: float = _PRESIGNED_URL_LIFETIME_SEC
    ) -> list[tuple[str, str]]:
        """Fetch presigned urls for the given files."""
        file_df = self._session.create_dataframe(files).to_df("name")
        try:
            presigned_urls: list[tuple[str, str]] = file_df.select_expr(
                f"name, get_presigned_url('{self.stage_name}', name, {url_lifetime}) as url"
            ).collect(
                statement_params=telemetry.get_function_usage_statement_params(
                    project=_PROJECT,
                    function_name=telemetry.get_statement_params_full_func_name(
                        inspect.currentframe(), self.__class__.__name__
                    ),
                    api_calls=[snowpark.DataFrame.collect],
                ),
            )
        except snowpark_exceptions.SnowparkSQLException as e:
            if e.sql_error_code in {fileset_errors.ERRNO_DOMAIN_NOT_EXIST, fileset_errors.ERRNO_STAGE_NOT_EXIST}:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.SNOWML_NOT_FOUND,
                    original_exception=fileset_errors.StageNotFoundError(
                        f"Stage {self.stage_name} does not exist or is not authorized."
                    ),
                )
            else:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWML_ERROR,
                    original_exception=fileset_errors.FileSetError(str(e)),
                )
        return presigned_urls


def _match_error_code(ex: snowpark_exceptions.SnowparkSQLException, error_code: int) -> bool:
    # Snowpark writes error code to message instead of populating e.sql_error_code
    error_code_str = str(error_code)
    return ex.sql_error_code == error_code_str or error_code_str in ex.message


@snowflake_plan.SnowflakePlan.Decorator.wrap_exception  # type: ignore[misc]
def _resolve_async_job(async_job: snowpark.AsyncJob) -> list[snowpark.Row]:
    # Make sure Snowpark exceptions are properly caught and converted by wrap_exception wrapper
    try:
        query_result = cast(list[snowpark.Row], async_job.result("row"))
        return query_result
    except snowpark_errors.DatabaseError as e:
        # HACK: Snowpark surfaces a generic exception if query doesn't complete immediately
        # assume it's due to FileNotFound
        if type(e) is snowpark_errors.DatabaseError and "results are unavailable" in str(e):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.SNOWML_NOT_FOUND,
                original_exception=fileset_errors.StageNotFoundError("Query failed."),
            ) from e
        assert e.msg is not None
        raise snowpark_exceptions.SnowparkSQLException(e.msg, conn_error=e) from e
