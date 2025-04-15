import collections
import logging
from functools import partial
from typing import Any, Callable, Optional, Union, cast

import fsspec

from snowflake import snowpark
from snowflake.connector import connection
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils import identifier
from snowflake.ml.fileset import stage_fs
from snowflake.ml.utils import connection_params
from snowflake.snowpark import context, exceptions as snowpark_exceptions

PROTOCOL_NAME = "sfc"

_SFFilePath = collections.namedtuple("_SFFilePath", ["database", "schema", "stage", "filepath"])
_PROJECT = "FileSet"

_FILESYSTEM_KWARGS_KEY = "kwargs"
_RECREATE_FROM_SERIALIZED = "recreate_from_serialized"


class SFFileSystem(fsspec.AbstractFileSystem):
    """A filesystem that allows user to access Snowflake stages and stage files with valid Snowflake locations.

    The file system is is based on fsspec (https://filesystem-spec.readthedocs.io/). It is a file system wrapper
    built on top of SFStageFileSystem. It takes Snowflake stage file path as the input and supports read operation.
    A valid Snowflake location will have the form "@{database_name}.{schema_name}.{stage_name}/{path_to_file}".

    Example 1: Create a file system object and do file operation
    --------
    >>> conn = snowflake.connector.connect(**connection_parameters)
    >>> sffs = SFFileSystem(sf_connection=conn)
    >>> sffs.ls("@MYDB.public.FOO/nytrain")
    ['@MYDB.public.FOO/nytrain/data_0_0_0.csv', '@MYDB.public.FOO/nytrain/data_0_0_1.csv']
    >>> with sffs.open('@MYDB.public.FOO/nytrain/nytrain/data_0_0_1.csv', mode='rb') as f:
    >>>     print(f.readline())
    b'2014-02-05 14:35:00.00000054,13,2014-02-05 14:35:00 UTC,-74.00688,40.73049,-74.00563,40.70676,2\n'

    Example 2: Use SFC protocol to create a file system object via fsspec
    --------
    >>> conn = snowflake.connector.connect(**connection_parameters)
    >>> sffs = fsspec.filesystem("sfc", sf_connection=conn)
    >>> sffs.ls("@MYDB.public.FOO/nytrain")
    ['@MYDB.public.FOO/nytrain/data_0_0_0.csv', '@MYDB.public.FOO/nytrain/data_0_0_1.csv']
    >>> with sffs.open('@MYDB.public.FOO/nytrain/nytrain/data_0_0_1.csv', mode='rb') as f:
    >>>     print(f.readline())
    b'2014-02-05 14:35:00.00000054,13,2014-02-05 14:35:00 UTC,-74.00688,40.73049,-74.00563,40.70676,2\n'

    Example 3: Use SFC protocol and open files via fsspec without creating a file system object explicitly
    --------
    >>> conn = snowflake.connector.connect(**connection_parameters)
    >>> with fsspec.open("sfc://@MYDB.public.FOO/nytrain/data_0_0_1.csv", mode='rb', sf_connection=conn) as f:
    >>>     print(f.readline())
    b'2014-02-05 14:35:00.00000054,13,2014-02-05 14:35:00 UTC,-74.00688,40.73049,-74.00563,40.70676,2\n'
    """

    protocol = PROTOCOL_NAME

    def __init__(
        self,
        sf_connection: Optional[connection.SnowflakeConnection] = None,
        snowpark_session: Optional[snowpark.Session] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize file system with a Snowflake Python connection.

        Args:
            sf_connection: A Snowflake python connection object. Either it or snowpark_session must be non-empty.
            snowpark_session: A Snowpark session. Either it or sf_connection must be non-empty.
            kwargs : Optional. Other parameters that can be passed on to fsspec. Currently supports:
                - skip_instance_cache: Int. Controls reuse of instances.
                - cache_type, cache_options, block_size: Configure file buffering.
                See more information of these options in https://filesystem-spec.readthedocs.io/en/latest/features.html

        Raises:
            ValueError: An error occurred when not exactly one of sf_connection and snowpark_session is given.
            SnowflakeMLException: A failure was encountered while recreating the SFFileSystem from a serialized state.
        """
        if kwargs.get(_RECREATE_FROM_SERIALIZED):
            try:
                snowpark_session = self._get_default_session()
            except Exception as e:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.SNOWML_DESERIALIZATION_FAILED,
                    original_exception=ValueError("Unable to deserialize SFFileSystem."),
                ) from e

        if sf_connection:
            self._session = snowpark.Session.builder.config("connection", sf_connection).create()
        elif snowpark_session:
            self._session = snowpark_session
        else:
            raise ValueError("Either sf_connection or snowpark_session has to be non-empty!")
        self._conn = self._session._conn._conn  # Telemetry wrappers expect connection under `conn_attr_name="_conn"``
        self._kwargs = kwargs
        self._stage_fs_set: dict[tuple[str, str, str], stage_fs.SFStageFileSystem] = {}

        super().__init__(**kwargs)

    def _get_default_session(self) -> snowpark.Session:
        """Create a Snowpark Session from default login options.

        Returns:
            An active Snowpark Session.

        Raises:
            ValueError: Snowflake Login Options could not be retrieved from default locations.
            ValueError: Snowflake Connection could not be created.

        """
        try:
            return context.get_active_session()
        except snowpark_exceptions.SnowparkSessionException:
            pass

        try:
            snowflake_config = connection_params.SnowflakeLoginOptions()
        except Exception as e:
            raise ValueError("Unable to retrieve Snowflake Login Options.") from e

        try:
            session = snowpark.Session.builder.configs(snowflake_config).create()
        except Exception as e:
            raise ValueError("Unable to create Snowflake connection.") from e

        assert isinstance(session, snowpark.Session)
        return session

    def __reduce__(self) -> tuple[Callable[[], type["SFFileSystem"]], tuple[()], dict[str, Any]]:
        """Returns a state dictionary for use in serialization.

        Returns:
            A tuple that is used for recreating the SFFileSystem. For more information, refer to
            https://docs.python.org/3/library/pickle.html#object.__reduce__
            A `partial` is used to generate a callable that accepts kwargs.
        """
        state_dictionary = {_FILESYSTEM_KWARGS_KEY: self._kwargs}

        return partial(self.__class__, **{_RECREATE_FROM_SERIALIZED: True}), (), state_dictionary

    def __setstate__(self, state_dict: dict[str, Any]) -> None:
        """Sets the dictionary state at deserialization time, and rebuilds a snowflake connection.

        Args:
            state_dict: State dictionary saved at serialization time.

        Raises:
            KeyError: The Kwargs key is not present in the state dictionary.
            ValueError: The value corresponding to the kwargs key is not a dictionary.

        """
        if _FILESYSTEM_KWARGS_KEY not in state_dict:
            raise KeyError(f"Serialized state dictionary missing key {_FILESYSTEM_KWARGS_KEY}.")

        kwargs_dict = state_dict.get(_FILESYSTEM_KWARGS_KEY)
        if not isinstance(kwargs_dict, dict):
            raise ValueError(f"The value corresponding to {_FILESYSTEM_KWARGS_KEY} is not a dictionary.")

        self._kwargs = kwargs_dict

    def _get_stage_fs(self, sf_file_path: _SFFilePath) -> stage_fs.SFStageFileSystem:
        """Get the stage file system for the given snowflake location.

        Args:
            sf_file_path: A SFFILEPATH namedtuple

        Returns:
            A SFStageFileSystem object which supports readonly file operations on Snowflake stages.
        """
        stage_fs_key = (sf_file_path.database, sf_file_path.schema, sf_file_path.stage)
        if stage_fs_key not in self._stage_fs_set:
            cnt_stage_fs = stage_fs.SFStageFileSystem(
                snowpark_session=self._session,
                db=sf_file_path.database,
                schema=sf_file_path.schema,
                stage=sf_file_path.stage,
                **self._kwargs,
            )
            self._stage_fs_set[stage_fs_key] = cnt_stage_fs
        return self._stage_fs_set[stage_fs_key]

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        func_params_to_log=["detail"],
        conn_attr_name="_conn",
    )
    def ls(self, path: str, detail: bool = False, **kwargs: Any) -> Union[list[str], list[dict[str, Any]]]:
        """Override fsspec `ls` method. List single "directory" with or without details.

        Args:
            path : location at which to list files.
                It should be in the format of "@{database}.{schema}.{stage}/{path}"
            detail : if True, each list item is a dict of file properties; otherwise, returns list of filenames.
            kwargs : additional arguments passed on.

        Returns:
            A list of filename if `detail` is false, or a list of dict if `detail` is true.

        Example:
        >>> sffs.ls("@MYDB.public.FOO/")
        ['@MYDB.public.FOO/nytrain']
        >>> sffs.ls("@MYDB.public.FOO/nytrain")
        ['@MYDB.public.FOO/nytrain/data_0_0_0.csv', '@MYDB.public.FOO/nytrain/data_0_0_1.csv']
        >>> sffs.ls("@MYDB.public.FOO/nytrain/")
        ['@MYDB.public.FOO/nytrain/data_0_0_0.csv', '@MYDB.public.FOO/nytrain/data_0_0_1.csv']
        """
        file_path = self._parse_file_path(path)
        stage_fs = self._get_stage_fs(file_path)
        stage_path_list = stage_fs.ls(file_path.filepath, detail=True, **kwargs)
        stage_path_list = cast(list[dict[str, Any]], stage_path_list)
        return self._decorate_ls_res(stage_fs, stage_path_list, detail)

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        conn_attr_name="_conn",
    )
    def optimize_read(self, files: Optional[list[str]] = None) -> None:
        """Prefetch and cache the presigned urls for all the given files to speed up the file opening.

        All the files introduced here will have their urls cached. Further open() on any of cached urls will lead to a
        batch refreshment of the cached urls in the same stage if that url is inactive.

        Args:
            files: A list of file paths that needs their presigned url cached.
        """
        if not files:
            return
        stage_fs_dict: dict[str, stage_fs.SFStageFileSystem] = {}
        stage_file_paths: dict[str, list[str]] = collections.defaultdict(list)
        for file in files:
            path_info = self._parse_file_path(file)
            fs = self._get_stage_fs(path_info)
            stage_fs_dict[fs.stage_name] = fs
            stage_file_paths[fs.stage_name].append(path_info.filepath)
        for k, v in stage_file_paths.items():
            stage_fs_dict[k].optimize_read(v)

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        conn_attr_name="_conn",
    )
    def _open(self, path: str, **kwargs: Any) -> fsspec.spec.AbstractBufferedFile:
        """Override fsspec `_open` method. Open a file for reading in 'rb' mode.

        The opened file will be readable for 4 hours. After that you will need to reopen it.

        Fsspec `open` API is built on `_open` method and supports non-binary mode.
        See more details in https://github.com/fsspec/filesystem_spec/blob/2022.10.0/fsspec/spec.py#L1019

        Args:
            path: Path of file in Snowflake stage.
                It should be in the format of "@{database}.{schema}.{stage}/{path}"
            **kwargs : Additional arguments passed on. See more in
                https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.open

        Returns:
            A fsspec AbstractBufferedFile which supports python file operations.
        """
        file_path = self._parse_file_path(path)
        stage_fs = self._get_stage_fs(file_path)
        return stage_fs._open(file_path.filepath, **kwargs)

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        conn_attr_name="_conn",
    )
    def info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Override fsspec `info` method. Give details of entry at path."""
        file_path = self._parse_file_path(path)
        stage_fs = self._get_stage_fs(file_path)
        res: dict[str, Any] = stage_fs.info(file_path.filepath, **kwargs)
        if res:
            res["name"] = self._stage_path_to_absolute_path(stage_fs, res["name"])
        return res

    def _decorate_ls_res(
        self,
        stage_fs: stage_fs.SFStageFileSystem,
        stage_path_list: list[dict[str, Any]],
        detail: bool,
    ) -> Union[list[str], list[dict[str, Any]]]:
        """Add the stage location as the prefix of file names returned by ls() of stagefs"""
        for path in stage_path_list:
            path["name"] = self._stage_path_to_absolute_path(stage_fs, path["name"])
        if detail:
            return stage_path_list
        else:
            return [path["name"] for path in stage_path_list]

    def _stage_path_to_absolute_path(self, stage_fs: stage_fs.SFStageFileSystem, path: str) -> str:
        """Convert the relative path in a stage to an absolute path starts with the location of the stage."""
        return stage_fs.stage_name + "/" + path

    @classmethod
    def _parse_file_path(cls, path: str) -> _SFFilePath:
        """Parse a snowflake location path.

        The following propertis will be extracted from the path input:
        - database
        - schema
        - stage
        - path (optional)

        Args:
            path: A string in the format of "@{database}.{schema}.{stage}/{path}".

                Example:
                    "@my_db.my_schema.my_stage/"
                    "@my_db.my_schema.my_stage/file1"
                    "@my_db.my_schema.my_stage/dir1/"
                    "@my_db.my_schema.my_stage/dir1/file2"

        Returns:
            A namedtuple consists of database name, schema name, stage name and path.

        Raises:
            SnowflakeMLException: An error occurred when invalid path is given.
        """
        sfc_prefix = f"{PROTOCOL_NAME}://"
        if path.startswith(sfc_prefix):
            path = path[len(sfc_prefix) :]
        if not path.startswith("@"):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.SNOWML_INVALID_STAGE,
                original_exception=ValueError(
                    'Invalid path. Expected path to start with "@". Example: @database.schema.stage/optional_path.'
                ),
            )
        try:
            res = identifier.parse_snowflake_stage_path(path[1:])
            if res[1] is None or res[0] is None or (res[3] and not res[3].startswith("/")):
                raise ValueError("Invalid path. Missing database or schema identifier.")
            logging.debug(f"Parsed path: {res}")
            return _SFFilePath(res[0], res[1], res[2], res[3][1:])
        except ValueError:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.SNOWML_INVALID_STAGE,
                original_exception=ValueError(
                    f"Invalid path. Expected format: @database.schema.stage/optional_path. Getting {path}"
                ),
            )


fsspec.register_implementation(PROTOCOL_NAME, SFFileSystem)
