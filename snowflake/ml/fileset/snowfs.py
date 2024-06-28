import collections
import logging
import re
from typing import Any, Optional

import fsspec

from snowflake import snowpark
from snowflake.connector import connection
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils import identifier
from snowflake.ml.fileset import embedded_stage_fs, sfcfs

_SFFileEntityPath = collections.namedtuple(
    "_SFFileEntityPath", ["domain", "name", "filepath", "version", "relative_path"]
)
_SNOWURL_PATTERN = re.compile(embedded_stage_fs._SNOWURL_ENTITY_PATTERN + embedded_stage_fs._SNOWURL_VERSION_PATTERN)


class SnowFileSystem(sfcfs.SFFileSystem):
    """A filesystem that allows user to access Snowflake embedded stage files with valid Snowflake locations.

    The file system is is based on fsspec (https://filesystem-spec.readthedocs.io/). It is a file system wrapper
    built on top of SFStageFileSystem. It takes Snowflake embedded stage path as the input and supports read operation.
    A valid Snowflake location will have the form "snow://{domain}/{entity_name}/versions/{version}/{path_to_file}".

    See `sfcfs.SFFileSystem` documentation for example usage patterns.
    """

    protocol = embedded_stage_fs.PROTOCOL_NAME
    _IS_BUGGED_VERSION = None

    def __init__(
        self,
        sf_connection: Optional[connection.SnowflakeConnection] = None,
        snowpark_session: Optional[snowpark.Session] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(sf_connection=sf_connection, snowpark_session=snowpark_session, **kwargs)

    def _get_stage_fs(
        self, sf_file_path: _SFFileEntityPath  # type: ignore[override]
    ) -> embedded_stage_fs.SFEmbeddedStageFileSystem:
        """Get the stage file system for the given snowflake location.

        Args:
            sf_file_path: The Snowflake path information.

        Returns:
            A SFEmbeddedStageFileSystem object which supports readonly file operations on Snowflake embedded stages.
        """
        stage_fs_key = (sf_file_path.domain, sf_file_path.name, sf_file_path.version)
        if stage_fs_key not in self._stage_fs_set:
            cnt_stage_fs = embedded_stage_fs.SFEmbeddedStageFileSystem(
                snowpark_session=self._session,
                domain=sf_file_path.domain,
                name=sf_file_path.name,
                **self._kwargs,
            )
            self._stage_fs_set[stage_fs_key] = cnt_stage_fs
        return self._stage_fs_set[stage_fs_key]

    def _stage_path_to_absolute_path(self, stage_fs: embedded_stage_fs.SFEmbeddedStageFileSystem, path: str) -> str:
        """Convert the relative path in a stage to an absolute path starts with the location of the stage."""
        # Strip protocol from absolute path, since backend needs snow:// prefix to resolve correctly
        # but fsspec logic strips protocol when doing any searching and globbing
        stage_name: str = self._strip_protocol(stage_fs.stage_name)
        abs_path = stage_name + "/" + path
        return abs_path

    @classmethod
    def _parse_file_path(cls, path: str) -> _SFFileEntityPath:  # type: ignore[override]
        """Parse a snowflake location path.

        The following properties will be extracted from the path input:
        - embedded stage domain
        - entity name
        - path (in format `versions/{version}/{relative_path}`)
        - entity version (optional)
        - relative file path (optional)

        Args:
            path: A string in the format of "snow://{domain}/{entity_name}/versions/{version}/{path_to_file}".

        Returns:
            A namedtuple consists of domain, entity name, filepath, version, and relative path, where
                filepath = "versions/{version}/{relative_path}"

        Raises:
            SnowflakeMLException: An error occurred when invalid path is given.
        """
        snowurl_match = _SNOWURL_PATTERN.fullmatch(path)
        if not snowurl_match:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.SNOWML_INVALID_STAGE,
                original_exception=ValueError(f"Invalid Snow URL: {path}"),
            )

        try:
            domain = snowurl_match.group("domain")
            parsed_name = identifier.parse_schema_level_object_identifier(snowurl_match.group("name"))
            name = identifier.get_schema_level_object_identifier(*parsed_name)
            filepath = snowurl_match.group("path")
            version = snowurl_match.group("version")
            relative_path = snowurl_match.group("relpath") or ""
            logging.debug(f"Parsed snow URL: {snowurl_match.groups()}")
            return _SFFileEntityPath(
                domain=domain, name=name, version=version, relative_path=relative_path, filepath=filepath
            )
        except ValueError as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.SNOWML_INVALID_STAGE,
                original_exception=e,
            )


fsspec.register_implementation(SnowFileSystem.protocol, SnowFileSystem)
