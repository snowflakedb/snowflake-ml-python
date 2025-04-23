import re
from collections import defaultdict
from typing import Any, Optional

from snowflake import snowpark
from snowflake.connector import connection
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
    fileset_errors,
)
from snowflake.ml._internal.utils import identifier
from snowflake.ml.fileset import stage_fs
from snowflake.snowpark import exceptions as snowpark_exceptions

PROTOCOL_NAME = "snow"
_SNOWURL_ENTITY_PATTERN = (
    f"(?:{PROTOCOL_NAME}://)?"
    r"(?<!@)(?P<domain>\w+)/"
    rf"(?P<name>(?:{identifier._SF_IDENTIFIER}\.){{,2}}{identifier._SF_IDENTIFIER})/"
)
_SNOWURL_VERSION_PATTERN = r"(?P<path>versions/(?:(?P<version>[^/]+)(?:/+(?P<relpath>.*))?)?)"
_SNOWURL_PATH_RE = re.compile(f"(?:{_SNOWURL_ENTITY_PATTERN})?" + _SNOWURL_VERSION_PATTERN)


class SFEmbeddedStageFileSystem(stage_fs.SFStageFileSystem):
    def __init__(
        self,
        *,
        domain: str,
        name: str,
        snowpark_session: Optional[snowpark.Session] = None,
        sf_connection: Optional[connection.SnowflakeConnection] = None,
        **kwargs: Any,
    ) -> None:

        (db, schema, object_name) = identifier.parse_schema_level_object_identifier(name)
        self._name = name  # TODO: Require or resolve FQN
        self._domain = domain

        super().__init__(
            db=db,
            schema=schema,
            stage=object_name,
            snowpark_session=snowpark_session,
            sf_connection=sf_connection,
            **kwargs,
        )

    @property
    def stage_name(self) -> str:
        """Get the Snowflake path to this stage.

        Returns:
            A string in the format of snow://<domain>/<name>
                Example: snow://dataset/my_dataset

        # noqa: DAR203
        """
        return f"snow://{self._domain}/{self._name}"

    def _stage_path_to_relative_path(self, stage_path: str) -> str:
        """Convert a stage file path which comes from the LIST query to a relative file path in that stage.

        The file path returned by LIST query always has the format "versions/<version>/<relative_file_path>".
                The full "versions/<version>/<relative_file_path>" is returned

        Args:
            stage_path: A string started with the name of the stage.

        Returns:
            A string of the relative stage path.
        """
        return stage_path

    def _fetch_presigned_urls(
        self, files: list[str], url_lifetime: float = stage_fs._PRESIGNED_URL_LIFETIME_SEC
    ) -> list[tuple[str, str]]:
        """Fetch presigned urls for the given files."""
        # SnowURL requires full snow://<domain>/<entity>/versions/<version> as the stage path arg to get_presigned_Url
        versions_dict = defaultdict(list)
        for file in files:
            match = _SNOWURL_PATH_RE.fullmatch(file)
            assert match is not None and match.group("relpath") is not None
            versions_dict[match.group("version")].append(match.group("relpath"))
        try:
            async_jobs: list[snowpark.AsyncJob] = []
            for version, version_files in versions_dict.items():
                for file in version_files:
                    stage_loc = f"{self.stage_name}/versions/{version}"
                    query_result = self._session.sql(
                        f"select '{version}/{file}' as name,"
                        f" get_presigned_url('{stage_loc}', '{file}', {url_lifetime}) as url"
                    ).collect(
                        block=False,
                        statement_params=telemetry.get_function_usage_statement_params(
                            project=stage_fs._PROJECT,
                            api_calls=[snowpark.DataFrame.collect],
                        ),
                    )
                    async_jobs.append(query_result)
            presigned_urls: list[tuple[str, str]] = [
                (r["NAME"], r["URL"]) for job in async_jobs for r in stage_fs._resolve_async_job(job)
            ]
            return presigned_urls
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

    @classmethod
    def _parent(cls, path: str) -> str:
        """Get parent of specified path up to minimally valid root path.

        For SnowURL, the minimum valid relative path is versions/<version>

        Args:
            path: File or directory path

        Returns:
            Parent path

        Examples:
        ----
        >>> fs._parent("versions/my_version/file.ext")
        "versions/my_version"
        >>> fs._parent("versions/my_version/subdir/file.ext")
        "versions/my_version/subdir"
        >>> fs._parent("versions/my_version/")
        "versions/my_version"
        >>> fs._parent("versions/my_version")
        "versions/my_version"
        """
        path_match = _SNOWURL_PATH_RE.fullmatch(path)
        if not path_match:
            return super()._parent(path)  # type: ignore[no-any-return]
        filepath: str = path_match.group("relpath") or ""
        root: str = path[: path_match.start("relpath")] if filepath else path
        if "/" in filepath:
            parent = filepath.rsplit("/", 1)[0]
            return root + parent
        else:
            return root.rstrip("/")
