import json
import warnings
from datetime import datetime
from typing import Any, Optional, Union

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.exceptions import (
    dataset_error_messages,
    dataset_errors,
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils import (
    formatting,
    identifier,
    mixins,
    query_result_checker,
    snowpark_dataframe_utils,
)
from snowflake.ml.dataset import dataset_metadata, dataset_reader
from snowflake.ml.lineage import lineage_node
from snowflake.snowpark import exceptions as snowpark_exceptions, functions

_PROJECT = "Dataset"
_TELEMETRY_STATEMENT_PARAMS = telemetry.get_function_usage_statement_params(_PROJECT)
_METADATA_MAX_QUERY_LENGTH = 10000
_DATASET_VERSION_NAME_COL = "version"


class DatasetVersion(mixins.SerializableSessionMixin):
    """Represents a version of a Snowflake Dataset"""

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def __init__(
        self,
        dataset: "Dataset",
        version: str,
    ) -> None:
        """Initialize a DatasetVersion object.

        Args:
            dataset: The parent Snowflake Dataset.
            version: Dataset version name.
        """
        self._parent = dataset
        self._version = version
        self._session: snowpark.Session = self._parent._session

        self._properties: Optional[dict[str, Any]] = None
        self._raw_metadata: Optional[dict[str, Any]] = None
        self._metadata: Optional[dataset_metadata.DatasetMetadata] = None

    @property
    def name(self) -> str:
        return self._version

    @property
    def created_on(self) -> datetime:
        timestamp = self._get_property("created_on")
        assert isinstance(timestamp, datetime)
        return timestamp

    @property
    def comment(self) -> Optional[str]:
        comment: Optional[str] = self._get_property("comment")
        return comment

    @property
    def label_cols(self) -> list[str]:
        metadata = self._get_metadata()
        if metadata is None or metadata.label_cols is None:
            return []
        return metadata.label_cols

    @property
    def exclude_cols(self) -> list[str]:
        metadata = self._get_metadata()
        if metadata is None or metadata.exclude_cols is None:
            return []
        return metadata.exclude_cols

    def _get_property(self, property_name: str, default: Any = None) -> Any:
        if self._properties is None:
            sql_result = (
                query_result_checker.SqlResultValidator(
                    self._session,
                    f"SHOW VERSIONS LIKE '{self._version}' IN DATASET {self._parent.fully_qualified_name}",
                    statement_params=_TELEMETRY_STATEMENT_PARAMS,
                )
                .has_column(_DATASET_VERSION_NAME_COL, allow_empty=False)
                .validate()
            )
            (match_row,) = (r for r in sql_result if r[_DATASET_VERSION_NAME_COL] == self._version)
            self._properties = match_row.as_dict(True)
        return self._properties.get(property_name, default)

    def _get_metadata(self) -> Optional[dataset_metadata.DatasetMetadata]:
        if self._raw_metadata is None:
            self._raw_metadata = json.loads(self._get_property("metadata", "{}"))
            try:
                self._metadata = (
                    dataset_metadata.DatasetMetadata.from_json(self._raw_metadata) if self._raw_metadata else None
                )
            except ValueError as e:
                warnings.warn(f"Metadata parsing failed with error: {e}", UserWarning, stacklevel=2)
        return self._metadata

    def url(self) -> str:
        """Returns the URL of the DatasetVersion contents in Snowflake.

        Returns:
            Snowflake URL string.
        """
        path = f"snow://dataset/{self._parent.fully_qualified_name}/versions/{self._version}/"
        return path

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def list_files(self, subdir: Optional[str] = None) -> list[snowpark.Row]:
        """Get the list of remote file paths for the current DatasetVersion."""
        return self._session.sql(f"LIST {self.url()}{subdir or ''}").collect(
            statement_params=_TELEMETRY_STATEMENT_PARAMS
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset='{self._parent.fully_qualified_name}', version='{self.name}')"


class Dataset(lineage_node.LineageNode):
    """Represents a Snowflake Dataset which is organized into versions."""

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def __init__(
        self,
        session: snowpark.Session,
        database: str,
        schema: str,
        name: str,
        selected_version: Optional[str] = None,
    ) -> None:
        """Initialize a lazily evaluated Dataset object"""
        self._db = database
        self._schema = schema
        self._name = name

        super().__init__(
            session,
            identifier.get_schema_level_object_identifier(database, schema, name),
            domain="dataset",
            version=selected_version,
        )

        self._version = DatasetVersion(self, selected_version) if selected_version else None
        self._reader: Optional[dataset_reader.DatasetReader] = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  name='{self._lineage_node_name}',\n"
            f"  version='{self._version._version if self._version else None}',\n"
            f")"
        )

    @property
    def fully_qualified_name(self) -> str:
        return self._lineage_node_name

    @property
    def selected_version(self) -> Optional[DatasetVersion]:
        return self._version

    @property
    def read(self) -> dataset_reader.DatasetReader:
        if not self.selected_version:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=RuntimeError("No Dataset version selected."),
            )
        if self._reader is None:
            self._reader = dataset_reader.DatasetReader.from_dataset(self, snowpark_session=self._session)
        return self._reader

    @staticmethod
    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def load(session: snowpark.Session, name: str) -> "Dataset":
        """
        Load an existing Snowflake Dataset. DatasetVersions can be created from the Dataset object
        using `Dataset.create_version()` and loaded with `Dataset.version()`.

        Args:
            session: Snowpark Session to interact with Snowflake backend.
            name: Name of dataset to load. May optionally be a schema-level identifier.

        Returns:
            Dataset object representing loaded dataset

        Raises:
            ValueError: name is not a valid Snowflake identifier
            DatasetNotExistError: Specified Dataset does not exist

        # noqa: DAR402
        """
        db, schema, ds_name = _get_schema_level_identifier(session, name)
        _validate_dataset_exists(session, db, schema, ds_name)
        return Dataset(session, db, schema, ds_name)

    @staticmethod
    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def create(session: snowpark.Session, name: str, exist_ok: bool = False) -> "Dataset":
        """
        Create a new Snowflake Dataset. DatasetVersions can created from the Dataset object
        using `Dataset.create_version()` and loaded with `Dataset.version()`.

        Args:
            session: Snowpark Session to interact with Snowflake backend.
            name: Name of dataset to create. May optionally be a schema-level identifier.
            exist_ok: If False, raises an exception if specified Dataset already exists

        Returns:
            Dataset object representing created dataset

        Raises:
            ValueError: name is not a valid Snowflake identifier
            DatasetExistError: Specified Dataset already exists
            DatasetError: Dataset creation failed

        # noqa: DAR401
        # noqa: DAR402
        """
        db, schema, ds_name = _get_schema_level_identifier(session, name)
        ds_fqn = identifier.get_schema_level_object_identifier(db, schema, ds_name)
        query = f"CREATE DATASET{' IF NOT EXISTS' if exist_ok else ''} {ds_fqn}"
        try:
            session.sql(query).collect(statement_params=_TELEMETRY_STATEMENT_PARAMS)
            return Dataset(session, db, schema, ds_name)
        except snowpark_exceptions.SnowparkSQLException as e:
            if e.sql_error_code == dataset_errors.ERRNO_OBJECT_ALREADY_EXISTS:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.OBJECT_ALREADY_EXISTS,
                    original_exception=dataset_errors.DatasetExistError(
                        dataset_error_messages.DATASET_ALREADY_EXISTS.format(name)
                    ),
                ) from e
            else:
                raise

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def list_versions(self, detailed: bool = False) -> Union[list[str], list[snowpark.Row]]:
        """Return list of versions"""
        versions = self._list_versions()
        versions.sort(key=lambda r: r[_DATASET_VERSION_NAME_COL])
        if not detailed:
            return [r[_DATASET_VERSION_NAME_COL] for r in versions]
        return versions

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def select_version(self, version: str) -> "Dataset":
        """Return a new Dataset instance with the specified version selected.

        Args:
            version: Dataset version name.

        Returns:
            Dataset object.
        """
        self._validate_version_exists(version)
        return Dataset(self._session, self._db, self._schema, self._name, version)

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def create_version(
        self,
        version: str,
        input_dataframe: snowpark.DataFrame,
        shuffle: bool = False,
        exclude_cols: Optional[list[str]] = None,
        label_cols: Optional[list[str]] = None,
        properties: Optional[dataset_metadata.DatasetPropertiesType] = None,
        partition_by: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> "Dataset":
        """Create a new version of the current Dataset.

        The result Dataset object captures the query result deterministically as stage files.

        Args:
            version: Dataset version name. Data contents are materialized to the Dataset entity.
            input_dataframe: A Snowpark DataFrame which yields the Dataset contents.
            shuffle: A boolean represents whether the data should be shuffled globally. Default to be false.
            exclude_cols: Name of column(s) in dataset to be excluded during training/testing (e.g. timestamp).
            label_cols: Name of column(s) in dataset that contains labels.
            properties: Custom metadata properties, saved under `DatasetMetadata.properties`
            partition_by: Optional SQL expression to use as the partitioning scheme within the new Dataset version.
            comment: A descriptive comment about this dataset.

        Returns:
            A Dataset object with the newly created version selected.

        Raises:
            SnowflakeMLException: The Dataset no longer exists.
            SnowflakeMLException: The specified Dataset version already exists.
            snowpark_exceptions.SnowparkSQLException: An error occurred during Dataset creation.

        Note: During the generation of stage files, data casting will occur. The casting rules are as follows::
            - Data casting:
                - DecimalType(NUMBER):
                    - If its scale is zero, cast to BIGINT
                    - If its scale is non-zero, cast to FLOAT
                - DoubleType(DOUBLE): Cast to FLOAT.
                - ByteType(TINYINT): Cast to SMALLINT.
                - ShortType(SMALLINT):Cast to SMALLINT.
                - IntegerType(INT): Cast to INT.
                - LongType(BIGINT): Cast to BIGINT.
            - No action:
                - FloatType(FLOAT): No action.
                - StringType(String): No action.
                - BinaryType(BINARY): No action.
                - BooleanType(BOOLEAN): No action.
            - Not supported:
                - ArrayType(ARRAY): Not supported. A warning will be logged.
                - MapType(OBJECT): Not supported. A warning will be logged.
                - TimestampType(TIMESTAMP): Not supported. A warning will be logged.
                - TimeType(TIME): Not supported. A warning will be logged.
                - DateType(DATE): Not supported. A warning will be logged.
                - VariantType(VARIANT): Not supported. A warning will be logged.
        """
        cast_ignore_cols = (exclude_cols or []) + (label_cols or [])
        casted_df = snowpark_dataframe_utils.cast_snowpark_dataframe(input_dataframe, ignore_columns=cast_ignore_cols)

        if shuffle:
            casted_df = casted_df.order_by(functions.random())

        source_query = json.dumps(input_dataframe.queries)
        if len(source_query) > _METADATA_MAX_QUERY_LENGTH:
            warnings.warn(
                "Source query exceeded max query length, dropping from metadata (limit=%d, actual=%d)"
                % (_METADATA_MAX_QUERY_LENGTH, len(source_query)),
                stacklevel=2,
            )
            source_query = "<query too long>"

        metadata = dataset_metadata.DatasetMetadata(
            source_query=source_query,
            owner=self._session.sql("SELECT CURRENT_USER()").collect(statement_params=_TELEMETRY_STATEMENT_PARAMS)[0][
                "CURRENT_USER()"
            ],
            exclude_cols=exclude_cols,
            label_cols=label_cols,
            properties=properties,
        )

        post_actions = casted_df._plan.post_actions
        try:
            # Execute all but the last query, final query gets passed to ALTER DATASET ADD VERSION
            query = casted_df._plan.queries[-1].sql.strip()
            if len(casted_df._plan.queries) > 1:
                casted_df._plan.queries = casted_df._plan.queries[:-1]
                casted_df._plan.post_actions = []
                casted_df.collect(statement_params=_TELEMETRY_STATEMENT_PARAMS)
            sql_command = "ALTER DATASET {} ADD VERSION '{}' FROM ({})".format(
                self.fully_qualified_name,
                version,
                query,
            )
            if partition_by:
                sql_command += f" PARTITION BY {partition_by}"
            if comment:
                sql_command += f" COMMENT={formatting.format_value_for_select(comment)}"
            sql_command += f" METADATA=$${metadata.to_json()}$$"
            self._session.sql(sql_command).collect(statement_params=_TELEMETRY_STATEMENT_PARAMS)

            return Dataset(self._session, self._db, self._schema, self._name, version)

        except snowpark_exceptions.SnowparkSQLException as e:
            if e.sql_error_code == dataset_errors.ERRNO_DATASET_NOT_EXIST:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.NOT_FOUND,
                    original_exception=dataset_errors.DatasetNotExistError(
                        dataset_error_messages.DATASET_NOT_EXIST.format(self.fully_qualified_name)
                    ),
                ) from e
            elif e.sql_error_code in {
                dataset_errors.ERRNO_DATASET_VERSION_ALREADY_EXISTS,
                dataset_errors.ERRNO_VERSION_ALREADY_EXISTS,
                dataset_errors.ERRNO_FILES_ALREADY_EXISTING,
            }:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.OBJECT_ALREADY_EXISTS,
                    original_exception=dataset_errors.DatasetExistError(
                        dataset_error_messages.DATASET_VERSION_ALREADY_EXISTS.format(self.fully_qualified_name, version)
                    ),
                ) from e
            else:
                raise
        finally:
            for action in post_actions:
                self._session.sql(action.sql.strip()).collect(statement_params=_TELEMETRY_STATEMENT_PARAMS)

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def delete_version(self, version_name: str) -> None:
        """Delete the Dataset version

        Args:
            version_name: Name of version to delete from Dataset

        Raises:
            SnowflakeMLException: An error occurred when the DatasetVersion cannot get deleted.
        """
        delete_sql = f"ALTER DATASET {self.fully_qualified_name} DROP VERSION '{version_name}'"
        try:
            self._session.sql(delete_sql).collect(
                statement_params=_TELEMETRY_STATEMENT_PARAMS,
            )
        except snowpark_exceptions.SnowparkClientException as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.SNOWML_DELETE_FAILED,
                original_exception=dataset_errors.DatasetCannotDeleteError(str(e)),
            ) from e
        return

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def delete(self) -> None:
        """Delete Dataset and all contained versions"""
        self._session.sql(f"DROP DATASET {self.fully_qualified_name}").collect(
            statement_params=_TELEMETRY_STATEMENT_PARAMS
        )

    def _list_versions(self, pattern: Optional[str] = None) -> list[snowpark.Row]:
        """Return list of versions"""
        try:
            pattern_clause = f" LIKE '{pattern}'" if pattern else ""
            return (
                query_result_checker.SqlResultValidator(
                    self._session,
                    f"SHOW VERSIONS{pattern_clause} IN DATASET {self.fully_qualified_name}",
                    statement_params=_TELEMETRY_STATEMENT_PARAMS,
                )
                .has_column(_DATASET_VERSION_NAME_COL, allow_empty=True)
                .validate()
            )
        except snowpark_exceptions.SnowparkSQLException as e:
            if e.sql_error_code == dataset_errors.ERRNO_OBJECT_NOT_EXIST:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.NOT_FOUND,
                    original_exception=dataset_errors.DatasetNotExistError(
                        dataset_error_messages.DATASET_NOT_EXIST.format(self.fully_qualified_name)
                    ),
                ) from e
            else:
                raise

    def _validate_version_exists(self, version: str) -> None:
        """Verify that the requested version exists. Raises DatasetNotExist if version not found"""
        matches = self._list_versions(version)
        matches = [m for m in matches if m[_DATASET_VERSION_NAME_COL] == version]  # Case sensitive match
        if len(matches) == 0:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=dataset_errors.DatasetNotExistError(
                    dataset_error_messages.DATASET_VERSION_NOT_EXIST.format(self.fully_qualified_name, version)
                ),
            )

    @staticmethod
    def _load_from_lineage_node(session: snowpark.Session, name: str, version: str) -> "Dataset":
        return Dataset.load(session, name).select_version(version)


lineage_node.DOMAIN_LINEAGE_REGISTRY["dataset"] = Dataset

# Utility methods


def _get_schema_level_identifier(session: snowpark.Session, dataset_name: str) -> tuple[str, str, str]:
    """Resolve a dataset name into a validated schema-level location identifier"""
    db, schema, object_name = identifier.parse_schema_level_object_identifier(dataset_name)
    db = db or session.get_current_database()
    schema = schema or session.get_current_schema()
    return str(db), str(schema), str(object_name)


def _validate_dataset_exists(session: snowpark.Session, db: str, schema: str, dataset_name: str) -> None:
    # FIXME: Once we switch version to SQL Identifiers we can just use version check with version=''
    dataset_name = identifier.resolve_identifier(dataset_name)
    if len(dataset_name) > 0 and dataset_name[0] == '"' and dataset_name[-1] == '"':
        dataset_name = identifier.get_unescaped_names(dataset_name)
    # Case sensitive match
    query = f"show datasets like '{dataset_name}' in schema {db}.{schema} starts with '{dataset_name}'"
    ds_matches = session.sql(query).count()
    if ds_matches == 0:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.NOT_FOUND,
            original_exception=dataset_errors.DatasetNotExistError(
                dataset_error_messages.DATASET_NOT_EXIST.format(dataset_name)
            ),
        )
