import datetime
import itertools
import json
import posixpath
from typing import Any, Dict, List, Union, cast

from _schema import (
    _ARTIFACT_TABLE_SCHEMA,
    _DEPLOYMENTS_TABLE_SCHEMA,
    _METADATA_TABLE_SCHEMA,
    _REGISTRY_TABLE_SCHEMA,
)
from absl.testing import absltest

from snowflake import connector, snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import formatting, identifier, uri
from snowflake.ml.model import _model
from snowflake.ml.registry import model_registry
from snowflake.ml.test_utils import mock_data_frame, mock_session

_DATABASE_NAME = identifier.get_inferred_name("_SYSTEM_MODEL_REGISTRY")
_SCHEMA_NAME = identifier.get_inferred_name("_SYSTEM_MODEL_REGISTRY_SCHEMA")
_REGISTRY_TABLE_NAME = identifier.get_inferred_name("_SYSTEM_REGISTRY_MODELS")
_ARTIFACTS_TABLE_NAME = identifier.get_inferred_name("_SYSTEM_REGISTRY_ARTIFACTS")
_METADATA_TABLE_NAME = identifier.get_inferred_name("_SYSTEM_REGISTRY_METADATA")
_DEPLOYMENTS_TABLE_NAME = identifier.get_inferred_name("_SYSTEM_REGISTRY_DEPLOYMENTS")
_FULLY_QUALIFIED_REGISTRY_TABLE_NAME = ".".join(
    [
        _DATABASE_NAME,
        _SCHEMA_NAME,
        _REGISTRY_TABLE_NAME,
    ]
)
_REGISTRY_SCHEMA_STRING = ", ".join([f"{k} {v}" for k, v in _REGISTRY_TABLE_SCHEMA])
_METADATA_INSERT_COLUMNS_STRING = ",".join(
    filter(lambda x: x != "SEQUENCE_ID", [item[0] for item in _METADATA_TABLE_SCHEMA])
)
_METADATA_SCHEMA_STRING = ", ".join(
    [f"{k} {v.format(registry_table_name=_FULLY_QUALIFIED_REGISTRY_TABLE_NAME)}" for k, v in _METADATA_TABLE_SCHEMA]
)
_DEPLOYMENTS_SCHEMA_STRING = ",".join(
    [f"{k} {v.format(registry_table_name=_FULLY_QUALIFIED_REGISTRY_TABLE_NAME)}" for k, v in _DEPLOYMENTS_TABLE_SCHEMA]
)
_ARTIFACTS_SCHEMA_STRING = ",".join(
    [f"{k} {v.format(registry_table_name=_FULLY_QUALIFIED_REGISTRY_TABLE_NAME)}" for k, v in _ARTIFACT_TABLE_SCHEMA]
)


class ModelRegistryTest(absltest.TestCase):
    """Testing ModelRegistry functions."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environemnts for testing."""
        self.maxDiff = None
        self._session = mock_session.MockSession(conn=None, test_case=self)
        self.event_id = "fedcba9876543210fedcba9876543210"
        self.model_id = "0123456789abcdef0123456789abcdef"
        self.model_name = "name"
        self.model_version = "abc"
        self.datetime = datetime.datetime(2022, 11, 4, 17, 1, 30, 153000)

        self._setup_mock_session()

    def tearDown(self) -> None:
        """Complete test case. Ensure all expected operations have been observed."""
        self._session.finalize()

    def _setup_mock_session(self) -> None:
        """Equip the mock session with mock variable/methods just for model registry."""
        self._session.get_current_database = absltest.mock.MagicMock(return_value=_DATABASE_NAME)
        self._session.get_current_schema = absltest.mock.MagicMock(return_value=_SCHEMA_NAME)
        self._session.use_database = absltest.mock.MagicMock()
        self._session.use_schema = absltest.mock.MagicMock()

    def _mock_show_database_exists(self) -> None:
        self.add_session_mock_sql(
            query=f"SHOW DATABASES LIKE '{_DATABASE_NAME}'",
            result=mock_data_frame.MockDataFrame(self.get_show_databases_success(name=_DATABASE_NAME)),
        )

    def _mock_show_database_not_exists(self, statement_params: Dict[str, str]) -> None:
        self.add_session_mock_sql(
            query=f"SHOW DATABASES LIKE '{_DATABASE_NAME}'",
            result=mock_data_frame.MockDataFrame([]).add_collect_result([], statement_params=statement_params),
        )

    def _mock_create_database_exists(self, statement_params: Dict[str, str]) -> None:
        self.add_session_mock_sql(
            query=f"CREATE DATABASE IF NOT EXISTS {_DATABASE_NAME}",
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status="MODEL_REGISTRY already exists, statement succeeded.")],
                collect_statement_params=statement_params,
            ),
        )

    def _mock_create_database_not_exists(self, statement_params: Dict[str, str]) -> None:
        self.add_session_mock_sql(
            query=f"CREATE DATABASE {_DATABASE_NAME}",
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status="Database MODEL_REGISTRY successfully created.")],
                collect_statement_params=statement_params,
            ),
        )

    def _mock_show_schema_exists(self, statement_params: Dict[str, str]) -> None:
        self.add_session_mock_sql(
            query=f"SHOW SCHEMAS LIKE '{_SCHEMA_NAME}' IN DATABASE {_DATABASE_NAME}",
            result=mock_data_frame.MockDataFrame(self.get_show_schemas_success(name=_SCHEMA_NAME)).add_collect_result(
                self.get_show_schemas_success(name=_SCHEMA_NAME),
                statement_params=statement_params,
            ),
        )

    def _mock_show_schema_not_exists(self, statement_params: Dict[str, str]) -> None:
        self.add_session_mock_sql(
            query=f"SHOW SCHEMAS LIKE '{_SCHEMA_NAME}' IN DATABASE {_DATABASE_NAME}",
            result=mock_data_frame.MockDataFrame([]).add_collect_result([], statement_params=statement_params),
        )

    def _mock_create_schema_not_exists(self, statement_params: Dict[str, str]) -> None:
        self.add_session_mock_sql(
            query=f"CREATE SCHEMA {_DATABASE_NAME}.{_SCHEMA_NAME}",
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status=f"SCHEMA {_SCHEMA_NAME} successfully created.")],
                collect_statement_params=statement_params,
            ),
        )

    def _mock_create_registry_table_exists(self, statement_params: Dict[str, str]) -> None:
        self.add_session_mock_sql(
            query=f"""CREATE TABLE IF NOT EXISTS {_DATABASE_NAME}.{_SCHEMA_NAME}.{_REGISTRY_TABLE_NAME}
                    ({_REGISTRY_SCHEMA_STRING})""",
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status=f"{_REGISTRY_TABLE_NAME} already exists, statement succeeded.")],
                collect_statement_params=statement_params,
            ),
        )

    def _mock_create_artifacts_table_exists(self, statement_params: Dict[str, str]) -> None:
        self.add_session_mock_sql(
            query=f"""CREATE TABLE IF NOT EXISTS {_DATABASE_NAME}.{_SCHEMA_NAME}.{_ARTIFACTS_TABLE_NAME}
                    ({_ARTIFACTS_SCHEMA_STRING})""",
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status=f"{_ARTIFACTS_TABLE_NAME} already exists, statement succeeded.")],
                collect_statement_params=statement_params,
            ),
        )

    def _mock_create_registry_table_not_exists(self, statement_params: Dict[str, str]) -> None:
        self.add_session_mock_sql(
            query=f"""CREATE TABLE IF NOT EXISTS {_DATABASE_NAME}.{_SCHEMA_NAME}.{_REGISTRY_TABLE_NAME}
                    ({_REGISTRY_SCHEMA_STRING})""",
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status=f"Table {_REGISTRY_TABLE_NAME} successfully created.")],
                collect_statement_params=statement_params,
            ),
        )

    def _mock_create_artifacts_table_not_exists(self, statement_params: Dict[str, str]) -> None:
        self.add_session_mock_sql(
            query=f"""CREATE TABLE IF NOT EXISTS {_DATABASE_NAME}.{_SCHEMA_NAME}.{_ARTIFACTS_TABLE_NAME}
                    ({_ARTIFACTS_SCHEMA_STRING})""",
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status=f"Table {_ARTIFACTS_TABLE_NAME} successfully created.")],
                collect_statement_params=statement_params,
            ),
        )

    def _mock_create_metadata_table_exists(self, statement_params: Dict[str, str]) -> None:
        self.add_session_mock_sql(
            query=f"""CREATE TABLE IF NOT EXISTS {_DATABASE_NAME}.{_SCHEMA_NAME}.{_METADATA_TABLE_NAME}
                    ({_METADATA_SCHEMA_STRING})""",
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status=f"{_METADATA_TABLE_NAME} already exists, statement succeeded.")],
                collect_statement_params=statement_params,
            ),
        )

    def _mock_create_metadata_table_not_exists(self, statement_params: Dict[str, str]) -> None:
        self.add_session_mock_sql(
            query=f"""CREATE TABLE IF NOT EXISTS {_DATABASE_NAME}.{_SCHEMA_NAME}.{_METADATA_TABLE_NAME}
                    ({_METADATA_SCHEMA_STRING})""",
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status=f"Table {_METADATA_TABLE_NAME} successfully created.")],
                collect_statement_params=statement_params,
            ),
        )

    def _mock_create_deployment_table_exists(self, statement_params: Dict[str, str]) -> None:
        self.add_session_mock_sql(
            query=f"""CREATE TABLE IF NOT EXISTS {_DATABASE_NAME}.{_SCHEMA_NAME}.{_DEPLOYMENTS_TABLE_NAME}
                    ({_DEPLOYMENTS_SCHEMA_STRING})""",
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status=f"{_DEPLOYMENTS_TABLE_NAME} already exists, statement succeeded.")],
                collect_statement_params=statement_params,
            ),
        )

    def _mock_create_deployment_table_not_exists(self, statement_params: Dict[str, str]) -> None:
        self.add_session_mock_sql(
            query=f"""CREATE TABLE IF NOT EXISTS {_DATABASE_NAME}.{_SCHEMA_NAME}.{_DEPLOYMENTS_TABLE_NAME}
                    ({_DEPLOYMENTS_SCHEMA_STRING})""",
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status=f"Table {_DEPLOYMENTS_TABLE_NAME} successfully created.")],
                collect_statement_params=statement_params,
            ),
        )

    def add_session_mock_sql(self, query: str, result: Any) -> None:
        self._session.add_mock_sql(query=query, result=result)

    def get_model_registry(self) -> model_registry.ModelRegistry:
        """Creates a valid model registry for testing."""
        self.setup_open_call()

        return model_registry.ModelRegistry(session=cast(snowpark.Session, self._session))

    def get_show_tables_success(
        self, name: str, database_name: str = _DATABASE_NAME, schema_name: str = _SCHEMA_NAME
    ) -> List[snowpark.Row]:
        """Helper method that returns a DataFrame that looks like the response of from a successful listing of
        tables."""
        return [
            snowpark.Row(
                created_on=self.datetime,
                name=name,
                database_name=database_name,
                schema_name=schema_name,
                kind="TABLE",
                comment="",
                cluster_by="",
                rows=0,
                bytes=0,
                owner="OWNER_ROLE",
                retention_time=1,
                change_tracking="OFF",
                is_external="N",
                enable_schema_evolution="N",
            )
        ]

    def get_show_schemas_success(
        self, name: str, database_name: str = _DATABASE_NAME, schema_name: str = _SCHEMA_NAME
    ) -> List[snowpark.Row]:
        """Helper method that returns a DataFrame that looks like the response of from a successful listing of
        schemas."""
        return [
            snowpark.Row(
                created_on=self.datetime,
                name=name,
                is_default="N",
                is_current="N",
                database_name=database_name,
                owner="OWNER_ROLE",
                comment="",
                options="",
                retention_time=1,
            )
        ]

    def get_show_databases_success(self, name: str) -> List[snowpark.Row]:
        """Helper method that returns a DataFrame that looks like the response of from a successful listing of
        databases."""
        return [
            snowpark.Row(
                created_on=self.datetime,
                name=name,
                is_default="N",
                is_current="N",
                origin="",
                owner="OWNER_ROLE",
                comment="",
                options="",
                retention_time=1,
            )
        ]

    def get_desc_registry_table_success(self) -> List[snowpark.Row]:
        """Helper method that returns a DataFrame that looks like the response of from a successful desc table."""
        return [
            snowpark.Row(name="CREATION_CONTEXT", type="VARCHAR"),
            snowpark.Row(name="CREATION_ENVIRONMENT_SPEC", type="OBJECT"),
            snowpark.Row(name="CREATION_ROLE", type="VARCHAR"),
            snowpark.Row(name="CREATION_TIME", type="TIMESTAMP_TZ"),
            snowpark.Row(name="ID", type="VARCHAR PRIMARY KEY RELY"),
            snowpark.Row(name="INPUT_SPEC", type="OBJECT"),
            snowpark.Row(name="NAME", type="VARCHAR"),
            snowpark.Row(name="OUTPUT_SPEC", type="OBJECT"),
            snowpark.Row(name="RUNTIME_ENVIRONMENT_SPEC", type="OBJECT"),
            snowpark.Row(name="TRAINING_DATASET_ID", type="VARCHAR"),
            snowpark.Row(name="TYPE", type="VARCHAR"),
            snowpark.Row(name="URI", type="VARCHAR"),
            snowpark.Row(name="VERSION", type="VARCHAR"),
        ]

    def setup_open_call(self) -> None:
        self.add_session_mock_sql(
            query=f"SHOW DATABASES LIKE '{_DATABASE_NAME}'",
            result=mock_data_frame.MockDataFrame(
                self.get_show_databases_success(name=_DATABASE_NAME)
            ).add_collect_result(self.get_show_databases_success(name=_DATABASE_NAME)),
        )
        self.add_session_mock_sql(
            query=f"SHOW SCHEMAS LIKE '{_SCHEMA_NAME}' IN DATABASE {_DATABASE_NAME}",
            result=mock_data_frame.MockDataFrame(self.get_show_schemas_success(name=_SCHEMA_NAME)).add_collect_result(
                self.get_show_schemas_success(name=_SCHEMA_NAME)
            ),
        )
        self.add_session_mock_sql(
            query=f"SHOW TABLES LIKE '{_REGISTRY_TABLE_NAME}' IN {_DATABASE_NAME}.{_SCHEMA_NAME}",
            result=mock_data_frame.MockDataFrame(
                self.get_show_tables_success(name=_REGISTRY_TABLE_NAME)
            ).add_collect_result(self.get_show_tables_success(name=_REGISTRY_TABLE_NAME)),
        )
        self.add_session_mock_sql(
            query=f"DESC TABLE {_FULLY_QUALIFIED_REGISTRY_TABLE_NAME}",
            result=mock_data_frame.MockDataFrame(self.get_desc_registry_table_success()).add_collect_result(
                self.get_desc_registry_table_success()
            ),
        )
        self.add_session_mock_sql(
            query=f"SHOW TABLES LIKE '{_METADATA_TABLE_NAME}' IN {_DATABASE_NAME}.{_SCHEMA_NAME}",
            result=mock_data_frame.MockDataFrame(
                self.get_show_tables_success(name=_METADATA_TABLE_NAME)
            ).add_collect_result(self.get_show_tables_success(name=_METADATA_TABLE_NAME)),
        )
        self.add_session_mock_sql(
            query=f"SHOW TABLES LIKE '{_DEPLOYMENTS_TABLE_NAME}' IN {_DATABASE_NAME}.{_SCHEMA_NAME}",
            result=mock_data_frame.MockDataFrame(
                self.get_show_tables_success(name=_DEPLOYMENTS_TABLE_NAME)
            ).add_collect_result(self.get_show_tables_success(name=_DEPLOYMENTS_TABLE_NAME)),
        )

    def setup_list_model_call(self) -> mock_data_frame.MockDataFrame:
        """Setup the expected calls originating from list_model."""
        result_df = mock_data_frame.MockDataFrame()

        self.add_session_mock_sql(
            query=(f"SELECT * FROM {_DATABASE_NAME}.{_SCHEMA_NAME}.{_REGISTRY_TABLE_NAME}_VIEW"),
            result=result_df,
        )
        # Returning result_df to allow the caller to add more expected operations.
        return result_df

    def setup_create_views_call(self) -> None:
        """Setup the expected calls originating from _create_views."""
        self.add_session_mock_sql(
            query=(
                f"""CREATE OR REPLACE VIEW {_DATABASE_NAME}.{_SCHEMA_NAME}.{_DEPLOYMENTS_TABLE_NAME}_VIEW
                    COPY GRANTS AS
                    SELECT
                        DEPLOYMENT_NAME,
                        MODEL_ID,
                        {_REGISTRY_TABLE_NAME}.NAME as MODEL_NAME,
                        {_REGISTRY_TABLE_NAME}.VERSION as MODEL_VERSION,
                        {_DEPLOYMENTS_TABLE_NAME}.CREATION_TIME as CREATION_TIME,
                        TARGET_METHOD,
                        TARGET_PLATFORM,
                        SIGNATURE,
                        OPTIONS,
                        STAGE_PATH,
                        ROLE
                    FROM {_DEPLOYMENTS_TABLE_NAME}
                    LEFT JOIN {_REGISTRY_TABLE_NAME}
                    ON {_DEPLOYMENTS_TABLE_NAME}.MODEL_ID = {_REGISTRY_TABLE_NAME}.ID
                """
            ),
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status=f"View {_DEPLOYMENTS_TABLE_NAME}_VIEW successfully created.")]
            ),
        )
        self.add_session_mock_sql(
            query=(
                f"""CREATE OR REPLACE VIEW {_DATABASE_NAME}.{_SCHEMA_NAME}.{_METADATA_TABLE_NAME}_LAST_DESCRIPTION
                    COPY GRANTS AS
                        SELECT DISTINCT
                            MODEL_ID,
                            (LAST_VALUE(VALUE) OVER (
                                PARTITION BY MODEL_ID ORDER BY SEQUENCE_ID))['DESCRIPTION']
                            as DESCRIPTION
                        FROM {_METADATA_TABLE_NAME} WHERE ATTRIBUTE_NAME = 'DESCRIPTION'"""
            ),
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status=f"View {_METADATA_TABLE_NAME}_LAST_DESCRIPTION successfully created.")]
            ),
        )
        self.add_session_mock_sql(
            query=(
                f"""CREATE OR REPLACE VIEW {_DATABASE_NAME}.{_SCHEMA_NAME}.{_METADATA_TABLE_NAME}_LAST_METRICS
                    COPY GRANTS AS
                        SELECT DISTINCT
                            MODEL_ID,
                            (LAST_VALUE(VALUE) OVER (
                                PARTITION BY MODEL_ID ORDER BY SEQUENCE_ID))['METRICS']
                            as METRICS
                        FROM {_METADATA_TABLE_NAME} WHERE ATTRIBUTE_NAME = 'METRICS'"""
            ),
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status=f"View {_METADATA_TABLE_NAME}_LAST_METRICS successfully created.")]
            ),
        )
        self.add_session_mock_sql(
            query=(
                f"""CREATE OR REPLACE VIEW {_DATABASE_NAME}.{_SCHEMA_NAME}.{_METADATA_TABLE_NAME}_LAST_TAGS
                    COPY GRANTS AS
                        SELECT DISTINCT
                            MODEL_ID,
                            (LAST_VALUE(VALUE) OVER (
                                PARTITION BY MODEL_ID ORDER BY SEQUENCE_ID))['TAGS']
                            as TAGS
                        FROM {_METADATA_TABLE_NAME} WHERE ATTRIBUTE_NAME = 'TAGS'"""
            ),
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status=f"View {_METADATA_TABLE_NAME}_LAST_TAGS successfully created.")]
            ),
        )
        self.add_session_mock_sql(
            query=(
                f"""CREATE OR REPLACE VIEW
                    {_DATABASE_NAME}.{_SCHEMA_NAME}.{_METADATA_TABLE_NAME}_LAST_REGISTRATION COPY GRANTS AS
                        SELECT DISTINCT
                            MODEL_ID, EVENT_TIMESTAMP as REGISTRATION_TIMESTAMP
                        FROM {_METADATA_TABLE_NAME} WHERE ATTRIBUTE_NAME = 'REGISTRATION'"""
            ),
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status=f"View {_METADATA_TABLE_NAME}_LAST_TAGS successfully created.")]
            ),
        )
        self.add_session_mock_sql(
            query=(
                f"""CREATE OR REPLACE VIEW  {_DATABASE_NAME}.{_SCHEMA_NAME}.{_REGISTRY_TABLE_NAME}_VIEW
                    COPY GRANTS AS
                    SELECT {_REGISTRY_TABLE_NAME}.*, {_METADATA_TABLE_NAME}_LAST_DESCRIPTION.DESCRIPTION
                            AS DESCRIPTION,
                        {_METADATA_TABLE_NAME}_LAST_METRICS.METRICS AS METRICS,
                        {_METADATA_TABLE_NAME}_LAST_TAGS.TAGS AS TAGS,
                        {_METADATA_TABLE_NAME}_LAST_REGISTRATION.REGISTRATION_TIMESTAMP AS REGISTRATION_TIMESTAMP
                    FROM {_REGISTRY_TABLE_NAME}
                        LEFT JOIN {_METADATA_TABLE_NAME}_LAST_DESCRIPTION ON
                            ({_METADATA_TABLE_NAME}_LAST_DESCRIPTION.MODEL_ID = {_REGISTRY_TABLE_NAME}.ID)
                        LEFT JOIN {_METADATA_TABLE_NAME}_LAST_METRICS
                            ON ({_METADATA_TABLE_NAME}_LAST_METRICS.MODEL_ID = {_REGISTRY_TABLE_NAME}.ID)
                        LEFT JOIN {_METADATA_TABLE_NAME}_LAST_TAGS
                            ON ({_METADATA_TABLE_NAME}_LAST_TAGS.MODEL_ID = {_REGISTRY_TABLE_NAME}.ID)
                        LEFT JOIN {_METADATA_TABLE_NAME}_LAST_REGISTRATION
                            ON ({_METADATA_TABLE_NAME}_LAST_REGISTRATION.MODEL_ID = {_REGISTRY_TABLE_NAME}.ID)
                """
            ),
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status=f"View {_REGISTRY_TABLE_NAME}_VIEW successfully created.")]
            ),
        )
        self.add_session_mock_sql(  # type: ignore
            query=(
                f"""CREATE OR REPLACE VIEW {_DATABASE_NAME}.{_SCHEMA_NAME}.{_ARTIFACTS_TABLE_NAME}_VIEW
                    COPY GRANTS AS
                    SELECT
                        {_REGISTRY_TABLE_NAME}.NAME AS MODEL_NAME,
                        {_REGISTRY_TABLE_NAME}.VERSION AS MODEL_VERSION,
                        {_ARTIFACTS_TABLE_NAME}.*
                    FROM {_REGISTRY_TABLE_NAME}
                    LEFT JOIN {_ARTIFACTS_TABLE_NAME}
                    ON ({_REGISTRY_TABLE_NAME}.TRAINING_DATASET_ID = {_ARTIFACTS_TABLE_NAME}.ID)
                    WHERE {_ARTIFACTS_TABLE_NAME}.TYPE = 'TRAINING_DATASET'
                """
            ),
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status=f"View {_ARTIFACTS_TABLE_NAME}_VIEW successfully created.")]
            ),
        ),

    def setup_open_existing(self) -> None:
        self.add_session_mock_sql(
            query=f"SHOW DATABASES LIKE '{_DATABASE_NAME}'",
            result=mock_data_frame.MockDataFrame(self.get_show_databases_success(name=_DATABASE_NAME)),
        )
        self.add_session_mock_sql(
            query=f"SHOW SCHEMAS LIKE '{_SCHEMA_NAME}' IN DATABASE {_DATABASE_NAME}",
            result=mock_data_frame.MockDataFrame(self.get_show_schemas_success(name=_SCHEMA_NAME)),
        )
        self.add_session_mock_sql(
            query=f"SHOW TABLES LIKE '{_REGISTRY_TABLE_NAME}' IN {_DATABASE_NAME}.{_SCHEMA_NAME}",
            result=mock_data_frame.MockDataFrame(self.get_show_tables_success(name=_REGISTRY_TABLE_NAME)),
        )
        self.add_session_mock_sql(
            query=f"DESC TABLE {_FULLY_QUALIFIED_REGISTRY_TABLE_NAME}",
            result=mock_data_frame.MockDataFrame(self.get_desc_registry_table_success()).add_collect_result(
                self.get_desc_registry_table_success()
            ),
        )
        self.add_session_mock_sql(
            query=f"SHOW TABLES LIKE '{_METADATA_TABLE_NAME}' IN {_DATABASE_NAME}.{_SCHEMA_NAME}",
            result=mock_data_frame.MockDataFrame(self.get_show_tables_success(name=_METADATA_TABLE_NAME)),
        )
        self.add_session_mock_sql(
            query=f"SHOW TABLES LIKE '{_DEPLOYMENTS_TABLE_NAME}' IN {_DATABASE_NAME}.{_SCHEMA_NAME}",
            result=mock_data_frame.MockDataFrame(self.get_show_tables_success(name=_DEPLOYMENTS_TABLE_NAME)),
        )

    def template_test_get_attribute(
        self, collection_res: List[snowpark.Row], use_id: bool = False
    ) -> mock_data_frame.MockDataFrame:
        expected_df = self.setup_list_model_call()
        expected_df.add_operation("filter")
        if not use_id:
            expected_df.add_operation("filter")
        expected_df.add_collect_result(collection_res)
        return expected_df

    def template_test_set_attribute(
        self,
        attribute_name: str,
        attribute_value: Union[str, Dict[Any, Any]],
        result_num_inserted: int = 1,
        use_id: bool = False,
    ) -> None:
        expected_df = self.setup_list_model_call()
        expected_df.add_operation("filter")
        if not use_id:
            expected_df.add_operation("filter")
        expected_df.add_collect_result(
            [
                snowpark.Row(
                    ID=self.model_id,
                    NAME="name",
                    VERSION="abc",
                    URI=f"sfc://{_DATABASE_NAME}.{_SCHEMA_NAME}.model_stage",
                )
            ]
        )

        self._session.add_operation("get_current_role", result="current_role")

        self.add_session_mock_sql(
            query=(
                f"""INSERT INTO {_DATABASE_NAME}.{_SCHEMA_NAME}.{_METADATA_TABLE_NAME}
                    ({_METADATA_INSERT_COLUMNS_STRING})
                         SELECT '{attribute_name}','{self.event_id}',CURRENT_TIMESTAMP(),'{self.model_id}','SET',
                         'current_role',OBJECT_CONSTRUCT('{attribute_name}',
                         {formatting.format_value_for_select(attribute_value)})"""
            ),
            result=mock_data_frame.MockDataFrame([snowpark.Row(**{"number of rows inserted": result_num_inserted})]),
        )

    def test_create_new(self) -> None:
        """Verify that we can create a new ModelRegistry database with the default names."""
        # "Create" calls.
        combinations = list(itertools.product([True, False], repeat=6))
        for (
            database_exists,
            schema_exists,
            registry_table_exists,
            metadata_table_exists,
            deployments_table_exists,
            artifacts_table_exists,
        ) in combinations:
            with self.subTest(
                msg=(
                    f"database_exists={database_exists}, "
                    f"schema_exists={schema_exists}, "
                    f"registry_table_exists={registry_table_exists}, "
                    f"metadata_table_exists={metadata_table_exists}, "
                    f"deployments_table_exists={deployments_table_exists}, "
                    f"artifacts_table_exists={artifacts_table_exists}"
                )
            ):
                statement_params = telemetry.get_function_usage_statement_params(
                    project="MLOps",
                    subproject="ModelRegistry",
                    function_name="snowflake.ml.registry.model_registry.create_model_registry",
                )
                if database_exists:
                    self._mock_show_database_exists()
                else:
                    self._mock_show_database_not_exists(statement_params)
                    self._mock_create_database_not_exists(statement_params)

                if schema_exists:
                    self._mock_show_schema_exists(statement_params)
                else:
                    self._mock_show_schema_not_exists(statement_params)
                    self._mock_create_schema_not_exists(statement_params)

                if registry_table_exists:
                    self._mock_create_registry_table_exists(statement_params)
                else:
                    self._mock_create_registry_table_not_exists(statement_params)

                if metadata_table_exists:
                    self._mock_create_metadata_table_exists(statement_params)
                else:
                    self._mock_create_metadata_table_not_exists(statement_params)

                if deployments_table_exists:
                    self._mock_create_deployment_table_exists(statement_params)
                else:
                    self._mock_create_deployment_table_not_exists(statement_params)

                if artifacts_table_exists:
                    self._mock_create_artifacts_table_exists(statement_params)
                else:
                    self._mock_create_artifacts_table_not_exists(statement_params)

                self.setup_create_views_call()

                model_registry.create_model_registry(
                    session=cast(snowpark.Session, self._session),
                    database_name=_DATABASE_NAME,
                    schema_name=_SCHEMA_NAME,
                )

    def test_create_if_not_exists(self) -> None:
        statement_params = telemetry.get_function_usage_statement_params(
            project="MLOps",
            subproject="ModelRegistry",
            function_name="snowflake.ml.registry.model_registry.create_model_registry",
        )
        # SQL queries issued by create_model_registry:
        self._mock_show_database_not_exists(statement_params)
        self._mock_create_database_not_exists(statement_params)
        self._mock_show_schema_not_exists(statement_params)
        self._mock_create_schema_not_exists(statement_params)
        self._mock_create_registry_table_not_exists(statement_params)
        self._mock_create_metadata_table_not_exists(statement_params)
        self._mock_create_deployment_table_not_exists(statement_params)
        self._mock_create_artifacts_table_not_exists(statement_params)
        self.setup_create_views_call()

        # 2. SQL queries issued by ModelRegistry constructor.
        self.setup_open_existing()

        registry = model_registry.ModelRegistry(
            session=cast(snowpark.Session, self._session),
            database_name=_DATABASE_NAME,
            schema_name=_SCHEMA_NAME,
            create_if_not_exists=True,
        )
        self.assertIsNotNone(registry)

    def test_open_existing(self) -> None:
        """Verify that we can open an existing ModelRegistry database with the default names."""
        self.setup_open_existing()
        model_registry.ModelRegistry(session=cast(snowpark.Session, self._session))

    def test_list_models(self) -> None:
        """Test the normal operation of list_models. We create a view and return the model metadata."""
        model_registry = self.get_model_registry()
        self.setup_list_model_call().add_collect_result([snowpark.Row(ID=self.model_id, NAME="model_name")])

        model_list = model_registry.list_models().collect()
        self.assertEqual(model_list, [snowpark.Row(ID=self.model_id, NAME="model_name")])

    def test_set_model_description(self) -> None:
        """Test that we can set the description for an existing model."""
        model_registry = self.get_model_registry()
        self.template_test_set_attribute("DESCRIPTION", "new_description")

        # Mock unique identifier for event id.
        with absltest.mock.patch.object(
            model_registry,
            "_get_new_unique_identifier",
            return_value=self.event_id,
        ):
            model_registry.set_model_description(
                model_name=self.model_name, model_version=self.model_version, description="new_description"
            )

    def test_get_model_description(self) -> None:
        """Test that we can get the description of an existing model from the registry."""
        model_registry = self.get_model_registry()
        self.template_test_get_attribute(
            [
                snowpark.Row(
                    ID=self.model_id,
                    NAME=self.model_name,
                    VERSION=self.model_version,
                    DESCRIPTION='"model_description"',
                )
            ]
        )

        model_description = model_registry.get_model_description(
            model_name=self.model_name, model_version=self.model_version
        )
        self.assertEqual(model_description, "model_description")

    def test_get_history(self) -> None:
        """Test that we can retrieve the history for the model history."""
        model_registry = self.get_model_registry()
        expected_collect_result = [
            snowpark.Row(
                EVENT_TIMESTAMP="ts",
                EVENT_ID=self.event_id,
                MODEL_ID=self.model_id,
                ROLE="role",
                OPERATION="SET",
                ATTRIBUTE_NAME="NAME",
                VALUE={"NAME": "name"},
            )
        ]

        expected_df = mock_data_frame.MockDataFrame()
        self._session.add_operation(
            operation="table",
            args=(f"{_DATABASE_NAME}.{_SCHEMA_NAME}.{_METADATA_TABLE_NAME}",),
            result=expected_df,
        )
        expected_df.add_operation("order_by", args=("EVENT_TIMESTAMP",))
        expected_df.add_operation(
            "select_expr",
            args=(
                "EVENT_TIMESTAMP",
                "EVENT_ID",
                "MODEL_ID",
                "ROLE",
                "OPERATION",
                "ATTRIBUTE_NAME",
                "VALUE[ATTRIBUTE_NAME]",
            ),
        )
        expected_df.add_collect_result(expected_collect_result)

        self.assertEqual(model_registry.get_history().collect(), expected_collect_result)

    def test_get_model_history(self) -> None:
        """Test that we can retrieve the history for a specific model."""
        model_registry = self.get_model_registry()
        self.template_test_get_attribute(
            [snowpark.Row(ID=self.model_id, NAME=self.model_name, VERSION=self.model_version)]
        )
        expected_collect_result = [
            snowpark.Row(
                EVENT_TIMESTAMP="ts",
                EVENT_ID=self.event_id,
                MODEL_ID=self.model_id,
                ROLE="role",
                OPERATION="SET",
                ATTRIBUTE_NAME="NAME",
                VALUE={"NAME": "name"},
            )
        ]

        expected_df = mock_data_frame.MockDataFrame()
        self._session.add_operation(
            operation="table",
            args=(f"{_DATABASE_NAME}.{_SCHEMA_NAME}.{_METADATA_TABLE_NAME}",),
            result=expected_df,
        )
        expected_df.add_operation("order_by", args=("EVENT_TIMESTAMP",))
        expected_df.add_operation(
            "select_expr",
            args=(
                "EVENT_TIMESTAMP",
                "EVENT_ID",
                "MODEL_ID",
                "ROLE",
                "OPERATION",
                "ATTRIBUTE_NAME",
                "VALUE[ATTRIBUTE_NAME]",
            ),
        )
        expected_df.add_operation(operation="filter", check_args=False, check_kwargs=False)
        expected_df.add_collect_result(expected_collect_result)

        self.assertEqual(
            model_registry.get_model_history(model_name=self.model_name, model_version=self.model_version).collect(),
            expected_collect_result,
        )

    def test_set_metric_no_existing(self) -> None:
        """Test that we can set a metric for an existing model that does not yet have any metrics set."""
        model_registry = self.get_model_registry()
        self.template_test_get_attribute(
            [snowpark.Row(ID=self.model_id, NAME=self.model_name, VERSION=self.model_version, METRICS=None)]
        )
        self.template_test_set_attribute("METRICS", {"voight-kampff": 0.9})

        # Mock unique identifier for event id.
        with absltest.mock.patch.object(
            model_registry,
            "_get_new_unique_identifier",
            return_value=self.event_id,
        ):
            model_registry.set_metric(
                model_name=self.model_name,
                model_version=self.model_version,
                metric_name="voight-kampff",
                metric_value=0.9,
            )

    def test_set_metric_with_existing(self) -> None:
        """Test that we can set a metric for an existing model that already has metrics."""
        model_registry = self.get_model_registry()
        self.template_test_get_attribute(
            [
                snowpark.Row(
                    ID=self.model_id, NAME=self.model_name, VERSION=self.model_version, METRICS='{"human-factor": 1.1}'
                )
            ]
        )
        self.template_test_set_attribute("METRICS", {"human-factor": 1.1, "voight-kampff": 0.9})

        # Mock unique identifier for event id.
        with absltest.mock.patch.object(
            model_registry,
            "_get_new_unique_identifier",
            return_value=self.event_id,
        ):
            model_registry.set_metric(
                model_name=self.model_name,
                model_version=self.model_version,
                metric_name="voight-kampff",
                metric_value=0.9,
            )

    def test_get_metrics(self) -> None:
        """Test that we can get the metrics for an existing model."""
        metrics_dict = {"human-factor": 1.1, "voight-kampff": 0.9}
        model_registry = self.get_model_registry()
        self.template_test_get_attribute(
            [
                snowpark.Row(
                    ID=self.model_id, NAME=self.model_name, VERSION=self.model_version, METRICS=json.dumps(metrics_dict)
                )
            ]
        )
        self.assertEqual(
            model_registry.get_metrics(model_name=self.model_name, model_version=self.model_version), metrics_dict
        )

    def test_get_metric_value(self) -> None:
        """Test that we can get a single metric value for an existing model."""
        metrics_dict = {"human-factor": 1.1, "voight-kampff": 0.9}
        model_registry = self.get_model_registry()
        self.template_test_get_attribute(
            [
                snowpark.Row(
                    ID=self.model_id, NAME=self.model_name, VERSION=self.model_version, METRICS=json.dumps(metrics_dict)
                )
            ]
        )
        self.assertEqual(
            model_registry.get_metric_value(
                model_name=self.model_name, model_version=self.model_version, metric_name="human-factor"
            ),
            1.1,
        )

    def test_private_insert_registry_entry(self) -> None:
        model_registry = self.get_model_registry()

        self.add_session_mock_sql(
            query=f"""
                INSERT INTO {_DATABASE_NAME}.{_SCHEMA_NAME}.{_REGISTRY_TABLE_NAME} ( ID,NAME,TYPE,URI,VERSION )
                SELECT 'id','name','type','uri','abc'
            """,
            result=mock_data_frame.MockDataFrame([snowpark.Row(**{"number of rows inserted": 1})]),
        )

        model_properties = {"ID": "id", "NAME": "name", "TYPE": "type", "URI": "uri"}

        model_registry._insert_registry_entry(id="id", name="name", version="abc", properties=model_properties)

    def test_get_tags(self) -> None:
        """Test that get_tags is working correctly with various types."""
        model_registry = self.get_model_registry()
        self.template_test_get_attribute(
            [
                snowpark.Row(
                    TAGS="""
                        {
                            "top_level": "string",
                            "nested": {
                               "float": 0.9,
                                "int": 23,
                                "nested_string": "string",
                                "empty_string": "",
                                "bool_true": true,
                                "bool_false": "false",
                                "1d_array": [
                                    1,
                                    2,
                                    3
                                ],
                                "2d_array":  [
                                    [
                                        90,
                                        0
                                    ],
                                    [
                                        3,
                                        7
                                    ]
                                ]
                            }
                        }""",
                )
            ]
        )
        tags = model_registry.get_tags(model_name=self.model_name, model_version=self.model_version)
        self.assertEqual(tags["top_level"], "string")
        self.assertEqual(tags["nested"]["float"], 0.9)

    def test_log_model_path(self) -> None:
        """Test _log_model_path().

        Validate _log_model_path() can perform stage file put operation with the expected stage path and call
        register_model() with the expected arguments.
        """
        model_registry = self.get_model_registry()

        model_name = "name"
        model_version = "abc"
        expected_stage_postfix = f"{self.model_id}".upper()

        self.add_session_mock_sql(
            query=f"CREATE OR REPLACE STAGE {_DATABASE_NAME}.{_SCHEMA_NAME}.SNOWML_MODEL_{expected_stage_postfix} "
            f"ENCRYPTION = (TYPE= 'SNOWFLAKE_SSE')",
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(**{"status": f"Stage area SNOWML_MODEL_{expected_stage_postfix} successfully created."})]
            ),
        )

        expected_stage_path = (
            f"{identifier.get_inferred_name(_DATABASE_NAME)}"
            + "."
            + f"{identifier.get_inferred_name(_SCHEMA_NAME)}"
            + "."
            + f"SNOWML_MODEL_{expected_stage_postfix}"
        )

        with absltest.mock.patch.object(
            model_registry,
            "_get_new_unique_identifier",
            return_value=self.model_id,
        ):
            model_id, stage_path = model_registry._log_model_path(
                model_name=model_name,
                model_version=model_version,
            )
            self.assertEqual(model_id, self.model_id)
            self.assertEqual(stage_path, expected_stage_path)

    def test_log_model(self) -> None:
        """Test log_model()"""
        model_registry = self.get_model_registry()

        model_name = "name"
        model_version = "abc"
        expected_stage_postfix = f"{self.model_id}".upper()

        expected_stage_path = (
            f"{identifier.get_inferred_name(_DATABASE_NAME)}"
            + "."
            + f"{identifier.get_inferred_name(_SCHEMA_NAME)}"
            + "."
            + f"SNOWML_MODEL_{expected_stage_postfix}"
        )
        model_path = posixpath.join(f"@{expected_stage_path}", f"{self.model_id}.zip")

        with absltest.mock.patch.object(
            model_registry,
            "_list_selected_models",
            return_value=absltest.mock.MagicMock(count=absltest.mock.MagicMock(return_value=0)),
        ):
            with absltest.mock.patch.object(
                model_registry,
                "_log_model_path",
                return_value=(self.model_id, expected_stage_path),
            ) as mock_path:
                mock_model = absltest.mock.MagicMock()
                mock_type = absltest.mock.MagicMock()
                mock_metadata = absltest.mock.MagicMock(model_type=mock_type)
                with absltest.mock.patch.object(
                    target=_model, attribute="save_model", return_value=mock_metadata
                ) as mock_save:
                    with absltest.mock.patch.object(
                        target=model_registry, attribute="_register_model_with_id", return_value=None
                    ) as mock_register:
                        with absltest.mock.patch.object(model_registry, "_get_model_id", return_value=self.model_id):
                            m_signatures = {"predict": None}
                            model_registry.log_model(
                                model_name=model_name,
                                model_version=model_version,
                                model=mock_model,
                                signatures=m_signatures,
                                description="description",
                                tags=None,
                            )
                            mock_path.assert_called_once_with(model_name=model_name, model_version=model_version)
                            mock_save.assert_called_once_with(
                                name=model_name,
                                session=self._session,
                                model_stage_file_path=model_path,
                                model=mock_model,
                                signatures=m_signatures,
                                metadata=None,
                                conda_dependencies=None,
                                pip_requirements=None,
                                sample_input=None,
                                code_paths=None,
                                options=None,
                            )
                            mock_register.assert_called_once_with(
                                model_name=model_name,
                                model_version=model_version,
                                model_id=self.model_id,
                                type=mock_type,
                                uri=uri.get_uri_from_snowflake_stage_path(model_path),
                                description="description",
                                tags=None,
                                training_dataset=None,
                            )

        with absltest.mock.patch.object(
            model_registry,
            "_list_selected_models",
            return_value=absltest.mock.MagicMock(count=absltest.mock.MagicMock(return_value=1)),
        ):
            with self.assertRaises(connector.DataError):
                model_registry.log_model(
                    model_name=model_name,
                    model_version=model_version,
                    model=mock_model,
                    signatures=m_signatures,
                    description="description",
                    tags=None,
                )

        self.add_session_mock_sql(
            query=f"DROP STAGE {_DATABASE_NAME}.{_SCHEMA_NAME}.SNOWML_MODEL_{expected_stage_postfix}",
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(**{"status": f"Stage area SNOWML_MODEL_{expected_stage_postfix} successfully dropped."})]
            ),
        )

        with absltest.mock.patch.object(
            model_registry,
            "_list_selected_models",
            return_value=absltest.mock.MagicMock(count=absltest.mock.MagicMock(return_value=0)),
        ):
            with absltest.mock.patch.object(
                model_registry,
                "_log_model_path",
                return_value=(self.model_id, expected_stage_path),
            ) as mock_path:
                mock_model = absltest.mock.MagicMock()
                mock_type = absltest.mock.MagicMock()
                mock_metadata = absltest.mock.MagicMock(model_type=mock_type)
                with absltest.mock.patch.object(target=_model, attribute="save_model") as mock_save:
                    mock_save.side_effect = ValueError("Mock Error")
                    with self.assertRaises(ValueError):
                        model_registry.log_model(
                            model_name=model_name,
                            model_version=model_version,
                            model=mock_model,
                            signatures=m_signatures,
                            description="description",
                            tags=None,
                        )

    def test_delete_model_with_artifact(self) -> None:
        """Test deleting a model and artifact from the registry."""
        model_registry = self.get_model_registry()
        self.setup_list_model_call().add_operation(operation="filter").add_operation(
            operation="filter"
        ).add_collect_result(
            [
                snowpark.Row(
                    ID=self.model_id,
                    NAME=self.model_name,
                    VERSION=self.model_version,
                    URI=f"sfc://{_DATABASE_NAME}.{_SCHEMA_NAME}.model_stage",
                )
            ],
        )
        self.add_session_mock_sql(
            query=f"""
                DELETE FROM {_DATABASE_NAME}.{_SCHEMA_NAME}.{_REGISTRY_TABLE_NAME} WHERE ID='{self.model_id}'
            """,
            result=mock_data_frame.MockDataFrame([snowpark.Row(**{"number of rows deleted": 1})]),
        )
        self.add_session_mock_sql(
            query=f"DROP STAGE {_DATABASE_NAME}.{_SCHEMA_NAME}.model_stage",
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(**{"status": f"'{_DATABASE_NAME}.{_SCHEMA_NAME}.model_stage' successfully dropped."})]
            ),
        )
        self.template_test_set_attribute(
            "DELETION",
            {
                "URI": f"sfc://{_DATABASE_NAME}.{_SCHEMA_NAME}.model_stage",
                "delete_artifact": True,
            },
            use_id=True,
        )

        with absltest.mock.patch.object(
            model_registry,
            "_get_new_unique_identifier",
            return_value=self.event_id,
        ):
            model_registry.delete_model(model_name="name", model_version="abc")


if __name__ == "__main__":
    absltest.main()
