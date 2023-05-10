import datetime
import json
from typing import Any, Dict, List, Union, cast

from _schema import _METADATA_TABLE_SCHEMA, _REGISTRY_TABLE_SCHEMA
from absl.testing import absltest

from snowflake import connector, snowpark
from snowflake.ml._internal.utils import formatting
from snowflake.ml.registry import model_registry
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.ml.utils import telemetry

_DATABASE_NAME = "MODEL_REGISTRY"
_SCHEMA_NAME = "PUBLIC"
_REGISTRY_TABLE_NAME = "MODELS"
_METADATA_TABLE_NAME = "METADATA"
_FULLY_QUALIFIED_REGISTRY_TABLE_NAME = f'"{_DATABASE_NAME}"."{_SCHEMA_NAME}"."{_REGISTRY_TABLE_NAME}"'
_REGISTRY_SCHEMA_STRING = ", ".join([f"{k} {v}" for k, v in _REGISTRY_TABLE_SCHEMA.items()])
_METADATA_INSERT_COLUMNS_STRING = ",".join(filter(lambda x: x != "SEQUENCE_ID", _METADATA_TABLE_SCHEMA.keys()))
_METADATA_SCHEMA_STRING = ", ".join(
    [
        f"{k} {v.format(registry_table_name=_FULLY_QUALIFIED_REGISTRY_TABLE_NAME)}"
        for k, v in _METADATA_TABLE_SCHEMA.items()
    ]
)


class ModelRegistryTest(absltest.TestCase):
    """Testing ModelRegistry functions."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environemnts for testing."""
        self.maxDiff = None
        self._session = mock_session.MockSession(conn=None, test_case=self)
        self.event_id = "fedcba9876543210fedcba9876543210"
        self.model_id = "0123456789abcdef0123456789abcdef"
        self.datetime = datetime.datetime(2022, 11, 4, 17, 1, 30, 153000)

    def tearDown(self) -> None:
        """Complete test case. Ensure all expected operations have been observed."""
        self._session.finalize()

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

    def setup_open_call(self) -> None:
        self.add_session_mock_sql(
            query="SHOW DATABASES LIKE 'MODEL_REGISTRY'",
            result=mock_data_frame.MockDataFrame(
                self.get_show_databases_success(name=_DATABASE_NAME)
            ).add_collect_result(self.get_show_databases_success(name=_DATABASE_NAME)),
        )
        self.add_session_mock_sql(
            query="SHOW SCHEMAS LIKE 'PUBLIC' IN DATABASE \"MODEL_REGISTRY\"",
            result=mock_data_frame.MockDataFrame(self.get_show_schemas_success(name=_SCHEMA_NAME)).add_collect_result(
                self.get_show_schemas_success(name=_SCHEMA_NAME)
            ),
        )
        self.add_session_mock_sql(
            query='SHOW TABLES LIKE \'MODELS\' IN "MODEL_REGISTRY"."PUBLIC"',
            result=mock_data_frame.MockDataFrame(
                self.get_show_tables_success(name=_REGISTRY_TABLE_NAME)
            ).add_collect_result(self.get_show_tables_success(name=_REGISTRY_TABLE_NAME)),
        )
        self.add_session_mock_sql(
            query='SHOW TABLES LIKE \'METADATA\' IN "MODEL_REGISTRY"."PUBLIC"',
            result=mock_data_frame.MockDataFrame(
                self.get_show_tables_success(name=_METADATA_TABLE_NAME)
            ).add_collect_result(self.get_show_tables_success(name=_METADATA_TABLE_NAME)),
        )

    def setup_list_model_call(self) -> mock_data_frame.MockDataFrame:
        """Setup the expected calls originating from list_model."""
        result_df = mock_data_frame.MockDataFrame()

        self.add_session_mock_sql(
            query=('SELECT * FROM "MODEL_REGISTRY"."PUBLIC"."MODELS_VIEW"'),
            result=result_df,
        )
        # Returning result_df to allow the caller to add more expected operations.
        return result_df

    def setup_create_views_call(self) -> None:
        """Setup the expected calls originating from _create_views."""
        self.add_session_mock_sql(
            query=(
                """CREATE OR REPLACE VIEW "MODEL_REGISTRY"."PUBLIC"."METADATA_LAST_DESCRIPTION" COPY GRANTS AS
                        SELECT DISTINCT
                            MODEL_ID,
                            (LAST_VALUE(VALUE) OVER (
                                PARTITION BY MODEL_ID ORDER BY SEQUENCE_ID))['DESCRIPTION']
                            as DESCRIPTION
                        FROM "METADATA" WHERE ATTRIBUTE_NAME = 'DESCRIPTION'"""
            ),
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status="View METADATA_LAST_DESCRIPTION successfully created.")]
            ),
        )
        self.add_session_mock_sql(
            query=(
                """CREATE OR REPLACE VIEW "MODEL_REGISTRY"."PUBLIC"."METADATA_LAST_METRICS" COPY GRANTS AS
                        SELECT DISTINCT
                            MODEL_ID,
                            (LAST_VALUE(VALUE) OVER (
                                PARTITION BY MODEL_ID ORDER BY SEQUENCE_ID))['METRICS']
                            as METRICS
                        FROM "METADATA" WHERE ATTRIBUTE_NAME = 'METRICS'"""
            ),
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status="View METADATA_LAST_METRICS successfully created.")]
            ),
        )
        self.add_session_mock_sql(
            query=(
                """CREATE OR REPLACE VIEW "MODEL_REGISTRY"."PUBLIC"."METADATA_LAST_NAME" COPY GRANTS AS
                        SELECT DISTINCT
                            MODEL_ID,
                            (LAST_VALUE(VALUE) OVER (
                                PARTITION BY MODEL_ID ORDER BY SEQUENCE_ID))['NAME']
                            as NAME
                        FROM "METADATA" WHERE ATTRIBUTE_NAME = 'NAME'"""
            ),
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status="View METADATA_LAST_NAME successfully created.")]
            ),
        )
        self.add_session_mock_sql(
            query=(
                """CREATE OR REPLACE VIEW "MODEL_REGISTRY"."PUBLIC"."METADATA_LAST_TAGS" COPY GRANTS AS
                        SELECT DISTINCT
                            MODEL_ID,
                            (LAST_VALUE(VALUE) OVER (
                                PARTITION BY MODEL_ID ORDER BY SEQUENCE_ID))['TAGS']
                            as TAGS
                        FROM "METADATA" WHERE ATTRIBUTE_NAME = 'TAGS'"""
            ),
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status="View METADATA_LAST_TAGS successfully created.")]
            ),
        )
        self.add_session_mock_sql(
            query=(
                """CREATE OR REPLACE VIEW "MODEL_REGISTRY"."PUBLIC"."METADATA_LAST_REGISTRATION" COPY GRANTS AS
                        SELECT DISTINCT
                            MODEL_ID, EVENT_TIMESTAMP as REGISTRATION_TIMESTAMP
                        FROM "METADATA" WHERE ATTRIBUTE_NAME = 'REGISTRATION'"""
            ),
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status="View METADATA_LAST_TAGS successfully created.")]
            ),
        )
        self.add_session_mock_sql(
            query=(
                """CREATE OR REPLACE VIEW "MODEL_REGISTRY"."PUBLIC"."MODELS_VIEW" COPY GRANTS AS
                    SELECT "MODELS".*, "METADATA_LAST_DESCRIPTION".DESCRIPTION AS DESCRIPTION,
                        "METADATA_LAST_METRICS".METRICS AS METRICS, "METADATA_LAST_NAME".NAME AS NAME,
                        "METADATA_LAST_TAGS".TAGS AS TAGS,
                        "METADATA_LAST_REGISTRATION".REGISTRATION_TIMESTAMP AS REGISTRATION_TIMESTAMP
                    FROM "MODELS"
                        LEFT JOIN "METADATA_LAST_DESCRIPTION" ON ("METADATA_LAST_DESCRIPTION".MODEL_ID = "MODELS".ID)
                        LEFT JOIN "METADATA_LAST_METRICS" ON ("METADATA_LAST_METRICS".MODEL_ID = "MODELS".ID)
                        LEFT JOIN "METADATA_LAST_NAME" ON ("METADATA_LAST_NAME".MODEL_ID = "MODELS".ID)
                        LEFT JOIN "METADATA_LAST_TAGS" ON ("METADATA_LAST_TAGS".MODEL_ID = "MODELS".ID)
                        LEFT JOIN "METADATA_LAST_REGISTRATION" ON ("METADATA_LAST_REGISTRATION".MODEL_ID = "MODELS".ID)
                """
            ),
            result=mock_data_frame.MockDataFrame([snowpark.Row(status="View MODELS_VIEW successfully created.")]),
        )

    def template_test_set_attribute(
        self, attribute_name: str, attribute_value: Union[str, Dict[Any, Any]], result_num_inserted: int = 1
    ) -> None:
        expected_df = self.setup_list_model_call()
        expected_df.add_operation("filter")
        expected_df.add_count_result(1)

        self._session.add_operation("get_current_role", result="current_role")

        self.add_session_mock_sql(
            query=(
                f"""INSERT INTO "MODEL_REGISTRY"."PUBLIC"."METADATA" ({_METADATA_INSERT_COLUMNS_STRING})
                         SELECT '{attribute_name}','{self.event_id}',CURRENT_TIMESTAMP(),'{self.model_id}','SET',
                         'current_role',OBJECT_CONSTRUCT('{attribute_name}',
                         {formatting.format_value_for_select(attribute_value)})"""
            ),
            result=mock_data_frame.MockDataFrame([snowpark.Row(**{"number of rows inserted": result_num_inserted})]),
        )

    def test_create_new(self) -> None:
        """Verify that we can create a new ModelRegistry database with the default names."""
        # "Create" calls.
        statement_params = telemetry.get_function_usage_statement_params(
            project="MLOps",
            subproject="ModelRegistry",
            function_name="snowflake.ml.registry.model_registry._create_registry_database",
        )
        self.add_session_mock_sql(
            query="SHOW DATABASES LIKE 'MODEL_REGISTRY'",
            result=mock_data_frame.MockDataFrame([]).add_collect_result([], statement_params=statement_params),
        )
        self.add_session_mock_sql(
            query='CREATE DATABASE "MODEL_REGISTRY"',
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status="Database MODEL_REGISTRY successfully created.")],
                collect_statement_params=statement_params,
            ),
        )
        self.add_session_mock_sql(
            query="SHOW SCHEMAS LIKE 'PUBLIC' IN DATABASE \"MODEL_REGISTRY\"",
            result=mock_data_frame.MockDataFrame(self.get_show_schemas_success(name=_SCHEMA_NAME)).add_collect_result(
                self.get_show_schemas_success(name=_SCHEMA_NAME),
                statement_params=statement_params,
            ),
        )
        self.add_session_mock_sql(
            query=f'CREATE TABLE "MODEL_REGISTRY"."PUBLIC"."MODELS" ({_REGISTRY_SCHEMA_STRING})',
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status="Table MODELS successfully created.")],
                collect_statement_params=statement_params,
            ),
        )
        self.add_session_mock_sql(
            query=f'CREATE TABLE "MODEL_REGISTRY"."PUBLIC"."METADATA" ({_METADATA_SCHEMA_STRING})',
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(status="Table METADATA successfully created.")],
                collect_statement_params=statement_params,
            ),
        )

        self.setup_create_views_call()
        model_registry.create_model_registry(
            session=cast(snowpark.Session, self._session),
            database_name=_DATABASE_NAME,
        )

    def test_open_existing(self) -> None:
        """Verify that we can open an existing ModelRegistry database with the default names."""
        self.add_session_mock_sql(
            query="SHOW DATABASES LIKE 'MODEL_REGISTRY'",
            result=mock_data_frame.MockDataFrame(self.get_show_databases_success(name=_DATABASE_NAME)),
        )
        self.add_session_mock_sql(
            query="SHOW SCHEMAS LIKE 'PUBLIC' IN DATABASE \"MODEL_REGISTRY\"",
            result=mock_data_frame.MockDataFrame(self.get_show_schemas_success(name=_SCHEMA_NAME)),
        )
        self.add_session_mock_sql(
            query='SHOW TABLES LIKE \'MODELS\' IN "MODEL_REGISTRY"."PUBLIC"',
            result=mock_data_frame.MockDataFrame(self.get_show_tables_success(name=_REGISTRY_TABLE_NAME)),
        )
        self.add_session_mock_sql(
            query='SHOW TABLES LIKE \'METADATA\' IN "MODEL_REGISTRY"."PUBLIC"',
            result=mock_data_frame.MockDataFrame(self.get_show_tables_success(name=_METADATA_TABLE_NAME)),
        )
        model_registry.ModelRegistry(session=cast(snowpark.Session, self._session))

    def test_create_existing_fail(self) -> None:
        """Verify that we skip creation in case we want to create a ModelRegistry that already exists."""
        self.add_session_mock_sql(
            query="SHOW DATABASES LIKE 'MODEL_REGISTRY'",
            result=mock_data_frame.MockDataFrame(self.get_show_databases_success(name=_DATABASE_NAME)),
        )
        self.assertFalse(
            model_registry.create_model_registry(
                session=cast(snowpark.Session, self._session),
                database_name=_DATABASE_NAME,
            )
        )

    def test_list_models(self) -> None:
        """Test the normal operation of list_models. We create a view and return the model metadata."""
        model_registry = self.get_model_registry()
        self.setup_list_model_call().add_collect_result([snowpark.Row(ID=self.model_id, NAME="model_name")])

        model_list = model_registry.list_models().collect()
        self.assertEqual(model_list, [snowpark.Row(ID=self.model_id, NAME="model_name")])

    def test_get_model_name(self) -> None:
        """Test that we can get the name of an existing model from the registry."""
        model_registry = self.get_model_registry()
        expected_df = self.setup_list_model_call()
        expected_df.add_operation(operation="filter")
        expected_df.add_operation(
            operation="select",
            args=("NAME",),
            result=mock_data_frame.MockDataFrame([snowpark.Row(ID=self.model_id, NAME="model_name")]),
        )

        model_name = model_registry.get_model_name(id=self.model_id)
        self.assertEqual(model_name, "model_name")

    def test_get_model_name_via_reference(self) -> None:
        """Test that we can get the name of an existing model from the registry via ModelReference."""
        model_registry_obj = self.get_model_registry()
        expected_df = self.setup_list_model_call()
        expected_df.add_operation(operation="filter")
        expected_df.add_operation(
            operation="select",
            args=("NAME",),
            result=mock_data_frame.MockDataFrame([snowpark.Row(ID=self.model_id, NAME="model_name")]),
        )

        model = model_registry.ModelReference(registry=model_registry_obj, id=self.model_id)

        self.assertEqual(model.get_model_name(), "model_name")  # type: ignore

    def test_get_model_name_not_found(self) -> None:
        """Test that if the model is not found, we return None."""
        model_registry = self.get_model_registry()
        expected_df = self.setup_list_model_call()
        expected_df.add_operation(operation="filter")
        expected_df.add_operation(operation="select", args=("NAME",))
        expected_df.add_collect_result([])

        model_name = model_registry.get_model_name(id=self.model_id)
        self.assertEqual(model_name, None)

    def test_set_model_name(self) -> None:
        """Test that we can set the name for an existing model."""
        model_registry = self.get_model_registry()
        self.template_test_set_attribute("NAME", "new_name")

        with absltest.mock.patch.object(
            model_registry,
            "_get_new_unique_identifier",
            return_value=self.event_id,
        ):
            model_registry.set_model_name(id=self.model_id, name="new_name")

    def test_set_model_name_fail(self) -> None:
        """Test that setting a name fails when the model foes not exist."""
        model_registry = self.get_model_registry()
        self.template_test_set_attribute("NAME", "new_name", result_num_inserted=0)

        with absltest.mock.patch.object(
            model_registry,
            "_get_new_unique_identifier",
            return_value=self.event_id,
        ):
            with self.assertRaises(connector.DataError):
                model_registry.set_model_name(id=self.model_id, name="new_name")

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
            model_registry.set_model_description(id=self.model_id, description="new_description")

    def test_get_model_description(self) -> None:
        """Test that we can get the description of an existing model from the registry."""
        model_registry = self.get_model_registry()
        expected_df = self.setup_list_model_call()
        expected_df.add_operation(operation="filter")
        expected_df.add_operation(
            operation="select",
            args=("DESCRIPTION",),
            result=mock_data_frame.MockDataFrame([snowpark.Row(ID=self.model_id, DESCRIPTION="model_description")]),
        )

        model_description = model_registry.get_model_description(id=self.model_id)
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
            operation="table", args=('"MODEL_REGISTRY"."PUBLIC"."METADATA"',), result=expected_df
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
            operation="table", args=('"MODEL_REGISTRY"."PUBLIC"."METADATA"',), result=expected_df
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

        self.assertEqual(model_registry.get_model_history(id=self.model_id).collect(), expected_collect_result)

    def test_set_metric_no_existing(self) -> None:
        """Test that we can set a metric for an existing model that does not yet have any metrics set."""
        model_registry = self.get_model_registry()
        expected_df = self.setup_list_model_call()
        expected_df.add_operation(operation="filter")
        expected_df.add_operation(
            operation="select",
            args=("METRICS",),
            result=mock_data_frame.MockDataFrame([snowpark.Row(ID=self.model_id, METRICS=None)]),
        )
        self.template_test_set_attribute("METRICS", {"voight-kampff": 0.9})

        # Mock unique identifier for event id.
        with absltest.mock.patch.object(
            model_registry,
            "_get_new_unique_identifier",
            return_value=self.event_id,
        ):
            model_registry.set_metric(id=self.model_id, name="voight-kampff", value=0.9)

    def test_set_metric_with_existing(self) -> None:
        """Test that we can set a metric for an existing model that already has metrics."""
        model_registry = self.get_model_registry()
        expected_df = self.setup_list_model_call()
        expected_df.add_operation(operation="filter")
        expected_df.add_operation(
            operation="select",
            args=("METRICS",),
            result=mock_data_frame.MockDataFrame([snowpark.Row(ID=self.model_id, METRICS='{"human-factor": 1.1}')]),
        )
        self.template_test_set_attribute("METRICS", {"human-factor": 1.1, "voight-kampff": 0.9})

        # Mock unique identifier for event id.
        with absltest.mock.patch.object(
            model_registry,
            "_get_new_unique_identifier",
            return_value=self.event_id,
        ):
            model_registry.set_metric(id=self.model_id, name="voight-kampff", value=0.9)

    def test_get_metrics(self) -> None:
        """Test that we can get the metrics for an existing model."""
        metrics_dict = {"human-factor": 1.1, "voight-kampff": 0.9}
        model_registry = self.get_model_registry()
        expected_df = self.setup_list_model_call()
        expected_df.add_operation(operation="filter")
        expected_df.add_operation(
            operation="select",
            args=("METRICS",),
            result=mock_data_frame.MockDataFrame([snowpark.Row(ID=self.model_id, METRICS=json.dumps(metrics_dict))]),
        )

        self.assertEqual(model_registry.get_metrics(id=self.model_id), metrics_dict)

    def test_get_metric_value(self) -> None:
        """Test that we can get a single metric value for an existing model."""
        metrics_dict = {"human-factor": 1.1, "voight-kampff": 0.9}
        model_registry = self.get_model_registry()
        expected_df = self.setup_list_model_call()
        expected_df.add_operation(operation="filter")
        expected_df.add_operation(
            operation="select",
            args=("METRICS",),
            result=mock_data_frame.MockDataFrame([snowpark.Row(ID=self.model_id, METRICS=json.dumps(metrics_dict))]),
        )

        self.assertEqual(model_registry.get_metric_value(id=self.model_id, name="human-factor"), 1.1)

    def test_private_insert_registry_entry(self) -> None:
        model_registry = self.get_model_registry()

        self.add_session_mock_sql(
            query="""INSERT INTO "MODEL_REGISTRY"."PUBLIC"."MODELS" ( ID,NAME,TYPE,URI ) SELECT 'id','name','type',"""
            + """'uri'""",
            result=mock_data_frame.MockDataFrame([snowpark.Row(**{"number of rows inserted": 1})]),
        )

        model_properties = {"ID": "id", "NAME": "name", "TYPE": "type", "URI": "uri"}

        model_registry._insert_registry_entry(id="id", properties=model_properties)

    def test_register_model(self) -> None:
        model_registry = self.get_model_registry()

        self._session.add_operation("get_current_role", result="current_role")

        mock_df = self.setup_list_model_call()
        mock_df.add_operation("filter", result=mock_data_frame.MockDataFrame([]))
        mock_df.add_count_result(0)

        self.add_session_mock_sql(
            query="""INSERT INTO "MODEL_REGISTRY"."PUBLIC"."MODELS" ( CREATION_ENVIRONMENT_SPEC,CREATION_ROLE,
                    CREATION_TIME,ID,INPUT_SPEC,OUTPUT_SPEC,TYPE,URI,VERSION )
                SELECT OBJECT_CONSTRUCT('python','3.8.13'),'current_role',CURRENT_TIMESTAMP(),
                    'id',null,null,'type','uri','abc'""",
            result=mock_data_frame.MockDataFrame([snowpark.Row(**{"number of rows inserted": 1})]),
        )

        # Mock calls to variable values: python version and internal _set_metadata_attribute.
        with absltest.mock.patch.object(model_registry, "_set_metadata_attribute", return_value=True):
            with absltest.mock.patch(
                "model_registry.sys.version_info", new_callable=absltest.mock.PropertyMock(return_value=(3, 8, 13))
            ):
                model_registry.register_model(
                    id="id", uri="uri", type="type", name="name", version="abc", tags={"tag_name": "tag_value"}
                )

    def test_register_model_no_tags(self) -> None:
        """Test registering a model without giving a tag."""
        model_registry = self.get_model_registry()

        self._session.add_operation("get_current_role", result="current_role")

        mock_df = self.setup_list_model_call()
        mock_df.add_operation("filter", result=mock_data_frame.MockDataFrame([]))
        mock_df.add_count_result(0)

        self.add_session_mock_sql(
            query=f"""INSERT INTO "MODEL_REGISTRY"."PUBLIC"."MODELS" ( CREATION_ENVIRONMENT_SPEC,CREATION_ROLE,
                    CREATION_TIME,ID,INPUT_SPEC,OUTPUT_SPEC,TYPE,URI,VERSION )
                SELECT OBJECT_CONSTRUCT('python','3.8.13'),'current_role',CURRENT_TIMESTAMP(),
                    '{self.model_id}',null,null,'type','uri','abc'""",
            result=mock_data_frame.MockDataFrame([snowpark.Row(**{"number of rows inserted": 1})]),
        )

        # Adding registration metadata.
        self.template_test_set_attribute(
            "REGISTRATION",
            {
                "CREATION_ENVIRONMENT_SPEC": {"python": "3.8.13"},
                "CREATION_ROLE": "current_role",
                "CREATION_TIME": formatting.SqlStr("CURRENT_TIMESTAMP()"),
                "ID": self.model_id,
                "INPUT_SPEC": None,
                "OUTPUT_SPEC": None,
                "TYPE": "type",
                "URI": "uri",
                "VERSION": "abc",
            },
        )

        self.template_test_set_attribute(
            "NAME",
            "name",
        )

        # Mock calls to variable values: internal set_model_name, python version, current time
        with absltest.mock.patch.object(
            model_registry,
            "_get_new_unique_identifier",
            return_value=self.event_id,
        ):
            with absltest.mock.patch(
                "model_registry.sys.version_info", new_callable=absltest.mock.PropertyMock(return_value=(3, 8, 13))
            ):
                model_registry.register_model(id=self.model_id, uri="uri", type="type", name="name", version="abc")

    def test_get_tags(self) -> None:
        """Test that get_tags is working correctly with various types."""
        model_registry = self.get_model_registry()
        self.setup_list_model_call().add_operation(operation="filter").add_operation(
            operation="select",
            args=("TAGS",),
            result=mock_data_frame.MockDataFrame(
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
                ],
            ),
        )
        model_registry.get_tags(id=self.model_id)

    def test_register_model_with_description(self) -> None:
        """Test registering a model with a description."""
        model_registry = self.get_model_registry()

        self._session.add_operation("get_current_role", result="current_role")

        mock_df = self.setup_list_model_call()
        mock_df.add_operation("filter", result=mock_data_frame.MockDataFrame([]))
        mock_df.add_count_result(0)

        self.add_session_mock_sql(
            query=f"""INSERT INTO "MODEL_REGISTRY"."PUBLIC"."MODELS" ( CREATION_ENVIRONMENT_SPEC,CREATION_ROLE,
                    CREATION_TIME,ID,INPUT_SPEC,OUTPUT_SPEC,TYPE,URI,VERSION )
                SELECT OBJECT_CONSTRUCT('python','3.8.13'),'current_role',CURRENT_TIMESTAMP(),
                    '{self.model_id}',null,null,'type','uri','abc'""",
            result=mock_data_frame.MockDataFrame([snowpark.Row(**{"number of rows inserted": 1})]),
        )

        # Adding registration metadata.
        self.template_test_set_attribute(
            "REGISTRATION",
            {
                "CREATION_ENVIRONMENT_SPEC": {"python": "3.8.13"},
                "CREATION_ROLE": "current_role",
                "CREATION_TIME": formatting.SqlStr("CURRENT_TIMESTAMP()"),
                "ID": self.model_id,
                "INPUT_SPEC": None,
                "OUTPUT_SPEC": None,
                "TYPE": "type",
                "URI": "uri",
                "VERSION": "abc",
            },
        )

        self.template_test_set_attribute(
            "DESCRIPTION",
            "Model B-263-54",
        )

        self.template_test_set_attribute(
            "NAME",
            "name",
        )

        # Mock calls to variable values: internal set_model_description, python version, current time
        with absltest.mock.patch.object(
            model_registry,
            "_get_new_unique_identifier",
            return_value=self.event_id,
        ):
            with absltest.mock.patch(
                "model_registry.sys.version_info", new_callable=absltest.mock.PropertyMock(return_value=(3, 8, 13))
            ):
                model_registry.register_model(
                    id=self.model_id,
                    uri="uri",
                    type="type",
                    version="abc",
                    name="name",
                    description="Model B-263-54",
                )

    def test_log_model_path_file(self) -> None:
        """Test log_model_path() when the model is a file.

        Validate log_model_path() can perform stage file put operation with the expected stage path and call
        register_model() with the expected arguments.
        """
        model_registry = self.get_model_registry()

        self.add_session_mock_sql(
            query=f"""CREATE OR REPLACE STAGE "{_DATABASE_NAME}"."{_SCHEMA_NAME}".SNOWML_MODEL_{self.model_id}""",
            result=mock_data_frame.MockDataFrame(
                [snowpark.Row(**{"status": f"Stage area SNOWML_MODEL_{self.model_id.upper()} successfully created."})]
            ),
        )

        # Mock the snowpark.session.file operation
        mock_sp_file_operation = absltest.mock.Mock()
        self._session.__setattr__("file", mock_sp_file_operation)

        expected_stage_path = f'"{_DATABASE_NAME}"."{_SCHEMA_NAME}".SNOWML_MODEL_{self.model_id.upper()}/data'

        with absltest.mock.patch("model_registry.os.path.isfile", return_value=True) as mock_isfile:
            with absltest.mock.patch.object(
                model_registry,
                "_get_new_unique_identifier",
                return_value=self.model_id,
            ):
                with absltest.mock.patch.object(
                    model_registry,
                    "register_model",
                    return_value=True,
                ):
                    model_registry.log_model_path(
                        path="path", type="type", name="name", version="abc", description="description"
                    )
                    mock_isfile.assert_called_once_with("path")
                    mock_sp_file_operation.put.assert_called_with("path", expected_stage_path)
                    assert isinstance(model_registry.register_model, absltest.mock.Mock)
                    model_registry.register_model.assert_called_with(
                        id=self.model_id,
                        type="type",
                        uri=f"sfc:MODEL_REGISTRY.PUBLIC.SNOWML_MODEL_{self.model_id.upper()}",
                        name="name",
                        version="abc",
                        description="description",
                        tags=None,
                    )

    def test_delete_model_with_artifact(self) -> None:
        """Test deleting a model and artifact from the registry."""
        model_registry = self.get_model_registry()
        self.setup_list_model_call().add_operation(operation="filter").add_collect_result(
            [snowpark.Row(URI="sfc://model_stage")],
        )
        self.add_session_mock_sql(
            query=f"""DELETE FROM "MODEL_REGISTRY"."PUBLIC"."MODELS" WHERE ID='{self.model_id}'""",
            result=mock_data_frame.MockDataFrame([snowpark.Row(**{"number of rows deleted": 1})]),
        )
        self.add_session_mock_sql(
            query="DROP STAGE model_stage",
            result=mock_data_frame.MockDataFrame([snowpark.Row(**{"status": "'model_stage' successfully dropped."})]),
        )
        self.template_test_set_attribute(
            "DELETION",
            {
                "URI": "sfc://model_stage",
                "delete_artifact": True,
            },
        )

        with absltest.mock.patch.object(
            model_registry,
            "_get_new_unique_identifier",
            return_value=self.event_id,
        ):
            model_registry.delete_model(id=self.model_id)


if __name__ == "__main__":
    absltest.main()
