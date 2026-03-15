import uuid

import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import custom_model, model_signature
from snowflake.ml.registry import registry
from snowflake.snowpark import exceptions as sp_exceptions
from tests.integ.snowflake.ml.test_utils import (
    common_test_base,
    db_manager,
    test_env_utils,
)


class DemoModelWithParams(custom_model.CustomModel):
    """Custom model that accepts inference parameters."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(
        self,
        input_df: pd.DataFrame,
        *,
        temperature: float = 1.0,
    ) -> pd.DataFrame:
        """Predict with an inference parameter.

        Args:
            input_df: Input features DataFrame
            temperature: A parameter that must be constant across all rows

        Returns:
            DataFrame with output and parameter info
        """
        return pd.DataFrame(
            {
                "output": input_df["feature"] * temperature,
                "temperature_used": [temperature] * len(input_df),
            }
        )


class DemoModelWithListParam(custom_model.CustomModel):
    """Custom model that accepts a list parameter (unhashable type)."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(
        self,
        input_df: pd.DataFrame,
        *,
        stop_words: list[str] = [],  # noqa: B006
    ) -> pd.DataFrame:
        """Predict with a list parameter.

        Args:
            input_df: Input features DataFrame
            stop_words: A list parameter for stop words

        Returns:
            DataFrame with output and parameter info
        """
        if stop_words is None:
            stop_words = []
        return pd.DataFrame(
            {
                "output": input_df["feature"],
                "stop_words_count": [len(stop_words)] * len(input_df),
            }
        )


class DemoModelWithDictParam(custom_model.CustomModel):
    """Custom model that accepts a dict parameter (structured config)."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(
        self,
        input_df: pd.DataFrame,
        *,
        config: dict = {"temperature": 1.0, "top_k": 50},  # noqa: B006
    ) -> pd.DataFrame:
        temperature = config.get("temperature", 1.0)
        top_k = config.get("top_k", 50)
        return pd.DataFrame(
            {
                "output": input_df["feature"] * temperature,
                "top_k_used": [top_k] * len(input_df),
            }
        )


class TestCustomModelWithParamsWarehouseInteg(common_test_base.CommonTestBase):
    """Integration tests for warehouse inference with model parameters.

    These tests verify that:
    1. Constant param values work correctly
    2. Varying param values across rows raise an appropriate error
    3. NULL/missing params fall back to defaults
    4. Explicit SQL NULL falls back to defaults
    5. List (unhashable) params work via the slow path
    """

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        super().setUp()

        self._run_id = uuid.uuid4().hex
        self._test_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self._run_id, "db").upper()
        self._test_schema = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "schema"
        ).upper()

        self._db_manager = db_manager.DBManager(self.session)
        self._db_manager.create_database(self._test_db)
        self._db_manager.create_schema(self._test_schema)
        self._db_manager.cleanup_databases(expire_hours=6)
        self.registry = registry.Registry(self.session)

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._test_db)
        super().tearDown()

    def _log_model_with_params(self) -> "registry.ModelVersion":
        """Log a model with params and return the model version."""
        model = DemoModelWithParams(custom_model.ModelContext())

        # Sample data for signature inference
        sample_input = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
        sample_output = model.predict(sample_input, temperature=1.0)

        # Define ParamSpec for the temperature parameter
        params = [
            model_signature.ParamSpec(
                name="temperature",
                dtype=model_signature.DataType.FLOAT,
                default_value=1.0,
            ),
        ]

        sig = model_signature.infer_signature(
            input_data=sample_input,
            output_data=sample_output,
            params=params,
        )

        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python!=1.12.0")
        ]

        mv = self.registry.log_model(
            model=model,
            model_name=f"model_params_test_{self._run_id}",
            version_name=f"v_{self._run_id}",
            conda_dependencies=conda_dependencies,
            signatures={"predict": sig},
            options={"embed_local_ml_library": True},
        )

        return mv

    def _log_model_with_list_param(self) -> "registry.ModelVersion":
        """Log a model with list param and return the model version."""
        model = DemoModelWithListParam(custom_model.ModelContext())

        sample_input = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
        sample_output = model.predict(sample_input, stop_words=["the", "a"])

        params = [
            model_signature.ParamSpec(
                name="stop_words",
                dtype=model_signature.DataType.STRING,
                default_value=[],
                shape=(-1,),
            ),
        ]

        sig = model_signature.infer_signature(
            input_data=sample_input,
            output_data=sample_output,
            params=params,
        )

        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python!=1.12.0")
        ]

        mv = self.registry.log_model(
            model=model,
            model_name=f"model_list_param_test_{self._run_id}",
            version_name=f"v_{self._run_id}",
            conda_dependencies=conda_dependencies,
            signatures={"predict": sig},
            options={"embed_local_ml_library": True},
        )

        return mv

    def _assert_constant_params(self, mv: "registry.ModelVersion") -> None:
        """Verify that warehouse inference works correctly when params are constant."""
        input_df = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})

        # Run with constant param value
        result = mv.run(input_df, function_name="predict", params={"temperature": 2.0})

        # Verify results
        self.assertEqual(len(result), 4)
        # output should be feature * temperature
        pd.testing.assert_series_equal(
            result["output"].reset_index(drop=True),
            pd.Series([2.0, 4.0, 6.0, 8.0], name="output"),
            check_dtype=False,
        )
        # temperature_used should all be 2.0
        self.assertTrue(all(result["temperature_used"] == 2.0))

    def _assert_varying_params_raises_error(self, mv: "registry.ModelVersion") -> None:
        """Verify that warehouse inference raises error when params differ across rows.

        This test calls the model function directly via SQL with varying param values
        to verify that the validation catches the inconsistency.

        We use UNIQUE param values for every row (temperature = id) so that ANY batch
        containing more than one row will have different param values.
        """
        model_name = mv.model_name
        version_name = mv.version_name

        # Call the model function with UNIQUE param values per row
        # temperature = id, so every single row has a different param value
        # Any batch with 2+ rows will definitely fail validation
        sql = f"""
            WITH MODEL_VERSION_ALIAS AS MODEL {model_name} VERSION {version_name},
            test_data AS (
                SELECT
                    SEQ4() + 1 AS id,
                    (SEQ4() + 1)::FLOAT AS feature
                FROM TABLE(GENERATOR(ROWCOUNT => 1000))
            )
            SELECT
                id,
                feature,
                MODEL_VERSION_ALIAS!PREDICT(
                    feature,
                    id::FLOAT  -- unique temperature for every row
                ) AS result
            FROM test_data
        """

        # Execute and expect an error
        with self.assertRaises(sp_exceptions.SnowparkSQLException) as context:
            self.session.sql(sql).collect()

        # Verify the error message mentions the param validation
        error_message = str(context.exception)
        self.assertIn("must be equal", error_message.lower())

    def _assert_null_params_uses_default(self, mv: "registry.ModelVersion") -> None:
        """Verify that NULL param values correctly fall back to defaults."""
        input_df = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

        # Run without specifying params (should use default temperature=1.0)
        result = mv.run(input_df, function_name="predict")

        # Verify results use default temperature of 1.0
        self.assertEqual(len(result), 3)
        pd.testing.assert_series_equal(
            result["output"].reset_index(drop=True),
            pd.Series([1.0, 2.0, 3.0], name="output"),
            check_dtype=False,
        )
        self.assertTrue(all(result["temperature_used"] == 1.0))

    def _assert_explicit_sql_null_uses_default(self, mv: "registry.ModelVersion") -> None:
        """Verify that explicit SQL NULL param values are detected and fall back to defaults.

        This exercises _is_null_param_value to ensure it correctly identifies NULL values
        passed via SQL and uses the default parameter value instead.
        """
        model_name = mv.model_name
        version_name = mv.version_name

        # Call model function with explicit NULL for the temperature param
        sql = f"""
            WITH MODEL_VERSION_ALIAS AS MODEL {model_name} VERSION {version_name},
            test_data AS (
                SELECT 1.0 AS feature UNION ALL
                SELECT 2.0 AS feature UNION ALL
                SELECT 3.0 AS feature
            )
            SELECT
                feature,
                MODEL_VERSION_ALIAS!PREDICT(
                    feature,
                    NULL  -- explicit NULL should use default temperature=1.0
                ):output::FLOAT AS output,
                MODEL_VERSION_ALIAS!PREDICT(
                    feature,
                    NULL
                ):temperature_used::FLOAT AS temperature_used
            FROM test_data
            ORDER BY feature
        """

        result = self.session.sql(sql).collect()

        # Verify results use default temperature of 1.0
        self.assertEqual(len(result), 3)
        for i, row in enumerate(result):
            expected_feature = float(i + 1)
            # output should be feature * default_temperature(1.0)
            self.assertAlmostEqual(row["OUTPUT"], expected_feature, places=5)
            # temperature_used should be the default 1.0
            self.assertAlmostEqual(row["TEMPERATURE_USED"], 1.0, places=5)

    def test_warehouse_inference_with_params(self) -> None:
        """Test warehouse inference with DemoModelWithParams across all param scenarios.

        Logs a single model and validates:
        - Constant params work correctly
        - Varying params across rows raise an appropriate error
        - NULL/missing params fall back to defaults
        - Explicit SQL NULL falls back to defaults
        """
        mv = self._log_model_with_params()

        self._assert_constant_params(mv)
        self._assert_varying_params_raises_error(mv)
        self._assert_null_params_uses_default(mv)
        self._assert_explicit_sql_null_uses_default(mv)

    def test_warehouse_inference_with_list_params(self) -> None:
        """Test warehouse inference with DemoModelWithListParam (unhashable type).

        Logs a single model and validates:
        - Constant list params work correctly (slow path via JSON serialization)
        - Varying list params across rows raise an appropriate error
        """
        mv = self._log_model_with_list_param()

        # Constant list param
        input_df = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
        result = mv.run(input_df, function_name="predict", params={"stop_words": ["the", "a", "an"]})

        # Verify results
        self.assertEqual(len(result), 4)
        # stop_words_count should all be 3 (length of the list)
        self.assertTrue(all(result["stop_words_count"] == 3))

        # Varying list params raise error
        model_name = mv.model_name
        version_name = mv.version_name

        # Call the model function with different list values per row using CASE
        # This creates varying list params that should fail validation
        sql = f"""
            WITH MODEL_VERSION_ALIAS AS MODEL {model_name} VERSION {version_name},
            test_data AS (
                SELECT
                    SEQ4() + 1 AS id,
                    (SEQ4() + 1)::FLOAT AS feature
                FROM TABLE(GENERATOR(ROWCOUNT => 1000))
            )
            SELECT
                id,
                feature,
                MODEL_VERSION_ALIAS!PREDICT(
                    feature,
                    ARRAY_CONSTRUCT('word_' || id::VARCHAR)  -- unique list for every row
                ) AS result
            FROM test_data
        """

        # Execute and expect an error
        with self.assertRaises(sp_exceptions.SnowparkSQLException) as context:
            self.session.sql(sql).collect()

        # Verify the error message mentions the param validation
        error_message = str(context.exception)
        self.assertIn("must be equal", error_message.lower())

    def test_warehouse_inference_with_dict_params_fails(self) -> None:
        """Demonstrate that dict parameters in predict signature do not work yet.

        CustomModel.__init__ validates parameter type annotations via _validate_parameter,
        which rejects dict as an unsupported type. The model cannot even be instantiated.
        """
        with self.assertRaises(TypeError) as context:
            DemoModelWithDictParam(custom_model.ModelContext())

        self.assertIn("unsupported type annotation", str(context.exception).lower())


class TestTableFunctionWithParamsWarehouseInteg(common_test_base.CommonTestBase):
    """Integration tests for table function inference with model parameters.

    These tests verify that models registered as TABLE_FUNCTION work correctly
    with ParamSpec parameters.
    """

    def setUp(self) -> None:
        super().setUp()

        self._run_id = uuid.uuid4().hex
        self._test_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self._run_id, "db").upper()
        self._test_schema = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "schema"
        ).upper()

        self._db_manager = db_manager.DBManager(self.session)
        self._db_manager.create_database(self._test_db)
        self._db_manager.create_schema(self._test_schema)
        self._db_manager.cleanup_databases(expire_hours=6)
        self.registry = registry.Registry(self.session)

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._test_db)
        super().tearDown()

    def _log_table_function_model_with_params(self) -> "registry.ModelVersion":
        """Log a table function model with params and return the model version."""
        model = DemoModelWithParams(custom_model.ModelContext())

        sample_input = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
        sample_output = model.predict(sample_input, temperature=1.5)

        params = [
            model_signature.ParamSpec(
                name="temperature",
                dtype=model_signature.DataType.FLOAT,
                default_value=1.5,
            ),
        ]

        sig = model_signature.infer_signature(
            input_data=sample_input,
            output_data=sample_output,
            params=params,
        )

        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python!=1.12.0")
        ]

        mv = self.registry.log_model(
            model=model,
            model_name=f"model_tf_params_test_{self._run_id}",
            version_name=f"v_{self._run_id}",
            conda_dependencies=conda_dependencies,
            signatures={"predict": sig},
            options={"embed_local_ml_library": True, "function_type": "TABLE_FUNCTION"},
        )

        return mv

    def test_table_function_inference_with_params(self) -> None:
        """Test table function inference across all param scenarios.

        Logs a single TABLE_FUNCTION model and validates:
        - Constant params work correctly
        - Varying params across rows raise an appropriate error
        - NULL/missing params fall back to defaults
        - Explicit SQL NULL falls back to defaults
        """
        mv = self._log_table_function_model_with_params()

        # Constant params
        input_df = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
        result = mv.run(input_df, function_name="predict", params={"temperature": 2.0})
        self.assertEqual(len(result), 4)
        pd.testing.assert_series_equal(
            result["output"].reset_index(drop=True),
            pd.Series([2.0, 4.0, 6.0, 8.0], name="output"),
            check_dtype=False,
        )
        self.assertTrue(all(result["temperature_used"] == 2.0))

        # Varying params raise error
        model_name = mv.model_name
        version_name = mv.version_name

        sql = f"""
            WITH MODEL_VERSION_ALIAS AS MODEL {model_name} VERSION {version_name},
            test_data AS (
                SELECT
                    SEQ4() + 1 AS id,
                    (SEQ4() + 1)::FLOAT AS feature
                FROM TABLE(GENERATOR(ROWCOUNT => 1000))
            )
            SELECT *
            FROM test_data,
                TABLE(MODEL_VERSION_ALIAS!PREDICT(feature, id::FLOAT) OVER (PARTITION BY 1))
        """

        with self.assertRaises(sp_exceptions.SnowparkSQLException) as context:
            self.session.sql(sql).collect()

        error_message = str(context.exception)
        self.assertIn("must be equal", error_message.lower())

        # NULL params use default
        input_df = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
        result = mv.run(input_df, function_name="predict")
        self.assertEqual(len(result), 3)
        pd.testing.assert_series_equal(
            result["output"].reset_index(drop=True),
            pd.Series([1.5, 3.0, 4.5], name="output"),
            check_dtype=False,
        )
        self.assertTrue(all(result["temperature_used"] == 1.5))

        # Explicit SQL NULL uses default
        sql = f"""
            WITH MODEL_VERSION_ALIAS AS MODEL {model_name} VERSION {version_name},
            test_data AS (
                SELECT 1.0::FLOAT AS feature UNION ALL
                SELECT 2.0::FLOAT AS feature UNION ALL
                SELECT 3.0::FLOAT AS feature
            )
            SELECT
                feature,
                tf.output::FLOAT AS output,
                tf.temperature_used::FLOAT AS temperature_used
            FROM test_data,
                TABLE(MODEL_VERSION_ALIAS!PREDICT(feature, NULL::FLOAT) OVER (PARTITION BY 1)) AS tf
            ORDER BY feature
        """

        result = self.session.sql(sql).collect()

        self.assertEqual(len(result), 3)
        for i, row in enumerate(result):
            expected_feature = float(i + 1)
            self.assertAlmostEqual(row["OUTPUT"], expected_feature * 1.5, places=5)
            self.assertAlmostEqual(row["TEMPERATURE_USED"], 1.5, places=5)


class DemoPartitionedModelWithParams(custom_model.CustomModel):
    """Custom partitioned model that accepts inference parameters."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.partitioned_api
    def predict(
        self,
        input_df: pd.DataFrame,
        *,
        temperature: float = 1.5,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "output": input_df["feature"] * temperature,
                "temperature_used": [temperature] * len(input_df),
            }
        )


class TestPartitionedModelWithParamsWarehouseInteg(common_test_base.CommonTestBase):
    """Integration tests for partitioned inference with model parameters.

    These tests verify that models using @partitioned_api work correctly
    with ParamSpec parameters.
    """

    def setUp(self) -> None:
        super().setUp()

        self._run_id = uuid.uuid4().hex
        self._test_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self._run_id, "db").upper()
        self._test_schema = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "schema"
        ).upper()

        self._db_manager = db_manager.DBManager(self.session)
        self._db_manager.create_database(self._test_db)
        self._db_manager.create_schema(self._test_schema)
        self._db_manager.cleanup_databases(expire_hours=6)
        self.registry = registry.Registry(self.session)

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._test_db)
        super().tearDown()

    def _log_partitioned_model_with_params(self) -> "registry.ModelVersion":
        """Log a partitioned model with params and return the model version."""
        model = DemoPartitionedModelWithParams(custom_model.ModelContext())

        sample_input = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
        sample_output = model.predict(sample_input, temperature=1.5)

        params = [
            model_signature.ParamSpec(
                name="temperature",
                dtype=model_signature.DataType.FLOAT,
                default_value=1.5,
            ),
        ]

        sig = model_signature.infer_signature(
            input_data=sample_input,
            output_data=sample_output,
            params=params,
        )

        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python!=1.12.0")
        ]

        mv = self.registry.log_model(
            model=model,
            model_name=f"model_part_params_test_{self._run_id}",
            version_name=f"v_{self._run_id}",
            conda_dependencies=conda_dependencies,
            signatures={"predict": sig},
            options={
                "embed_local_ml_library": True,
                "method_options": {"predict": {"function_type": "TABLE_FUNCTION"}},
            },
        )

        return mv

    def test_partitioned_inference_with_params(self) -> None:
        """Test partitioned inference across all param scenarios.

        Logs a single partitioned model and validates:
        - Constant params work correctly
        - Varying params across rows raise an appropriate error
        - NULL/missing params fall back to defaults
        - Explicit SQL NULL falls back to defaults
        """
        mv = self._log_partitioned_model_with_params()

        # Constant params
        input_df = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
        result = mv.run(input_df, function_name="predict", params={"temperature": 2.0})
        self.assertEqual(len(result), 4)
        result_sorted = result.sort_values("output").reset_index(drop=True)
        pd.testing.assert_series_equal(
            result_sorted["output"],
            pd.Series([2.0, 4.0, 6.0, 8.0], name="output"),
            check_dtype=False,
        )
        self.assertTrue(all(result_sorted["temperature_used"] == 2.0))

        # Varying params raise error
        model_name = mv.model_name
        version_name = mv.version_name

        sql = f"""
            WITH MODEL_VERSION_ALIAS AS MODEL {model_name} VERSION {version_name},
            test_data AS (
                SELECT
                    SEQ4() + 1 AS id,
                    (SEQ4() + 1)::FLOAT AS feature
                FROM TABLE(GENERATOR(ROWCOUNT => 1000))
            )
            SELECT *
            FROM test_data,
                TABLE(MODEL_VERSION_ALIAS!PREDICT(feature, id::FLOAT) OVER (PARTITION BY 1))
        """

        with self.assertRaises(sp_exceptions.SnowparkSQLException) as context:
            self.session.sql(sql).collect()

        error_message = str(context.exception)
        self.assertIn("must be equal", error_message.lower())

        # NULL params use default
        input_df = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
        result = mv.run(input_df, function_name="predict")

        self.assertEqual(len(result), 3)
        result_sorted = result.sort_values("output").reset_index(drop=True)
        pd.testing.assert_series_equal(
            result_sorted["output"],
            pd.Series([1.5, 3.0, 4.5], name="output"),
            check_dtype=False,
        )
        self.assertTrue(all(result_sorted["temperature_used"] == 1.5))

        # Explicit SQL NULL uses default
        sql = f"""
            WITH MODEL_VERSION_ALIAS AS MODEL {model_name} VERSION {version_name},
            test_data AS (
                SELECT 1.0::FLOAT AS feature UNION ALL
                SELECT 2.0::FLOAT AS feature UNION ALL
                SELECT 3.0::FLOAT AS feature
            )
            SELECT
                tf.output::FLOAT AS output,
                tf.temperature_used::FLOAT AS temperature_used
            FROM test_data,
                TABLE(MODEL_VERSION_ALIAS!PREDICT(feature, NULL::FLOAT) OVER (PARTITION BY 1)) AS tf
        """

        result = self.session.sql(sql).collect()

        self.assertEqual(len(result), 3)
        outputs = sorted(row["OUTPUT"] for row in result)
        expected_outputs = [1.5, 3.0, 4.5]
        for actual, expected in zip(outputs, expected_outputs):
            self.assertAlmostEqual(actual, expected, places=5)
        for row in result:
            self.assertAlmostEqual(row["TEMPERATURE_USED"], 1.5, places=5)


if __name__ == "__main__":
    absltest.main()
