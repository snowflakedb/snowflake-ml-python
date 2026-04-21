import datetime
import uuid
from typing import Any

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
        config: dict = {  # noqa: B006
            "temperature": 1.0,
            "top_k": 50,
            "nested_list": [[1, 2], [3, 4]],
            "nested_dict": [{"a": 1, "b": 2}, {"a": 1, "b": 2}],
        },
    ) -> pd.DataFrame:
        temperature = config.get("temperature", 1.0)
        top_k = config.get("top_k", 50)
        nested_list = config.get("nested_list", [[1, 2], [3, 4]])
        nested_dict = config.get("nested_dict", [{"a": 1, "b": 2}, {"a": 1, "b": 2}])
        return pd.DataFrame(
            {
                "output": input_df["feature"] * temperature,
                "top_k_used": [top_k] * len(input_df),
                "nested_list_used": [nested_list] * len(input_df),
                "nested_dict_used": [nested_dict] * len(input_df),
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

    def _assert_unknown_param_raises_error(self, mv: "registry.ModelVersion") -> None:
        """Verify that mv.run() with an unknown param name raises an error."""
        input_df = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

        with self.assertRaises(ValueError) as context:
            mv.run(input_df, function_name="predict", params={"unknown_param": 42})

        self.assertIn("Unknown parameter(s)", str(context.exception))

    def _assert_wrong_type_raises_error(self, mv: "registry.ModelVersion") -> None:
        """Verify that mv.run() with a wrong param type raises an error."""
        input_df = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

        with self.assertRaises(ValueError) as context:
            mv.run(input_df, function_name="predict", params={"temperature": "not_a_float"})

        self.assertIn("not compatible with dtype", str(context.exception))

    def _assert_case_insensitive_param_succeeds(self, mv: "registry.ModelVersion") -> None:
        """Verify that mv.run() with a case-insensitive param name succeeds."""
        input_df = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

        result = mv.run(input_df, function_name="predict", params={"TEMPERATURE": 2.0})

        self.assertEqual(len(result), 3)
        pd.testing.assert_series_equal(
            result["output"].reset_index(drop=True),
            pd.Series([2.0, 4.0, 6.0], name="output"),
            check_dtype=False,
        )
        self.assertTrue(all(result["temperature_used"] == 2.0))

    def _assert_duplicate_case_params_raises_error(self, mv: "registry.ModelVersion") -> None:
        """Verify that mv.run() with duplicate param names differing only in case raises an error."""
        input_df = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

        with self.assertRaises(ValueError) as context:
            mv.run(
                input_df,
                function_name="predict",
                params={"Temperature": 1.0, "temperature": 2.0},
            )

        self.assertIn("Duplicate parameter(s)", str(context.exception))

    def test_warehouse_inference_with_params(self) -> None:
        """Test warehouse inference with DemoModelWithParams across all param scenarios.

        Logs a single model and validates:
        - Constant params work correctly
        - Varying params across rows raise an appropriate error
        - NULL/missing params fall back to defaults
        - Explicit SQL NULL falls back to defaults
        - Unknown param name raises error
        - Wrong param type raises error
        - Case-insensitive param name succeeds
        - Duplicate case param names raise error
        """
        mv = self._log_model_with_params()

        self._assert_constant_params(mv)
        self._assert_varying_params_raises_error(mv)
        self._assert_null_params_uses_default(mv)
        self._assert_explicit_sql_null_uses_default(mv)
        self._assert_unknown_param_raises_error(mv)
        self._assert_wrong_type_raises_error(mv)
        self._assert_case_insensitive_param_succeeds(mv)
        self._assert_duplicate_case_params_raises_error(mv)

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


class TestCustomModelWithDictParamsWarehouseInteg(common_test_base.CommonTestBase):
    """Integration tests for warehouse inference with dict (ParamGroupSpec) parameters."""

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

    def _log_model_with_dict_params(self) -> "registry.ModelVersion":
        model = DemoModelWithDictParam(custom_model.ModelContext())

        sample_input = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
        sample_output = model.predict(
            sample_input,
            config={
                "temperature": 1.0,
                "top_k": 50,
                "nested_list": [[1, 2], [3, 4]],
                "nested_dict": [{"a": 1, "b": 2}, {"a": 1, "b": 2}],
            },
        )

        params = [
            model_signature.ParamGroupSpec(
                name="config",
                specs=[
                    model_signature.ParamSpec(
                        name="temperature",
                        dtype=model_signature.DataType.FLOAT,
                        default_value=1.0,
                    ),
                    model_signature.ParamSpec(
                        name="top_k",
                        dtype=model_signature.DataType.INT32,
                        default_value=50,
                    ),
                    model_signature.ParamSpec(
                        name="nested_list",
                        dtype=model_signature.DataType.INT64,
                        default_value=[[1, 2], [3, 4]],
                        shape=(2, 2),
                    ),
                    model_signature.ParamGroupSpec(
                        name="nested_dict",
                        specs=[
                            model_signature.ParamSpec(
                                name="a",
                                dtype=model_signature.DataType.INT64,
                                default_value=1,
                            ),
                            model_signature.ParamSpec(
                                name="b",
                                dtype=model_signature.DataType.INT64,
                                default_value=2,
                            ),
                        ],
                        shape=(2,),
                    ),
                ],
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
            model_name=f"model_dict_params_test_{self._run_id}",
            version_name=f"v_{self._run_id}",
            conda_dependencies=conda_dependencies,
            signatures={"predict": sig},
            options={"embed_local_ml_library": True},
        )
        return mv

    def test_dict_params_with_override(self) -> None:
        """Test that dict params can be overridden at runtime via mv.run()."""
        mv = self._log_model_with_dict_params()

        input_df = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
        result = mv.run(
            input_df,
            function_name="predict",
            params={
                "config": {
                    "temperature": 2.0,
                    "top_k": 10,
                    "nested_list": [[10, 20], [30, 40]],
                    "nested_dict": [{"a": 10, "b": 20}, {"a": 30, "b": 40}],
                },
            },
        )

        self.assertEqual(len(result), 3)
        result_sorted = result.sort_values("output").reset_index(drop=True)
        pd.testing.assert_series_equal(
            result_sorted["output"],
            pd.Series([2.0, 4.0, 6.0], name="output"),
            check_dtype=False,
        )
        self.assertTrue(all(result_sorted["top_k_used"] == 10))
        for _, row in result_sorted.iterrows():
            self.assertEqual(row["nested_list_used"], [[10, 20], [30, 40]])
            self.assertEqual(row["nested_dict_used"], [{"a": 10, "b": 20}, {"a": 30, "b": 40}])

    def test_dict_params_with_defaults(self) -> None:
        """Test that dict params use defaults when not provided."""
        mv = self._log_model_with_dict_params()

        input_df = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
        result = mv.run(input_df, function_name="predict")

        self.assertEqual(len(result), 3)
        result_sorted = result.sort_values("output").reset_index(drop=True)
        pd.testing.assert_series_equal(
            result_sorted["output"],
            pd.Series([1.0, 2.0, 3.0], name="output"),
            check_dtype=False,
        )
        self.assertTrue(all(result_sorted["top_k_used"] == 50))
        self.assertEqual(result_sorted["nested_list_used"].tolist(), [[[1, 2], [3, 4]]] * 3)
        self.assertEqual(result_sorted["nested_dict_used"].tolist(), [[{"a": 1, "b": 2}, {"a": 1, "b": 2}]] * 3)

    def test_dict_params_partial_override(self) -> None:
        """Test that partial dict override deep-merges with defaults."""
        mv = self._log_model_with_dict_params()

        input_df = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
        result = mv.run(input_df, function_name="predict", params={"config": {"temperature": 3.0}})

        self.assertEqual(len(result), 3)
        result_sorted = result.sort_values("output").reset_index(drop=True)
        pd.testing.assert_series_equal(
            result_sorted["output"],
            pd.Series([3.0, 6.0, 9.0], name="output"),
            check_dtype=False,
        )
        self.assertTrue(all(result_sorted["top_k_used"] == 50))
        self.assertEqual(result_sorted["nested_list_used"].tolist(), [[[1, 2], [3, 4]]] * 3)
        self.assertEqual(result_sorted["nested_dict_used"].tolist(), [[{"a": 1, "b": 2}, {"a": 1, "b": 2}]] * 3)


_DEFAULT_TIMESTAMP = datetime.datetime(2024, 1, 1, 12, 0, 0)
_DEFAULT_WEIGHTS = [1.0, 2.0, 3.0]
_DEFAULT_NESTED_LIST = [[1, 2], [3, 4]]
_DEFAULT_CONFIG = {"mode": "standard", "threshold": 0.5}


class DemoModelWithAllDataTypes(custom_model.CustomModel):
    """Custom model that accepts scalar, shaped, and dict params."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(
        self,
        input_df: pd.DataFrame,
        *,
        int_param: int = 42,
        bool_param: bool = True,
        string_param: str = "default",
        bytes_param: bytes = b"default",
        timestamp_param: datetime.datetime = _DEFAULT_TIMESTAMP,
        weights_param: list[float] = _DEFAULT_WEIGHTS,  # noqa: B006
        nested_list: list[list[int]] = _DEFAULT_NESTED_LIST,  # noqa: B006
        config: dict = _DEFAULT_CONFIG,  # noqa: B006
    ) -> pd.DataFrame:
        n = len(input_df)
        bytes_str = bytes_param.hex() if isinstance(bytes_param, bytes) else str(bytes_param)
        ts_str = str(timestamp_param)
        mode = config.get("mode", "standard")
        threshold = config.get("threshold", 0.5)
        return pd.DataFrame(
            {
                "input_value": input_df["value"].tolist(),
                "received_int": [int_param] * n,
                "received_bool": [bool_param] * n,
                "received_string": [string_param] * n,
                "received_bytes": [bytes_str] * n,
                "received_timestamp": [ts_str] * n,
                "received_weights": [weights_param] * n,
                "received_nested_list": [nested_list] * n,
                "received_mode": [mode] * n,
                "received_threshold": [threshold] * n,
            }
        )


class TestCustomModelWithAllDataTypesWarehouseInteg(common_test_base.CommonTestBase):
    """Integration tests for warehouse inference with multiple param data types.

    Tests that INT, BOOL, STRING, BYTES, TIMESTAMP, shaped float list,
    shaped 2D int list, and dict (ParamGroupSpec) params are correctly
    handled through the warehouse UDF path.
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

    def _log_model_with_all_data_types(self) -> "registry.ModelVersion":
        """Log a model with scalar, shaped, and dict data type params."""
        model = DemoModelWithAllDataTypes(custom_model.ModelContext())

        sig = model_signature.ModelSignature(
            inputs=[model_signature.FeatureSpec(name="value", dtype=model_signature.DataType.FLOAT)],
            outputs=[
                model_signature.FeatureSpec(name="input_value", dtype=model_signature.DataType.FLOAT),
                model_signature.FeatureSpec(name="received_int", dtype=model_signature.DataType.INT64),
                model_signature.FeatureSpec(name="received_bool", dtype=model_signature.DataType.BOOL),
                model_signature.FeatureSpec(name="received_string", dtype=model_signature.DataType.STRING),
                model_signature.FeatureSpec(name="received_bytes", dtype=model_signature.DataType.STRING),
                model_signature.FeatureSpec(name="received_timestamp", dtype=model_signature.DataType.STRING),
                model_signature.FeatureSpec(
                    name="received_weights",
                    dtype=model_signature.DataType.DOUBLE,
                    shape=(3,),
                ),
                model_signature.FeatureSpec(
                    name="received_nested_list",
                    dtype=model_signature.DataType.INT64,
                    shape=(2, 2),
                ),
                model_signature.FeatureSpec(name="received_mode", dtype=model_signature.DataType.STRING),
                model_signature.FeatureSpec(
                    name="received_threshold",
                    dtype=model_signature.DataType.DOUBLE,
                ),
            ],
            params=[
                model_signature.ParamSpec(
                    name="int_param",
                    dtype=model_signature.DataType.INT64,
                    default_value=42,
                ),
                model_signature.ParamSpec(
                    name="bool_param",
                    dtype=model_signature.DataType.BOOL,
                    default_value=True,
                ),
                model_signature.ParamSpec(
                    name="string_param",
                    dtype=model_signature.DataType.STRING,
                    default_value="default",
                ),
                model_signature.ParamSpec(
                    name="bytes_param",
                    dtype=model_signature.DataType.BYTES,
                    default_value=b"default",
                ),
                model_signature.ParamSpec(
                    name="timestamp_param",
                    dtype=model_signature.DataType.TIMESTAMP_NTZ,
                    default_value=_DEFAULT_TIMESTAMP,
                ),
                model_signature.ParamSpec(
                    name="weights_param",
                    dtype=model_signature.DataType.DOUBLE,
                    default_value=list(_DEFAULT_WEIGHTS),
                    shape=(3,),
                ),
                model_signature.ParamSpec(
                    name="nested_list",
                    dtype=model_signature.DataType.INT64,
                    default_value=_DEFAULT_NESTED_LIST,
                    shape=(2, 2),
                ),
                model_signature.ParamGroupSpec(
                    name="config",
                    specs=[
                        model_signature.ParamSpec(
                            name="mode",
                            dtype=model_signature.DataType.STRING,
                            default_value="standard",
                        ),
                        model_signature.ParamSpec(
                            name="threshold",
                            dtype=model_signature.DataType.DOUBLE,
                            default_value=0.5,
                        ),
                    ],
                ),
            ],
        )

        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python!=1.12.0")
        ]

        mv = self.registry.log_model(
            model=model,
            model_name=f"model_all_types_test_{self._run_id}",
            version_name=f"v_{self._run_id}",
            conda_dependencies=conda_dependencies,
            signatures={"predict": sig},
            options={"embed_local_ml_library": True},
        )
        return mv

    def _assert_params_match(self, result: pd.DataFrame, expected: dict[str, Any], label: str = "") -> None:
        """Verify that received param values match expected values."""
        tag = f"[{label}] " if label else ""
        row = result.iloc[0]
        for key, expected_value in expected.items():
            actual = row[key]
            if isinstance(expected_value, float):
                self.assertAlmostEqual(float(actual), expected_value, places=5, msg=f"{tag}{key}")
            elif isinstance(expected_value, list):
                actual_list = actual.tolist() if hasattr(actual, "tolist") else actual
                self.assertEqual(actual_list, expected_value, f"{tag}{key}")
            else:
                self.assertEqual(actual, expected_value, f"{tag}{key}")

    def test_warehouse_inference_all_data_types(self) -> None:
        """Test warehouse inference with all param data types (scalar + shaped + dict).

        Logs a single model and validates:
        - All params provided — verify each received value
        - No params — verify defaults for every type
        - Partial params — verify mix of overrides + defaults
        - Dict param partial override — verify deep-merge with defaults
        - Direct SQL with typed params (INT, BOOL, STRING)
        """
        mv = self._log_model_with_all_data_types()
        input_df = pd.DataFrame({"value": [10.0]})

        # --- All params provided ---
        custom_ts = datetime.datetime(2025, 6, 15, 8, 30, 0)
        result = mv.run(
            input_df,
            function_name="predict",
            params={
                "int_param": 99,
                "bool_param": False,
                "string_param": "custom",
                "bytes_param": b"hello",
                "timestamp_param": custom_ts,
                "weights_param": [4.5, 3.5, 2.5],
                "nested_list": [[4, 3], [2, 1]],
                "config": {"mode": "turbo", "threshold": 0.9},
            },
        )
        self.assertEqual(len(result), 1)
        self._assert_params_match(
            result,
            {
                "input_value": 10.0,
                "received_int": 99,
                "received_bool": False,
                "received_string": "custom",
                "received_bytes": b"hello".hex(),
                "received_timestamp": str(custom_ts),
                "received_weights": [4.5, 3.5, 2.5],
                "received_nested_list": [[4, 3], [2, 1]],
                "received_mode": "turbo",
                "received_threshold": 0.9,
            },
            label="full_params",
        )

        # --- No params (defaults) ---
        result = mv.run(input_df, function_name="predict")
        self.assertEqual(len(result), 1)
        self._assert_params_match(
            result,
            {
                "input_value": 10.0,
                "received_int": 42,
                "received_bool": True,
                "received_string": "default",
                "received_bytes": b"default".hex(),
                "received_timestamp": str(_DEFAULT_TIMESTAMP),
                "received_weights": _DEFAULT_WEIGHTS,
                "received_nested_list": _DEFAULT_NESTED_LIST,
                "received_mode": "standard",
                "received_threshold": 0.5,
            },
            label="default_params",
        )

        # --- Partial params (only int and string overridden) ---
        result = mv.run(
            input_df,
            function_name="predict",
            params={"int_param": 7, "string_param": "partial"},
        )
        self.assertEqual(len(result), 1)
        self._assert_params_match(
            result,
            {
                "received_int": 7,
                "received_bool": True,
                "received_string": "partial",
                "received_bytes": b"default".hex(),
                "received_timestamp": str(_DEFAULT_TIMESTAMP),
                "received_weights": _DEFAULT_WEIGHTS,
                "received_nested_list": _DEFAULT_NESTED_LIST,
                "received_mode": "standard",
                "received_threshold": 0.5,
            },
            label="partial_params",
        )

        # --- Dict param partial override (only threshold overridden) ---
        result = mv.run(
            input_df,
            function_name="predict",
            params={"config": {"threshold": 0.99}},
        )
        self.assertEqual(len(result), 1)
        self._assert_params_match(
            result,
            {
                "received_int": 42,
                "received_mode": "standard",
                "received_threshold": 0.99,
            },
            label="dict_partial_override",
        )

        # --- Direct SQL with typed params (INT, BOOL, STRING; NULL for bytes/timestamp) ---
        model_name = mv.model_name
        version_name = mv.version_name

        sql = f"""
            WITH MODEL_VERSION_ALIAS AS MODEL {model_name} VERSION {version_name},
            test_data AS (
                SELECT 10.0::FLOAT AS value
            )
            SELECT
                MODEL_VERSION_ALIAS!PREDICT(
                    value, 99, false, 'custom', NULL, NULL, NULL, NULL, NULL
                ):received_int::INT AS received_int,
                MODEL_VERSION_ALIAS!PREDICT(
                    value, 99, false, 'custom', NULL, NULL, NULL, NULL, NULL
                ):received_bool::BOOLEAN AS received_bool,
                MODEL_VERSION_ALIAS!PREDICT(
                    value, 99, false, 'custom', NULL, NULL, NULL, NULL, NULL
                ):received_string::STRING AS received_string
            FROM test_data
        """

        rows = self.session.sql(sql).collect()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["RECEIVED_INT"], 99)
        self.assertEqual(row["RECEIVED_BOOL"], False)
        self.assertEqual(row["RECEIVED_STRING"], "custom")


if __name__ == "__main__":
    absltest.main()
