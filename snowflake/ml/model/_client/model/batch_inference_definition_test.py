from typing import Any, Optional
from unittest import mock

import yaml
from absl.testing import absltest

from snowflake.ml.model._client.model.batch_inference_specs import (
    InputSpec,
    JobSpec,
    OutputSpec,
)


class BatchInferenceDefinitionTest(absltest.TestCase):
    def _create_mock_model_version(
        self, target_method: str = "predict", session_warehouse: Optional[str] = "SESSION_WH"
    ) -> mock.MagicMock:
        mv = mock.MagicMock()
        mv.fully_qualified_model_name = "MY_DB.MY_SCHEMA.MY_MODEL"
        mv.version_name = "V1"
        mv._get_function_info.return_value = {"target_method": target_method}
        mv._service_ops._session.get_current_warehouse.return_value = session_warehouse
        return mv

    def _create_mock_dataframe(self) -> mock.MagicMock:
        df = mock.MagicMock()
        df.queries = {
            "queries": ["CREATE TEMP TABLE t1 AS SELECT 1", "SELECT * FROM t1"],
            "post_actions": ["DROP TABLE IF EXISTS t1"],
        }
        return df

    def _extract_spec_from_sql(self, sql: str) -> dict[str, Any]:
        """Extract and parse the YAML spec from a CALL SYSTEM$DEPLOY_MODEL($$...$$) SQL string."""
        prefix = "CALL SYSTEM$DEPLOY_MODEL($$"
        suffix = "$$)"
        self.assertTrue(sql.startswith(prefix), f"SQL does not start with {prefix}: {sql[:50]}")
        self.assertTrue(sql.endswith(suffix), f"SQL does not end with {suffix}: {sql[-10:]}")
        yaml_str = sql[len(prefix) : -len(suffix)]
        result: dict[str, Any] = yaml.safe_load(yaml_str)
        return result

    def test_constructor_stores_parameters(self) -> None:
        mv = self._create_mock_model_version()
        df = self._create_mock_dataframe()
        output_spec = OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/")
        input_spec = InputSpec(params={"temperature": 0.7})
        job_spec = JobSpec(job_name_prefix="my_prefix", gpu_requests="1", warehouse="MY_WH")

        from snowflake.ml.model._client.model.batch_inference_definition import (
            BatchInferenceDefinition,
        )

        defn = BatchInferenceDefinition(
            model_version=mv,
            X=df,
            compute_pool="GPU_POOL",
            output_spec=output_spec,
            input_spec=input_spec,
            job_spec=job_spec,
        )

        self.assertEqual(defn._fully_qualified_model_name, "MY_DB.MY_SCHEMA.MY_MODEL")
        self.assertEqual(defn._version_name, "V1")
        self.assertEqual(defn._compute_pool, "GPU_POOL")
        self.assertEqual(defn._queries, ["CREATE TEMP TABLE t1 AS SELECT 1", "SELECT * FROM t1"])
        self.assertEqual(defn._post_actions, ["DROP TABLE IF EXISTS t1"])
        self.assertIs(defn._output_spec, output_spec)
        self.assertIs(defn._input_spec, input_spec)
        self.assertIs(defn._job_spec, job_spec)

    def test_constructor_defaults(self) -> None:
        mv = self._create_mock_model_version()
        df = self._create_mock_dataframe()
        output_spec = OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/")

        from snowflake.ml.model._client.model.batch_inference_definition import (
            BatchInferenceDefinition,
        )

        defn = BatchInferenceDefinition(
            model_version=mv,
            X=df,
            compute_pool="GPU_POOL",
            output_spec=output_spec,
        )

        self.assertIsNotNone(defn._input_spec)
        self.assertIsNotNone(defn._job_spec)
        self.assertIsNone(defn._inference_engine_options)

    def test_to_sql_basic_structure(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_definition import (
            BatchInferenceDefinition,
        )

        mv = self._create_mock_model_version()
        df = self._create_mock_dataframe()
        output_spec = OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/")
        job_spec = JobSpec(job_name="MY_JOB", gpu_requests="1", warehouse="MY_WH")

        defn = BatchInferenceDefinition(
            model_version=mv,
            X=df,
            compute_pool="GPU_POOL",
            output_spec=output_spec,
            job_spec=job_spec,
        )

        sql = defn.to_sql()

        # Verify it's a CALL statement and extract spec
        spec = self._extract_spec_from_sql(sql)

        # Verify model section
        self.assertEqual(len(spec["models"]), 1)
        self.assertEqual(spec["models"][0]["name"], "MY_DB.MY_SCHEMA.MY_MODEL")
        self.assertEqual(spec["models"][0]["version"], "V1")

        # Verify job section
        job = spec["job"]
        self.assertEqual(job["compute_pool"], "GPU_POOL")
        self.assertEqual(job["function_name"], "predict")
        self.assertTrue(job["sync"])
        self.assertEqual(job["gpu"], "1")
        self.assertEqual(job["warehouse"], "MY_WH")

        # Verify job name is fully qualified
        self.assertEqual(job["name"], "MY_DB.MY_SCHEMA.MY_JOB")

        # Verify input has queries
        self.assertEqual(
            job["input"]["queries"],
            [
                "CREATE TEMP TABLE t1 AS SELECT 1",
                "SELECT * FROM t1",
            ],
        )
        self.assertEqual(job["input"]["post_actions"], ["DROP TABLE IF EXISTS t1"])

        # Verify output
        self.assertEqual(job["output"]["output_stage_location"], "@MY_DB.MY_SCHEMA.MY_STAGE/output/")
        self.assertEqual(job["output"]["completion_filename"], "_SUCCESS")

    def test_to_sql_with_name_prefix(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_definition import (
            BatchInferenceDefinition,
        )

        mv = self._create_mock_model_version()
        df = self._create_mock_dataframe()
        output_spec = OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/")
        job_spec = JobSpec(job_name_prefix="MY_PREFIX", warehouse="MY_WH")

        defn = BatchInferenceDefinition(
            model_version=mv,
            X=df,
            compute_pool="GPU_POOL",
            output_spec=output_spec,
            job_spec=job_spec,
        )

        sql = defn.to_sql()
        spec = self._extract_spec_from_sql(sql)

        # name should be absent, name_prefix should be fully qualified with trailing _
        self.assertNotIn("name", spec["job"])
        self.assertEqual(spec["job"]["name_prefix"], "MY_DB.MY_SCHEMA.MY_PREFIX_")

    def test_missing_warehouse_and_no_session_warehouse_raises(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_definition import (
            BatchInferenceDefinition,
        )

        mv = self._create_mock_model_version(session_warehouse=None)
        df = self._create_mock_dataframe()
        output_spec = OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/")
        job_spec = JobSpec()  # no warehouse

        with self.assertRaises(ValueError):
            BatchInferenceDefinition(
                model_version=mv,
                X=df,
                compute_pool="GPU_POOL",
                output_spec=output_spec,
                job_spec=job_spec,
            )

    def test_warehouse_falls_back_to_session_warehouse(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_definition import (
            BatchInferenceDefinition,
        )

        mv = self._create_mock_model_version(session_warehouse="SESSION_WH")
        df = self._create_mock_dataframe()
        output_spec = OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/")
        job_spec = JobSpec()  # no warehouse — should fall back to session

        defn = BatchInferenceDefinition(
            model_version=mv,
            X=df,
            compute_pool="GPU_POOL",
            output_spec=output_spec,
            job_spec=job_spec,
        )

        spec = self._extract_spec_from_sql(defn.to_sql())
        self.assertEqual(spec["job"]["warehouse"], "SESSION_WH")

    def test_to_sql_with_inference_engine(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_definition import (
            BatchInferenceDefinition,
        )
        from snowflake.ml.model.inference_engine import InferenceEngine

        mv = self._create_mock_model_version()
        df = self._create_mock_dataframe()
        output_spec = OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/")
        job_spec = JobSpec(warehouse="MY_WH")

        defn = BatchInferenceDefinition(
            model_version=mv,
            X=df,
            compute_pool="GPU_POOL",
            output_spec=output_spec,
            job_spec=job_spec,
            inference_engine_options={
                "engine": InferenceEngine.VLLM,
                "engine_args_override": ["--max-model-len=2048"],
            },
        )

        sql = defn.to_sql()
        spec = self._extract_spec_from_sql(sql)

        # Should have inference_engine_spec, NOT image_build
        self.assertNotIn("image_build", spec)
        engine_spec = spec["job"]["inference_engine_spec"]
        self.assertEqual(engine_spec["inference_engine_name"], "vllm")
        self.assertEqual(engine_spec["inference_engine_args"], ["--max-model-len=2048"])

    def test_to_sql_without_inference_engine_has_image_build(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_definition import (
            BatchInferenceDefinition,
        )

        mv = self._create_mock_model_version()
        df = self._create_mock_dataframe()
        output_spec = OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/")
        job_spec = JobSpec(warehouse="MY_WH", image_repo="my_repo", force_rebuild=True)

        defn = BatchInferenceDefinition(
            model_version=mv,
            X=df,
            compute_pool="GPU_POOL",
            output_spec=output_spec,
            job_spec=job_spec,
        )

        sql = defn.to_sql()
        spec = self._extract_spec_from_sql(sql)

        self.assertIn("image_build", spec)
        self.assertEqual(spec["image_build"]["compute_pool"], "GPU_POOL")
        self.assertEqual(spec["image_build"]["image_repo"], "my_repo")
        self.assertTrue(spec["image_build"]["force_rebuild"])

    def test_to_sql_output_stage_trailing_slash(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_definition import (
            BatchInferenceDefinition,
        )

        mv = self._create_mock_model_version()
        df = self._create_mock_dataframe()
        # No trailing slash
        output_spec = OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output")
        job_spec = JobSpec(warehouse="MY_WH")

        defn = BatchInferenceDefinition(
            model_version=mv,
            X=df,
            compute_pool="GPU_POOL",
            output_spec=output_spec,
            job_spec=job_spec,
        )

        sql = defn.to_sql()
        spec = self._extract_spec_from_sql(sql)

        # Should have trailing slash added
        self.assertEqual(spec["job"]["output"]["output_stage_location"], "@MY_DB.MY_SCHEMA.MY_STAGE/output/")

    def test_to_sql_no_job_name_or_prefix(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_definition import (
            BatchInferenceDefinition,
        )

        mv = self._create_mock_model_version()
        df = self._create_mock_dataframe()
        output_spec = OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/")
        job_spec = JobSpec(warehouse="MY_WH")  # no job_name, no job_name_prefix

        defn = BatchInferenceDefinition(
            model_version=mv,
            X=df,
            compute_pool="GPU_POOL",
            output_spec=output_spec,
            job_spec=job_spec,
        )

        spec = self._extract_spec_from_sql(defn.to_sql())

        self.assertNotIn("name", spec["job"])
        self.assertNotIn("name_prefix", spec["job"])

    def test_to_sql_with_full_input_spec(self) -> None:
        import base64
        import json

        from snowflake.ml.model._client.model.batch_inference_definition import (
            BatchInferenceDefinition,
        )
        from snowflake.ml.model._client.model.batch_inference_specs import (
            FileEncoding,
            InputFormat,
        )

        mv = self._create_mock_model_version()
        df = self._create_mock_dataframe()
        output_spec = OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/")
        input_spec = InputSpec(
            params={"temperature": 0.7, "top_k": 50},
            column_handling={
                "image_col": {
                    "input_format": InputFormat.FULL_STAGE_PATH,
                    "convert_to": FileEncoding.BASE64,
                }
            },
            partition_column="batch_id",
        )
        job_spec = JobSpec(warehouse="MY_WH")

        defn = BatchInferenceDefinition(
            model_version=mv,
            X=df,
            compute_pool="GPU_POOL",
            output_spec=output_spec,
            input_spec=input_spec,
            job_spec=job_spec,
        )

        spec = self._extract_spec_from_sql(defn.to_sql())
        input_section = spec["job"]["input"]

        # params should be base64-encoded
        params = json.loads(base64.b64decode(input_section["params"]))
        self.assertEqual(params["temperature"], 0.7)
        self.assertEqual(params["top_k"], 50)

        # column_handling should be base64-encoded
        self.assertIn("column_handling", input_section)
        decoded = json.loads(base64.b64decode(input_section["column_handling"]))
        self.assertEqual(decoded["image_col"]["input_format"], "full_stage_path")
        self.assertEqual(decoded["image_col"]["convert_to"], "base64")

        # partition_column should become a list
        self.assertEqual(input_section["partition_columns"], ["batch_id"])

    def test_to_sql_with_full_job_spec(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_definition import (
            BatchInferenceDefinition,
        )

        mv = self._create_mock_model_version(target_method="CUSTOM_FN")
        df = self._create_mock_dataframe()
        output_spec = OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/")
        job_spec = JobSpec(
            warehouse="MY_WH",
            function_name="custom_fn",
            cpu_requests="4",
            memory_requests="16Gi",
            gpu_requests="2",
            num_workers=8,
            max_batch_rows=1024,
            replicas=3,
        )

        defn = BatchInferenceDefinition(
            model_version=mv,
            X=df,
            compute_pool="GPU_POOL",
            output_spec=output_spec,
            job_spec=job_spec,
        )

        # Verify _get_function_info was called with the custom function name
        mv._get_function_info.assert_called_once_with(function_name="custom_fn")

        spec = self._extract_spec_from_sql(defn.to_sql())
        job = spec["job"]

        self.assertEqual(job["function_name"], "CUSTOM_FN")
        self.assertEqual(job["cpu"], "4")
        self.assertEqual(job["memory"], "16Gi")
        self.assertEqual(job["gpu"], "2")
        self.assertEqual(job["num_workers"], 8)
        self.assertEqual(job["max_batch_rows"], 1024)
        self.assertEqual(job["replicas"], 3)


if __name__ == "__main__":
    absltest.main()
