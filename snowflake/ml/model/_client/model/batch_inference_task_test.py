from datetime import timedelta
from typing import Any, Optional
from unittest import mock

import yaml
from absl.testing import absltest

try:
    from snowflake.core.task.dagv1 import DAG, DAGTask

    _HAS_SNOWFLAKE_CORE = True
except ModuleNotFoundError:
    _HAS_SNOWFLAKE_CORE = False

from snowflake.ml.model._client.model.batch_inference_specs import (
    InputSpec,
    JobSpec,
    OutputSpec,
)


@absltest.skipUnless(_HAS_SNOWFLAKE_CORE, "snowflake.core not installed")
class BatchInferenceTaskTest(absltest.TestCase):
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

    def _extract_spec_from_sql(self, sql: Any) -> dict[str, Any]:
        prefix = "CALL SYSTEM$DEPLOY_MODEL($$"
        suffix = "$$)"
        self.assertIsInstance(sql, str)
        self.assertTrue(sql.startswith(prefix), f"SQL does not start with {prefix}: {sql[:50]}")
        self.assertTrue(sql.endswith(suffix), f"SQL does not end with {suffix}: {sql[-10:]}")
        yaml_str = sql[len(prefix) : -len(suffix)]
        result: dict[str, Any] = yaml.safe_load(yaml_str)
        return result

    def _make_dag(self) -> "DAG":
        return DAG(
            "test_dag",
            schedule=timedelta(days=1),
            warehouse="DAG_WH",
            stage_location="@MY_DB.MY_SCHEMA.MY_STAGE",
        )

    # ------ Ported from BatchInferenceDefinitionTest ------

    def test_constructor_stores_parameters(self) -> None:
        mv = self._create_mock_model_version()
        df = self._create_mock_dataframe()
        output_spec = OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/")
        input_spec = InputSpec(params={"temperature": 0.7})
        job_spec = JobSpec(job_name_prefix="my_prefix", gpu_requests="1", warehouse="MY_WH")

        from snowflake.ml.model._client.model.batch_inference_task import (
            BatchInferenceTask,
        )

        with self._make_dag():
            task = BatchInferenceTask(
                "batch_inference",
                model_version=mv,
                X=df,
                compute_pool="GPU_POOL",
                output_spec=output_spec,
                input_spec=input_spec,
                job_spec=job_spec,
            )

        self.assertEqual(task._fully_qualified_model_name, "MY_DB.MY_SCHEMA.MY_MODEL")
        self.assertEqual(task._version_name, "V1")
        self.assertEqual(task._compute_pool, "GPU_POOL")
        self.assertEqual(task._queries, ["CREATE TEMP TABLE t1 AS SELECT 1", "SELECT * FROM t1"])
        self.assertEqual(task._post_actions, ["DROP TABLE IF EXISTS t1"])
        self.assertIs(task._output_spec, output_spec)
        self.assertIs(task._input_spec, input_spec)
        self.assertIs(task._job_spec, job_spec)

    def test_constructor_defaults(self) -> None:
        mv = self._create_mock_model_version()
        df = self._create_mock_dataframe()
        output_spec = OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/")

        from snowflake.ml.model._client.model.batch_inference_task import (
            BatchInferenceTask,
        )

        with self._make_dag():
            task = BatchInferenceTask(
                "batch_inference",
                model_version=mv,
                X=df,
                compute_pool="GPU_POOL",
                output_spec=output_spec,
            )

        self.assertIsNotNone(task._input_spec)
        self.assertIsNotNone(task._job_spec)
        self.assertIsNone(task._inference_engine_options)

    def test_to_sql_basic_structure(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_task import (
            BatchInferenceTask,
        )

        mv = self._create_mock_model_version()
        df = self._create_mock_dataframe()
        output_spec = OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/")
        job_spec = JobSpec(job_name="MY_JOB", gpu_requests="1", warehouse="MY_WH")

        with self._make_dag():
            task = BatchInferenceTask(
                "batch_inference",
                model_version=mv,
                X=df,
                compute_pool="GPU_POOL",
                output_spec=output_spec,
                job_spec=job_spec,
            )

        spec = self._extract_spec_from_sql(task.definition)

        self.assertEqual(len(spec["models"]), 1)
        self.assertEqual(spec["models"][0]["name"], "MY_DB.MY_SCHEMA.MY_MODEL")
        self.assertEqual(spec["models"][0]["version"], "V1")

        job = spec["job"]
        self.assertEqual(job["compute_pool"], "GPU_POOL")
        self.assertEqual(job["function_name"], "predict")
        self.assertTrue(job["sync"])
        self.assertEqual(job["gpu"], "1")
        self.assertEqual(job["warehouse"], "MY_WH")

        self.assertEqual(job["name"], "MY_DB.MY_SCHEMA.MY_JOB")

        self.assertEqual(
            job["input"]["queries"],
            [
                "CREATE TEMP TABLE t1 AS SELECT 1",
                "SELECT * FROM t1",
            ],
        )
        self.assertEqual(job["input"]["post_actions"], ["DROP TABLE IF EXISTS t1"])

        self.assertEqual(job["output"]["output_stage_location"], "@MY_DB.MY_SCHEMA.MY_STAGE/output/")
        self.assertEqual(job["output"]["completion_filename"], "_SUCCESS")

    def test_to_sql_with_name_prefix(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_task import (
            BatchInferenceTask,
        )

        mv = self._create_mock_model_version()
        df = self._create_mock_dataframe()
        output_spec = OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/")
        job_spec = JobSpec(job_name_prefix="MY_PREFIX", warehouse="MY_WH")

        with self._make_dag():
            task = BatchInferenceTask(
                "batch_inference",
                model_version=mv,
                X=df,
                compute_pool="GPU_POOL",
                output_spec=output_spec,
                job_spec=job_spec,
            )

        spec = self._extract_spec_from_sql(task.definition)

        self.assertNotIn("name", spec["job"])
        self.assertEqual(spec["job"]["name_prefix"], "MY_DB.MY_SCHEMA.MY_PREFIX_")

    def test_missing_warehouse_and_no_session_warehouse_raises(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_task import (
            BatchInferenceTask,
        )

        mv = self._create_mock_model_version(session_warehouse=None)
        df = self._create_mock_dataframe()
        output_spec = OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/")
        job_spec = JobSpec()  # no warehouse

        with self._make_dag():
            with self.assertRaises(ValueError):
                BatchInferenceTask(
                    "batch_inference",
                    model_version=mv,
                    X=df,
                    compute_pool="GPU_POOL",
                    output_spec=output_spec,
                    job_spec=job_spec,
                )

    def test_warehouse_falls_back_to_session_warehouse(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_task import (
            BatchInferenceTask,
        )

        mv = self._create_mock_model_version(session_warehouse="SESSION_WH")
        df = self._create_mock_dataframe()
        output_spec = OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/")
        job_spec = JobSpec()  # no warehouse — should fall back to session

        with self._make_dag():
            task = BatchInferenceTask(
                "batch_inference",
                model_version=mv,
                X=df,
                compute_pool="GPU_POOL",
                output_spec=output_spec,
                job_spec=job_spec,
            )

        spec = self._extract_spec_from_sql(task.definition)
        self.assertEqual(spec["job"]["warehouse"], "SESSION_WH")

    def test_to_sql_with_inference_engine(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_task import (
            BatchInferenceTask,
        )
        from snowflake.ml.model.inference_engine import InferenceEngine

        mv = self._create_mock_model_version()
        df = self._create_mock_dataframe()
        output_spec = OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/")
        job_spec = JobSpec(warehouse="MY_WH")

        with self._make_dag():
            task = BatchInferenceTask(
                "batch_inference",
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

        spec = self._extract_spec_from_sql(task.definition)

        self.assertNotIn("image_build", spec)
        engine_spec = spec["job"]["inference_engine_spec"]
        self.assertEqual(engine_spec["inference_engine_name"], "vllm")
        self.assertEqual(engine_spec["inference_engine_args"], ["--max-model-len=2048"])

    def test_to_sql_without_inference_engine_has_image_build(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_task import (
            BatchInferenceTask,
        )

        mv = self._create_mock_model_version()
        df = self._create_mock_dataframe()
        output_spec = OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/")
        job_spec = JobSpec(warehouse="MY_WH", image_repo="my_repo", force_rebuild=True)

        with self._make_dag():
            task = BatchInferenceTask(
                "batch_inference",
                model_version=mv,
                X=df,
                compute_pool="GPU_POOL",
                output_spec=output_spec,
                job_spec=job_spec,
            )

        spec = self._extract_spec_from_sql(task.definition)

        self.assertIn("image_build", spec)
        self.assertEqual(spec["image_build"]["compute_pool"], "GPU_POOL")
        self.assertEqual(spec["image_build"]["image_repo"], "my_repo")
        self.assertTrue(spec["image_build"]["force_rebuild"])

    def test_to_sql_output_stage_trailing_slash(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_task import (
            BatchInferenceTask,
        )

        mv = self._create_mock_model_version()
        df = self._create_mock_dataframe()
        # No trailing slash
        output_spec = OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output")
        job_spec = JobSpec(warehouse="MY_WH")

        with self._make_dag():
            task = BatchInferenceTask(
                "batch_inference",
                model_version=mv,
                X=df,
                compute_pool="GPU_POOL",
                output_spec=output_spec,
                job_spec=job_spec,
            )

        spec = self._extract_spec_from_sql(task.definition)

        self.assertEqual(spec["job"]["output"]["output_stage_location"], "@MY_DB.MY_SCHEMA.MY_STAGE/output/")

    def test_to_sql_no_job_name_or_prefix(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_task import (
            BatchInferenceTask,
        )

        mv = self._create_mock_model_version()
        df = self._create_mock_dataframe()
        output_spec = OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/")
        job_spec = JobSpec(warehouse="MY_WH")  # no job_name, no job_name_prefix

        with self._make_dag():
            task = BatchInferenceTask(
                "batch_inference",
                model_version=mv,
                X=df,
                compute_pool="GPU_POOL",
                output_spec=output_spec,
                job_spec=job_spec,
            )

        spec = self._extract_spec_from_sql(task.definition)

        self.assertNotIn("name", spec["job"])
        self.assertNotIn("name_prefix", spec["job"])

    def test_to_sql_with_full_input_spec(self) -> None:
        import base64
        import json

        from snowflake.ml.model._client.model.batch_inference_specs import (
            FileEncoding,
            InputFormat,
        )
        from snowflake.ml.model._client.model.batch_inference_task import (
            BatchInferenceTask,
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

        with self._make_dag():
            task = BatchInferenceTask(
                "batch_inference",
                model_version=mv,
                X=df,
                compute_pool="GPU_POOL",
                output_spec=output_spec,
                input_spec=input_spec,
                job_spec=job_spec,
            )

        spec = self._extract_spec_from_sql(task.definition)
        input_section = spec["job"]["input"]

        params = json.loads(base64.b64decode(input_section["params"]))
        self.assertEqual(params["temperature"], 0.7)
        self.assertEqual(params["top_k"], 50)

        self.assertIn("column_handling", input_section)
        decoded = json.loads(base64.b64decode(input_section["column_handling"]))
        self.assertEqual(decoded["image_col"]["input_format"], "full_stage_path")
        self.assertEqual(decoded["image_col"]["convert_to"], "base64")

        self.assertEqual(input_section["partition_columns"], ["batch_id"])

    def test_to_sql_with_full_job_spec(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_task import (
            BatchInferenceTask,
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

        with self._make_dag():
            task = BatchInferenceTask(
                "batch_inference",
                model_version=mv,
                X=df,
                compute_pool="GPU_POOL",
                output_spec=output_spec,
                job_spec=job_spec,
            )

        mv._get_function_info.assert_called_once_with(function_name="custom_fn")

        spec = self._extract_spec_from_sql(task.definition)
        job = spec["job"]

        self.assertEqual(job["function_name"], "CUSTOM_FN")
        self.assertEqual(job["cpu"], "4")
        self.assertEqual(job["memory"], "16Gi")
        self.assertEqual(job["gpu"], "2")
        self.assertEqual(job["num_workers"], 8)
        self.assertEqual(job["max_batch_rows"], 1024)
        self.assertEqual(job["replicas"], 3)

    def test_to_sql_with_base_stage_location(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_task import (
            BatchInferenceTask,
        )

        mv = self._create_mock_model_version()
        df = self._create_mock_dataframe()
        output_spec = OutputSpec(base_stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/base")
        job_spec = JobSpec(job_name_prefix="MY_PREFIX", warehouse="MY_WH")

        with self._make_dag():
            task = BatchInferenceTask(
                "batch_inference",
                model_version=mv,
                X=df,
                compute_pool="GPU_POOL",
                output_spec=output_spec,
                job_spec=job_spec,
            )

        spec = self._extract_spec_from_sql(task.definition)

        output = spec["job"]["output"]
        self.assertNotIn("output_stage_location", output)
        self.assertEqual(output["base_stage_location"], "@MY_DB.MY_SCHEMA.MY_STAGE/base/")
        self.assertEqual(output["completion_filename"], "_SUCCESS")

        input_section = spec["job"]["input"]
        self.assertNotIn("input_stage_location", input_section)
        self.assertEqual(input_section["queries"], ["CREATE TEMP TABLE t1 AS SELECT 1", "SELECT * FROM t1"])
        self.assertEqual(input_section["post_actions"], ["DROP TABLE IF EXISTS t1"])

    # ------ DAGTask-specific tests ------

    def test_constructor_returns_dagtask_with_sql_definition(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_task import (
            BatchInferenceTask,
        )

        with self._make_dag():
            task = BatchInferenceTask(
                "batch_inference",
                model_version=self._create_mock_model_version(),
                X=self._create_mock_dataframe(),
                compute_pool="CPU_POOL",
                output_spec=OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/o/"),
                job_spec=JobSpec(warehouse="MY_WH"),
            )

        self.assertIsInstance(task, DAGTask)
        self.assertEqual(task.name, "batch_inference")
        self.assertIsInstance(task.definition, str)
        assert isinstance(task.definition, str)  # mypy narrowing
        self.assertTrue(task.definition.startswith("CALL SYSTEM$DEPLOY_MODEL($$"))

    def test_chaining_with_rshift_registers_predecessor(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_task import (
            BatchInferenceTask,
        )

        with self._make_dag():
            data_prep = DAGTask("data_preparation", definition="SELECT 1")
            bi = BatchInferenceTask(
                "batch_inference",
                model_version=self._create_mock_model_version(),
                X=self._create_mock_dataframe(),
                compute_pool="CPU_POOL",
                output_spec=OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/o/"),
                job_spec=JobSpec(warehouse="MY_WH"),
            )
            data_prep >> bi

        self.assertIn(data_prep, bi.predecessors)

    def test_passes_dagtask_kwargs(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_task import (
            BatchInferenceTask,
        )

        with self._make_dag():
            task = BatchInferenceTask(
                "batch_inference",
                model_version=self._create_mock_model_version(),
                X=self._create_mock_dataframe(),
                compute_pool="CPU_POOL",
                output_spec=OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/o/"),
                job_spec=JobSpec(warehouse="MY_WH"),
                condition="SYSTEM$STREAM_HAS_DATA('MY_STREAM')",
                comment="my batch task",
                warehouse="OVERRIDE_WH",
                session_parameters={"QUERY_TAG": "BATCH"},
                user_task_timeout_ms=600000,
                target_completion_interval=timedelta(minutes=30),
            )

        self.assertEqual(task.condition, "SYSTEM$STREAM_HAS_DATA('MY_STREAM')")
        self.assertEqual(task.comment, "my batch task")
        self.assertEqual(task.warehouse, "OVERRIDE_WH")
        self.assertEqual(task.session_parameters, {"QUERY_TAG": "BATCH"})
        self.assertEqual(task.user_task_timeout_ms, 600000)
        self.assertEqual(task.target_completion_interval, timedelta(minutes=30))

    def test_definition_kwarg_rejected(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_task import (
            BatchInferenceTask,
        )

        with self._make_dag():
            with self.assertRaises(TypeError):
                BatchInferenceTask(
                    "batch_inference",
                    model_version=self._create_mock_model_version(),
                    X=self._create_mock_dataframe(),
                    compute_pool="CPU_POOL",
                    output_spec=OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/o/"),
                    job_spec=JobSpec(warehouse="MY_WH"),
                    definition="SELECT 1",
                )

    def test_outside_dag_context_raises(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_task import (
            BatchInferenceTask,
        )

        with self.assertRaises(ValueError):
            BatchInferenceTask(
                "batch_inference",
                model_version=self._create_mock_model_version(),
                X=self._create_mock_dataframe(),
                compute_pool="CPU_POOL",
                output_spec=OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/o/"),
                job_spec=JobSpec(warehouse="MY_WH"),
            )

    def test_explicit_dag_kwarg_without_context(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_task import (
            BatchInferenceTask,
        )

        dag = self._make_dag()

        task = BatchInferenceTask(
            "batch_inference",
            model_version=self._create_mock_model_version(),
            X=self._create_mock_dataframe(),
            compute_pool="CPU_POOL",
            output_spec=OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/o/"),
            job_spec=JobSpec(warehouse="MY_WH"),
            dag=dag,
        )

        self.assertIs(task.dag, dag)


if __name__ == "__main__":
    absltest.main()
