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

from snowflake.ml.model._client.model.batch_inference_job_specs import (
    ImageBuildSpec,
    InferenceSpec,
    InputSpec,
    OutputSpec,
    ResourcesSpec,
    SaveMode,
)

_QUERY = "SELECT C1, C2 FROM MY_DB.MY_SCHEMA.MY_TABLE"


@absltest.skipUnless(_HAS_SNOWFLAKE_CORE, "snowflake.core not installed")
class BatchInferenceTaskV2Test(absltest.TestCase):
    def _create_mock_model_version(
        self,
        target_method: str = "predict",
        *,
        current_database: Optional[str] = "MY_DB",
        current_schema: Optional[str] = "MY_SCHEMA",
    ) -> mock.MagicMock:
        mv = mock.MagicMock()
        mv.fully_qualified_model_name = "MY_DB.MY_SCHEMA.MY_MODEL"
        mv.version_name = "V1"
        mv._validate_batch_inference_request.return_value = {"target_method": target_method}
        session = mv._service_ops._session
        session.get_current_database.return_value = current_database
        session.get_current_schema.return_value = current_schema
        return mv

    def _make_dag(self) -> "DAG":
        return DAG(
            "test_dag",
            schedule=timedelta(days=1),
            warehouse="DAG_WH",
            stage_location="@MY_DB.MY_SCHEMA.MY_STAGE",
        )

    def _build_task(self, **kwargs: Any) -> Any:
        from snowflake.ml.model._client.model.batch_inference_task_v2 import (
            BatchInferenceTask,
        )

        kwargs.setdefault("model_version", self._create_mock_model_version())
        kwargs.setdefault("compute_pool", "MY_POOL")
        kwargs.setdefault("output_spec", OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/"))
        if "query" not in kwargs and "input_stage_location" not in kwargs:
            kwargs["query"] = _QUERY
        with self._make_dag():
            return BatchInferenceTask("batch_inference", **kwargs)

    def _spec_from_sql(self, sql: str) -> dict[str, Any]:
        """Extract and parse the WITH SPECIFICATION YAML body out of the command text."""
        marker = "WITH SPECIFICATION '"
        start = sql.index(marker) + len(marker)
        end = sql.index("'\nFROM ", start)
        yaml_str = sql[start:end].replace("\\'", "'")
        parsed: dict[str, Any] = yaml.safe_load(yaml_str)
        return parsed

    def test_clause_order_and_content(self) -> None:
        task = self._build_task()
        lines = [line for line in task.definition.splitlines() if not line.startswith(" ")]

        self.assertEqual(lines[0], "EXECUTE INFERENCE JOB SERVICE")
        self.assertEqual(lines[1], "IN COMPUTE POOL MY_POOL")
        self.assertIn(f"FROM ({_QUERY})", task.definition)
        self.assertIn("MODEL = MY_DB.MY_SCHEMA.MY_MODEL", task.definition)
        self.assertIn("VERSION = V1", task.definition)
        self.assertIn("ASYNC = FALSE", task.definition)

    def test_no_name_clause(self) -> None:
        """The server names each run; a literal NAME would collide on the second firing."""
        task = self._build_task()
        self.assertNotIn("NAME = ", task.definition)

    def test_no_replicas_clause_by_default(self) -> None:
        self.assertNotIn("REPLICAS", self._build_task().definition)

    def test_replicas_clause(self) -> None:
        self.assertIn("REPLICAS = 3", self._build_task(replicas=3).definition)

    def test_function_uses_target_method(self) -> None:
        """FUNCTION must be the model's target method, not the upper-cased SQL function name."""
        mv = self._create_mock_model_version(target_method="predict")
        task = self._build_task(model_version=mv, function_name="PREDICT")

        self.assertIn("FUNCTION = 'predict'", task.definition)
        mv._validate_batch_inference_request.assert_called_once()
        self.assertEqual(mv._validate_batch_inference_request.call_args.kwargs["function_name"], "PREDICT")

    def test_query_renders_parenthesized_subquery(self) -> None:
        self.assertIn(f"FROM ({_QUERY})", self._build_task(query=_QUERY).definition)

    def test_input_stage_location_renders_stage_path(self) -> None:
        task = self._build_task(input_stage_location="@MY_DB.MY_SCHEMA.OTHER_STAGE/input/")
        self.assertIn("FROM @MY_DB.MY_SCHEMA.OTHER_STAGE/input/", task.definition)

    def test_input_stage_location_gets_trailing_slash(self) -> None:
        task = self._build_task(input_stage_location="@MY_DB.MY_SCHEMA.OTHER_STAGE/input")
        self.assertIn("FROM @MY_DB.MY_SCHEMA.OTHER_STAGE/input/", task.definition)

    def test_output_stage_location_normalized(self) -> None:
        task = self._build_task(output_spec=OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output"))
        spec = self._spec_from_sql(task.definition)
        self.assertEqual(spec["output"]["stage_location"], "@MY_DB.MY_SCHEMA.MY_STAGE/output/")

    def test_spec_body_only_has_output_by_default(self) -> None:
        spec = self._spec_from_sql(self._build_task().definition)
        self.assertEqual(list(spec.keys()), ["output"])
        self.assertEqual(spec["output"]["mode"], "error")

    def test_spec_body_with_all_blocks(self) -> None:
        task = self._build_task(
            output_spec=OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/", mode=SaveMode.OVERWRITE),
            input_spec=InputSpec(params={"threshold": 0.5}),
            resources_spec=ResourcesSpec(cpu_requests="2", memory_requests="8Gi"),
            inference_spec=InferenceSpec(num_workers=4, max_batch_rows=1024),
            image_build_spec=ImageBuildSpec(image_repo="my_repo", force_rebuild=True),
        )
        spec = self._spec_from_sql(task.definition)

        self.assertEqual(list(spec.keys()), ["input", "output", "resources", "inference", "image_build"])
        self.assertEqual(spec["input"]["params"], {"threshold": 0.5})
        self.assertEqual(spec["output"]["mode"], "overwrite")
        self.assertEqual(spec["resources"], {"cpu_requests": "2", "memory_requests": "8Gi"})
        self.assertEqual(spec["inference"], {"num_workers": 4, "max_batch_rows": 1024})
        self.assertEqual(spec["image_build"], {"image_repo": "my_repo", "force_rebuild": True})

    def test_neither_input_source_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Exactly one of query or input_stage_location"):
            self._build_task(query=None, input_stage_location=None)

    def test_both_input_sources_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "Exactly one of query or input_stage_location"):
            self._build_task(query=_QUERY, input_stage_location="@MY_DB.MY_SCHEMA.OTHER_STAGE/input/")

    def test_output_stage_without_at_prefix_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a stage path starting with '@'"):
            self._build_task(output_spec=OutputSpec(stage_location="MY_DB.MY_SCHEMA.MY_STAGE/output/"))

    def test_unparsable_output_stage_raises(self) -> None:
        """Starts with '@' but is not a well-formed stage path, so parsing rejects it."""
        with self.assertRaisesRegex(ValueError, "is not a valid Snowflake stage path"):
            self._build_task(output_spec=OutputSpec(stage_location="@"))

    def test_input_stage_inside_output_stage_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be inside output_spec.stage_location"):
            self._build_task(
                query=None,
                input_stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/nested/",
                output_spec=OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/"),
            )

    def test_validation_failure_propagates(self) -> None:
        mv = self._create_mock_model_version()
        mv._validate_batch_inference_request.side_effect = ValueError("partition_column is not supported")
        with self.assertRaisesRegex(ValueError, "partition_column is not supported"):
            self._build_task(model_version=mv)

    def test_definition_kwarg_rejected(self) -> None:
        with self.assertRaises(TypeError):
            self._build_task(definition="SELECT 1")

    def test_outside_dag_context_raises(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_task_v2 import (
            BatchInferenceTask,
        )

        with self.assertRaises(ValueError):
            BatchInferenceTask(
                "batch_inference",
                model_version=self._create_mock_model_version(),
                compute_pool="MY_POOL",
                output_spec=OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/"),
                query=_QUERY,
            )

    def test_explicit_dag_kwarg_without_context(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_task_v2 import (
            BatchInferenceTask,
        )

        dag = self._make_dag()
        task = BatchInferenceTask(
            "batch_inference",
            model_version=self._create_mock_model_version(),
            compute_pool="MY_POOL",
            output_spec=OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/"),
            query=_QUERY,
            dag=dag,
        )
        self.assertIs(task.dag, dag)

    def test_is_dagtask_and_chains(self) -> None:
        from snowflake.ml.model._client.model.batch_inference_task_v2 import (
            BatchInferenceTask,
        )

        with self._make_dag():
            prep = DAGTask("prep", definition="SELECT 1")
            task = BatchInferenceTask(
                "batch_inference",
                model_version=self._create_mock_model_version(),
                compute_pool="MY_POOL",
                output_spec=OutputSpec(stage_location="@MY_DB.MY_SCHEMA.MY_STAGE/output/"),
                query=_QUERY,
            )
            prep >> task

        self.assertIsInstance(task, DAGTask)
        self.assertEqual(task.name, "batch_inference")
        self.assertIn(prep, task.predecessors)

    def test_passes_dagtask_kwargs(self) -> None:
        task = self._build_task(
            condition="SYSTEM$STREAM_HAS_DATA('MY_STREAM')",
            comment="my batch task",
            warehouse="OVERRIDE_WH",
            session_parameters={"QUERY_TAG": "BATCH"},
            user_task_timeout_ms=600000,
        )

        self.assertEqual(task.condition, "SYSTEM$STREAM_HAS_DATA('MY_STREAM')")
        self.assertEqual(task.comment, "my batch task")
        self.assertEqual(task.warehouse, "OVERRIDE_WH")
        self.assertEqual(task.session_parameters, {"QUERY_TAG": "BATCH"})
        self.assertEqual(task.user_task_timeout_ms, 600000)


if __name__ == "__main__":
    absltest.main()
