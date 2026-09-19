import inspect
from collections.abc import Callable
from typing import Any, cast
from unittest import mock

from absl.testing import absltest

from snowflake import connector
from snowflake.connector import telemetry as connector_telemetry
from snowflake.ml._internal import platform_capabilities as pc, telemetry
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.model import (
    batch_inference_job_specs,
    model_version_impl,
    telemetry_params,
)
from snowflake.ml.model._client.ops import model_ops, service_ops
from snowflake.ml.registry import registry
from snowflake.ml.test_utils import mock_session
from snowflake.ml.test_utils.mock_progress import create_mock_progress_status
from snowflake.snowpark import Session
from snowflake.snowpark._internal import server_connection


def _assert_allowlist_subset_of_signature(func: Callable[..., Any], allowlist: list[str]) -> None:
    sig = inspect.signature(func)
    missing = [name for name in allowlist if name not in sig.parameters]
    if missing:
        raise AssertionError(f"{func.__qualname__} is missing allowlisted params: {missing}")


class ModelTelemetryParamsTest(absltest.TestCase):
    def test_create_service_allowlist_has_no_stale_keys(self) -> None:
        self.assertNotIn("image_repo_database", telemetry_params.CREATE_SERVICE_FUNC_PARAMS_TO_LOG)
        self.assertNotIn("image_repo_schema", telemetry_params.CREATE_SERVICE_FUNC_PARAMS_TO_LOG)
        _assert_allowlist_subset_of_signature(
            model_version_impl.ModelVersion.create_service,
            telemetry_params.CREATE_SERVICE_FUNC_PARAMS_TO_LOG,
        )

    def test_run_and_run_batch_allowlists_match_signatures(self) -> None:
        _assert_allowlist_subset_of_signature(
            model_version_impl.ModelVersion.run,
            telemetry_params.RUN_FUNC_PARAMS_TO_LOG,
        )
        _assert_allowlist_subset_of_signature(
            model_version_impl.ModelVersion.run_batch,
            telemetry_params.RUN_BATCH_FUNC_PARAMS_TO_LOG,
        )
        _assert_allowlist_subset_of_signature(
            registry.Registry.log_model,
            telemetry_params.LOG_MODEL_FUNC_PARAMS_TO_LOG,
        )

    def test_create_service_func_params_include_new_fields(self) -> None:
        params = telemetry._get_func_params(
            model_version_impl.ModelVersion.create_service,
            telemetry_params.CREATE_SERVICE_FUNC_PARAMS_TO_LOG,
            args=(),
            kwargs={
                "service_name": "SERVICE",
                "service_compute_pool": "POOL",
                "ingress_enabled": True,
                "min_instances": 1,
                "max_instances": 3,
                "cpu_requests": "1",
                "memory_requests": "4Gi",
                "gpu_requests": "1",
                "num_workers": 2,
                "max_batch_rows": 32,
                "force_rebuild": True,
                "build_external_access_integrations": ["EAI"],
                "block": False,
                "autocapture": True,
                "inference_engine_options": {"engine": "vllm"},
            },
        )
        self.assertEqual(params["ingress_enabled"], "True")
        self.assertEqual(params["min_instances"], "1")
        self.assertEqual(params["max_instances"], "3")
        self.assertEqual(params["cpu_requests"], "'1'")
        self.assertEqual(params["memory_requests"], "'4Gi'")
        self.assertEqual(params["autocapture"], "True")
        self.assertEqual(params["inference_engine_options"], "{'engine': 'vllm'}")
        self.assertEqual(params["force_rebuild"], "True")
        self.assertNotIn("image_repo_database", params)
        self.assertNotIn("feature_sources_per_function", params)

    def test_run_func_params_include_partition_and_strict_flags(self) -> None:
        params = telemetry._get_func_params(
            model_version_impl.ModelVersion.run,
            telemetry_params.RUN_FUNC_PARAMS_TO_LOG,
            args=(),
            kwargs={
                "X": None,
                "service_name": "SVC",
                "function_name": "predict",
                "partition_column": "USER_ID",
                "strict_input_validation": True,
                "params": {"temperature": 0.2},
            },
        )
        self.assertEqual(params["partition_column"], "'USER_ID'")
        self.assertEqual(params["strict_input_validation"], "True")
        self.assertEqual(params["service_name"], "'SVC'")
        self.assertEqual(params["params"], "{'temperature': 0.2}")
        self.assertNotIn("X", params)

    def test_run_batch_func_params_include_function_and_async(self) -> None:
        output_spec = batch_inference_job_specs.OutputSpec(stage_location="@db.public.stage/out/")
        params = telemetry._get_func_params(
            model_version_impl.ModelVersion.run_batch,
            telemetry_params.RUN_BATCH_FUNC_PARAMS_TO_LOG,
            args=(),
            kwargs={
                "compute_pool": "POOL",
                "output_spec": output_spec,
                "function_name": "predict",
                "job_name": "JOB",
                "async_": False,
                "input_stage_location": "@db.public.stage/in/",
                "replicas": 2,
            },
        )
        self.assertEqual(params["function_name"], "'predict'")
        self.assertEqual(params["job_name"], "'JOB'")
        self.assertEqual(params["async_"], "False")
        self.assertEqual(params["input_stage_location"], "'@db.public.stage/in/'")
        self.assertEqual(params["replicas"], "2")
        self.assertNotIn("X", params)

    def test_log_model_func_params_include_task_and_options(self) -> None:
        params = telemetry._get_func_params(
            registry.Registry.log_model,
            telemetry_params.LOG_MODEL_FUNC_PARAMS_TO_LOG,
            args=(),
            kwargs={
                "model_name": "MODEL",
                "task": "TABULAR_REGRESSION",
                "options": {"enable_explainability": True},
            },
        )
        self.assertEqual(params["task"], "'TABULAR_REGRESSION'")
        self.assertEqual(params["options"], "{'enable_explainability': True}")
        self.assertNotIn("sample_input_data", params)

    def test_delete_service_and_get_model_func_params(self) -> None:
        delete_params = telemetry._get_func_params(
            model_version_impl.ModelVersion.delete_service,
            ["service_name"],
            args=(),
            kwargs={"service_name": "SERVICE"},
        )
        self.assertEqual(delete_params["service_name"], "'SERVICE'")

        get_model_params = telemetry._get_func_params(
            registry.Registry.get_model,
            ["model_name"],
            args=(),
            kwargs={"model_name": "MODEL"},
        )
        self.assertEqual(get_model_params["model_name"], "'MODEL'")

    def test_hf_allowlist_covers_service_and_log_args(self) -> None:
        allowlist = telemetry_params.HF_LOG_MODEL_AND_CREATE_SERVICE_FUNC_PARAMS_TO_LOG
        self.assertIn("model_name", allowlist)
        self.assertIn("ingress_enabled", allowlist)
        self.assertIn("inference_engine_options", allowlist)
        self.assertIn("autocapture", allowlist)


class ModelTelemetryEmitTest(absltest.TestCase):
    def setUp(self) -> None:
        telemetry.clear_cached_conn()
        self.mock_session = mock_session.MockSession(conn=None, test_case=self)
        self.c_session = cast(Session, self.mock_session)
        self.mock_server_conn = mock.MagicMock(spec=server_connection.ServerConnection)
        self.mock_snowflake_conn = mock.MagicMock(spec=connector.SnowflakeConnection)
        self.mock_telemetry = mock.MagicMock(spec=connector_telemetry.TelemetryClient)
        self.mock_snowflake_conn._telemetry = self.mock_telemetry
        self.mock_snowflake_conn._session_parameters = {}
        self.mock_snowflake_conn.is_closed.return_value = False
        self.mock_server_conn._conn = self.mock_snowflake_conn
        with (
            mock.patch.object(model_version_impl.ModelVersion, "_get_functions", return_value=[]),
            pc.PlatformCapabilities.mock_features({"ENABLE_INLINE_DEPLOYMENT_SPEC_FROM_CLIENT_VERSION": "1.8.6"}),
        ):
            self.m_mv = model_version_impl.ModelVersion._ref(
                model_ops.ModelOperator(
                    self.c_session,
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                ),
                service_ops=service_ops.ServiceOperator(
                    self.c_session,
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                ),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            )
        peft_check = mock.patch.object(self.m_mv, "_is_peft_adapter_version", return_value=False, autospec=True)
        peft_check.start()
        self.addCleanup(peft_check.stop)

    def _func_params_from_last_event(self) -> dict[str, Any]:
        message = self.mock_telemetry.try_add_log_to_batch.call_args.args[0].to_dict()["message"]
        return cast(dict[str, Any], message["data"][telemetry.TelemetryField.KEY_FUNC_PARAMS.value])

    @mock.patch("snowflake.ml._internal.telemetry._get_snowflake_connection")
    def test_create_service_emits_expanded_func_params(self, mock_get_conn: mock.MagicMock) -> None:
        mock_get_conn.return_value = self.mock_snowflake_conn
        mock_progress_status = create_mock_progress_status()
        with (
            mock.patch.object(self.m_mv._service_ops, "create_service"),
            mock.patch("snowflake.ml.model.event_handler.ModelEventHandler") as mock_event_handler_cls,
            mock.patch.object(self.m_mv, "_can_run_on_gpu", return_value=True),
        ):
            mock_event_handler_cls.return_value.status.return_value.__enter__.return_value = mock_progress_status
            self.m_mv.create_service(
                service_name="SERVICE",
                service_compute_pool="POOL",
                ingress_enabled=True,
                min_instances=1,
                max_instances=2,
                cpu_requests="2",
                memory_requests="8Gi",
                gpu_requests="1",
                num_workers=1,
                max_batch_rows=64,
                force_rebuild=True,
                block=True,
                autocapture=False,
            )

        func_params = self._func_params_from_last_event()
        self.assertEqual(func_params["service_name"], "'SERVICE'")
        self.assertEqual(func_params["ingress_enabled"], "True")
        self.assertEqual(func_params["min_instances"], "1")
        self.assertEqual(func_params["autocapture"], "False")
        self.assertNotIn("image_repo_database", func_params)

    @mock.patch("snowflake.ml._internal.telemetry._get_snowflake_connection")
    def test_delete_service_emits_service_name(self, mock_get_conn: mock.MagicMock) -> None:
        mock_get_conn.return_value = self.mock_snowflake_conn
        with mock.patch.object(self.m_mv._model_ops, "delete_service"):
            self.m_mv.delete_service("SERVICE")
        func_params = self._func_params_from_last_event()
        self.assertEqual(func_params["service_name"], "'SERVICE'")

    @mock.patch("snowflake.ml._internal.telemetry._get_snowflake_connection")
    def test_run_emits_partition_column(self, mock_get_conn: mock.MagicMock) -> None:
        mock_get_conn.return_value = self.mock_snowflake_conn
        from snowflake.ml.model import model_signature
        from snowflake.ml.model._model_composer.model_manifest import (
            model_manifest_schema,
        )

        sig = model_signature.ModelSignature(
            inputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input")],
            outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
        )
        self.m_mv._functions = [
            model_manifest_schema.ModelFunctionInfo(
                {
                    "name": '"predict"',
                    "target_method": "predict",
                    "target_method_function_type": "FUNCTION",
                    "signature": sig,
                    "is_partitioned": True,
                    "is_object_output": False,
                }
            ),
        ]
        with (
            mock.patch.object(self.m_mv._model_ops, "invoke_method", return_value=mock.MagicMock()),
            mock.patch.object(
                self.m_mv._model_ops,
                "_fetch_model_spec_and_target_platforms",
                return_value=(mock.MagicMock(), ["WAREHOUSE"]),
            ),
        ):
            self.m_mv.run(
                mock.MagicMock(),
                function_name='"predict"',
                partition_column="USER_ID",
                strict_input_validation=True,
            )
        func_params = self._func_params_from_last_event()
        self.assertEqual(func_params["function_name"], "'\"predict\"'")
        self.assertEqual(func_params["partition_column"], "'USER_ID'")
        self.assertEqual(func_params["strict_input_validation"], "True")


if __name__ == "__main__":
    absltest.main()
