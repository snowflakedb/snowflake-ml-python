import json
import logging
import os
import tempfile
import time
from typing import Any
from unittest import mock

import pandas as pd
import requests
import yaml
from absl.testing import absltest
from sklearn import datasets, linear_model

from snowflake.ml._internal import platform_capabilities
from snowflake.ml.model import ModelVersion, PeftAdapter, openai_signatures
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model.inference_engine import InferenceEngine
from snowflake.ml.model.models import huggingface_pipeline
from snowflake.ml.registry import registry
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)
from tests.integ.snowflake.ml.test_utils import (
    db_manager,
    lora_adapter_account_gate,
    test_env_utils,
)

_TINY_GPT2 = "hf-internal-testing/tiny-gpt2-with-chatml-template"
_SMOLLM2 = "HuggingFaceTB/SmolLM2-135M-Instruct"
_SMOLLM2_ADAPTER = "Miladsaeedi70/smollm2-135m-scientific-sft-lora"
_SMOLLM2_ADAPTER_REVISION = "c344de0f75239d4de9341f7d38a580cdd8b10d25"

_LORA_ADAPTERS_ENABLED_PATCHER = mock.patch.object(
    platform_capabilities.PlatformCapabilities,
    "is_lora_adapters_enabled",
    return_value=True,
    autospec=True,
)
_LORA_ADAPTERS_ENABLED_PATCHER.start()


def _write_stub_adapter_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 8}, f)
    with open(os.path.join(path, "adapter_model.safetensors"), "wb") as f:
        f.write(b"")
    return path


class TestRegistryLoraAdapterDeploymentInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    cache_dir: tempfile.TemporaryDirectory
    _original_cache_dir: str | None = None
    _original_hf_accept_encoding: str | None = None
    _original_hf_home: str | None = None
    _original_hf_endpoint: str | None = None

    @classmethod
    def setUpClass(cls) -> None:
        import huggingface_hub

        super().setUpClass()
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls._original_cache_dir = os.getenv("TRANSFORMERS_CACHE", None)
        cls._original_hf_home = os.getenv("HF_HOME", None)
        os.environ["TRANSFORMERS_CACHE"] = cls.cache_dir.name
        os.environ["HF_HOME"] = cls.cache_dir.name
        if "HF_ENDPOINT" in os.environ:
            cls._original_hf_endpoint = os.environ["HF_ENDPOINT"]
            del os.environ["HF_ENDPOINT"]
        hf_session = huggingface_hub.get_session()
        cls._original_hf_accept_encoding = hf_session.headers.get("Accept-Encoding")
        hf_session.headers["Accept-Encoding"] = "identity"

    @classmethod
    def tearDownClass(cls) -> None:
        import huggingface_hub

        if cls._original_cache_dir is not None:
            os.environ["TRANSFORMERS_CACHE"] = cls._original_cache_dir
        else:
            os.environ.pop("TRANSFORMERS_CACHE", None)
        if cls._original_hf_home is not None:
            os.environ["HF_HOME"] = cls._original_hf_home
        else:
            os.environ.pop("HF_HOME", None)
        cls.cache_dir.cleanup()
        if cls._original_hf_endpoint is not None:
            os.environ["HF_ENDPOINT"] = cls._original_hf_endpoint
        hf_session = huggingface_hub.get_session()
        if cls._original_hf_accept_encoding is None:
            hf_session.headers.pop("Accept-Encoding", None)
        else:
            hf_session.headers["Accept-Encoding"] = cls._original_hf_accept_encoding
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        lora_adapter_account_gate.skip_unless_lora_adapters_account_enabled(self.session)
        self.session.sql("ALTER SESSION SET SPCS_MODEL_AUTO_POPULATE_GPU_FROM_COMPUTE_POOL = TRUE").collect()

    def tearDown(self) -> None:
        self.session.sql("ALTER SESSION UNSET SPCS_MODEL_AUTO_POPULATE_GPU_FROM_COMPUTE_POOL").collect()
        super().tearDown()

    def _name(self, suffix: str) -> str:
        return db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self._run_id, suffix).upper()

    def _service_fqn(self, service_name: str) -> str:
        if "." in service_name:
            return service_name
        return f"{self._test_db}.{self._test_schema}.{service_name}"

    def _chat_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "Complete the sentence."}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "A descendant of the Lost City of Atlantis, who swam to Earth while saying, "
                                    ),
                                }
                            ],
                        },
                    ],
                    "temperature": 0.9,
                    "max_completion_tokens": 64,
                    "stop": None,
                    "n": 1,
                    "stream": False,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "response_format": None,
                }
            ]
        )

    def _assert_chat_res(self, res: pd.DataFrame) -> None:
        pd.testing.assert_index_equal(
            res.columns,
            pd.Index(["id", "object", "created", "model", "choices", "usage"], dtype="object"),
            check_order=False,
        )
        for row in res["choices"]:
            self.assertIsInstance(row, list)
            self.assertGreater(len(row), 0)
            self.assertIn("message", row[0])
            self.assertIn("content", row[0]["message"])
            self.assertGreater(len(row[0]["message"]["content"]), 0)

    def _assert_sql_chat(self, rows: Any) -> None:
        self.assertGreater(len(rows), 0)
        value = rows[0][0]
        if isinstance(value, str):
            value = json.loads(value)
        self.assertIsInstance(value, dict)
        choices = value.get("choices") or value.get("CHOICES") or []
        self.assertGreater(len(choices), 0)
        message = choices[0].get("message") or choices[0].get("MESSAGE") or {}
        content = message.get("content") or message.get("CONTENT") or ""
        self.assertGreater(len(str(content)), 0)

    def _serving_conda_dependencies(self) -> list[str]:
        return [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python"),
            "pytorch==2.6.0",
        ]

    def _log_serving_base(self, *, model_name: str, version_name: str) -> ModelVersion:
        model = huggingface_pipeline.HuggingFacePipelineModel(
            task="text-generation",
            model=_SMOLLM2,
            download_snapshot=False,
        )
        return self.registry.log_model(
            model=model,
            model_name=model_name,
            version_name=version_name,
            signatures=openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE,
            pip_requirements=["transformers==5.3.0", "torch==2.6.0"],
            options={
                "embed_local_ml_library": True,
                "cuda_version": model_env.DEFAULT_CUDA_VERSION,
            },
            target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
            conda_dependencies=self._serving_conda_dependencies(),
        )

    def _log_serving_adapter(self, *, base_mv: ModelVersion, model_name: str, version_name: str) -> ModelVersion:
        return self.registry.log_model(
            model=PeftAdapter(
                base_model=base_mv,
                adapter_repo=_SMOLLM2_ADAPTER,
                revision=_SMOLLM2_ADAPTER_REVISION,
            ),
            model_name=model_name,
            version_name=version_name,
        )

    def _log_tiny_with_model_base(self, *, model_name: str, version_name: str) -> ModelVersion:
        import transformers

        model = transformers.pipeline(
            task="text-generation",
            model=_TINY_GPT2,
            max_length=200,
        )
        return self.registry.log_model(
            model=model,
            model_name=model_name,
            version_name=version_name,
            signatures=openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE,
            pip_requirements=["transformers==5.3.0", "torch==2.6.0"],
            target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
            options={"embed_local_ml_library": True},
        )

    def _log_stub_adapter(self, *, base_mv: ModelVersion, model_name: str, version_name: str) -> ModelVersion:
        with tempfile.TemporaryDirectory() as tmp:
            return self.registry.log_model(
                model=PeftAdapter(base_model=base_mv, adapter_path=_write_stub_adapter_dir(tmp)),
                model_name=model_name,
                version_name=version_name,
            )

    def _vllm_engine_options(self, *, extra_args: list[str] | None = None) -> dict[str, Any]:
        options = self._get_inference_engine_options_for_inference_engine(
            InferenceEngine.VLLM,
            {"engine_args_override": extra_args or []},
        )
        assert options is not None
        return options

    def _service_owner(self, mv: ModelVersion) -> ModelVersion:
        if mv._is_peft_adapter_version():
            return mv._resolve_adapter_pin()
        return mv

    def _skip_unless_image_override(self) -> None:
        if self._has_image_override() or self.PROXY_IMAGE_PATH:
            return
        self.skipTest(
            "Short adapter aliases need the V4 proxy rewrite, which is not in the "
            "production serving image. This case runs under Jenkins image override."
        )

    def _log_service_diagnostics(self, service_name: str) -> None:
        service_fqn = self._service_fqn(service_name)
        for attempt in range(2):
            if attempt:
                time.sleep(30)
            try:
                container_rows = self.session.sql(f"SHOW SERVICE CONTAINERS IN SERVICE {service_fqn}").collect()
            except Exception:
                logging.exception("Unable to read service container statuses for %s", service_fqn)
                return

            logging.error(
                "Service container statuses for %s (attempt %s): %s",
                service_fqn,
                attempt + 1,
                [row.as_dict() for row in container_rows],
            )
            for container_row in container_rows:
                status = {str(key).lower(): value for key, value in container_row.as_dict().items()}
                container_name = status.get("container_name") or status.get("name")
                if not container_name:
                    continue
                instance_id = str(status.get("instance_id", "0"))
                escaped_container_name = str(container_name).replace("'", "''")
                try:
                    log_rows = self.session.sql(
                        f"CALL SYSTEM$GET_SERVICE_LOGS('{service_fqn}', '{instance_id}', '{escaped_container_name}')"
                    ).collect()
                    logging.error(
                        "Service logs for %s instance %s container %s (attempt %s):\n%s",
                        service_fqn,
                        instance_id,
                        container_name,
                        attempt + 1,
                        log_rows[0][0] if log_rows else "<empty>",
                    )
                except Exception:
                    logging.exception(
                        "Unable to read service logs for %s instance %s container %s",
                        service_fqn,
                        instance_id,
                        container_name,
                    )

    def _create_lora_service(
        self,
        mv: ModelVersion,
        *,
        service_name: str,
        adapters: dict[str, ModelVersion] | list[ModelVersion] | None = None,
        engine_args_override: list[str] | None = None,
        autocapture: bool | None = None,
    ) -> str:
        create_kwargs: dict[str, Any] = {
            "service_name": service_name,
            "service_compute_pool": self._TEST_GPU_COMPUTE_POOL,
            "ingress_enabled": True,
            "force_rebuild": True,
            "inference_engine_options": self._vllm_engine_options(extra_args=engine_args_override),
        }
        if adapters is not None:
            create_kwargs["adapters"] = adapters
        if autocapture is not None:
            create_kwargs["autocapture"] = autocapture
        try:
            mv.create_service(**create_kwargs)
            self._wait_for_service_status(self._service_owner(mv))
        except Exception:
            self._log_service_diagnostics(service_name)
            raise
        return service_name

    def _sql_call(self, service_fqn: str, *, model: str | None = None) -> Any:
        messages = self._chat_df()["messages"].iloc[0]
        messages_sql = json.dumps(messages).replace("'", "''")
        model_arg = f", model => '{model}'" if model is not None else ""
        query = f"""SELECT {service_fqn}!"__CALL__"(PARSE_JSON('{messages_sql}'){model_arg})"""
        return self.session.sql(query).collect()

    def _rest_call(
        self,
        mv: ModelVersion,
        *,
        params: dict[str, Any] | None = None,
        expect_ok: bool = True,
    ) -> Any:
        endpoint = self._ensure_ingress_url(self._service_owner(mv))
        payload = self._build_rest_inference_request_payload(
            registry_model_deployment_test_base.RestInferencePayloadFormat.DATAFRAME_SPLIT,
            self._chat_df(),
            params,
        )
        res = requests.post(
            f"https://{endpoint}/--call--",
            json=payload,
            auth=self._get_auth_for_inference(endpoint),
        )
        if expect_ok:
            res.raise_for_status()
            return pd.DataFrame([row[1] for row in res.json()["data"]])
        return res

    def _listed_service_names(self, mv: ModelVersion) -> list[str]:
        services = mv.list_services()
        if services.empty:
            return []
        return [str(name) for name in services["name"].tolist()]

    def _assert_lists_service(self, mv: ModelVersion, service_name: str) -> None:
        listed = self._listed_service_names(mv)
        self.assertTrue(
            listed,
            f"list_services() was empty; expected {service_name!r}",
        )
        expected = service_name.upper().strip('"')
        fqn = self._service_fqn(service_name).upper()
        matched = any(
            name.upper().strip('"') in (expected, fqn) or name.upper().endswith("." + expected) for name in listed
        )
        self.assertTrue(matched, f"service {service_name!r} not in list_services names {listed}")

    def _show_model_type(self, model_name: str) -> str:
        rows = self.session.sql(
            f"SHOW MODELS LIKE '{model_name}' IN SCHEMA {self._test_db}.{self._test_schema}"
        ).collect()
        self.assertGreater(len(rows), 0, f"SHOW MODELS LIKE {model_name!r} returned no rows")
        row = rows[0].as_dict()
        for key, value in row.items():
            if key.lower() == "model_type":
                return str(value)
        self.fail(f"SHOW MODELS row has no model_type column: {sorted(row)}")

    def _service_status(self, mv: ModelVersion) -> str:
        services = mv.list_services()
        if services.empty:
            return "PENDING"
        return str(services.loc[0, "status"])

    def _sql_service_status(self, service_name: str) -> str:
        unqualified = service_name.rsplit(".", 1)[-1].strip('"')
        rows = self.session.sql(
            f"SHOW SERVICES LIKE '{unqualified}' IN SCHEMA {self._test_db}.{self._test_schema}"
        ).collect()
        if not rows:
            return "PENDING"
        row = rows[0].as_dict()
        for key, value in row.items():
            if key.lower() == "status":
                return str(value)
        return "UNKNOWN"

    def _describe_service_spec(self, service_name: str) -> str:
        rows = self.session.sql(f"DESCRIBE SERVICE {self._service_fqn(service_name)}").collect()
        self.assertGreater(len(rows), 0, f"DESCRIBE SERVICE {service_name!r} returned no rows")
        row = rows[0].as_dict()
        for key, value in row.items():
            if key.lower() == "spec":
                return str(value)
        self.fail(f"DESCRIBE SERVICE row has no spec column: {sorted(row)}")

    def _lora_url_map_from_spec(self, spec_yaml: str) -> Any:
        parsed = yaml.safe_load(spec_yaml) or {}
        spec = parsed.get("spec") or parsed
        for container in spec.get("containers") or []:
            env = container.get("env") or {}
            raw = env.get("LORA_ADAPTERS_SNOW_URL_MAP")
            if raw is None:
                continue
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return raw
            return raw
        return None

    def _assert_base_only_spec(self, service_name: str) -> None:
        spec = self._describe_service_spec(service_name)
        self.assertNotIn("--enable-lora", spec)
        payload = self._lora_url_map_from_spec(spec)
        self.assertTrue(payload in (None, [], {}), f"base-only spec still has attach map: {payload!r}")

    def _assert_spec_adapters_list(self, service_name: str, *, expected_aliases: list[str]) -> None:
        spec = self._describe_service_spec(service_name)
        payload = self._lora_url_map_from_spec(spec)
        self.assertIsInstance(payload, list, f"attach map must be a list, got {type(payload).__name__}: {payload!r}")
        aliases = []
        for item in payload:
            self.assertIsInstance(item, dict)
            self.assertIn("name", item)
            self.assertIn("version", item)
            aliases.append(str(item.get("alias") or ""))
        for alias in expected_aliases:
            self.assertIn(alias, aliases)

    def test_explicit_aliases_serve_adapter(self) -> None:
        self._skip_unless_image_override()
        base = self._log_serving_base(model_name=self._name("BASE"), version_name="V1")
        support = self._log_serving_adapter(base_mv=base, model_name=self._name("SUPPORT"), version_name="V1")
        sql_gen = self._log_serving_adapter(base_mv=base, model_name=self._name("SQL_GEN"), version_name="V1")
        svc = self._create_lora_service(
            base,
            service_name=self._name("SVC"),
            adapters={"support": support, "sql_gen": sql_gen},
        )
        self._assert_spec_adapters_list(svc, expected_aliases=["support", "sql_gen"])
        fqn = self._service_fqn(svc)
        self._assert_sql_chat(self._sql_call(fqn, model="support"))
        rest = self._rest_call(base, params={"model": "sql_gen"})
        self._assert_chat_res(rest)
        self._assert_sql_chat(self._sql_call(fqn, model=None))
        self._assert_chat_res(self._rest_call(base))

    def test_list_adapters_omits_alias_and_serves(self) -> None:
        base = self._log_serving_base(model_name=self._name("BASE"), version_name="V1")
        support = self._log_serving_adapter(base_mv=base, model_name=self._name("SUPPORT"), version_name="V1")
        svc = self._create_lora_service(
            base,
            service_name=self._name("SVC"),
            adapters=[support],
        )
        fqn = self._service_fqn(svc)
        alias = f"{support.fully_qualified_model_name}/VERSIONS/{support.version_name}"
        self._assert_spec_adapters_list(svc, expected_aliases=[alias])
        self._assert_sql_chat(self._sql_call(fqn, model=alias))
        self._assert_sql_chat(self._sql_call(fqn, model=None))

    def test_sql_per_row_model_selects_adapter(self) -> None:
        self._skip_unless_image_override()
        base = self._log_serving_base(model_name=self._name("BASE"), version_name="V1")
        support = self._log_serving_adapter(base_mv=base, model_name=self._name("SUPPORT"), version_name="V1")
        sql_gen = self._log_serving_adapter(base_mv=base, model_name=self._name("SQL_GEN"), version_name="V1")
        svc = self._create_lora_service(
            base,
            service_name=self._name("SVC"),
            adapters={"support": support, "sql_gen": sql_gen},
        )
        messages = json.dumps(self._chat_df()["messages"].iloc[0]).replace("'", "''")
        table_name = self._name("TICKETS")
        self.session.sql(
            f"""
            CREATE TEMPORARY TABLE {self._test_db}.{self._test_schema}.{table_name} (MODEL_ALIAS VARCHAR)
            AS SELECT * FROM VALUES ('support'), ('sql_gen')
            """
        ).collect()
        rows = self.session.sql(
            f"""
            SELECT {self._service_fqn(svc)}!"__CALL__"(
                PARSE_JSON('{messages}'),
                model => MODEL_ALIAS
            )
            FROM {self._test_db}.{self._test_schema}.{table_name}
            """
        ).collect()
        self.assertEqual(len(rows), 2)
        for row in rows:
            self._assert_sql_chat([row])

    def test_omit_adapters_passthrough_unchanged(self) -> None:
        model = huggingface_pipeline.HuggingFacePipelineModel(
            task="text-generation",
            model=_SMOLLM2,
            download_snapshot=False,
        )
        mv = self._test_registry_model_deployment(
            model=model,
            prediction_assert_fns={
                "__call__": (self._chat_df(), self._assert_chat_res),
            },
            options={"cuda_version": model_env.DEFAULT_CUDA_VERSION},
            pip_requirements=["transformers==5.3.0", "torch==2.6.0"],
            conda_dependencies=self._serving_conda_dependencies(),
            signatures=openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE,
            inference_engine_options=self._vllm_engine_options(),
            rest_inference_formats=[
                registry_model_deployment_test_base.RestInferencePayloadFormat.DATAFRAME_SPLIT,
            ],
            service_compute_pool=self._TEST_GPU_COMPUTE_POOL,
        )
        self._assert_base_only_spec(str(mv.list_services().loc[0, "name"]))

    def test_empty_adapters_map_passthrough_unchanged(self) -> None:
        base = self._log_serving_base(model_name=self._name("BASE"), version_name="V1")
        svc = self._create_lora_service(base, service_name=self._name("SVC"), adapters={})
        res = base.run(self._chat_df(), function_name="__call__", service_name=svc)
        self._assert_chat_res(res)
        self._assert_sql_chat(self._sql_call(self._service_fqn(svc), model=None))
        self._assert_chat_res(self._rest_call(base))
        self._assert_base_only_spec(svc)

    def test_empty_adapters_list_passthrough_unchanged(self) -> None:
        base = self._log_serving_base(model_name=self._name("BASE"), version_name="V1")
        svc = self._create_lora_service(base, service_name=self._name("SVC"), adapters=[])
        res = base.run(self._chat_df(), function_name="__call__", service_name=svc)
        self._assert_chat_res(res)
        self._assert_sql_chat(self._sql_call(self._service_fqn(svc), model=None))
        self._assert_chat_res(self._rest_call(base))
        self._assert_base_only_spec(svc)

    def test_unknown_model_passes_through(self) -> None:
        base = self._log_serving_base(model_name=self._name("BASE"), version_name="V1")
        support = self._log_serving_adapter(base_mv=base, model_name=self._name("SUPPORT"), version_name="V1")
        sql_gen = self._log_serving_adapter(base_mv=base, model_name=self._name("SQL_GEN"), version_name="V1")
        svc = self._create_lora_service(
            base,
            service_name=self._name("SVC"),
            adapters={"support": support, "sql_gen": sql_gen},
        )
        unknown = "not_a_key"
        with self.assertRaisesRegex(Exception, unknown):
            self._sql_call(self._service_fqn(svc), model=unknown)

        rest = self._rest_call(base, params={"model": unknown}, expect_ok=False)
        self.assertIn(unknown, rest.text)
        self.assertNotIn("support", rest.text)
        self.assertNotIn("sql_gen", rest.text)

    def test_attach_set_exceeds_max_loras_lru(self) -> None:
        base = self._log_serving_base(model_name=self._name("BASE"), version_name="V1")
        adapter_a = self._log_serving_adapter(base_mv=base, model_name=self._name("A"), version_name="V1")
        adapter_b = self._log_serving_adapter(base_mv=base, model_name=self._name("B"), version_name="V1")
        svc = self._create_lora_service(
            base,
            service_name=self._name("SVC"),
            adapters=[adapter_a, adapter_b],
            engine_args_override=["--max-loras=1"],
        )
        fqn = self._service_fqn(svc)
        alias_a = f"{adapter_a.fully_qualified_model_name}/VERSIONS/{adapter_a.version_name}"
        alias_b = f"{adapter_b.fully_qualified_model_name}/VERSIONS/{adapter_b.version_name}"
        self._assert_sql_chat(self._sql_call(fqn, model=alias_a))
        self._assert_sql_chat(self._sql_call(fqn, model=alias_b))
        self._assert_chat_res(
            base.run(self._chat_df(), function_name="__call__", service_name=svc, params={"model": alias_a})
        )

    def test_adapter_create_service_forwards_to_pin(self) -> None:
        base = self._log_serving_base(model_name=self._name("BASE"), version_name="V1")
        adapter_name = self._name("SUPPORT")
        adapter = self._log_serving_adapter(base_mv=base, model_name=adapter_name, version_name="V1")
        svc = self._create_lora_service(adapter, service_name=self._name("SVC"))
        fqn = self._service_fqn(svc)
        alias = f"{adapter.fully_qualified_model_name}/VERSIONS/{adapter.version_name}"
        self._assert_spec_adapters_list(svc, expected_aliases=[alias])
        self._assert_sql_chat(self._sql_call(fqn, model=alias))
        self._assert_sql_chat(self._sql_call(fqn, model=None))
        self._assert_lists_service(base, svc)

    def test_mixed_version_create_service_on_each(self) -> None:
        model_name = self._name("MIXED")
        v1 = self._log_serving_base(model_name=model_name, version_name="V1")
        v2 = self._log_serving_adapter(base_mv=v1, model_name=model_name, version_name="V2")
        svc_v1 = self._create_lora_service(v1, service_name=self._name("SVC_V1"))
        svc_v2 = self._create_lora_service(v2, service_name=self._name("SVC_V2"))
        self.assertEqual(self._show_model_type(model_name), "USER_MODEL")
        alias = f"{v2.fully_qualified_model_name}/VERSIONS/{v2.version_name}"
        self._assert_spec_adapters_list(svc_v2, expected_aliases=[alias])
        self._assert_sql_chat(self._sql_call(self._service_fqn(svc_v1), model=None))
        self._assert_sql_chat(self._sql_call(self._service_fqn(svc_v2), model=alias))
        self._assert_sql_chat(self._sql_call(self._service_fqn(svc_v2), model=None))

    def test_adapter_run_service_name(self) -> None:
        base = self._log_serving_base(model_name=self._name("BASE"), version_name="V1")
        adapter = self._log_serving_adapter(base_mv=base, model_name=self._name("SUPPORT"), version_name="V1")
        svc = self._create_lora_service(
            base,
            service_name=self._name("SVC"),
            adapters=[adapter],
        )
        res = adapter.run(self._chat_df(), function_name="__call__", service_name=svc)
        self._assert_chat_res(res)

    def test_illegal_alias_charset_rejected(self) -> None:
        base = self._log_tiny_with_model_base(model_name=self._name("BASE"), version_name="V1")
        adapter = self._log_stub_adapter(base_mv=base, model_name=self._name("ADAPTER"), version_name="V1")
        # Empty aliases are rejected by GS deployment-spec validation.
        with self.assertRaisesRegex(Exception, r"(?i)invalid|alias|reserved"):
            base.create_service(
                service_name=self._name("SVC"),
                service_compute_pool=self._TEST_CPU_COMPUTE_POOL,
                adapters={"": adapter},
            )

    def test_slash_and_fqn_aliases_pass_client_charset(self) -> None:
        base = self._log_tiny_with_model_base(model_name=self._name("BASE"), version_name="V1")
        adapter = self._log_stub_adapter(base_mv=base, model_name=self._name("ADAPTER"), version_name="V1")
        fqn_alias = f"{adapter.fully_qualified_model_name}/VERSIONS/{adapter.version_name}"
        for alias in ("support", "a/b", "db.schema.model", fqn_alias):
            with self.subTest(alias=alias):
                try:
                    # Charset must not reject. This CPU create may fail later
                    # (G6 schema / image); that is not a charset failure.
                    base.create_service(
                        service_name=self._name("SVC"),
                        service_compute_pool=self._TEST_CPU_COMPUTE_POOL,
                        adapters={alias: adapter},
                    )
                except Exception as exc:
                    self.assertNotRegex(str(exc), r"Adapter alias")

    def test_adapter_create_service_rejects_extra_adapters(self) -> None:
        base = self._log_tiny_with_model_base(model_name=self._name("BASE"), version_name="V1")
        adapter = self._log_stub_adapter(base_mv=base, model_name=self._name("ADAPTER"), version_name="V1")
        extra = self._log_stub_adapter(base_mv=base, model_name=self._name("EXTRA"), version_name="V1")
        with self.assertRaisesRegex(ValueError, r"does not accept the adapters argument"):
            adapter.create_service(
                service_name=self._name("SVC"),
                service_compute_pool=self._TEST_CPU_COMPUTE_POOL,
                adapters={"other": extra},
            )

    def test_garden_base_plus_adapter_attach(self) -> None:
        garden_name = os.getenv("LORA_IT_GARDEN_MODEL")
        garden_version = os.getenv("LORA_IT_GARDEN_VERSION")
        if not garden_name:
            self.skipTest("LORA_IT_GARDEN_MODEL is unset")
        garden_adapter_dir = os.getenv("LORA_IT_GARDEN_ADAPTER_DIR")
        if not garden_adapter_dir:
            self.skipTest("LORA_IT_GARDEN_ADAPTER_DIR is unset")
        garden_reg = registry.Registry(self.session, database_name="SNOWFLAKE", schema_name="MODELS")
        garden_model = garden_reg.get_model(garden_name)
        garden_mv = garden_model.version(garden_version) if garden_version else garden_model.default
        adapter = self.registry.log_model(
            model=PeftAdapter(base_model=garden_mv, adapter_path=garden_adapter_dir),
            model_name=self._name("GARDEN_ADAPTER"),
            version_name="V1",
        )
        service_name = self._name("GARDEN_SVC")
        try:
            svc = self._create_lora_service(
                garden_mv,
                service_name=service_name,
                adapters=[adapter],
                autocapture=False,
            )
        except Exception as err:
            msg = str(err)
            self.assertNotRegex(msg, r"ENABLE_LORA_ADAPTERS|Cannot pin an adapter|Adapter alias")
            self.skipTest(f"Garden SPCS deploy failed (garden gates, not LoRA): {err}")
        fqn = self._service_fqn(svc)
        alias = f"{adapter.fully_qualified_model_name}/VERSIONS/{adapter.version_name}"
        self._assert_sql_chat(self._sql_call(fqn, model=None))
        self._assert_sql_chat(self._sql_call(fqn, model=alias))

    def test_revoke_adapter_read_fails_next_restart(self) -> None:
        admin_role = self.session.get_current_role().strip('"')
        deployer = self._name("DEPLOYER")
        current_user = self.session.get_current_user().strip('"')
        warehouse = self.session.get_current_warehouse()
        self._db_manager.create_role(deployer)
        self.session.sql(f"GRANT ROLE {deployer} TO USER {current_user}").collect()
        self.session.sql(f"GRANT USAGE ON DATABASE {self._test_db} TO ROLE {deployer}").collect()
        self.session.sql(f"GRANT USAGE ON SCHEMA {self._test_db}.{self._test_schema} TO ROLE {deployer}").collect()
        try:
            self.session.sql(f"GRANT USAGE ON WAREHOUSE {warehouse} TO ROLE {deployer}").collect()
        except Exception as err:
            logging.error("Failed to grant warehouse to %s: %s", deployer, err)

        base = self._log_serving_base(model_name=self._name("BASE"), version_name="V1")
        adapter = self._log_serving_adapter(base_mv=base, model_name=self._name("ADAPTER"), version_name="V1")
        adapter_fqn = adapter.fully_qualified_model_name
        base_fqn = base.fully_qualified_model_name
        self.session.sql(f"GRANT READ ON MODEL {base_fqn} TO ROLE {deployer}").collect()
        self.session.sql(f"GRANT READ ON MODEL {adapter_fqn} TO ROLE {deployer}").collect()
        image_repo = f"{self._test_db}.{self._test_schema}.{self._test_image_repo}"
        for grant_sql in (
            f"GRANT USAGE ON COMPUTE POOL {self._TEST_GPU_COMPUTE_POOL} TO ROLE {deployer}",
            f"GRANT CREATE SERVICE ON SCHEMA {self._test_db}.{self._test_schema} TO ROLE {deployer}",
            f"GRANT BIND SERVICE ENDPOINT ON ACCOUNT TO ROLE {deployer}",
            f"GRANT READ, WRITE ON IMAGE REPOSITORY {image_repo} TO ROLE {deployer}",
            f"GRANT SERVICE READ, SERVICE WRITE ON IMAGE REPOSITORY {image_repo} TO ROLE {deployer}",
        ):
            try:
                self.session.sql(grant_sql).collect()
            except Exception as err:
                logging.warning("Grant failed (%s): %s", grant_sql, err)

        service_name = self._name("SVC")
        base_name = base.model_name
        adapter_name = adapter.model_name
        base_version = base.version_name
        adapter_version = adapter.version_name

        def _deploy_and_wait() -> None:
            reg = registry.Registry(self.session)
            pin = reg.get_model(base_name).version(base_version)
            attached = reg.get_model(adapter_name).version(adapter_version)
            pin.create_service(
                service_name=service_name,
                service_compute_pool=self._TEST_GPU_COMPUTE_POOL,
                ingress_enabled=True,
                force_rebuild=True,
                inference_engine_options=self._vllm_engine_options(),
                adapters=[attached],
            )
            self._wait_for_service_status(pin)

        def _restart_after_revoke() -> None:
            self.session.sql(f"ALTER SERVICE {self._service_fqn(service_name)} SUSPEND").collect()
            self.session.sql(f"ALTER SERVICE {self._service_fqn(service_name)} RESUME").collect()
            deadline = time.time() + 1800.0
            last_status = "UNKNOWN"
            while time.time() < deadline:
                last_status = self._sql_service_status(service_name)
                if last_status not in ("RUNNING", "PENDING"):
                    break
                time.sleep(10)
            self.assertNotEqual(
                last_status,
                "RUNNING",
                f"service still RUNNING after adapter READ revoke and restart: {last_status}",
            )

        def _drop_service() -> None:
            self.session.sql(f"DROP SERVICE IF EXISTS {self._service_fqn(service_name)}").collect()

        def _revoke_adapter_read() -> None:
            self.session.sql(f"REVOKE READ ON MODEL {adapter_fqn} FROM ROLE {deployer}").collect()

        try:
            self._run_as_role(deployer, _deploy_and_wait)
            _revoke_adapter_read()
            self._run_as_role(deployer, _restart_after_revoke)
        finally:
            self.session.use_role(admin_role)
            try:
                self._run_as_role(deployer, _drop_service)
            except Exception as err:
                logging.warning("Failed to drop test service as its owner: %s", err)
            try:
                self._db_manager.drop_role(deployer, if_exists=True)
            except Exception as err:
                logging.warning("Failed to drop test deployer role: %s", err)

    def test_shared_base_rejection_unchanged(self) -> None:
        shared_fqn = os.getenv("LORA_IT_SHARED_MODEL")
        if not shared_fqn:
            self.skipTest("LORA_IT_SHARED_MODEL is unset")
        parts = [part.strip('"') for part in shared_fqn.split(".")]
        if len(parts) == 3:
            shared_reg = registry.Registry(self.session, database_name=parts[0], schema_name=parts[1])
            shared_model = shared_reg.get_model(parts[2])
        elif len(parts) == 1:
            shared_model = self.registry.get_model(parts[0])
        else:
            self.fail(f"LORA_IT_SHARED_MODEL must be MODEL or DB.SCHEMA.MODEL, got {shared_fqn!r}")
        version = os.getenv("LORA_IT_SHARED_VERSION")
        shared_mv = shared_model.version(version) if version else shared_model.default
        adapter = self._log_stub_adapter(base_mv=shared_mv, model_name=self._name("SHARED_ADAPTER"), version_name="V1")
        with self.assertRaisesRegex(Exception, r"MODEL_SPCS_DEPLOY_SHARED_MODEL_NOT_SUPPORTED"):
            shared_mv.create_service(
                service_name=self._name("SHARED_SVC"),
                service_compute_pool=self._TEST_GPU_COMPUTE_POOL,
                adapters={"support": adapter},
            )

    def test_pre_with_model_base_deploys_without_adapters(self) -> None:
        # CPU python inference, same shape as the huggingface CPU neighbor. GPU
        # + cuda_version races the LoRA vLLM shards for pool capacity.
        base = self._log_tiny_with_model_base(model_name=self._name("BASE"), version_name="V1")
        service_name = self._name("SVC")
        base.create_service(
            service_name=service_name,
            service_compute_pool=self._TEST_CPU_COMPUTE_POOL,
        )
        self._wait_for_service_status(base)
        self._assert_base_only_spec(service_name)

    def test_base_drop_dangles_then_attach_fails(self) -> None:
        base_name = self._name("BASE")
        adapter_name = self._name("ADAPTER")
        base = self._log_tiny_with_model_base(model_name=base_name, version_name="V1")
        adapter = self._log_stub_adapter(base_mv=base, model_name=adapter_name, version_name="V1")
        self.registry.delete_model(base_name)
        # Catalog still lists the adapter. version() resolves signatures from the
        # pin, so it fail-closes once the base is gone; SHOW VERSIONS does not.
        dangling = self.registry.get_model(adapter_name)
        self.assertFalse(dangling.show_versions().empty)
        other = self._log_tiny_with_model_base(model_name=self._name("OTHER"), version_name="V1")
        with self.assertRaisesRegex(Exception, r"(?i)pin|exist|not found|mismatch|adapter|lineage"):
            other.create_service(
                service_name=self._name("DANGLE_SVC"),
                service_compute_pool=self._TEST_CPU_COMPUTE_POOL,
                adapters={"support": adapter},
            )

    def test_adapter_drop_while_attached_succeeds(self) -> None:
        base = self._log_serving_base(model_name=self._name("BASE"), version_name="V1")
        adapter_name = self._name("ADAPTER")
        adapter = self._log_serving_adapter(base_mv=base, model_name=adapter_name, version_name="V1")
        svc = self._create_lora_service(
            base,
            service_name=self._name("SVC"),
            adapters=[adapter],
        )
        self.registry.delete_model(adapter_name)
        self.assertEqual(self._service_status(base), "RUNNING")
        base.delete_service(svc)

    def test_adapter_list_services_includes_attached_service(self) -> None:
        base = self._log_serving_base(model_name=self._name("BASE"), version_name="V1")
        adapter = self._log_serving_adapter(base_mv=base, model_name=self._name("ADAPTER"), version_name="V1")
        svc = self._create_lora_service(
            base,
            service_name=self._name("SVC"),
            adapters=[adapter],
        )
        self._assert_lists_service(adapter, svc)
        adapter.delete_service(svc)

    def test_flag_off_refuses_nonempty_adapters(self) -> None:
        base = self._log_tiny_with_model_base(model_name=self._name("BASE"), version_name="V1")
        adapter = self._log_stub_adapter(base_mv=base, model_name=self._name("ADAPTER"), version_name="V1")
        with mock.patch.object(
            platform_capabilities.PlatformCapabilities,
            "is_lora_adapters_enabled",
            return_value=False,
        ):
            with self.assertRaisesRegex(ValueError, r"ENABLE_LORA_ADAPTERS"):
                base.create_service(
                    service_name=self._name("SVC"),
                    service_compute_pool=self._TEST_CPU_COMPUTE_POOL,
                    adapters={"support": adapter},
                )
        self.assertTrue(base.list_services().empty)

    def test_flag_off_refuses_adapter_create_service(self) -> None:
        base = self._log_tiny_with_model_base(model_name=self._name("BASE"), version_name="V1")
        adapter = self._log_stub_adapter(base_mv=base, model_name=self._name("ADAPTER"), version_name="V1")
        with mock.patch.object(
            platform_capabilities.PlatformCapabilities,
            "is_lora_adapters_enabled",
            return_value=False,
        ):
            with self.assertRaisesRegex(ValueError, r"ENABLE_LORA_ADAPTERS"):
                adapter.create_service(
                    service_name=self._name("SVC"),
                    service_compute_pool=self._TEST_CPU_COMPUTE_POOL,
                )

    @mock.patch.object(
        platform_capabilities.PlatformCapabilities,
        "is_lora_adapters_enabled",
        return_value=False,
    )
    def test_flag_off_non_lora_create_and_drop_unchanged(self, _mock_enabled: mock.MagicMock) -> None:
        iris_x, iris_y = datasets.load_iris(return_X_y=True)
        classifier = linear_model.LogisticRegression()
        classifier.fit(iris_x, iris_y)
        mv = self._test_registry_model_deployment(
            model=classifier,
            sample_input_data=iris_x,
            prediction_assert_fns={
                "predict": (
                    iris_x,
                    lambda res: self.assertEqual(len(res), len(iris_x)),
                ),
            },
            options={"enable_explainability": False},
        )
        svc = mv.list_services().loc[0, "name"]
        mv.delete_service(svc)
        self.assertTrue(mv.list_services().empty)
        self.registry.delete_model(mv.model_name)


if __name__ == "__main__":
    absltest.main()
