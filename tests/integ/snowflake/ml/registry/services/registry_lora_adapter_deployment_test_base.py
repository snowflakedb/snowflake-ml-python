import json
import logging
import os
import tempfile
import time
from typing import Any
from unittest import mock

# Bazel sandboxes mount $HOME read-only. huggingface_hub resolves HF_HOME at
# import time, so these must be set before snowml huggingface modules are imported.
_ORIGINAL_TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE")
_ORIGINAL_HF_HOME = os.getenv("HF_HOME")
_ORIGINAL_HF_HUB_CACHE = os.getenv("HF_HUB_CACHE")
_HF_CACHE_DIR = tempfile.TemporaryDirectory()
os.environ["HF_HOME"] = _HF_CACHE_DIR.name
os.environ["HF_HUB_CACHE"] = os.path.join(_HF_CACHE_DIR.name, "hub")
os.environ["TRANSFORMERS_CACHE"] = _HF_CACHE_DIR.name

import pandas as pd  # noqa: E402
import requests  # noqa: E402
import yaml  # noqa: E402

from snowflake.ml._internal import platform_capabilities  # noqa: E402
from snowflake.ml.model import (  # noqa: E402
    ModelVersion,
    PeftAdapter,
    openai_signatures,
)
from snowflake.ml.model._packager.model_env import model_env  # noqa: E402
from snowflake.ml.model.inference_engine import InferenceEngine  # noqa: E402
from snowflake.ml.model.models import huggingface_pipeline  # noqa: E402
from tests.integ.snowflake.ml.registry.services import (  # noqa: E402
    registry_model_deployment_test_base,
)
from tests.integ.snowflake.ml.test_utils import (  # noqa: E402
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


def write_stub_adapter_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 8}, f)
    with open(os.path.join(path, "adapter_model.safetensors"), "wb") as f:
        f.write(b"")
    return path


class RegistryLoraAdapterDeploymentTestBase(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    """Helpers for LoRA adapter SPCS serving integration tests."""

    cache_dir: tempfile.TemporaryDirectory
    _original_hf_accept_encoding: str | None = None
    _original_hf_endpoint: str | None = None

    @classmethod
    def setUpClass(cls) -> None:
        import huggingface_hub

        super().setUpClass()
        cls.cache_dir = _HF_CACHE_DIR
        os.environ["TRANSFORMERS_CACHE"] = cls.cache_dir.name
        os.environ["HF_HOME"] = cls.cache_dir.name
        os.environ["HF_HUB_CACHE"] = os.path.join(cls.cache_dir.name, "hub")
        if "HF_ENDPOINT" in os.environ:
            cls._original_hf_endpoint = os.environ["HF_ENDPOINT"]
            del os.environ["HF_ENDPOINT"]
        hf_session = huggingface_hub.get_session()
        cls._original_hf_accept_encoding = hf_session.headers.get("Accept-Encoding")
        hf_session.headers["Accept-Encoding"] = "identity"

    @classmethod
    def tearDownClass(cls) -> None:
        import huggingface_hub

        if _ORIGINAL_TRANSFORMERS_CACHE is not None:
            os.environ["TRANSFORMERS_CACHE"] = _ORIGINAL_TRANSFORMERS_CACHE
        else:
            os.environ.pop("TRANSFORMERS_CACHE", None)
        if _ORIGINAL_HF_HOME is not None:
            os.environ["HF_HOME"] = _ORIGINAL_HF_HOME
        else:
            os.environ.pop("HF_HOME", None)
        if _ORIGINAL_HF_HUB_CACHE is not None:
            os.environ["HF_HUB_CACHE"] = _ORIGINAL_HF_HUB_CACHE
        else:
            os.environ.pop("HF_HUB_CACHE", None)
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

    def _served_name(self, mv: ModelVersion) -> str:
        return f"{mv.fully_qualified_model_name}/VERSIONS/{mv.version_name}"

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
                model=PeftAdapter(base_model=base_mv, adapter_path=write_stub_adapter_dir(tmp)),
                model_name=model_name,
                version_name=version_name,
            )

    def _vllm_engine_options(self, *, extra_args: list[str] | None = None) -> dict[str, Any]:
        args = ["--max-model-len=1024"]
        if extra_args:
            args.extend(extra_args)
        options = self._get_inference_engine_options_for_inference_engine(
            InferenceEngine.VLLM,
            {"engine_args_override": args},
        )
        assert options is not None
        return options

    def _service_owner(self, mv: ModelVersion) -> ModelVersion:
        if mv._is_peft_adapter_version():
            return mv._resolve_adapter_pin()
        return mv

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

    def _drop_lora_service(self, service_name: str) -> None:
        try:
            self.session.sql(f"DROP SERVICE IF EXISTS {self._service_fqn(service_name)}").collect()
        except Exception as err:
            logging.warning("Failed to drop service %s: %s", service_name, err)

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
