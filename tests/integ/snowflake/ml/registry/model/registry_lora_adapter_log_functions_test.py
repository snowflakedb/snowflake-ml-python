import json
import os
import tempfile
from unittest import mock

from absl.testing import absltest

from snowflake.ml._internal import platform_capabilities
from snowflake.ml.model import PeftAdapter, openai_signatures, target_platform
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import db_manager, lora_adapter_account_gate

_TINY_GPT2 = "hf-internal-testing/tiny-gpt2-with-chatml-template"

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


class TestRegistryLoraAdapterLogFunctionsInteg(registry_model_test_base.RegistryModelTestBase):
    cache_dir: tempfile.TemporaryDirectory
    _original_cache_dir: str | None = None
    _original_hf_home: str | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls._original_cache_dir = os.getenv("TRANSFORMERS_CACHE", None)
        cls._original_hf_home = os.getenv("HF_HOME", None)
        os.environ["TRANSFORMERS_CACHE"] = cls.cache_dir.name
        os.environ["HF_HOME"] = cls.cache_dir.name

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._original_cache_dir:
            os.environ["TRANSFORMERS_CACHE"] = cls._original_cache_dir
        else:
            os.environ.pop("TRANSFORMERS_CACHE", None)
        if cls._original_hf_home:
            os.environ["HF_HOME"] = cls._original_hf_home
        else:
            os.environ.pop("HF_HOME", None)
        cls.cache_dir.cleanup()

    def setUp(self) -> None:
        super().setUp()
        lora_adapter_account_gate.skip_unless_lora_adapters_account_enabled(self.session)

    def _name(self, suffix: str) -> str:
        return db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self._run_id, suffix).upper()

    def test_log_peft_adapter_show_functions_matches_pin(self) -> None:
        """log_model of a signature-empty PeftAdapter must return, and show_functions must match the pin.

        G4 copies pin methods into the catalog while C1 stages empty spec signatures. The client
        resolves signatures from the lineage pin (blob options only if that edge is not visible
        yet) so ModelVersion construction does not KeyError on __CALL__. Reloading through
        registry.get_model(...).version(...) uses the same path.
        """
        import transformers

        # SPCS-only: this case only needs G4 catalog methods after commit, not a warehouse embed of
        # the local snowflake-ml-python library.
        spcs_only = [target_platform.TargetPlatform.SNOWPARK_CONTAINER_SERVICES]
        base_mv = self.registry.log_model(
            model=transformers.pipeline(
                task="text-generation",
                model=_TINY_GPT2,
                max_length=200,
            ),
            model_name=self._name("BASE"),
            version_name="V1",
            signatures=openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE,
            target_platforms=spcs_only,
        )
        with tempfile.TemporaryDirectory() as adapter_dir:
            adapter_mv = self.registry.log_model(
                model=PeftAdapter(
                    base_model=base_mv,
                    adapter_path=_write_stub_adapter_dir(adapter_dir),
                ),
                model_name=self._name("ADAPTER"),
                version_name="V1",
                target_platforms=spcs_only,
            )
        base_by_method = {fn["target_method"]: fn["signature"] for fn in base_mv.show_functions()}
        adapter_by_method = {fn["target_method"]: fn["signature"] for fn in adapter_mv.show_functions()}
        self.assertEqual(adapter_by_method, base_by_method)
        reloaded = self.registry.get_model(adapter_mv.model_name).version(adapter_mv.version_name)
        reloaded_by_method = {fn["target_method"]: fn["signature"] for fn in reloaded.show_functions()}
        self.assertEqual(reloaded_by_method, base_by_method)


if __name__ == "__main__":
    absltest.main()
