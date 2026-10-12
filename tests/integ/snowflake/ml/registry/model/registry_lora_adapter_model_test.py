import json
import logging
import os
import tempfile
from typing import Callable, TypeVar

from absl.testing import absltest

from snowflake.ml.model import (
    ModelVersion,
    PeftAdapter,
    model_signature,
    openai_signatures,
)
from snowflake.ml.registry import registry
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import (
    db_manager,
    lora_adapter_account_gate,
    lora_adapters_enabled_patch,
)

lora_adapters_enabled_patch.enable()

_TINY_GPT2 = "hf-internal-testing/tiny-gpt2-with-chatml-template"

T = TypeVar("T")


def _write_stub_adapter_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 8}, f)
    with open(os.path.join(path, "adapter_model.safetensors"), "wb") as f:
        f.write(b"")
    return path


class TestRegistryLoraAdapterModelInteg(registry_model_test_base.RegistryModelTestBase):
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

    def _run_as_role(self, role: str, fn: Callable[[], T]) -> T:
        prev_role = self.session.get_current_role()
        try:
            self.session.sql("USE SECONDARY ROLES NONE").collect()
            self.session.use_role(role)
            return fn()
        finally:
            self.session.use_role(prev_role)
            self.session.sql("USE SECONDARY ROLES ALL").collect()

    def _log_customer_base(
        self,
        *,
        model_name: str,
        version_name: str,
        signatures: dict[str, model_signature.ModelSignature] | None = None,
    ) -> ModelVersion:
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
            signatures=signatures or openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE,
        )

    def _log_adapter(
        self,
        *,
        base_mv: ModelVersion,
        model_name: str,
        version_name: str,
        adapter_dir: str,
    ) -> ModelVersion:
        return self.registry.log_model(
            model=PeftAdapter(base_model=base_mv, adapter_path=adapter_dir),
            model_name=model_name,
            version_name=version_name,
        )

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

    def _chat_df_messages(self) -> dict[str, object]:
        return {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            ]
        }

    def test_log_peft_adapter_gs_catalog_signature_parity(self) -> None:
        """Post-commit GS catalog materialization from a signature-empty PeftAdapter upload (G4).

        C1 stages no methods/signatures; after commit, adapter ``show_functions()`` must match the
        pin on both ``target_method`` and the full ``ModelSignature`` object for each method.
        """
        base_mv = self._log_customer_base(model_name=self._name("BASE"), version_name="V1")
        with tempfile.TemporaryDirectory() as adapter_dir:
            adapter_mv = self._log_adapter(
                base_mv=base_mv,
                model_name=self._name("ADAPTER"),
                version_name="V1",
                adapter_dir=_write_stub_adapter_dir(adapter_dir),
            )
        base_by_method = {fn["target_method"]: fn["signature"] for fn in base_mv.show_functions()}
        adapter_by_method = {fn["target_method"]: fn["signature"] for fn in adapter_mv.show_functions()}
        self.assertEqual(adapter_by_method, base_by_method)

    def test_adapter_as_base_rejected(self) -> None:
        base_mv = self._log_customer_base(model_name=self._name("BASE"), version_name="V1")
        with tempfile.TemporaryDirectory() as first_dir:
            adapter_mv = self._log_adapter(
                base_mv=base_mv,
                model_name=self._name("ADAPTER"),
                version_name="V1",
                adapter_dir=_write_stub_adapter_dir(first_dir),
            )
        with tempfile.TemporaryDirectory() as nested_dir:
            with self.assertRaisesRegex(Exception, r"(?i)adapter|pin|398542|base_model"):
                self.registry.log_model(
                    model=PeftAdapter(
                        base_model=adapter_mv,
                        adapter_path=_write_stub_adapter_dir(nested_dir),
                    ),
                    model_name=self._name("NESTED"),
                    version_name="V1",
                )

    def test_pre_with_model_pin_rejected_at_log(self) -> None:
        """G4 commit rejects a pin whose ``__CALL__`` has no ``model`` input ParamSpec.

        C1 no longer fail-fasts this in the packager; the log must still fail at commit.
        """
        base_mv = self._log_customer_base(
            model_name=self._name("PRE"),
            version_name="V1",
            signatures=openai_signatures.OPENAI_CHAT_SIGNATURE,
        )
        with tempfile.TemporaryDirectory() as adapter_dir:
            with self.assertRaisesRegex(Exception, r"(?i)model.?input|398548|signature|WITH_PARAMS|pin"):
                self._log_adapter(
                    base_mv=base_mv,
                    model_name=self._name("ADAPTER"),
                    version_name="V1",
                    adapter_dir=_write_stub_adapter_dir(adapter_dir),
                )

    def test_lineage_base_to_adapter_after_log(self) -> None:
        base_mv = self._log_customer_base(model_name=self._name("BASE"), version_name="V1")
        with tempfile.TemporaryDirectory() as adapter_dir:
            adapter_mv = self._log_adapter(
                base_mv=base_mv,
                model_name=self._name("ADAPTER"),
                version_name="V1",
                adapter_dir=_write_stub_adapter_dir(adapter_dir),
            )
        upstream = adapter_mv.lineage(direction="upstream", domain_filter={"model"})
        self.assertLen(upstream, 1)
        self.assertIsInstance(upstream[0], ModelVersion)
        self.assertEqual(upstream[0].model_name, base_mv.model_name)
        self.assertEqual(upstream[0].version_name, base_mv.version_name)
        adapters = base_mv.get_adapters()
        adapter_ids = {(a.model_name, a.version_name) for a in adapters}
        self.assertIn((adapter_mv.model_name, adapter_mv.version_name), adapter_ids)

    @absltest.skip("SNOW-4157123: GS MASKED MODEL lineage neighbor aborts DGQL with 518001")
    def test_get_adapters_omits_masked(self) -> None:
        admin_role = self.session.get_current_role().strip('"')
        usage_role = self._name("USAGE_ROLE")
        self._db_manager.create_role(usage_role)
        current_user = self.session.get_current_user().strip('"')
        self.session.sql(f"GRANT ROLE {usage_role} TO USER {current_user}").collect()
        self.session.sql(f"GRANT VIEW LINEAGE ON ACCOUNT TO ROLE {usage_role}").collect()
        self.session.sql(f"GRANT USAGE ON DATABASE {self._test_db} TO ROLE {usage_role}").collect()
        self.session.sql(f"GRANT USAGE ON SCHEMA {self._test_db}.{self._test_schema} TO ROLE {usage_role}").collect()
        warehouse = self.session.get_current_warehouse()
        try:
            self.session.sql(f"GRANT USAGE ON WAREHOUSE {warehouse} TO ROLE {usage_role}").collect()
        except Exception:
            logging.error("Failed to grant warehouse %s to role %s", warehouse, usage_role)

        base_name = self._name("BASE")
        adapter_a_name = self._name("ADAPTER_A")
        adapter_b_name = self._name("ADAPTER_B")
        base_mv = self._log_customer_base(model_name=base_name, version_name="V1")
        with tempfile.TemporaryDirectory() as dir_a:
            adapter_a = self._log_adapter(
                base_mv=base_mv,
                model_name=adapter_a_name,
                version_name="V1",
                adapter_dir=_write_stub_adapter_dir(dir_a),
            )
        with tempfile.TemporaryDirectory() as dir_b:
            self._log_adapter(
                base_mv=base_mv,
                model_name=adapter_b_name,
                version_name="V1",
                adapter_dir=_write_stub_adapter_dir(dir_b),
            )

        self.session.sql(f"GRANT USAGE ON MODEL {base_mv.fully_qualified_model_name} TO ROLE {usage_role}").collect()
        self.session.sql(f"GRANT READ ON MODEL {base_mv.fully_qualified_model_name} TO ROLE {usage_role}").collect()
        self.session.sql(f"GRANT USAGE ON MODEL {adapter_a.fully_qualified_model_name} TO ROLE {usage_role}").collect()
        self.session.sql(f"GRANT READ ON MODEL {adapter_a.fully_qualified_model_name} TO ROLE {usage_role}").collect()

        def _as_limited_role() -> None:
            reg = registry.Registry(self.session)
            visible = reg.get_model(base_name).version("V1").get_adapters()
            self.assertLen(visible, 1)
            self.assertEqual(visible[0].model_name, adapter_a.model_name)
            self.assertEqual(visible[0].version_name, adapter_a.version_name)
            for node in visible:
                self.assertIsInstance(node, ModelVersion)
                self.assertNotIn("***", node.model_name)
                self.assertNotEqual(node.model_name.upper(), "MASKED")

        try:
            self._run_as_role(usage_role, _as_limited_role)
        finally:
            self.session.use_role(admin_role)
            self._db_manager.drop_role(usage_role, if_exists=True)

    def test_adapter_warehouse_run_and_load_rejected(self) -> None:
        import pandas as pd

        base_mv = self._log_customer_base(model_name=self._name("BASE"), version_name="V1")
        with tempfile.TemporaryDirectory() as adapter_dir:
            adapter_mv = self._log_adapter(
                base_mv=base_mv,
                model_name=self._name("ADAPTER"),
                version_name="V1",
                adapter_dir=_write_stub_adapter_dir(adapter_dir),
            )
        x_df = pd.DataFrame.from_records([self._chat_df_messages()])
        with self.assertRaisesRegex(Exception, r"(?i)warehouse|adapter|398538|cannot be executed"):
            adapter_mv.run(x_df, function_name="__call__")
        with self.assertRaisesRegex(Exception, r"cannot be loaded"):
            adapter_mv.load()

    def test_show_models_model_type_stays_user_model(self) -> None:
        base_name = self._name("BASE")
        adapter_name = self._name("ADAPTER")
        base_mv = self._log_customer_base(model_name=base_name, version_name="V1")
        with tempfile.TemporaryDirectory() as adapter_dir:
            self._log_adapter(
                base_mv=base_mv,
                model_name=adapter_name,
                version_name="V1",
                adapter_dir=_write_stub_adapter_dir(adapter_dir),
            )
        self.assertEqual(self._show_model_type(base_name), "USER_MODEL")
        self.assertEqual(self._show_model_type(adapter_name), "USER_MODEL")


if __name__ == "__main__":
    absltest.main()
