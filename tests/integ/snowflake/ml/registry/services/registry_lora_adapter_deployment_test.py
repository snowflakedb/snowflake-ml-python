import logging
import os
import time
from unittest import mock

from absl.testing import absltest
from sklearn import datasets, linear_model

from snowflake.ml._internal import platform_capabilities
from snowflake.ml.model import PeftAdapter
from snowflake.ml.registry import registry
from tests.integ.snowflake.ml.registry.services import (
    registry_lora_adapter_deployment_test_base,
)


class TestRegistryLoraAdapterDeploymentInteg(
    registry_lora_adapter_deployment_test_base.RegistryLoraAdapterDeploymentTestBase
):
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
        try:
            fqn = self._service_fqn(svc)
            alias_a = self._served_name(adapter_a)
            alias_b = self._served_name(adapter_b)
            self._assert_sql_chat(self._sql_call(fqn, model=alias_a))
            self._assert_sql_chat(self._sql_call(fqn, model=alias_b))
            self._assert_chat_res(
                base.run(self._chat_df(), function_name="__call__", service_name=svc, params={"model": alias_a})
            )
        finally:
            self._drop_lora_service(svc)

    def test_mixed_version_create_service_on_each(self) -> None:
        model_name = self._name("MIXED")
        v1 = self._log_serving_base(model_name=model_name, version_name="V1")
        v2 = self._log_serving_adapter(base_mv=v1, model_name=model_name, version_name="V2")
        svc_v1 = None
        svc_v2 = None
        try:
            svc_v1 = self._create_lora_service(v1, service_name=self._name("SVC_V1"))
            self.assertEqual(self._show_model_type(model_name), "USER_MODEL")
            self._assert_base_only_spec(svc_v1)
            res = v1.run(self._chat_df(), function_name="__call__", service_name=svc_v1)
            self._assert_chat_res(res)
            self._assert_sql_chat(self._sql_call(self._service_fqn(svc_v1), model=None))
            self._assert_chat_res(self._rest_call(v1))
            self._drop_lora_service(svc_v1)
            svc_v1 = None

            svc_v2 = self._create_lora_service(v2, service_name=self._name("SVC_V2"))
            alias = self._served_name(v2)
            self._assert_spec_adapters_list(svc_v2, expected_aliases=[alias])
            self._assert_sql_chat(self._sql_call(self._service_fqn(svc_v2), model=alias))
            self._assert_sql_chat(self._sql_call(self._service_fqn(svc_v2), model=None))
            self._assert_lists_service(v1, svc_v2)
        finally:
            if svc_v2 is not None:
                self._drop_lora_service(svc_v2)
            if svc_v1 is not None:
                self._drop_lora_service(svc_v1)

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
        fqn_alias = self._served_name(adapter)
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
        svc = None
        try:
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
            alias = self._served_name(adapter)
            self._assert_sql_chat(self._sql_call(fqn, model=None))
            self._assert_sql_chat(self._sql_call(fqn, model=alias))
        finally:
            if svc is not None:
                self._drop_lora_service(svc)

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
        # Session image overrides resolve to a shared account image repository.
        # create_service as deployer still uses those URLs, not the per-test repo.
        for override_repo in self._override_image_repository_fqns():
            database_name, schema_name, _repo_name = override_repo.split(".")
            self.session.sql(f"GRANT USAGE ON DATABASE {database_name} TO ROLE {deployer}").collect()
            self.session.sql(f"GRANT USAGE ON SCHEMA {database_name}.{schema_name} TO ROLE {deployer}").collect()
            self.session.sql(f"GRANT READ ON IMAGE REPOSITORY {override_repo} TO ROLE {deployer}").collect()
            self.session.sql(f"GRANT SERVICE READ ON IMAGE REPOSITORY {override_repo} TO ROLE {deployer}").collect()
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

        def _revoke_adapter_read() -> None:
            self.session.sql(f"REVOKE READ ON MODEL {adapter_fqn} FROM ROLE {deployer}").collect()

        try:
            self._run_as_role(deployer, _deploy_and_wait)
            _revoke_adapter_read()
            self._run_as_role(deployer, _restart_after_revoke)
        finally:
            self.session.use_role(admin_role)
            try:
                self._run_as_role(deployer, lambda: self._drop_lora_service(service_name))
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
