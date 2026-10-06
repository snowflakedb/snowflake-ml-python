import http
import json
import logging
from typing import Any

import requests
import retrying
from absl.testing import absltest

from snowflake.ml.model import ModelVersion
from tests.integ.snowflake.ml.registry.services import (
    registry_lora_adapter_deployment_test_base,
    registry_model_deployment_test_base,
)

_SHARED_ENV_ATTRS = (
    "session",
    "private_key",
    "pat_token",
    "snowflake_account_url",
    "_run_id",
    "_test_db",
    "_test_schema",
    "_test_image_repo",
    "_test_stage",
    "_db_manager",
    "registry",
    "_base_mv",
    "_support_mv",
    "_sql_gen_mv",
    "_service_name",
)


def _retry_if_list_models_not_ready(result: requests.Response) -> bool:
    if registry_model_deployment_test_base.RegistryModelDeploymentTestBase.retry_if_result_status_retriable(result):
        return True
    return result.status_code in (
        http.HTTPStatus.BAD_GATEWAY,
        http.HTTPStatus.NOT_FOUND,
    )


class TestRegistryLoraAdapterSharedServingInteg(
    registry_lora_adapter_deployment_test_base.RegistryLoraAdapterDeploymentTestBase
):
    """LoRA serving checks against one class-level aliased vLLM service."""

    _shared_ready: bool = False
    _deploy_failed: bool = False
    _base_mv: ModelVersion | None = None
    _support_mv: ModelVersion | None = None
    _sql_gen_mv: ModelVersion | None = None
    _service_name: str | None = None

    def setUp(self) -> None:
        cls = type(self)
        if cls._shared_ready:
            self._restore_shared_env()
            return
        if cls._deploy_failed:
            self.skipTest("Shared LoRA service deployment failed in a previous setUp.")
        super().setUp()
        self._stash_shared_env()
        try:
            self._deploy_shared_service()
            self._stash_shared_env()
            cls._shared_ready = True
        except Exception:
            cls._deploy_failed = True
            raise

    def tearDown(self) -> None:
        if type(self)._shared_ready:
            return
        if getattr(self, "session", None) is None:
            return
        super().tearDown()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._shared_ready:
            db_manager = getattr(cls, "_db_manager", None)
            test_db = getattr(cls, "_test_db", None)
            if db_manager is not None and test_db:
                try:
                    db_manager.drop_database(test_db)
                except Exception:
                    logging.exception("Failed to drop shared LoRA test database")
            session = getattr(cls, "session", None)
            if session is not None:
                try:
                    session.close()
                except Exception:
                    logging.exception("Failed to close shared LoRA test session")
        super().tearDownClass()

    def _stash_shared_env(self) -> None:
        cls = type(self)
        for attr in _SHARED_ENV_ATTRS:
            if hasattr(self, attr):
                setattr(cls, attr, getattr(self, attr))

    def _restore_shared_env(self) -> None:
        cls = type(self)
        for attr in _SHARED_ENV_ATTRS:
            setattr(self, attr, getattr(cls, attr))

    def _deploy_shared_service(self) -> None:
        base = self._log_serving_base(model_name=self._name("BASE"), version_name="V1")
        support = self._log_serving_adapter(base_mv=base, model_name=self._name("SUPPORT"), version_name="V1")
        sql_gen = self._log_serving_adapter(base_mv=base, model_name=self._name("SQL_GEN"), version_name="V1")
        svc = self._create_lora_service(
            base,
            service_name=self._name("SVC"),
            adapters={"support": support, "sql_gen": sql_gen},
        )
        self._base_mv = base
        self._support_mv = support
        self._sql_gen_mv = sql_gen
        self._service_name = svc

    def _require_shared_service(self) -> tuple[ModelVersion, ModelVersion, ModelVersion, str]:
        assert self._base_mv is not None
        assert self._support_mv is not None
        assert self._sql_gen_mv is not None
        assert self._service_name is not None
        return self._base_mv, self._support_mv, self._sql_gen_mv, self._service_name

    def test_explicit_aliases_serve_adapter(self) -> None:
        base, _support, _sql_gen, svc = self._require_shared_service()
        self._assert_spec_adapters_list(svc, expected_aliases=["support", "sql_gen"])
        fqn = self._service_fqn(svc)
        self._assert_sql_chat(self._sql_call(fqn, model="support"))
        rest = self._rest_call(base, params={"model": "sql_gen"})
        self._assert_chat_res(rest)
        self._assert_sql_chat(self._sql_call(fqn, model=None))
        self._assert_chat_res(self._rest_call(base))

    def test_default_served_name_selects_adapter(self) -> None:
        _base, support, _sql_gen, svc = self._require_shared_service()
        fqn = self._service_fqn(svc)
        alias = self._served_name(support)
        self._assert_sql_chat(self._sql_call(fqn, model=alias))
        self._assert_sql_chat(self._sql_call(fqn, model=None))

    def test_sql_per_row_model_selects_adapter(self) -> None:
        _base, _support, _sql_gen, svc = self._require_shared_service()
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

    def test_unknown_model_passes_through(self) -> None:
        base, _support, _sql_gen, svc = self._require_shared_service()
        unknown = "not_a_key"
        with self.assertRaisesRegex(Exception, unknown):
            self._sql_call(self._service_fqn(svc), model=unknown)

        rest = self._rest_call(base, params={"model": unknown}, expect_ok=False)
        self.assertIn(unknown, rest.text)
        self.assertNotIn("support", rest.text)
        self.assertNotIn("sql_gen", rest.text)

    def test_adapter_run_service_name(self) -> None:
        _base, support, _sql_gen, svc = self._require_shared_service()
        res = support.run(self._chat_df(), function_name="__call__", service_name=svc)
        self._assert_chat_res(res)

    def test_adapter_list_services_includes_attached_service(self) -> None:
        base, support, _sql_gen, svc = self._require_shared_service()
        self._assert_lists_service(support, svc)
        self._assert_lists_service(base, svc)

    def test_list_models_includes_base_alias_and_default_served_name(self) -> None:
        base, support, sql_gen, _svc = self._require_shared_service()
        payload = self._list_models_payload(base)
        self.assertEqual(payload["object"], "list")
        cards_by_id = {card["id"]: card for card in payload["data"]}
        ids = list(cards_by_id)
        self.assertIn(base.model_name, ids)
        self.assertNotIn(base.fully_qualified_model_name, ids)
        support_served = self._served_name(support)
        sql_gen_served = self._served_name(sql_gen)
        self.assertIn(support_served, ids)
        self.assertIn("support", ids)
        self.assertIn(sql_gen_served, ids)
        self.assertIn("sql_gen", ids)
        self.assertNotIn(support.model_name, ids)
        self.assertNotIn(sql_gen.model_name, ids)

        support_alias_card = {key: value for key, value in cards_by_id["support"].items() if key != "id"}
        support_served_card = {key: value for key, value in cards_by_id[support_served].items() if key != "id"}
        self.assertEqual(support_alias_card, support_served_card)
        sql_gen_alias_card = {key: value for key, value in cards_by_id["sql_gen"].items() if key != "id"}
        sql_gen_served_card = {key: value for key, value in cards_by_id[sql_gen_served].items() if key != "id"}
        self.assertEqual(sql_gen_alias_card, sql_gen_served_card)

    def test_z_adapter_drop_while_attached(self) -> None:
        # absltest runs methods alphabetically; this must follow the read-only
        # cases that still need both adapters on the shared service.
        base, _support, sql_gen, _svc = self._require_shared_service()
        self.registry.delete_model(sql_gen.model_name)
        self.assertEqual(self._service_status(base), "RUNNING")

    def _list_models_payload(self, base: ModelVersion) -> dict[str, Any]:
        endpoint = self._ensure_ingress_url(base)
        auth_handler = self._get_auth_for_inference(endpoint)
        response = retrying.retry(
            stop_max_attempt_number=20,
            stop_max_delay=120000,
            wait_exponential_multiplier=100,
            wait_exponential_max=4000,
            retry_on_result=_retry_if_list_models_not_ready,
        )(requests.get)(
            f"https://{endpoint}/v1/models",
            auth=auth_handler,
        )
        response.raise_for_status()
        payload = response.json()
        self.assertIsInstance(payload, dict)
        return payload


if __name__ == "__main__":
    absltest.main()
