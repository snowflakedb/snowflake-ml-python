"""Manual e2e test for ModelVersion.create_service(feature_sources_per_function=...)."""

import logging
import os
import time
import unittest
from typing import Optional

import pandas as pd
from absl.testing import absltest

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_store import CreationMode, FeatureStore
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    OnlineConfig,
    OnlineStoreType,
    StoreType,
)
from snowflake.ml.model import custom_model
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)

logger = logging.getLogger(__name__)


def _wait_online_service_running(fs: FeatureStore, *, timeout_s: float = 900.0, poll_interval_s: float = 30.0) -> None:
    """Poll ``fs.get_online_service_status()`` until RUNNING with a query endpoint."""
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        st = fs.get_online_service_status()
        last = st.status
        if st.status == "RUNNING" and any(ep.name == "query" for ep in st.endpoints):
            logger.info("OnlineService RUNNING+query.")
            return
        if st.status == "ERROR":
            raise RuntimeError(f"Online Service entered ERROR state (message={st.message!r}).")
        time.sleep(poll_interval_s)
    raise RuntimeError(f"Online Service did not reach RUNNING+query within {timeout_s:.0f}s (last={last!r}).")


class _PassthroughAmountModel(custom_model.CustomModel):
    """Echoes input AMOUNT unchanged so the FR round-trip assertion can be bit-exact."""

    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"AMOUNT_ECHOED": input_df["AMOUNT"].astype(float)})


@unittest.skipUnless(
    os.environ.get("SNOWFLAKE_PAT", "").strip(),
    "SNOWFLAKE_PAT must be set for OFT vnext / Postgres Online Service Query API.",
)
class TestRegistryFeatureRetrievalDeploymentInteg(registry_model_deployment_test_base.RegistryModelDeploymentTestBase):
    """E2E: registered online FV + create_service(feature_sources_per_function=...) + REST inference."""

    _ENTITY_KEY = "USER_ID"
    _FEATURE_COL = "AMOUNT"
    _TIMESTAMP_COL = "EVENT_TIME"
    _USER_ID_VALUE = "U_FR_E2E"
    _AMOUNT_VALUE = 888.0

    _FR_FEATURE_FLAG = "FEATURE_ONLINE_INFERENCE_FEATURE_RETRIEVAL"

    def setUp(self) -> None:
        # Requires BUILDER/BASE_*/MODEL_LOGGER/PROXY_IMAGE_PATH so an FR-aware proxy image is wired in.
        if not self._has_image_override():
            self.skipTest("Skipping test: image override environment variables not set.")

        super().setUp()

        # Session-level opt-in; assumes ACCOUNTADMIN has already enabled the flag at account level.
        try:
            self.session.sql(f"ALTER SESSION SET {self._FR_FEATURE_FLAG} = ENABLED").collect()
        except Exception as exc:
            self.skipTest(f"Could not opt into {self._FR_FEATURE_FLAG}: {exc!r}")

        # FS schema lives inside the per-test DB so super().tearDown() drops it with the registry.
        self._fs_schema = f"FS_{self._run_id.upper()}"
        self.session.sql(f"CREATE SCHEMA IF NOT EXISTS {self._test_db}.{self._fs_schema}").collect()

        self.fs = FeatureStore(
            session=self.session,
            database=self._test_db,
            name=self._fs_schema,
            default_warehouse=self._TEST_SPCS_WH,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._user_entity = Entity(name="USER", join_keys=[self._ENTITY_KEY], desc="User entity for FR E2E test")
        self.fs.register_entity(self._user_entity)

        self._fv_name = f"FR_E2E_FV_{self._run_id.upper()}"
        src_table = f"{self._test_db}.{self._fs_schema}.SRC_{self._run_id.upper()}"
        self.session.sql(
            f"CREATE OR REPLACE TABLE {src_table} ("
            f"{self._ENTITY_KEY} STRING, {self._TIMESTAMP_COL} TIMESTAMP_NTZ, {self._FEATURE_COL} DOUBLE)"
        ).collect()
        self.session.sql(
            f"INSERT INTO {src_table} VALUES "
            f"('{self._USER_ID_VALUE}', CURRENT_TIMESTAMP()::TIMESTAMP_NTZ, {self._AMOUNT_VALUE})"
        ).collect()

        # Online Service must be RUNNING before register_feature_view(online=True).
        current_role = self.session.get_current_role()
        assert current_role is not None
        producer_role = current_role.strip('"')
        self._consumer_role = f"SNOWML_FR_E2E_C_{self._run_id.upper()}"
        self.session.sql(f"CREATE ROLE IF NOT EXISTS {SqlIdentifier(self._consumer_role)}").collect()
        self.session.sql(f"GRANT ROLE {SqlIdentifier(self._consumer_role)} TO ROLE {current_role}").collect()

        os_started = time.time()
        try:
            self.fs.create_online_service(producer_role, self._consumer_role)
        except Exception as exc:
            # Likely reused from a prior run; fall through and let the poll confirm health.
            logger.info("create_online_service raised (assuming reuse): %r", exc)
        _wait_online_service_running(self.fs)
        logger.info("Online Service ready (+%.1fs).", time.time() - os_started)

        feature_df = self.session.table(src_table).select(self._ENTITY_KEY, self._TIMESTAMP_COL, self._FEATURE_COL)
        fv = FeatureView(
            name=self._fv_name,
            entities=[self._user_entity],
            feature_df=feature_df,
            timestamp_col=self._TIMESTAMP_COL,
            refresh_freq="1 minute",
            online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        )
        self._registered_fv = self.fs.register_feature_view(fv, "V1")
        self.assertTrue(self._registered_fv.online)

        # Make sure the row is queryable end-to-end before any inference request hits FR.
        self._wait_offline_rows()
        self._wait_online_visible()

    def tearDown(self) -> None:
        # Drop the FS online service before super().tearDown drops the test DB.
        fs = getattr(self, "fs", None)
        if fs is not None:
            try:
                fs.drop_online_service()
            except Exception:
                logger.warning("drop_online_service failed", exc_info=True)
            try:
                fs._clear(dryrun=False)
            except Exception:
                logger.warning("fs._clear failed", exc_info=True)
        consumer_role = getattr(self, "_consumer_role", None)
        if consumer_role is not None:
            try:
                self.session.sql(f"DROP ROLE IF EXISTS {SqlIdentifier(consumer_role)}").collect()
            except Exception:
                logger.warning("DROP ROLE failed", exc_info=True)
        try:
            self.session.sql(f"ALTER SESSION UNSET {self._FR_FEATURE_FLAG}").collect()
        except Exception:
            logger.warning("ALTER SESSION UNSET %s failed", self._FR_FEATURE_FLAG, exc_info=True)
        super().tearDown()

    def _wait_offline_rows(self, timeout_s: float = 300.0) -> None:
        deadline = time.time() + timeout_s
        last_exc: Optional[Exception] = None
        while time.time() < deadline:
            try:
                if self.fs.read_feature_view(self._registered_fv, store_type=StoreType.OFFLINE).count() > 0:
                    return
            except Exception as exc:
                last_exc = exc
            time.sleep(10)
        self.fail(f"Offline DT did not materialize any rows within {timeout_s:.0f}s (last_exc={last_exc!r}).")

    def _wait_online_visible(self, timeout_s: float = 600.0) -> None:
        """Per-key poll until the seed row is visible via OFT vnext (Postgres FVs require keys)."""
        deadline = time.time() + timeout_s
        last_exc: Optional[Exception] = None
        while time.time() < deadline:
            try:
                pdf = self.fs.read_feature_view(
                    self._registered_fv,
                    keys=[[self._USER_ID_VALUE]],
                    store_type=StoreType.ONLINE,
                    as_pandas=True,
                )
                if len(pdf) > 0:
                    return
            except Exception as exc:
                last_exc = exc
            time.sleep(10)
        self.fail(f"Online store did not return seed key within {timeout_s:.0f}s (last_exc={last_exc!r}).")

    def test_create_service_with_feature_retrieval(self) -> None:
        """Request omits AMOUNT; proxy must fetch it from FS and echo it back."""
        sample_input = pd.DataFrame({self._ENTITY_KEY: [self._USER_ID_VALUE], self._FEATURE_COL: [0.0]})

        model_name = f"FR_E2E_MODEL_{self._run_id.upper()}"
        version_name = f"V_{self._run_id.upper()}"
        service_name = f"FR_E2E_SVC_{self._run_id.upper()}"

        mv = self.registry.log_model(
            model=_PassthroughAmountModel(custom_model.ModelContext()),
            model_name=model_name,
            version_name=version_name,
            sample_input_data=sample_input,
            options={"enable_explainability": False},
        )

        mv.create_service(
            service_name=service_name,
            service_compute_pool=self._TEST_CPU_COMPUTE_POOL,
            ingress_enabled=True,
            min_instances=1,
            max_instances=1,
            feature_sources_per_function={"predict": [self._registered_fv]},
        )
        self._wait_for_service_status(mv)
        endpoint = self._ensure_ingress_url(mv)

        # Send only USER_ID so the proxy must fetch AMOUNT from FS.
        request_payload = self._build_rest_inference_request_payload(
            registry_model_deployment_test_base.RestInferencePayloadFormat.DATAFRAME_SPLIT,
            test_input=pd.DataFrame({self._ENTITY_KEY: [self._USER_ID_VALUE]}),
            params=None,
        )
        res_df = self._inference_using_rest_api(
            request_payload,
            endpoint=endpoint,
            jwt_token_generator=self._get_jwt_token_generator(),
            target_method="predict",
        )

        self.assertEqual(len(res_df), 1, f"Expected exactly one prediction row, got {res_df!r}")
        echoed = float(res_df.iloc[0]["AMOUNT_ECHOED"])
        self.assertAlmostEqual(
            echoed,
            self._AMOUNT_VALUE,
            places=3,
            msg=(
                f"AMOUNT round-trip mismatch: wrote {self._AMOUNT_VALUE}, got {echoed!r} from inference "
                f"(was AMOUNT actually fetched from FS by the proxy?). Full response: {res_df.to_dict()!r}"
            ),
        )


if __name__ == "__main__":
    absltest.main()
