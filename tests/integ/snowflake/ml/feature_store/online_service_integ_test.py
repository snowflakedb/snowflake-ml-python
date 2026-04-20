"""Integration tests for Online Service APIs (create/get/drop)."""

import time
import uuid

from absl.testing import absltest
from fs_integ_test_base import FeatureStoreIntegTestBase

from snowflake.ml.feature_store import feature_store
from snowflake.ml.feature_store.feature_store import FeatureStore


class OnlineServiceIntegTest(FeatureStoreIntegTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.fs = FeatureStore(
            session=self._session,
            database=self.test_db,
            name=self.test_schema,
            default_warehouse=self._test_warehouse_name,
            creation_mode=feature_store.CreationMode.CREATE_IF_NOT_EXIST,
        )

    def tearDown(self) -> None:
        # Drop Online Service before base tearDown removes the database (network rules / policies).
        try:
            self.fs.drop_online_service()
        except Exception:
            pass
        super().tearDown()

    def test_create_online_service_poll_until_running_then_drop(self) -> None:
        producer = f"SML_FSRT_P_{uuid.uuid4().hex[:8]}".upper()
        consumer = f"SML_FSRT_C_{uuid.uuid4().hex[:8]}".upper()
        self._session.sql(f"CREATE ROLE IF NOT EXISTS {producer}").collect()
        self._session.sql(f"CREATE ROLE IF NOT EXISTS {consumer}").collect()

        try:
            self.fs.create_online_service(producer, consumer)

            deadline = time.time() + 900.0
            last = ""
            while time.time() < deadline:
                st = self.fs.get_online_service_status()
                last = st.status
                if st.status == "RUNNING" and any(ep.name == "query" for ep in st.endpoints):
                    break
                time.sleep(5)
            else:
                self.fail(f"Online Service did not become RUNNING with query endpoint (last status={last!r}).")

            self.fs.drop_online_service()
        finally:
            self._session.sql(f"DROP ROLE IF EXISTS {producer}").collect()
            self._session.sql(f"DROP ROLE IF EXISTS {consumer}").collect()


if __name__ == "__main__":
    absltest.main()
