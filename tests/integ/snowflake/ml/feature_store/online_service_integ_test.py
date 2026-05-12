"""Integration tests for Online Service APIs (create/get/drop)."""

import logging
import uuid

from absl.testing import absltest
from feature_store_streaming_fv_integ_base import (
    wait_online_service_running_with_query_endpoint,
)
from fs_integ_test_base import FeatureStoreIntegTestBase

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store import feature_store
from snowflake.ml.feature_store.feature_store import FeatureStore

logger = logging.getLogger(__name__)


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
        # Drop Online Service before base tearDown removes the database.
        try:
            self.fs.drop_online_service()
        except Exception:
            pass
        super().tearDown()

    def test_create_online_service_poll_until_running_then_drop(self) -> None:
        # Producer must own the test schema; use the session role (mirrors bundle runner).
        producer = self._session.get_current_role().strip('"')
        consumer = f"SNOWML_TEST_SPEC_OFT_C_{uuid.uuid4().hex[:8]}".upper()
        self._session.sql(f"CREATE ROLE IF NOT EXISTS {SqlIdentifier(consumer)}").collect()
        self._session.sql(f"GRANT ROLE {SqlIdentifier(consumer)} TO ROLE {self._session.get_current_role()}").collect()

        try:
            # This test verifies the full create/poll/drop cycle, so disable the helper's
            # reuse-if-already-running fast path.
            wait_online_service_running_with_query_endpoint(
                session=self._session,
                fs=self.fs,
                producer_role=producer,
                consumer_role=consumer,
                reuse_if_running=False,
            )
            self.fs.drop_online_service()
        finally:
            self._session.sql(f"DROP ROLE IF EXISTS {SqlIdentifier(consumer)}").collect()


if __name__ == "__main__":
    absltest.main()
