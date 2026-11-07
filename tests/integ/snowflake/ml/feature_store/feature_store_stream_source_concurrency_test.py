"""Integration tests for stream source reference tracking under concurrent writers.

Each worker uses its own Snowpark session, so writes reach Snowflake concurrently
the way separate users or CI jobs would.
"""

import threading
import time
from typing import Callable

from absl.testing import absltest
from common_utils import (
    create_random_schema,
    delete_streaming_fv_metadata,
    register_streaming_fv_metadata,
)

from fs_integ_test_base import FeatureStoreIntegTestBase
from snowflake.ml.feature_store.feature_store import CreationMode, FeatureStore
from snowflake.ml.feature_store.metadata_manager import (
    FeatureStoreMetadataManager,
    StreamingMetadata,
)
from snowflake.ml.feature_store.stream_source import StreamSource
from snowflake.snowpark.types import StringType, StructField, StructType
from tests.integ.snowflake.ml.test_utils import test_env_utils

_NUM_WORKERS = 8
_SOURCE_NAME = "TXN_EVENTS"


class FeatureStoreStreamSourceConcurrencyTest(FeatureStoreIntegTestBase):
    def setUp(self) -> None:
        super().setUp()
        schema = create_random_schema(self._session, "SS_CONC_TEST", database=self.test_db)
        self._fs = FeatureStore(
            self._session,
            self.test_db,
            schema,
            default_warehouse=self._test_warehouse_name,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )
        self._fs.register_stream_source(StreamSource(_SOURCE_NAME, StructType([StructField("user_id", StringType())])))

    def _worker_managers(self, n: int) -> list[FeatureStoreMetadataManager]:
        base = self._fs._metadata_manager
        managers = []
        for _ in range(n):
            session = test_env_utils.get_available_session()
            self.addCleanup(session.close)
            managers.append(
                FeatureStoreMetadataManager(
                    session, base._database, base._schema, base._fs_object_tag_path, base._telemetry_stmp
                )
            )
        return managers

    def _run_concurrently(self, fns: list[Callable[[], None]]) -> None:
        start = threading.Barrier(len(fns))
        errors: list[BaseException] = []

        def run(fn: Callable[[], None]) -> None:
            start.wait()
            try:
                fn()
            except BaseException as e:  # surfaced to the main thread below
                errors.append(e)

        threads = [threading.Thread(target=run, args=(fn,)) for fn in fns]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]

    def test_concurrent_registers_count_every_reference(self) -> None:
        mgrs = self._worker_managers(_NUM_WORKERS)
        self._run_concurrently(
            [lambda m=m, i=i: register_streaming_fv_metadata(m, f"FV_{i}", _SOURCE_NAME) for i, m in enumerate(mgrs)]
        )

        self.assertEqual(self._fs._metadata_manager.get_stream_source_ref_count(_SOURCE_NAME), _NUM_WORKERS)
        # Older SDK clients read the stored value, so it must not lose an update either.
        stored = self._fs._metadata_manager.get_stream_source_metadata(_SOURCE_NAME)
        assert stored is not None
        self.assertEqual(stored["ref_count"], _NUM_WORKERS)

    def test_concurrent_deletes_release_every_reference(self) -> None:
        for i in range(_NUM_WORKERS):
            register_streaming_fv_metadata(self._fs._metadata_manager, f"FV_{i}", _SOURCE_NAME)

        mgrs = self._worker_managers(_NUM_WORKERS)
        self._run_concurrently(
            [lambda m=m, i=i: delete_streaming_fv_metadata(m, f"FV_{i}", _SOURCE_NAME) for i, m in enumerate(mgrs)]
        )

        self.assertEqual(self._fs._metadata_manager.get_stream_source_ref_count(_SOURCE_NAME), 0)
        stored = self._fs._metadata_manager.get_stream_source_metadata(_SOURCE_NAME)
        assert stored is not None
        self.assertEqual(stored["ref_count"], 0)
        self._fs.delete_stream_source(_SOURCE_NAME)
        self.assertFalse(self._fs._metadata_manager.stream_source_exists(_SOURCE_NAME))

    def test_sync_repairs_a_drifted_stored_ref_count(self) -> None:
        """Older SDK clients read the stored value; a recount replaces whatever drifted value is there."""
        mgr = self._fs._metadata_manager
        register_streaming_fv_metadata(mgr, "FV_0", _SOURCE_NAME)
        drifted = mgr.get_stream_source_metadata(_SOURCE_NAME)
        assert drifted is not None
        drifted["ref_count"] = 7
        mgr.save_stream_source(_SOURCE_NAME, drifted)

        mgr.sync_stream_source_ref_count(_SOURCE_NAME)

        stored = mgr.get_stream_source_metadata(_SOURCE_NAME)
        assert stored is not None
        self.assertEqual(stored["ref_count"], 1)

    # Register vs delete: hold one guarded statement open in a transaction, run the other.
    _HOLD_SECONDS = 5

    def _hold_open(self, mgr: FeatureStoreMetadataManager, op: Callable[[], object]) -> None:
        """Run ``op`` inside an explicit transaction and keep it uncommitted for a few seconds."""
        opened = threading.Event()

        def run() -> None:
            mgr._session.sql("BEGIN").collect()
            try:
                op()
                opened.set()
                time.sleep(self._HOLD_SECONDS)
                mgr._session.sql("COMMIT").collect()
            except BaseException:
                mgr._session.sql("ROLLBACK").collect()
                opened.set()
                raise

        thread = threading.Thread(target=run)
        thread.start()
        self.addCleanup(thread.join)
        self.assertTrue(opened.wait(timeout=120))

    def test_register_waiting_on_an_open_source_delete_writes_nothing(self) -> None:
        holder, writer = self._worker_managers(2)
        self._hold_open(holder, lambda: holder.delete_stream_source_metadata(_SOURCE_NAME))

        written = writer.save_streaming_metadata(
            "FV_0", "V1", StreamingMetadata(stream_source_name=_SOURCE_NAME, transformation_fn_name="fn")
        )

        self.assertFalse(written)
        self.assertFalse(self._fs._metadata_manager.stream_source_exists(_SOURCE_NAME))
        self.assertEqual(self._fs._metadata_manager.get_stream_source_ref_count(_SOURCE_NAME), 0)

    def test_source_delete_waiting_on_an_open_register_is_refused(self) -> None:
        holder, deleter = self._worker_managers(2)
        self._hold_open(
            holder,
            lambda: holder.save_streaming_metadata(
                "FV_0", "V1", StreamingMetadata(stream_source_name=_SOURCE_NAME, transformation_fn_name="fn")
            ),
        )

        deleted = deleter.delete_stream_source_metadata(_SOURCE_NAME)

        self.assertFalse(deleted)
        self.assertTrue(self._fs._metadata_manager.stream_source_exists(_SOURCE_NAME))
        self.assertEqual(self._fs._metadata_manager.list_stream_source_references(_SOURCE_NAME), [("FV_0", "V1")])


if __name__ == "__main__":
    absltest.main()
