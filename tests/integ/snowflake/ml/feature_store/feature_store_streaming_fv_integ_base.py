"""Shared integration test fixtures for streaming feature views (class-scoped Online Service, FeatureStore).

See ``feature_store_streaming_fv_test`` for the main streaming FV suite. Other modules (e.g. stream ingest)
reuse this base to avoid duplicating Online Service provisioning.

**Standalone mode** (default): each test class creates unique databases, schemas, roles,
and an Online Service via ``setUpClass``, and tears everything down in ``tearDownClass``.

**Module-level runner mode**: when ``_module_state`` is set on the class (by a runner such as
``feature_store_spec_oft_e2e_integ_test``), ``setUpClass`` pulls shared session/DB/FS/entity
from that dict and skips independent provisioning.  ``tearDownClass`` is a no-op (the runner
handles cleanup).
"""

import logging
import os
import time
import uuid
from typing import Optional

import pandas as pd
from fs_integ_test_base import FeatureStoreIntegTestBase

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_store import CreationMode, FeatureStore
from snowflake.ml.feature_store.feature_view import StoreType
from snowflake.ml.feature_store.online_service import (
    endpoint_url,
    fetch_online_service_status,
)
from snowflake.ml.feature_store.stream_source import StreamSource
from snowflake.snowpark._internal import utils as snowpark_utils
from snowflake.snowpark.types import (
    BooleanType,
    DecimalType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampTimeZone,
    TimestampType,
)
from tests.integ.snowflake.ml.test_utils import (
    db_manager,
    external_volume_manager,
    test_env_utils,
)

logger = logging.getLogger(__name__)


def identity_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Identity transform for ``StreamConfig`` / ``map_in_pandas``.

    Selects ``USER_ID``, ``EVENT_TIME``, then feature columns so UDF / OFT schema
    matches online service layout (timestamp adjacent to entity keys).

    Args:
        df: Input pandas batch from the stream.

    Returns:
        DataFrame containing ``USER_ID``, ``EVENT_TIME``, and ``AMOUNT``.
    """
    return df[["USER_ID", "EVENT_TIME", "AMOUNT"]]


def all_types_identity_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Identity transform that passes through all 6 supported column types."""
    return df[["USER_ID", "EVENT_TIME", "SCORE", "RANK", "PRICE", "IS_ACTIVE"]]


# Force cloudpickle to serialize by value (bytecode) rather than by module reference.
# Without this, Snowflake warehouse workers fail with ModuleNotFoundError because
# test module names are not importable on the server.  When __module__ is "__main__",
# cloudpickle's importability check fails (the function isn't in __main__.__dict__),
# so it falls back to by-value serialization — which is what we need.
identity_transform.__module__ = "__main__"
all_types_identity_transform.__module__ = "__main__"


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


class StreamingFeatureViewIntegTestBase(FeatureStoreIntegTestBase):
    """Shared fixtures: class-scoped DB, Feature Store, and Online Service.

    **Standalone** (default): each test class creates unique databases, schemas,
    consumer roles, and an Online Service via ``setUpClass``, and drops them all
    in ``tearDownClass``.

    **Module-level runner**: when ``_module_state`` is set on the class (by a
    runner such as ``feature_store_spec_oft_e2e_integ_test``), ``setUpClass``
    pulls shared session/DB/FS/entity from that dict and skips independent
    provisioning.  ``tearDownClass`` is a no-op (the runner handles cleanup).
    """

    _module_state: dict | None = None

    @classmethod
    def _init_from_module_state(cls) -> None:
        """Populate class attrs from the module-level runner state."""
        ms = cls._module_state
        assert ms is not None
        cls._session = ms["session"]
        cls._dbm = ms["dbm"]
        cls._evm = ms["evm"]
        cls._test_db = ms["test_db"]
        cls._dummy_db = ms["dummy_db"]
        cls._test_warehouse_name = ms["warehouse"]
        cls._test_schema = ms["test_schema"]
        cls._alt_warehouse_name_value = None
        cls._alt_warehouse_created = False
        cls.test_db = cls._test_db
        cls.test_schema = cls._test_schema
        cls.warehouse = cls._test_warehouse_name
        cls.fs = ms["fs"]
        cls.user_entity = ms["user_entity"]
        try:
            cls.fs.register_entity(cls.user_entity)
        except Exception:
            pass

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        if cls._module_state is not None:
            cls._init_from_module_state()
            return

        try:
            cls._session = test_env_utils.get_available_session()
            cls._dbm = db_manager.DBManager(cls._session)
            cls._evm = external_volume_manager.ExternalVolumeManager(cls._session)

            cls._dbm.cleanup_databases(expire_hours=6)
            cls._dbm.cleanup_warehouses(expire_hours=6)
            cls._dbm.cleanup_roles(expire_hours=6)

            run_id = uuid.uuid4().hex[:6]
            cls._test_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(run_id, "FS_DB").upper()
            cls._dummy_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
                run_id, "FS_DUMMY_DB"
            ).upper()

            session_warehouse = cls._session.get_current_warehouse()
            if not session_warehouse:
                raise RuntimeError("No warehouse is configured in the current session.")
            cls._test_warehouse_name = session_warehouse.strip('"')

            cls._alt_warehouse_name_value: str | None = None
            cls._alt_warehouse_created = False

            cls._dbm.create_database(cls._test_db, data_retention_time_in_days=1)
            cls._dbm.create_database(cls._dummy_db, data_retention_time_in_days=1)
            cls._dbm.use_warehouse(cls._test_warehouse_name)
            cls._dbm.use_database(cls._test_db)

            full_qual_schema = cls._dbm.create_random_schema(db_name=cls._test_db)
            cls._test_schema = full_qual_schema.split(".")[-1]
            cls.test_db = cls._test_db
            cls.test_schema = cls._test_schema
            cls.warehouse = cls._test_warehouse_name

            cls.fs = FeatureStore(
                session=cls._session,
                database=cls.test_db,
                name=cls.test_schema,
                default_warehouse=cls.warehouse,
                creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
            )

            cls.user_entity = Entity(name="user_entity", join_keys=["USER_ID"], desc="User entity for streaming SFV")
            cls.fs.register_entity(cls.user_entity)

            cls._provision_online_service_for_postgres_tests_class()
        except Exception:
            cls._cleanup_class_scoped_fixtures()
            raise

    @classmethod
    def _cleanup_class_scoped_fixtures(cls) -> None:
        """Tear down class-scoped Online Service, consumer role, feature store contents, and DBs."""
        try:
            if getattr(cls, "fs", None) is not None:
                try:
                    cls.fs.drop_online_service()
                except Exception:
                    pass
                try:
                    cls.fs._clear(dryrun=False)
                except Exception:
                    pass
        finally:
            session = getattr(cls, "_session", None)
            consumer = getattr(cls, "_online_service_consumer_role", None)
            if consumer and session is not None:
                try:
                    session.sql(f"DROP ROLE IF EXISTS {SqlIdentifier(consumer)}").collect()
                except Exception:
                    pass
            dbm = getattr(cls, "_dbm", None)
            if dbm is not None:
                for db_attr in ("_test_db", "_dummy_db"):
                    db_name = getattr(cls, db_attr, None)
                    if db_name:
                        try:
                            dbm.drop_database(db_name, if_exists=True)
                        except Exception:
                            pass
            if getattr(cls, "_alt_warehouse_created", False) and dbm is not None:
                try:
                    dbm.drop_warehouse(cls._alt_warehouse_name_value, if_exists=True)
                except Exception:
                    pass
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._module_state is not None:
            return
        try:
            cls._cleanup_class_scoped_fixtures()
        finally:
            super().tearDownClass()

    @classmethod
    def _wait_online_service_running_with_query_endpoint(cls) -> None:
        deadline = time.time() + 900.0
        while time.time() < deadline:
            st = cls.fs.get_online_service_status()
            if st.status == "RUNNING" and any(ep.name == "query" for ep in st.endpoints):
                return
            time.sleep(5)
        raise RuntimeError("Online Service did not reach RUNNING with a query endpoint within timeout.")

    @classmethod
    def _provision_online_service_for_postgres_tests_class(cls) -> None:
        producer = cls._session.get_current_role().strip('"')
        consumer = f"SML_SFVRT_C_{uuid.uuid4().hex[:8]}".upper()
        cls._online_service_producer_role = producer
        cls._online_service_consumer_role = consumer
        cls._session.sql(f"CREATE ROLE IF NOT EXISTS {SqlIdentifier(consumer)}").collect()
        cls._session.sql(f"GRANT ROLE {SqlIdentifier(consumer)} TO ROLE {cls._session.get_current_role()}").collect()

        st = fetch_online_service_status(
            cls._session,
            cls.fs._config.database,
            cls.fs._config.schema,
            statement_params=cls.fs._telemetry_stmp,
        )
        if st.status == "RUNNING" and any(ep.name == "query" for ep in st.endpoints):
            return

        cls.fs.create_online_service(producer, consumer)
        cls._wait_online_service_running_with_query_endpoint()

    def setUp(self) -> None:
        self._session = type(self)._session
        self._dbm = type(self)._dbm
        self._evm = type(self)._evm
        self._test_db = type(self)._test_db
        self._dummy_db = type(self)._dummy_db
        self._test_warehouse_name = type(self)._test_warehouse_name
        self._test_schema = type(self)._test_schema
        self._alt_warehouse_name_value = type(self)._alt_warehouse_name_value
        self._alt_warehouse_created = type(self)._alt_warehouse_created
        self.test_db = type(self).test_db
        self.test_schema = type(self).test_schema
        self.warehouse = type(self).warehouse
        self.fs = type(self).fs
        self.user_entity = type(self).user_entity

    def tearDown(self) -> None:
        if os.environ.get("SKIP_FV_TEARDOWN"):
            return
        stmp = self.fs._telemetry_stmp
        try:
            for row in self.fs.list_feature_views().collect(statement_params=stmp):
                name = SqlIdentifier(row["NAME"], case_sensitive=True).identifier()
                ver = row["VERSION"]
                try:
                    self.fs.delete_feature_view(name, str(ver))
                except Exception:
                    pass
        except Exception:
            pass
        try:
            for row in self.fs.list_stream_sources().collect(statement_params=stmp):
                try:
                    self.fs.delete_stream_source(str(row["NAME"]))
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self.fs.register_entity(self.user_entity)
        except Exception:
            pass

    def _create_feature_store(self) -> FeatureStore:
        return self.fs

    # ------------------------------------------------------------------
    # Shared polling helpers
    # ------------------------------------------------------------------

    def _wait_offline_dt_rows(self, fs: FeatureStore, fv_name: str, version: str, timeout: float = 300.0) -> None:
        """Poll until offline DT has at least one row."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                fv_live = fs.get_feature_view(fv_name, version)
                off = fs.read_feature_view(fv_live, store_type=StoreType.OFFLINE)
                if off.count() > 0:
                    return
            except Exception:
                pass
            time.sleep(5)
        self.fail(f"Timed out waiting for {fv_name}/{version} offline DT to return rows.")

    def _poll_online_read(
        self,
        fs: FeatureStore,
        fv_name: str,
        version: str,
        keys: list,
        validate_fn=None,
        timeout: float = 300.0,
        desc: str = "",
    ) -> None:
        """Poll spec OFT online read until rows appear and optional validation passes.

        Args:
            fs: Feature store client.
            fv_name: Feature view name.
            version: Feature view version.
            keys: Entity keys passed to ``read_feature_view``.
            validate_fn: ``callable(pdf) -> None``. Raise ``AssertionError`` to retry.
            timeout: Seconds to poll before failing.
            desc: Label for the failure message.

        Returns:
            None when online read returns rows and ``validate_fn`` passes (if set).
        """
        fv_live = fs.get_feature_view(fv_name, version)
        deadline = time.time() + timeout
        last_err: Optional[str] = None
        while time.time() < deadline:
            try:
                out = fs.read_feature_view(fv_live, keys=keys, store_type=StoreType.ONLINE)
                if out.count() > 0:
                    if validate_fn is not None:
                        pdf = out.to_pandas()
                        validate_fn(pdf)
                    return
            except Exception as e:
                last_err = str(e)
            time.sleep(10)
        label = f" ({desc})" if desc else ""
        self.fail(f"Online read for {fv_name}/{version}{label} timed out; last_err={last_err!r}")

    def _wait_online_service_ingest_endpoint(self, timeout_s: float = 180.0) -> None:
        """Poll until the Online Service exposes an ingest endpoint."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            st = self.fs.get_online_service_status()
            if st.status == "RUNNING" and endpoint_url(st, "ingest"):
                return
            time.sleep(3)
        self.fail(f"Online Service did not expose an ingest endpoint within {timeout_s}s")

    def _stream_ingest_with_retry(self, fs: FeatureStore, stream_name: str, records, timeout_s: float = 180.0) -> None:
        """Retry ``stream_ingest`` until the Online Service accepts it.

        Replaces the former blind ``sleep(120)`` post-registration propagation wait.

        Args:
            fs: Feature store client.
            stream_name: Registered stream source name.
            records: Payload for ``stream_ingest``.
            timeout_s: Seconds to retry before failing.

        Returns:
            None when ingest succeeds.
        """
        deadline = time.time() + timeout_s
        last_err: Optional[str] = None
        while time.time() < deadline:
            try:
                fs.stream_ingest(stream_name, records)
                return
            except Exception as e:
                last_err = str(e)
            time.sleep(5)
        self.fail(f"stream_ingest not accepted within {timeout_s}s; last_err={last_err!r}")

    def _make_stream_source(self, fs: FeatureStore, stream_name: str) -> None:
        fs.register_stream_source(
            StreamSource(
                name=stream_name,
                schema=StructType(
                    [
                        StructField("USER_ID", StringType()),
                        StructField("EVENT_TIME", TimestampType(TimestampTimeZone.NTZ)),
                        StructField("AMOUNT", DoubleType()),
                    ]
                ),
                desc="Transaction events stream",
            )
        )

    def _create_backfill_table(self, fs: FeatureStore, suffix: str) -> str:
        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BACKFILL_{suffix}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR,
                EVENT_TIME TIMESTAMP_NTZ,
                AMOUNT FLOAT
            )
        """
        ).collect()

        self._session.sql(
            f"""
            INSERT INTO {table_name} VALUES
            ('u1', '2024-01-01 00:00:00', 100.0),
            ('u2', '2024-01-01 01:00:00', 750.0),
            ('u3', '2024-01-01 02:00:00', 50.0)
        """
        ).collect()

        return table_name

    def _make_all_types_stream_source(self, fs: FeatureStore, stream_name: str) -> None:
        """Register a stream source with all 6 supported column types."""
        fs.register_stream_source(
            StreamSource(
                name=stream_name,
                schema=StructType(
                    [
                        StructField("USER_ID", StringType()),
                        StructField("EVENT_TIME", TimestampType(TimestampTimeZone.NTZ)),
                        StructField("SCORE", DoubleType()),
                        StructField("RANK", LongType()),
                        StructField("PRICE", DecimalType(10, 2)),
                        StructField("IS_ACTIVE", BooleanType()),
                    ]
                ),
                desc="All-types stream for type coverage testing",
            )
        )

    def _create_all_types_backfill_table(self, fs: FeatureStore, suffix: str) -> str:
        """Create a backfill table with all 6 supported column types."""
        table_name = f"{self.test_db}.{fs._config.schema.identifier()}.BACKFILL_ALL_TYPES_{suffix}"
        self._session.sql(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                USER_ID VARCHAR,
                EVENT_TIME TIMESTAMP_NTZ,
                SCORE FLOAT,
                RANK INT,
                PRICE NUMBER(10,2),
                IS_ACTIVE BOOLEAN
            )
        """
        ).collect()
        self._session.sql(
            f"""
            INSERT INTO {table_name} VALUES
            ('u1', '2024-01-01 00:00:00', 3.14, 42, 99.95, TRUE),
            ('u2', '2024-01-01 01:00:00', 2.72, 7, 49.99, FALSE),
            ('u3', '2024-01-01 02:00:00', 1.41, 1, 9.99, TRUE)
        """
        ).collect()
        return table_name

    def _stream_source_ref_key(self, stream_name: str) -> str:
        return SqlIdentifier(stream_name).resolved()

    def _streaming_backfill_query_status(self, query_id: str) -> tuple[Optional[str], Optional[str]]:
        """Return ``(EXECUTION_STATUS, ERROR_MESSAGE)`` from session query history, or ``(None, None)`` if absent."""
        safe = snowpark_utils.escape_single_quotes(query_id)  # type: ignore[no-untyped-call]
        rows = self._session.sql(
            f"""
            SELECT EXECUTION_STATUS, ERROR_MESSAGE
            FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY_BY_SESSION(RESULT_LIMIT => 10000))
            WHERE QUERY_ID = '{safe}'
            ORDER BY START_TIME DESC
            LIMIT 1
            """
        ).collect()
        if not rows:
            return None, None
        row = rows[0].as_dict()
        status: Optional[str] = None
        err: Optional[str] = None
        for k, v in row.items():
            ku = k.upper()
            if ku == "EXECUTION_STATUS" and v is not None:
                status = str(v).upper()
            elif ku == "ERROR_MESSAGE" and v is not None:
                err = str(v)
        return status, err

    def _wait_udf_and_backfill(
        self,
        fq_udf: str,
        timeout_s: float = 240.0,
        *,
        feature_store: Optional[FeatureStore] = None,
        streaming_fv_metadata_name: Optional[str] = None,
        streaming_fv_version: Optional[str] = None,
    ) -> None:
        """Wait until the UDF-transformed (DT) table is non-empty after backfill.

        The ``$BACKFILL`` table is intentionally NOT checked: the OFT refresh may
        delete it before we get a chance to poll.

        When ``feature_store``, ``streaming_fv_metadata_name`` (use ``str(registered.name)``), and
        ``streaming_fv_version`` are provided, polls ``INFORMATION_SCHEMA.QUERY_HISTORY_BY_SESSION`` for the
        async backfill ``query_id`` stored in streaming metadata (``StreamingMetadata.backfill_query_id``)
        until ``EXECUTION_STATUS`` is terminal, then verifies the DT has rows. That avoids relying only on
        blind table polling for the long-running ``INSERT ALL`` job.

        If metadata has no ``backfill_query_id`` or query-history lookup fails, falls back to row-count polling
        for the full timeout.

        Args:
            fq_udf: Fully qualified UDF-transformed dynamic table name.
            timeout_s: Max seconds to wait for backfill completion and DT rows.
            feature_store: Optional client used to read streaming metadata and
                ``backfill_query_id``.
            streaming_fv_metadata_name: Feature view name for metadata lookup
                (typically ``str(registered.name)``).
            streaming_fv_version: Feature view version for metadata lookup.

        Raises:
            AssertionError: Via ``self.fail`` on missing metadata, failed or
                unfinished backfill query, or empty DT after ``timeout_s``.
        """
        deadline = time.time() + timeout_s
        query_id: Optional[str] = None
        polled_query = False
        qid_for_hint: Optional[str] = None
        if feature_store is not None and streaming_fv_metadata_name and streaming_fv_version:
            meta = feature_store._metadata_manager.get_streaming_metadata(
                streaming_fv_metadata_name,
                streaming_fv_version,
            )
            if meta is None:
                self.fail(
                    f"No streaming metadata for feature view {streaming_fv_metadata_name!r} "
                    f"version {streaming_fv_version!r}"
                )
            query_id = meta.backfill_query_id

        if query_id:
            qid_for_hint = query_id
            try:
                last_status: Optional[str] = None
                while time.time() < deadline:
                    status, err_msg = self._streaming_backfill_query_status(query_id)
                    last_status = status
                    if status is None:
                        time.sleep(2)
                        continue
                    if status in ("SUCCESS", "SUCCESS_WITH_ERRORS"):
                        polled_query = True
                        break
                    if "FAIL" in status or status.startswith("CANCEL") or status.startswith("ABORT"):
                        detail = f" {err_msg}" if err_msg else ""
                        self.fail(f"Streaming backfill query {query_id} ended with {status}.{detail}")
                    time.sleep(2)
                else:
                    self.fail(
                        f"Backfill query {query_id} did not report SUCCESS within {timeout_s}s "
                        f"(last_status={last_status!r})"
                    )
            except AssertionError:
                raise
            except Exception as ex:
                logger.warning(
                    "Could not poll QUERY_HISTORY_BY_SESSION for backfill query_id=%r: %s; "
                    "falling back to table row polling.",
                    query_id,
                    ex,
                )

        while time.time() < deadline:
            udf_count = self._session.table(fq_udf).count()
            if udf_count > 0:
                return
            time.sleep(2)

        hint = f" (backfill_query_id={qid_for_hint!r}, used_query_poll={polled_query})" if qid_for_hint else ""
        self.fail(f"Backfill did not populate DT within {timeout_s}s{hint}")
