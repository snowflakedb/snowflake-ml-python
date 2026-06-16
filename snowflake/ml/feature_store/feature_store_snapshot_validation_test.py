"""Unit tests for snapshot config validation gates."""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock

from absl.testing import absltest, parameterized

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store import feature_view_append_only_validation
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewStatus,
    FeatureViewVersion,
)
from snowflake.ml.feature_store.metadata_manager import AppendOnlyMetadata


def _create_feature_store_with_mocks() -> Any:
    """Create a FeatureStore with mocked dependencies (bypassing __init__).

    Note: This manually sets internal attributes. Update if FeatureStore.__init__ changes.
    """
    from snowflake.ml.feature_store.feature_store import (
        FeatureStore,
        _FeatureStoreConfig,
    )

    fs = object.__new__(FeatureStore)
    fs._session = MagicMock()
    fs._session.get_current_role.return_value = "ROLE_1"
    fs._session.get_current_warehouse.return_value = "WH_1"
    fs._metadata_manager = MagicMock()
    fs._config = _FeatureStoreConfig(
        database=SqlIdentifier("TEST_DB"),
        schema=SqlIdentifier("TEST_SCHEMA"),
    )
    fs._default_warehouse = SqlIdentifier("WH_1")
    fs._telemetry_stmp = {}
    fs._asof_join_enabled = None
    return fs


def _make_valid_snapshot_fv() -> FeatureView:
    """Build a valid snapshot-enabled FeatureView that passes all validations."""
    mock_df = MagicMock()
    mock_df.columns = ["GUEST_ID", "SNAPSHOT_TS", "N_RSRVS_30_DAY"]
    mock_df.queries = {"queries": ["SELECT * FROM source"]}

    from snowflake.snowpark.types import TimestampType

    ts_field = MagicMock()
    ts_field.datatype = TimestampType()
    mock_df.schema.__getitem__ = lambda self, key: ts_field

    entity = Entity(name="guest", join_keys=["GUEST_ID"])
    fv = FeatureView(
        name="SNAP_FV",
        entities=[entity],
        feature_df=mock_df,
        timestamp_col="SNAPSHOT_TS",
        refresh_freq="0 0 * * * UTC",
        refresh_mode="FULL",
        append_only=True,
    )
    fv._version = FeatureViewVersion("V1")
    fv._status = FeatureViewStatus.ACTIVE
    fv._database = SqlIdentifier("TEST_DB")
    fv._schema = SqlIdentifier("TEST_SCHEMA")
    return fv


def _make_non_snapshot_fv() -> FeatureView:
    """Build a managed DT FeatureView that is NOT snapshot-enabled."""
    mock_df = MagicMock()
    mock_df.columns = ["GUEST_ID", "SNAPSHOT_TS", "SCORE"]
    mock_df.queries = {"queries": ["SELECT * FROM source"]}

    from snowflake.snowpark.types import TimestampType

    ts_field = MagicMock()
    ts_field.datatype = TimestampType()
    mock_df.schema.__getitem__ = lambda self, key: ts_field

    entity = Entity(name="guest", join_keys=["GUEST_ID"])
    fv = FeatureView(
        name="REGULAR_FV",
        entities=[entity],
        feature_df=mock_df,
        timestamp_col="SNAPSHOT_TS",
        refresh_freq="1 day",
    )
    fv._version = FeatureViewVersion("V1")
    fv._status = FeatureViewStatus.ACTIVE
    fv._database = SqlIdentifier("TEST_DB")
    fv._schema = SqlIdentifier("TEST_SCHEMA")
    return fv


class ValidateSnapshotConfigForRegisterTest(parameterized.TestCase):
    """Tests for ``feature_view_append_only_validation.validate_snapshot_config_for_register``."""

    def test_valid_snapshot_fv_passes(self) -> None:
        """A properly configured snapshot FV should pass validation."""
        fv = _make_valid_snapshot_fv()
        feature_view_append_only_validation.validate_snapshot_config_for_register(fv)

    def test_rejects_overwrite(self) -> None:
        """append_only feature views reject overwrite=True at the method level."""
        fv = _make_valid_snapshot_fv()

        with self.assertRaises(Exception) as cm:
            feature_view_append_only_validation.validate_snapshot_config_for_register(fv, overwrite=True)
        self.assertIn("append_only", str(cm.exception).lower())
        self.assertIn("overwrite", str(cm.exception).lower())

    def test_rejects_streaming_fv(self) -> None:
        """Snapshot accumulation is not supported for streaming feature views."""
        fv = _make_valid_snapshot_fv()
        fv._stream_config = MagicMock()

        with self.assertRaises(Exception) as cm:
            feature_view_append_only_validation.validate_snapshot_config_for_register(fv)
        self.assertIn("streaming", str(cm.exception).lower())

    def test_rejects_rollup_fv(self) -> None:
        """Snapshot accumulation is not supported for rollup feature views."""
        fv = _make_valid_snapshot_fv()
        fv._rollup_config = MagicMock()

        with self.assertRaises(Exception) as cm:
            feature_view_append_only_validation.validate_snapshot_config_for_register(fv)
        self.assertIn("rollup", str(cm.exception).lower())

    def test_rejects_tiled_fv(self) -> None:
        """Snapshot accumulation is not supported for tiled feature views."""
        fv = _make_valid_snapshot_fv()
        fv._feature_granularity = "1 hour"
        fv._aggregation_specs = [MagicMock()]

        with self.assertRaises(Exception) as cm:
            feature_view_append_only_validation.validate_snapshot_config_for_register(fv)
        self.assertIn("tiled", str(cm.exception).lower())

    def test_rejects_auto_refresh_mode(self) -> None:
        """Snapshot accumulation requires refresh_mode='FULL'."""
        fv = _make_valid_snapshot_fv()
        fv._refresh_mode = "AUTO"

        with self.assertRaises(Exception) as cm:
            feature_view_append_only_validation.validate_snapshot_config_for_register(fv)
        self.assertIn("refresh_mode", str(cm.exception))
        self.assertIn("FULL", str(cm.exception))

    def test_rejects_incremental_refresh_mode(self) -> None:
        """Snapshot accumulation rejects refresh_mode='INCREMENTAL'."""
        fv = _make_valid_snapshot_fv()
        fv._refresh_mode = "INCREMENTAL"

        with self.assertRaises(Exception) as cm:
            feature_view_append_only_validation.validate_snapshot_config_for_register(fv)
        self.assertIn("INCREMENTAL", str(cm.exception))

    def test_rejects_none_refresh_mode(self) -> None:
        """Snapshot accumulation rejects refresh_mode=None (defaults to AUTO)."""
        fv = _make_valid_snapshot_fv()
        fv._refresh_mode = None

        with self.assertRaises(Exception) as cm:
            feature_view_append_only_validation.validate_snapshot_config_for_register(fv)
        self.assertIn("refresh_mode", str(cm.exception))

    def test_rejects_none_refresh_freq(self) -> None:
        """Snapshot accumulation requires refresh_freq (cannot be a view)."""
        fv = _make_valid_snapshot_fv()
        fv._refresh_freq = None

        with self.assertRaises(Exception) as cm:
            feature_view_append_only_validation.validate_snapshot_config_for_register(fv)
        self.assertIn("refresh_freq", str(cm.exception))

    def test_constructor_rejects_append_only_with_none_refresh_freq(self) -> None:
        """FeatureView(append_only=True, refresh_freq=None) must raise during __init__.

        ``FeatureView.__init__`` invokes ``feature_view_append_only_validation.validate_snapshot_config_for_register``,
        so the invalid combination must be rejected at construction time — before
        the FV ever reaches ``register_feature_view``. This test pins that contract
        end-to-end through the public constructor (rather than calling
        ``feature_view_append_only_validation.validate_snapshot_config_for_register`` directly), so future refactors
        that move the validation out of ``__init__`` can't silently regress.
        """
        mock_df = MagicMock()
        mock_df.columns = ["GUEST_ID", "SNAPSHOT_TS", "N_RSRVS_30_DAY"]
        mock_df.queries = {"queries": ["SELECT * FROM source"]}

        from snowflake.snowpark.types import TimestampType

        ts_field = MagicMock()
        ts_field.datatype = TimestampType()
        mock_df.schema.__getitem__ = lambda self, key: ts_field

        entity = Entity(name="guest", join_keys=["GUEST_ID"])

        with self.assertRaises(Exception) as cm:
            FeatureView(
                name="SNAP_FV",
                entities=[entity],
                feature_df=mock_df,
                timestamp_col="SNAPSHOT_TS",
                refresh_freq=None,
                refresh_mode="FULL",
                append_only=True,
            )
        self.assertIn("Snapshot accumulation requires refresh_freq", str(cm.exception))

    def test_rejects_duration_refresh_freq(self) -> None:
        """Snapshot accumulation requires cron expression, not a duration like '1 day'."""
        fv = _make_valid_snapshot_fv()
        fv._refresh_freq = "1 day"

        with self.assertRaises(Exception) as cm:
            feature_view_append_only_validation.validate_snapshot_config_for_register(fv)
        self.assertIn("cron", str(cm.exception).lower())

    @parameterized.named_parameters(  # type: ignore[misc]
        ("upper", "DOWNSTREAM"),
        ("lower", "downstream"),
        ("mixed", "Downstream"),
    )
    def test_rejects_downstream_refresh_freq(self, refresh_freq: str) -> None:
        """Snapshot accumulation rejects DOWNSTREAM refresh_freq, case-insensitively.

        ``refresh_freq`` is stored verbatim on the FV (no normalization in the
        constructor or setter), so the snapshot-config gate must recognise
        ``"DOWNSTREAM"`` regardless of case — otherwise a lowercase
        ``"downstream"`` would slip past ``_is_cron_refresh_freq`` and be
        treated as a (malformed) cron expression downstream.
        """
        fv = _make_valid_snapshot_fv()
        fv._refresh_freq = refresh_freq

        with self.assertRaises(Exception) as cm:
            feature_view_append_only_validation.validate_snapshot_config_for_register(fv)
        self.assertIn("cron", str(cm.exception).lower())

    def test_rejects_none_feature_df(self) -> None:
        """Snapshot accumulation requires feature_df for schema."""
        fv = _make_valid_snapshot_fv()
        fv._feature_df = None

        with self.assertRaises(Exception) as cm:
            feature_view_append_only_validation.validate_snapshot_config_for_register(fv)
        self.assertIn("feature DataFrame", str(cm.exception))

    def test_rejects_none_timestamp_col(self) -> None:
        """Snapshot accumulation requires timestamp_col."""
        fv = _make_valid_snapshot_fv()
        fv._timestamp_col = None

        with self.assertRaises(Exception) as cm:
            feature_view_append_only_validation.validate_snapshot_config_for_register(fv)
        self.assertIn("timestamp_col", str(cm.exception))

    def test_rejects_empty_entities(self) -> None:
        """Snapshot accumulation requires at least one entity."""
        fv = _make_valid_snapshot_fv()
        fv._entities = []

        with self.assertRaises(Exception) as cm:
            feature_view_append_only_validation.validate_snapshot_config_for_register(fv)
        self.assertIn("entity", str(cm.exception).lower())


class ValidateSnapshotConfigForUpdateTest(parameterized.TestCase):
    """Tests for ``feature_view_append_only_validation.validate_snapshot_config_for_update``.

    The update gate intentionally has narrower scope than the register gate:
    structural invariants (``is_batch_view``, ``feature_df``, ``timestamp_col``,
    ``entities``, ``refresh_mode``) cannot change for an already-registered FV
    and so are not re-checked. Only the cron ``refresh_freq`` contract — the
    one thing ``update_feature_view`` can mutate — is enforced here.
    """

    def test_valid_snapshot_fv_passes(self) -> None:
        """A valid snapshot FV passes the update-time gate."""
        fv = _make_valid_snapshot_fv()
        feature_view_append_only_validation.validate_snapshot_config_for_update(fv)

    def test_non_append_only_is_noop(self) -> None:
        """Non-append_only FVs short-circuit out of the update gate."""
        fv = _make_non_snapshot_fv()
        # No raise even though refresh_freq is a duration ("1 day") — the
        # gate only matters for append_only FVs.
        feature_view_append_only_validation.validate_snapshot_config_for_update(fv)

    def test_rejects_none_refresh_freq(self) -> None:
        """An append_only FV with refresh_freq=None is rejected at update time."""
        fv = _make_valid_snapshot_fv()
        fv._refresh_freq = None

        with self.assertRaises(Exception) as cm:
            feature_view_append_only_validation.validate_snapshot_config_for_update(fv)
        self.assertIn("Snapshot accumulation requires refresh_freq", str(cm.exception))

    def test_rejects_duration_refresh_freq(self) -> None:
        """An append_only FV with a duration refresh_freq is rejected at update time."""
        fv = _make_valid_snapshot_fv()
        fv._refresh_freq = "1 day"

        with self.assertRaises(Exception) as cm:
            feature_view_append_only_validation.validate_snapshot_config_for_update(fv)
        self.assertIn("cron", str(cm.exception).lower())

    @parameterized.named_parameters(  # type: ignore[misc]
        ("upper", "DOWNSTREAM"),
        ("lower", "downstream"),
        ("mixed", "Downstream"),
    )
    def test_rejects_downstream_refresh_freq(self, refresh_freq: str) -> None:
        """An append_only FV with a DOWNSTREAM refresh_freq is rejected at update time."""
        fv = _make_valid_snapshot_fv()
        fv._refresh_freq = refresh_freq

        with self.assertRaises(Exception) as cm:
            feature_view_append_only_validation.validate_snapshot_config_for_update(fv)
        self.assertIn("cron", str(cm.exception).lower())

    def test_does_not_re_check_structural_invariants(self) -> None:
        """Structural invariants are register-time only — not re-checked at update.

        Pin the narrow scope: even though these mutations would all be rejected
        by ``feature_view_append_only_validation.validate_snapshot_config_for_register``, the update-time gate
        intentionally ignores them. ``update_feature_view``'s public surface
        does not let a caller mutate any of these; the original registration is
        the source of truth.
        """
        fv = _make_valid_snapshot_fv()
        fv._refresh_mode = "AUTO"
        fv._feature_df = None
        fv._timestamp_col = None
        fv._entities = []

        # No raise — the update-time gate only checks refresh_freq.
        feature_view_append_only_validation.validate_snapshot_config_for_update(fv)


class RegisterFeatureViewSnapshotValidationTest(absltest.TestCase):
    """Tests for register_feature_view snapshot-related validation gates."""

    def test_rejects_append_only_with_overwrite(self) -> None:
        """register_feature_view rejects append_only=True combined with overwrite=True."""
        fs = _create_feature_store_with_mocks()
        fv = _make_valid_snapshot_fv()
        fv._status = FeatureViewStatus.DRAFT

        with self.assertRaises(Exception) as cm:
            fs.register_feature_view(feature_view=fv, version="V1", overwrite=True)
        self.assertIn("append_only", str(cm.exception).lower())
        self.assertIn("overwrite", str(cm.exception).lower())

    def test_accepts_append_only_without_overwrite(self) -> None:
        """register_feature_view accepts append_only=True when overwrite=False."""
        fs = _create_feature_store_with_mocks()
        fv = _make_valid_snapshot_fv()
        fv._status = FeatureViewStatus.DRAFT

        fs._validate_entity_exists = MagicMock(return_value=True)
        fs._create_offline_feature_view = MagicMock(return_value=[])
        fs._finalize_feature_view_registration = MagicMock()
        fs.get_feature_view = MagicMock(return_value=fv)

        result = fs.register_feature_view(feature_view=fv, version="V1", overwrite=False)
        self.assertIsNotNone(result)

    def test_register_rejects_append_only_with_none_refresh_freq(self) -> None:
        """register_feature_view re-runs snapshot validation as defense-in-depth.

        ``FeatureView.__init__`` is the primary gate for the
        ``append_only=True, refresh_freq=None`` combination, but a caller can
        bypass it by constructing a valid snapshot FV and then mutating
        ``_refresh_freq`` directly on the instance. ``register_feature_view``
        re-invokes ``feature_view_append_only_validation.validate_snapshot_config_for_register`` before any
        Snowflake-side mutation so this combination still fails fast — pin that
        contract so future refactors don't silently bypass it.
        """
        fs = _create_feature_store_with_mocks()
        fv = _make_valid_snapshot_fv()
        fv._status = FeatureViewStatus.DRAFT
        # Bypass the constructor's snapshot validation by patching after construction.
        fv._refresh_freq = None

        # Mocks that should never be reached because validation must fail first.
        fs._validate_entity_exists = MagicMock(return_value=True)
        fs._create_offline_feature_view = MagicMock(return_value=[])
        fs._finalize_feature_view_registration = MagicMock()
        fs.get_feature_view = MagicMock(return_value=fv)

        with self.assertRaises(Exception) as cm:
            fs.register_feature_view(feature_view=fv, version="V1", overwrite=False)

        self.assertIn("Snapshot accumulation requires refresh_freq", str(cm.exception))
        # The validation must fail before any Snowflake-side mutation is attempted.
        fs._validate_entity_exists.assert_not_called()
        fs._create_offline_feature_view.assert_not_called()
        fs._finalize_feature_view_registration.assert_not_called()


class UpdateFeatureViewSnapshotValidationTest(absltest.TestCase):
    """Tests for update_feature_view snapshot-related validation gates."""

    def test_rejects_explicit_none_refresh_freq_on_append_only(self) -> None:
        """update_feature_view rejects explicit refresh_freq=None on an append-only FV."""
        fs = _create_feature_store_with_mocks()
        fv = _make_valid_snapshot_fv()

        fs._validate_feature_view_name_and_version_input = MagicMock(return_value=fv)

        with self.assertRaises(Exception) as cm:
            fs.update_feature_view(name=fv, refresh_freq=None)
        self.assertIn("refresh_freq", str(cm.exception).lower())

    def test_none_refresh_freq_raises_via_validate_snapshot_config_for_update(self) -> None:
        """update_feature_view(refresh_freq=None) on an append_only FV must raise via
        ``validate_snapshot_config_for_update``, before any side effects.

        The validation block copies the FV, sets ``_refresh_freq=None`` on the copy,
        and invokes ``validate_snapshot_config_for_update`` — which rejects ``None``
        because append_only feature views require a cron schedule. This test pins
        that exact contract so future refactors don't silently bypass it.
        """
        fs = _create_feature_store_with_mocks()
        fv = _make_valid_snapshot_fv()

        fs._validate_feature_view_name_and_version_input = MagicMock(return_value=fv)
        fs._plan_feature_view_update_operations = MagicMock(return_value=([], []))
        fs._execute_atomic_operations = MagicMock()
        fs._create_dynamic_table = MagicMock()
        fs._create_online_feature_table = MagicMock()
        fs._create_scheduled_refresh_task = MagicMock()
        fs.get_feature_view = MagicMock(return_value=fv)

        with self.assertRaises(Exception) as cm:
            fs.update_feature_view(name=fv, refresh_freq=None)

        # The exact message emitted by validate_snapshot_config_for_update for refresh_freq=None.
        self.assertIn("Snapshot accumulation requires refresh_freq", str(cm.exception))
        # The validation must fail before any Snowflake-side mutation or planning is attempted.
        fs._plan_feature_view_update_operations.assert_not_called()
        fs._execute_atomic_operations.assert_not_called()
        fs._create_dynamic_table.assert_not_called()
        # The caller's FV must remain untouched on this failure path.
        self.assertEqual(fv.refresh_freq, "0 0 * * * UTC")

    def test_omitted_refresh_freq_skips_snapshot_validation_on_append_only(self) -> None:
        """Omitting ``refresh_freq`` on an append_only FV must skip snapshot validation.

        The ``_UNSET`` sentinel distinguishes "caller omitted the argument" from
        "caller explicitly passed ``refresh_freq=None``". Only the latter triggers
        ``validate_snapshot_config_for_update``; the former preserves the FV's
        existing cron schedule without running any check. This is the paired
        complement to ``test_none_refresh_freq_raises_via_validate_snapshot_config_for_update``.
        """
        fs = _create_feature_store_with_mocks()
        fv = _make_valid_snapshot_fv()
        original_refresh_freq = fv.refresh_freq

        fs._validate_feature_view_name_and_version_input = MagicMock(return_value=fv)
        fs._plan_feature_view_update_operations = MagicMock(return_value=([], []))
        fs._execute_atomic_operations = MagicMock()
        fs.get_feature_view = MagicMock(return_value=fv)

        # No refresh_freq argument — validation must be skipped and the update must succeed.
        result = fs.update_feature_view(name=fv, desc="new description")
        self.assertIsNotNone(result)

        # The planner was called with actual_refresh_freq=None (no change) and the
        # new desc — confirming the validation_fv branch was not taken.
        plan_args, _ = fs._plan_feature_view_update_operations.call_args
        self.assertIsNone(plan_args[1], "actual_refresh_freq must be None when caller omits refresh_freq")
        self.assertEqual(plan_args[3], "new description")

        # Caller's FV is untouched — the existing cron schedule is preserved.
        self.assertEqual(fv.refresh_freq, original_refresh_freq)

    def test_rejects_duration_refresh_freq_update_on_append_only(self) -> None:
        """update_feature_view rejects changing refresh_freq to a duration on an append-only FV."""
        fs = _create_feature_store_with_mocks()
        fv = _make_valid_snapshot_fv()

        fs._validate_feature_view_name_and_version_input = MagicMock(return_value=fv)

        with self.assertRaises(Exception) as cm:
            fs.update_feature_view(name=fv, refresh_freq="1 day")
        self.assertIn("cron", str(cm.exception).lower())

    def test_rejects_downstream_refresh_freq_update_on_append_only(self) -> None:
        """update_feature_view rejects changing refresh_freq to DOWNSTREAM on an append-only FV."""
        fs = _create_feature_store_with_mocks()
        fv = _make_valid_snapshot_fv()

        fs._validate_feature_view_name_and_version_input = MagicMock(return_value=fv)

        with self.assertRaises(Exception) as cm:
            fs.update_feature_view(name=fv, refresh_freq="DOWNSTREAM")
        self.assertIn("cron", str(cm.exception).lower())

    def test_accepts_valid_cron_refresh_freq_update_on_append_only(self) -> None:
        """update_feature_view accepts a valid cron refresh_freq on an append-only FV."""
        fs = _create_feature_store_with_mocks()
        fv = _make_valid_snapshot_fv()

        fs._validate_feature_view_name_and_version_input = MagicMock(return_value=fv)
        fs._plan_feature_view_update_operations = MagicMock(return_value=([], []))
        fs._execute_atomic_operations = MagicMock()
        fs.get_feature_view = MagicMock(return_value=fv)

        result = fs.update_feature_view(name=fv, refresh_freq="30 6 * * * UTC")
        self.assertIsNotNone(result)

    def test_accepts_desc_only_update_on_append_only(self) -> None:
        """update_feature_view with only desc change skips snapshot validation entirely."""
        fs = _create_feature_store_with_mocks()
        fv = _make_valid_snapshot_fv()

        fs._validate_feature_view_name_and_version_input = MagicMock(return_value=fv)
        fs._plan_feature_view_update_operations = MagicMock(return_value=([], []))
        fs._execute_atomic_operations = MagicMock()
        fs.get_feature_view = MagicMock(return_value=fv)

        result = fs.update_feature_view(name=fv, desc="updated description")
        self.assertIsNotNone(result)

    def test_accepts_cron_update_on_non_append_only(self) -> None:
        """update_feature_view accepts a CRON refresh_freq on a non-append_only FV.

        Non-append_only FVs use a companion Task to drive cron schedules just
        like append_only FVs, so there is no reason to reject this update.
        Also verifies that the new cron value is propagated to the planner
        and reflected when the feature view is read back.
        """
        fs = _create_feature_store_with_mocks()
        fv = _make_non_snapshot_fv()
        new_cron = "0 0 * * * UTC"

        # Simulate a real round-trip: get_feature_view returns a copy that
        # reflects the new refresh_freq applied by the update.
        updated_fv = copy.copy(fv)
        updated_fv._refresh_freq = new_cron

        fs._validate_feature_view_name_and_version_input = MagicMock(return_value=fv)
        fs._plan_feature_view_update_operations = MagicMock(return_value=([], []))
        fs._execute_atomic_operations = MagicMock()
        fs.get_feature_view = MagicMock(return_value=updated_fv)

        result = fs.update_feature_view(name=fv, refresh_freq=new_cron)
        self.assertIsNotNone(result)
        self.assertEqual(result.refresh_freq, new_cron)

        # The planner should be called with the new cron value as actual_refresh_freq.
        plan_args, _ = fs._plan_feature_view_update_operations.call_args
        self.assertEqual(plan_args[1], new_cron)

        # Reading the feature view back should reflect the updated schedule.
        read_back = fs.get_feature_view(name=fv.name, version=str(fv.version))
        self.assertEqual(read_back.refresh_freq, new_cron)

    def test_updated_feature_df_does_not_mutate_caller_fv(self) -> None:
        """update_feature_view(updated_feature_df=...) must not mutate the caller's FV instance.

        The DF path builds a desired-state copy and recreates the DT/OFT/Task against it.
        The original ``feature_view`` argument supplied by the caller must keep its
        registered state — otherwise the caller's object is silently rewritten and the
        downstream planner (if it ran) would see stale state in its rollback SQL.
        """
        from snowflake.snowpark.types import (
            StringType,
            StructField,
            StructType,
            TimestampType,
        )

        fs = _create_feature_store_with_mocks()
        fv = _make_valid_snapshot_fv()

        original_refresh_freq = fv.refresh_freq
        original_desc = fv.desc
        original_warehouse = fv.warehouse
        original_feature_df = fv._feature_df
        original_query = fv._query
        original_feature_desc = fv._feature_desc
        original_cluster_by = fv._cluster_by

        new_df = MagicMock()
        new_df.session = fs._session
        new_df.queries = {"queries": ["SELECT GUEST_ID, SNAPSHOT_TS, NEW_COL FROM updated_source"]}
        new_df.columns = ["GUEST_ID", "SNAPSHOT_TS", "NEW_COL"]
        new_df.schema = StructType(
            [
                StructField("GUEST_ID", StringType()),
                StructField("SNAPSHOT_TS", TimestampType()),
                StructField("NEW_COL", StringType()),
            ]
        )

        # Snapshot table query: schema matches the DT so extend-only is a no-op.
        snapshot_table_mock = MagicMock()
        snapshot_table_mock.schema = StructType(
            [
                StructField("GUEST_ID", StringType()),
                StructField("SNAPSHOT_TS", TimestampType()),
                StructField("NEW_COL", StringType()),
            ]
        )
        fs._session.table = MagicMock(return_value=snapshot_table_mock)

        fs._validate_feature_view_name_and_version_input = MagicMock(return_value=fv)
        fs._create_dynamic_table = MagicMock()
        fs._create_online_feature_table = MagicMock()
        fs._create_scheduled_refresh_task = MagicMock()
        # The function returns the canonical FV via get_feature_view; for this test
        # we don't care what it returns, only that the caller's FV stays untouched.
        fs.get_feature_view = MagicMock(return_value=copy.copy(fv))
        # The planner must NOT be called on the DF path: the recreate above already
        # applied refresh_freq / warehouse / desc / online_config.
        fs._plan_feature_view_update_operations = MagicMock(return_value=([], []))

        fs.update_feature_view(
            name=fv,
            refresh_freq="30 6 * * * UTC",
            desc="updated description",
            updated_feature_df=new_df,
        )

        # Caller's FV state is untouched.
        self.assertEqual(fv.refresh_freq, original_refresh_freq)
        self.assertEqual(fv.desc, original_desc)
        self.assertEqual(fv.warehouse, original_warehouse)
        self.assertIs(fv._feature_df, original_feature_df)
        self.assertEqual(fv._query, original_query)
        self.assertEqual(fv._feature_desc, original_feature_desc)
        self.assertEqual(fv._cluster_by, original_cluster_by)

        # The recreate must use the desired-state copy (a different object) which
        # carries the new schema, refresh_freq, and desc.
        self.assertTrue(fs._create_dynamic_table.called)
        recreate_fv = fs._create_dynamic_table.call_args[0][1]
        self.assertIsNot(recreate_fv, fv, "Recreate must use a desired-state copy, not the registered FV")
        self.assertEqual(recreate_fv.refresh_freq, "30 6 * * * UTC")
        self.assertEqual(recreate_fv.desc, "updated description")
        self.assertIs(recreate_fv._feature_df, new_df)

        # Planner is skipped on the DF path — the recreate already applied the new state.
        fs._plan_feature_view_update_operations.assert_not_called()

    def test_updated_feature_df_preserves_existing_feature_descriptions(self) -> None:
        """Schema evolution should keep descriptions on existing features and leave new ones blank."""
        from snowflake.snowpark.types import (
            StringType,
            StructField,
            StructType,
            TimestampType,
        )

        fs = _create_feature_store_with_mocks()
        fv = _make_valid_snapshot_fv()
        fv.attach_feature_desc({"N_RSRVS_30_DAY": "existing description"})

        new_df = MagicMock()
        new_df.session = fs._session
        new_df.queries = {"queries": ["SELECT GUEST_ID, SNAPSHOT_TS, N_RSRVS_30_DAY, NEW_COL FROM updated_source"]}
        new_df.columns = ["GUEST_ID", "SNAPSHOT_TS", "N_RSRVS_30_DAY", "NEW_COL"]
        new_df.schema = StructType(
            [
                StructField("GUEST_ID", StringType()),
                StructField("SNAPSHOT_TS", TimestampType()),
                StructField("N_RSRVS_30_DAY", StringType()),
                StructField("NEW_COL", StringType()),
            ]
        )

        snapshot_table_mock = MagicMock()
        snapshot_table_mock.schema = StructType(
            [
                StructField("GUEST_ID", StringType()),
                StructField("SNAPSHOT_TS", TimestampType()),
                StructField("N_RSRVS_30_DAY", StringType()),
                StructField("NEW_COL", StringType()),
            ]
        )
        fs._session.table = MagicMock(return_value=snapshot_table_mock)

        fs._validate_feature_view_name_and_version_input = MagicMock(return_value=fv)
        fs._create_dynamic_table = MagicMock()
        fs._create_online_feature_table = MagicMock()
        fs._create_scheduled_refresh_task = MagicMock()
        fs.get_feature_view = MagicMock(return_value=copy.copy(fv))

        fs.update_feature_view(name=fv, updated_feature_df=new_df)

        recreate_fv = fs._create_dynamic_table.call_args[0][1]
        assert recreate_fv._feature_desc is not None
        self.assertEqual(recreate_fv._feature_desc[SqlIdentifier("N_RSRVS_30_DAY")], "existing description")
        self.assertEqual(recreate_fv._feature_desc[SqlIdentifier("NEW_COL")], "")

    def test_updated_feature_df_uses_caller_mutated_feature_descriptions(self) -> None:
        """When the FeatureView overload carries a new description, the recreate path should keep it."""
        from snowflake.snowpark.types import (
            StringType,
            StructField,
            StructType,
            TimestampType,
        )

        fs = _create_feature_store_with_mocks()
        fv = _make_valid_snapshot_fv()
        fv.attach_feature_desc({"N_RSRVS_30_DAY": "original description"})
        fv.attach_feature_desc({"N_RSRVS_30_DAY": "updated description"})

        new_df = MagicMock()
        new_df.session = fs._session
        new_df.queries = {"queries": ["SELECT GUEST_ID, SNAPSHOT_TS, N_RSRVS_30_DAY, NEW_COL FROM updated_source"]}
        new_df.columns = ["GUEST_ID", "SNAPSHOT_TS", "N_RSRVS_30_DAY", "NEW_COL"]
        new_df.schema = StructType(
            [
                StructField("GUEST_ID", StringType()),
                StructField("SNAPSHOT_TS", TimestampType()),
                StructField("N_RSRVS_30_DAY", StringType()),
                StructField("NEW_COL", StringType()),
            ]
        )

        snapshot_table_mock = MagicMock()
        snapshot_table_mock.schema = StructType(
            [
                StructField("GUEST_ID", StringType()),
                StructField("SNAPSHOT_TS", TimestampType()),
                StructField("N_RSRVS_30_DAY", StringType()),
                StructField("NEW_COL", StringType()),
            ]
        )
        fs._session.table = MagicMock(return_value=snapshot_table_mock)

        fs._validate_feature_view_name_and_version_input = MagicMock(return_value=fv)
        fs._create_dynamic_table = MagicMock()
        fs._create_online_feature_table = MagicMock()
        fs._create_scheduled_refresh_task = MagicMock()
        fs.get_feature_view = MagicMock(return_value=copy.copy(fv))

        fs.update_feature_view(name=fv, updated_feature_df=new_df)

        recreate_fv = fs._create_dynamic_table.call_args[0][1]
        assert recreate_fv._feature_desc is not None
        self.assertEqual(recreate_fv._feature_desc[SqlIdentifier("N_RSRVS_30_DAY")], "updated description")
        self.assertEqual(recreate_fv._feature_desc[SqlIdentifier("NEW_COL")], "")


class BuildOfflineUpdateQueriesRollbackTest(parameterized.TestCase):
    """Tests for the rollback SQL produced by _build_offline_update_queries.

    The planner is contracted to receive the *registered* FeatureView and the *new* values
    as separate arguments. Rollback SQL is built from ``feature_view.refresh_freq`` /
    ``.warehouse`` / ``.desc`` so it must reflect the pre-update state. Mutating the FV
    before calling the planner silently breaks rollback (it "restores" the new values).
    """

    def test_cron_to_cron_update_rollback_references_old_cron(self) -> None:
        """Updating one cron expression to another must roll back to the original cron."""
        fs = _create_feature_store_with_mocks()
        fv = _make_non_snapshot_fv()
        # _make_non_snapshot_fv() builds a duration FV; flip it to cron so old_is_cron is True.
        fv._refresh_freq = "0 0 * * * UTC"
        fv._warehouse = SqlIdentifier("WH_1")
        fv._desc = "original desc"

        new_cron = "30 6 * * * UTC"

        operations, rollback_ops = fs._build_offline_update_queries(
            fv, refresh_freq=new_cron, warehouse=None, desc="new desc"
        )

        # Forward operations apply the new cron to the companion Task.
        forward_sql = " ".join(sql for _, sql in operations)
        self.assertIn(f"USING CRON {new_cron}", forward_sql)

        # Rollback operations must reference the *old* cron, not the new one.
        rollback_sql = " ".join(sql for _, sql in rollback_ops)
        self.assertIn("USING CRON 0 0 * * * UTC", rollback_sql)
        self.assertNotIn(f"USING CRON {new_cron}", rollback_sql)
        self.assertIn("COMMENT = 'original desc'", rollback_sql)

    @parameterized.named_parameters(  # type: ignore[misc]
        ("upper", "DOWNSTREAM"),
        ("lower", "downstream"),
        ("mixed", "Downstream"),
    )
    def test_downstream_update_is_case_insensitive(self, new_freq: str) -> None:
        """Updating a cron FV to ``DOWNSTREAM`` must classify as non-cron regardless of case.

        ``refresh_freq`` is stored verbatim on the FV (no normalization in the constructor
        or setter), so the planner's cron-vs-duration classifier must be case-insensitive.
        Otherwise a lowercase/mixed-case ``"downstream"`` would be treated as a cron
        expression and the planner would emit a ``CREATE OR REPLACE TASK ... SCHEDULE =
        'USING CRON downstream'`` which Snowflake would reject at execution.

        The expected transition is cron → DOWNSTREAM: drop the companion Task and set
        ``TARGET_LAG`` to the literal value the user provided (Snowflake accepts any case).
        """
        fs = _create_feature_store_with_mocks()
        fv = _make_non_snapshot_fv()
        fv._refresh_freq = "0 0 * * * UTC"
        fv._warehouse = SqlIdentifier("WH_1")
        fv._desc = "original desc"

        operations, rollback_ops = fs._build_offline_update_queries(
            fv, refresh_freq=new_freq, warehouse=None, desc="new desc"
        )

        forward_sqls = [sql for _, sql in operations]
        # No forward task creation/schedule mutation — DOWNSTREAM is not a cron.
        self.assertFalse(
            any("CREATE OR REPLACE TASK" in sql or "SET SCHEDULE" in sql for sql in forward_sqls),
            f"Lowercase/mixed-case '{new_freq}' must not be classified as cron. Got: {forward_sqls}",
        )
        # Cron → non-cron: drop the orphaned task.
        self.assertTrue(
            any("DROP TASK IF EXISTS" in sql for sql in forward_sqls),
            f"Cron → DOWNSTREAM transition must drop the companion task. Got: {forward_sqls}",
        )

        # Rollback restores the original cron schedule on the Task.
        rollback_sql = " ".join(sql for _, sql in rollback_ops)
        self.assertIn("USING CRON 0 0 * * * UTC", rollback_sql)

    @parameterized.named_parameters(  # type: ignore[misc]
        ("upper", "DOWNSTREAM"),
        ("lower", "downstream"),
        ("mixed", "Downstream"),
    )
    def test_old_downstream_to_cron_update_is_case_insensitive(self, old_freq: str) -> None:
        """A registered FV with case-variant ``DOWNSTREAM`` must be classified as non-cron.

        The registered FV's ``refresh_freq`` is stored verbatim from whatever the caller
        passed at registration time. If the planner treats lowercase ``"downstream"`` as
        a cron expression, the DOWNSTREAM → cron transition would emit ``ALTER TASK ...
        SET SCHEDULE`` against a Task that doesn't exist, instead of the correct
        ``CREATE OR REPLACE TASK``.
        """
        fs = _create_feature_store_with_mocks()
        fv = _make_non_snapshot_fv()
        fv._refresh_freq = old_freq
        fv._warehouse = SqlIdentifier("WH_1")
        fv._desc = "original desc"

        new_cron = "30 6 * * * UTC"
        operations, _rollback_ops = fs._build_offline_update_queries(
            fv, refresh_freq=new_cron, warehouse=None, desc="new desc"
        )
        forward_sqls = [sql for _, sql in operations]

        # Non-cron → cron: must CREATE the task (not ALTER an existing one).
        self.assertTrue(
            any("CREATE OR REPLACE TASK" in sql for sql in forward_sqls),
            f"Old refresh_freq '{old_freq}' must be classified as non-cron, "
            f"so the transition emits CREATE OR REPLACE TASK. Got: {forward_sqls}",
        )
        self.assertFalse(
            any("SET SCHEDULE" in sql for sql in forward_sqls),
            f"Old refresh_freq '{old_freq}' is non-cron; ALTER TASK SET SCHEDULE would "
            f"target a Task that doesn't exist. Got: {forward_sqls}",
        )


class RefreshFeatureViewSnapshotTest(absltest.TestCase):
    """Tests for refresh_feature_view rejection when snapshot is enabled."""

    def test_rejects_offline_refresh_when_append_only(self) -> None:
        """Manual offline refresh is not supported for append-only feature views."""
        fs = _create_feature_store_with_mocks()
        fv = _make_valid_snapshot_fv()

        fs._validate_feature_view_name_and_version_input = MagicMock(return_value=fv)

        with self.assertRaises(Exception) as cm:
            fs.refresh_feature_view(feature_view=fv, store_type="OFFLINE")
        self.assertIn("append_only", str(cm.exception).lower())
        self.assertIn("Manual refresh", str(cm.exception))

    def test_rejects_online_refresh_when_append_only(self) -> None:
        """Manual online refresh is not supported for append-only feature views."""
        fs = _create_feature_store_with_mocks()
        fv = _make_valid_snapshot_fv()

        fs._validate_feature_view_name_and_version_input = MagicMock(return_value=fv)

        with self.assertRaises(Exception) as cm:
            fs.refresh_feature_view(feature_view=fv, store_type="ONLINE")
        self.assertIn("append_only", str(cm.exception).lower())
        self.assertIn("Manual refresh", str(cm.exception))

    def test_allows_refresh_when_no_snapshot(self) -> None:
        """Manual refresh is allowed for non-snapshot feature views."""
        fs = _create_feature_store_with_mocks()
        fv = _make_non_snapshot_fv()

        fs._validate_feature_view_name_and_version_input = MagicMock(return_value=fv)
        fs._update_feature_view_status = MagicMock()

        fs.refresh_feature_view(feature_view=fv)
        fs._update_feature_view_status.assert_called_once()


class ComposeFeatureViewAppendOnlyTest(absltest.TestCase):
    """Tests that _compose_feature_view correctly propagates append_only from metadata."""

    def test_append_only_metadata_roundtrip(self) -> None:
        metadata = AppendOnlyMetadata(backup_source="DB.SCH.HISTORY")
        restored = AppendOnlyMetadata.from_dict(metadata.to_dict())
        self.assertEqual(restored.backup_source, "DB.SCH.HISTORY")

    def test_append_only_metadata_roundtrip_without_backup_source(self) -> None:
        metadata = AppendOnlyMetadata()
        restored = AppendOnlyMetadata.from_dict(metadata.to_dict())
        self.assertIsNone(restored.backup_source)

    def test_compose_feature_view_preserves_append_only(self) -> None:
        """_compose_feature_view should set append_only=True when metadata has is_append_only=True."""
        from unittest.mock import patch

        from snowflake.ml.feature_store.feature_store import _FeatureStoreObjTypes
        from snowflake.ml.feature_store.feature_view import _FeatureViewMetadata

        fs = _create_feature_store_with_mocks()

        metadata = _FeatureViewMetadata(
            entities=["GUEST"],
            timestamp_col="SNAPSHOT_TS",
            is_append_only=True,
        )

        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: {
            "name": "SNAP_FV$V1",
            "text": "CREATE DYNAMIC TABLE x initialize = 'ON_SCHEDULE' AS SELECT 1",
            "comment": "test",
            "target_lag": "0 0 * * * UTC",
            "scheduling_state": "ACTIVE",
            "warehouse": "WH_1",
            "refresh_mode": "FULL",
            "refresh_mode_reason": "",
            "owner": "ROLE_1",
            "cluster_by": "",
        }[key]

        entity_row = MagicMock()
        entity_row.__getitem__ = lambda self, key: {
            "NAME": "GUEST",
            "JOIN_KEYS": '["GUEST_ID"]',
            "DESC": "",
        }[key]

        mock_df = MagicMock()
        mock_df.columns = ["GUEST_ID", "SNAPSHOT_TS", "SCORE"]
        mock_df.queries = {"queries": ["SELECT 1"]}

        from snowflake.snowpark.types import TimestampType

        ts_field = MagicMock()
        ts_field.datatype = TimestampType()
        mock_df.schema.__getitem__ = lambda self, key: ts_field

        fs._session.sql.return_value = mock_df

        fs._lookup_feature_view_metadata = MagicMock(return_value=(metadata, "SELECT 1"))
        fs._determine_online_config_from_oft = MagicMock(return_value='{"enable": false}')
        fs._metadata_manager.get_feature_specs.return_value = None
        fs._metadata_manager.get_append_only_metadata.return_value = {"backup_source": "DB.SCH.HISTORY"}
        fs._extract_cluster_by_columns = MagicMock(return_value=None)

        with patch.object(FeatureView, "_construct_feature_view", wraps=FeatureView._construct_feature_view) as mock_ct:
            fv = fs._compose_feature_view(mock_row, _FeatureStoreObjTypes.MANAGED_FEATURE_VIEW, [entity_row])
            _, kwargs = mock_ct.call_args
            self.assertTrue(kwargs.get("append_only", False))
            self.assertEqual(kwargs.get("backup_source"), "DB.SCH.HISTORY")

        self.assertTrue(fv.append_only)
        self.assertEqual(fv.backup_source, "DB.SCH.HISTORY")


class AppendOnlyJoinValidationTest(absltest.TestCase):
    """Tests that append_only FVs require spine_timestamp_col in generate_training_set/generate_dataset."""

    def _spine_df(self) -> Any:
        spine_df = MagicMock()
        spine_df.columns = ["GUEST_ID"]
        spine_df.queries = {"queries": ["SELECT GUEST_ID FROM spine"]}
        return spine_df

    def test_append_only_without_spine_timestamp_raises(self) -> None:
        """_join_features rejects append_only FVs when spine_timestamp_col is None."""
        fs = _create_feature_store_with_mocks()
        fv = _make_valid_snapshot_fv()
        fv._status = FeatureViewStatus.ACTIVE
        fv._version = FeatureViewVersion("v1")

        with self.assertRaisesRegex(Exception, "append_only.*requires spine_timestamp_col"):
            fs._join_features(
                spine_df=self._spine_df(),
                features=[fv],
                spine_timestamp_col=None,
                include_feature_view_timestamp_col=False,
            )

    def test_append_only_detected_when_not_first_in_features_list(self) -> None:
        """The detection loop must scan past non-append_only FVs to find a later append_only FV.

        Regression guard: an early ``break`` after detecting a non-append_only FV would misclassify
        this list as "no append_only present" and silently let the join proceed without
        ``spine_timestamp_col`` — bypassing snapshot history.
        """
        fs = _create_feature_store_with_mocks()
        regular_fv = _make_non_snapshot_fv()
        snapshot_fv = _make_valid_snapshot_fv()

        with self.assertRaisesRegex(Exception, "SNAP_FV.*append_only.*requires spine_timestamp_col"):
            fs._join_features(
                spine_df=self._spine_df(),
                features=[regular_fv, snapshot_fv],
                spine_timestamp_col=None,
                include_feature_view_timestamp_col=False,
            )

    def test_first_append_only_fv_named_in_error_when_multiple_present(self) -> None:
        """When multiple append_only FVs are present, the error names the first one.

        The ``first_append_only_fv is None`` guard in the detection loop must hold the
        first match; without it the error would name an arbitrary append_only FV (e.g.
        the last one) and obscure which FV in the caller's list triggered the check.
        """
        fs = _create_feature_store_with_mocks()
        first_snapshot_fv = _make_valid_snapshot_fv()
        first_snapshot_fv._name = SqlIdentifier("FIRST_SNAP_FV")
        second_snapshot_fv = _make_valid_snapshot_fv()
        second_snapshot_fv._name = SqlIdentifier("SECOND_SNAP_FV")

        with self.assertRaisesRegex(Exception, "FIRST_SNAP_FV.*append_only.*requires spine_timestamp_col"):
            fs._join_features(
                spine_df=self._spine_df(),
                features=[first_snapshot_fv, second_snapshot_fv],
                spine_timestamp_col=None,
                include_feature_view_timestamp_col=False,
            )

    def test_append_only_detected_through_feature_view_slice(self) -> None:
        """The detection loop must unwrap ``FeatureViewSlice`` to inspect ``append_only``.

        Callers can pass slices (the result of ``feature_view.slice([...])``) instead of
        bare FeatureViews. A type confusion that treats slices as never-append_only would
        silently bypass snapshot history when the caller selects a column subset.
        """
        from snowflake.ml.feature_store.feature_view import FeatureViewSlice

        fs = _create_feature_store_with_mocks()
        snapshot_fv = _make_valid_snapshot_fv()
        snapshot_slice = FeatureViewSlice(
            feature_view_ref=snapshot_fv,
            names=[SqlIdentifier("N_RSRVS_30_DAY")],
        )

        with self.assertRaisesRegex(Exception, "SNAP_FV.*append_only.*requires spine_timestamp_col"):
            fs._join_features(
                spine_df=self._spine_df(),
                features=[snapshot_slice],
                spine_timestamp_col=None,
                include_feature_view_timestamp_col=False,
            )

    def test_tiled_error_takes_precedence_over_append_only_when_both_present(self) -> None:
        """When both tiled and append_only FVs are present, the tiled error wins.

        The ``if spine_timestamp_col is None:`` block intentionally raises the tiled
        ValueError before the append_only check. This ordering pins behavior so a future
        reorder is a deliberate decision rather than a silent regression.

        ``join_method='cte'`` is required to clear the unrelated tiled-FV ``join_method``
        gate that fires earlier; this test isolates the ``spine_timestamp_col`` precedence.
        """
        fs = _create_feature_store_with_mocks()
        snapshot_fv = _make_valid_snapshot_fv()
        tiled_fv = _make_non_snapshot_fv()
        # Make ``is_tiled`` return True without invoking the full tile-aggregation setup;
        # the property is derived from these two private fields being non-None.
        tiled_fv._feature_granularity = MagicMock()
        tiled_fv._aggregation_specs = MagicMock()
        assert tiled_fv.is_tiled

        with self.assertRaisesRegex(ValueError, "Tiled feature views require a spine_timestamp_col"):
            fs._join_features(
                spine_df=self._spine_df(),
                features=[snapshot_fv, tiled_fv],
                spine_timestamp_col=None,
                include_feature_view_timestamp_col=False,
                join_method="cte",
            )


class RecreateAppendOnlyFeatureViewAtomicallyTest(absltest.TestCase):
    """Tests for the rollback semantics of _recreate_append_only_feature_view_atomically.

    The four-step recreate (snapshot ALTER ADD COLUMN, DT CREATE OR REPLACE, online table
    CREATE OR REPLACE, refresh task CREATE OR REPLACE) must run compensating actions in
    reverse on any forward failure so the feature view stays consistent. These tests
    exercise the failure paths and verify the registered compensating actions fire.
    """

    def _make_old_snapshot_schema(self) -> Any:
        from snowflake.snowpark.types import (
            StringType,
            StructField,
            StructType,
            TimestampType,
        )

        return StructType(
            [
                StructField("GUEST_ID", StringType()),
                StructField("SNAPSHOT_TS", TimestampType()),
                StructField("N_RSRVS_30_DAY", StringType()),
            ]
        )

    def _new_schema_appending_one_col(self) -> Any:
        from snowflake.snowpark.types import LongType, StringType, TimestampType

        return {
            "GUEST_ID": StringType(),
            "SNAPSHOT_TS": TimestampType(),
            "N_RSRVS_30_DAY": StringType(),
            "NEW_COL": LongType(),
        }

    def _setup(self) -> tuple[Any, FeatureView, FeatureView, dict[str, Any]]:
        fs = _create_feature_store_with_mocks()
        old_fv = _make_valid_snapshot_fv()
        new_fv = copy.copy(old_fv)
        new_fv._desc = "updated"

        snapshot_table_mock = MagicMock()
        snapshot_table_mock.schema = self._make_old_snapshot_schema()
        fs._session.table = MagicMock(return_value=snapshot_table_mock)

        # Default: every sql(...).collect(...) succeeds.
        fs._session.sql = MagicMock(return_value=MagicMock(collect=MagicMock(return_value=None)))

        new_schema = self._new_schema_appending_one_col()
        return fs, old_fv, new_fv, new_schema

    @staticmethod
    def _executed_sql(fs: Any) -> list[str]:
        return [c.args[0] for c in fs._session.sql.call_args_list]

    def test_dt_recreate_failure_drops_added_snapshot_columns(self) -> None:
        """If DT CREATE OR REPLACE fails, the snapshot ADD COLUMN must be reversed."""
        fs, old_fv, new_fv, new_schema = self._setup()

        fs._create_dynamic_table = MagicMock(side_effect=RuntimeError("DT recreate failed"))
        fs._create_online_feature_table = MagicMock()
        fs._create_scheduled_refresh_task = MagicMock()

        with self.assertRaisesRegex(Exception, "DT recreate failed"):
            fs._recreate_append_only_feature_view_atomically(
                old_feature_view=old_fv,
                new_feature_view=new_fv,
                feature_view_name=SqlIdentifier("SNAP_FV$V1"),
                fully_qualified_name="TEST_DB.TEST_SCHEMA.SNAP_FV$V1",
                version="V1",
                new_schema=new_schema,
            )

        executed = self._executed_sql(fs)
        add_idx = next((i for i, s in enumerate(executed) if 'ADD COLUMN "NEW_COL"' in s), -1)
        drop_idx = next((i for i, s in enumerate(executed) if 'DROP COLUMN "NEW_COL"' in s), -1)
        self.assertGreaterEqual(add_idx, 0, f"Expected ADD COLUMN, got {executed}")
        self.assertGreater(drop_idx, add_idx, f"Expected DROP COLUMN after ADD COLUMN, got {executed}")

        fs._create_online_feature_table.assert_not_called()
        fs._create_scheduled_refresh_task.assert_not_called()

    def test_partial_dt_recreate_failure_still_runs_dt_rollback(self) -> None:
        """If DT recreate mutates Snowflake before raising, rollback must still restore the old DT."""
        fs, old_fv, new_fv, new_schema = self._setup()

        dt_calls = {"n": 0}

        def _dt_side_effect(*args: Any, **kwargs: Any) -> None:
            del kwargs
            dt_calls["n"] += 1
            marker = "FORWARD_DT_PARTIAL" if dt_calls["n"] == 1 else "ROLLBACK_DT_RESTORE"
            fs._session.sql(f"SELECT '{marker}'").collect()
            if dt_calls["n"] == 1:
                raise RuntimeError("DT recreate post-create failure")

        fs._create_dynamic_table = MagicMock(side_effect=_dt_side_effect)
        fs._create_online_feature_table = MagicMock()
        fs._create_scheduled_refresh_task = MagicMock()

        with self.assertRaisesRegex(Exception, "DT recreate post-create failure"):
            fs._recreate_append_only_feature_view_atomically(
                old_feature_view=old_fv,
                new_feature_view=new_fv,
                feature_view_name=SqlIdentifier("SNAP_FV$V1"),
                fully_qualified_name="TEST_DB.TEST_SCHEMA.SNAP_FV$V1",
                version="V1",
                new_schema=new_schema,
            )

        self.assertEqual(fs._create_dynamic_table.call_count, 2)
        forward_fv = fs._create_dynamic_table.call_args_list[0].args[1]
        rollback_fv = fs._create_dynamic_table.call_args_list[1].args[1]
        self.assertIs(forward_fv, new_fv)
        self.assertIs(rollback_fv, old_fv)

        executed = self._executed_sql(fs)
        forward_idx = next((i for i, s in enumerate(executed) if "FORWARD_DT_PARTIAL" in s), -1)
        rollback_idx = next((i for i, s in enumerate(executed) if "ROLLBACK_DT_RESTORE" in s), -1)
        drop_idx = next((i for i, s in enumerate(executed) if 'DROP COLUMN "NEW_COL"' in s), -1)
        self.assertGreaterEqual(forward_idx, 0, f"Expected forward DT marker, got {executed}")
        self.assertGreater(
            rollback_idx, forward_idx, f"Expected DT rollback after partial forward mutate, got {executed}"
        )
        self.assertGreater(drop_idx, rollback_idx, f"Expected snapshot rollback after DT rollback, got {executed}")

        fs._create_online_feature_table.assert_not_called()
        fs._create_scheduled_refresh_task.assert_not_called()

    def test_task_recreate_failure_runs_dt_and_schema_rollback(self) -> None:
        """If task CREATE OR REPLACE fails, both the DT and snapshot mutations must be reversed."""
        fs, old_fv, new_fv, new_schema = self._setup()

        fs._create_dynamic_table = MagicMock()
        fs._create_online_feature_table = MagicMock()
        fs._create_scheduled_refresh_task = MagicMock(side_effect=RuntimeError("task recreate failed"))

        with self.assertRaisesRegex(Exception, "task recreate failed"):
            fs._recreate_append_only_feature_view_atomically(
                old_feature_view=old_fv,
                new_feature_view=new_fv,
                feature_view_name=SqlIdentifier("SNAP_FV$V1"),
                fully_qualified_name="TEST_DB.TEST_SCHEMA.SNAP_FV$V1",
                version="V1",
                new_schema=new_schema,
            )

        # _create_dynamic_table called twice: once forward (with new_fv) and once for rollback (with old_fv).
        self.assertEqual(fs._create_dynamic_table.call_count, 2)
        forward_fv = fs._create_dynamic_table.call_args_list[0].args[1]
        rollback_fv = fs._create_dynamic_table.call_args_list[1].args[1]
        self.assertIs(forward_fv, new_fv)
        self.assertIs(rollback_fv, old_fv)

        # Schema ADD/DROP must both have been issued.
        executed = self._executed_sql(fs)
        self.assertTrue(any('ADD COLUMN "NEW_COL"' in s for s in executed))
        self.assertTrue(any('DROP COLUMN "NEW_COL"' in s for s in executed))

    def test_partial_task_recreate_failure_rolls_back_dt_before_task(self) -> None:
        """If task recreate mutates Snowflake before raising, rollback must restore the old DT first."""
        fs, old_fv, new_fv, new_schema = self._setup()

        dt_calls = {"n": 0}
        task_calls = {"n": 0}

        def _dt_side_effect(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            dt_calls["n"] += 1
            marker = "FORWARD_DT_RECREATE" if dt_calls["n"] == 1 else "ROLLBACK_DT_RESTORE"
            fs._session.sql(f"SELECT '{marker}'").collect()

        def _task_side_effect(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            task_calls["n"] += 1
            marker = "FORWARD_TASK_PARTIAL" if task_calls["n"] == 1 else "ROLLBACK_TASK_RESTORE"
            fs._session.sql(f"SELECT '{marker}'").collect()
            if task_calls["n"] == 1:
                raise RuntimeError("task recreate post-create failure")

        fs._create_dynamic_table = MagicMock(side_effect=_dt_side_effect)
        fs._create_online_feature_table = MagicMock()
        fs._create_scheduled_refresh_task = MagicMock(side_effect=_task_side_effect)

        with self.assertRaisesRegex(Exception, "task recreate post-create failure"):
            fs._recreate_append_only_feature_view_atomically(
                old_feature_view=old_fv,
                new_feature_view=new_fv,
                feature_view_name=SqlIdentifier("SNAP_FV$V1"),
                fully_qualified_name="TEST_DB.TEST_SCHEMA.SNAP_FV$V1",
                version="V1",
                new_schema=new_schema,
            )

        self.assertEqual(fs._create_dynamic_table.call_count, 2)
        self.assertEqual(fs._create_scheduled_refresh_task.call_count, 2)

        executed = self._executed_sql(fs)
        rollback_dt_idx = next((i for i, s in enumerate(executed) if "ROLLBACK_DT_RESTORE" in s), -1)
        rollback_task_idx = next((i for i, s in enumerate(executed) if "ROLLBACK_TASK_RESTORE" in s), -1)
        drop_idx = next((i for i, s in enumerate(executed) if 'DROP COLUMN "NEW_COL"' in s), -1)
        self.assertGreaterEqual(rollback_dt_idx, 0, f"Expected DT rollback marker, got {executed}")
        self.assertGreater(
            rollback_task_idx, rollback_dt_idx, f"Expected DT rollback before task rollback, got {executed}"
        )
        self.assertGreater(
            drop_idx, rollback_task_idx, f"Expected snapshot rollback after task rollback, got {executed}"
        )

    def test_partial_schema_evolution_failure_does_not_register_outer_compensating_action(self) -> None:
        """If snapshot ALTER ADD COLUMN itself fails, inline rollback runs and no DT change is attempted."""
        from snowflake.snowpark.types import LongType, StringType, TimestampType

        fs, old_fv, new_fv, _ = self._setup()

        # Two-column extension: first ADD succeeds, second ADD fails. The inline rollback
        # in step 1 should attempt DROP for both (the second DROP will fail because the
        # column doesn't exist; that failure is logged, not raised).
        new_schema = {
            "GUEST_ID": StringType(),
            "SNAPSHOT_TS": TimestampType(),
            "N_RSRVS_30_DAY": StringType(),
            "NEW_COL_1": LongType(),
            "NEW_COL_2": LongType(),
        }

        call_count = {"n": 0}

        def _sql_side_effect(sql: str) -> Any:
            call_count["n"] += 1
            mock = MagicMock()
            if 'ADD COLUMN "NEW_COL_2"' in sql:
                mock.collect = MagicMock(side_effect=RuntimeError("second ADD failed"))
            else:
                mock.collect = MagicMock(return_value=None)
            return mock

        fs._session.sql = MagicMock(side_effect=_sql_side_effect)

        fs._create_dynamic_table = MagicMock()
        fs._create_online_feature_table = MagicMock()
        fs._create_scheduled_refresh_task = MagicMock()

        with self.assertRaisesRegex(Exception, "second ADD failed"):
            fs._recreate_append_only_feature_view_atomically(
                old_feature_view=old_fv,
                new_feature_view=new_fv,
                feature_view_name=SqlIdentifier("SNAP_FV$V1"),
                fully_qualified_name="TEST_DB.TEST_SCHEMA.SNAP_FV$V1",
                version="V1",
                new_schema=new_schema,
            )

        # No CREATE OR REPLACE was attempted because step 1 failed.
        fs._create_dynamic_table.assert_not_called()
        fs._create_online_feature_table.assert_not_called()
        fs._create_scheduled_refresh_task.assert_not_called()

        # Inline rollback issued DROP COLUMN for the column that succeeded.
        executed = [c.args[0] for c in fs._session.sql.call_args_list]
        self.assertTrue(
            any('DROP COLUMN "NEW_COL_1"' in s for s in executed),
            f"Expected inline DROP COLUMN for NEW_COL_1, got {executed}",
        )

    def test_successful_recreate_does_not_run_any_rollback(self) -> None:
        """A successful recreate path should not issue any DROP COLUMN."""
        fs, old_fv, new_fv, new_schema = self._setup()

        fs._create_dynamic_table = MagicMock()
        fs._create_online_feature_table = MagicMock()
        fs._create_scheduled_refresh_task = MagicMock()

        fs._recreate_append_only_feature_view_atomically(
            old_feature_view=old_fv,
            new_feature_view=new_fv,
            feature_view_name=SqlIdentifier("SNAP_FV$V1"),
            fully_qualified_name="TEST_DB.TEST_SCHEMA.SNAP_FV$V1",
            version="V1",
            new_schema=new_schema,
        )

        # Forward DT/task happened exactly once each; no rollback recreate.
        self.assertEqual(fs._create_dynamic_table.call_count, 1)
        self.assertEqual(fs._create_scheduled_refresh_task.call_count, 1)

        executed = self._executed_sql(fs)
        self.assertTrue(any('ADD COLUMN "NEW_COL"' in s for s in executed))
        self.assertFalse(
            any('DROP COLUMN "NEW_COL"' in s for s in executed),
            f"Did not expect DROP COLUMN on success, got {executed}",
        )

    def test_oft_recreate_failure_with_online_recreates_online_from_old_fv(self) -> None:
        """When both old and new FVs are online and OFT recreation fails, the online
        compensating action must recreate the online table from the old FV.

        OFT is the final forward step (sequenced after DT and task to mirror the
        register path), so its failure is the case that exercises every prior
        compensating action — DT, task, and OFT itself — running in dependency
        order.
        """
        from snowflake.ml.feature_store.feature_view import OnlineConfig

        fs, old_fv, new_fv, new_schema = self._setup()

        # Both FVs online; new differs from old only in description.
        old_fv._online_config = OnlineConfig(enable=True)
        new_fv._online_config = OnlineConfig(enable=True)

        fs._create_dynamic_table = MagicMock()
        fs._create_online_feature_table = MagicMock(side_effect=RuntimeError("OFT recreate failed"))
        fs._create_scheduled_refresh_task = MagicMock()

        with self.assertRaisesRegex(Exception, "OFT recreate failed"):
            fs._recreate_append_only_feature_view_atomically(
                old_feature_view=old_fv,
                new_feature_view=new_fv,
                feature_view_name=SqlIdentifier("SNAP_FV$V1"),
                fully_qualified_name="TEST_DB.TEST_SCHEMA.SNAP_FV$V1",
                version="V1",
                new_schema=new_schema,
            )

        # Online recreate happened twice: once forward (new_fv) and once for rollback (old_fv).
        self.assertEqual(fs._create_online_feature_table.call_count, 2)
        forward_online_fv = fs._create_online_feature_table.call_args_list[0].args[0]
        rollback_online_fv = fs._create_online_feature_table.call_args_list[1].args[0]
        self.assertIs(forward_online_fv, new_fv)
        self.assertIs(rollback_online_fv, old_fv)

        # DT recreate happened twice: forward (new_fv) and rollback (old_fv).
        self.assertEqual(fs._create_dynamic_table.call_count, 2)
        forward_dt_fv = fs._create_dynamic_table.call_args_list[0].args[1]
        rollback_dt_fv = fs._create_dynamic_table.call_args_list[1].args[1]
        self.assertIs(forward_dt_fv, new_fv)
        self.assertIs(rollback_dt_fv, old_fv)

        # Task recreate happened twice: forward (new_fv) and rollback (old_fv).
        self.assertEqual(fs._create_scheduled_refresh_task.call_count, 2)
        forward_task_fv = fs._create_scheduled_refresh_task.call_args_list[0].args[1]
        rollback_task_fv = fs._create_scheduled_refresh_task.call_args_list[1].args[1]
        self.assertIs(forward_task_fv, new_fv)
        self.assertIs(rollback_task_fv, old_fv)

        # Schema ADD/DROP both issued.
        executed = self._executed_sql(fs)
        self.assertTrue(any('ADD COLUMN "NEW_COL"' in s for s in executed))
        self.assertTrue(any('DROP COLUMN "NEW_COL"' in s for s in executed))

    def test_rollback_runs_in_dependency_order_dt_oft_task_schema(self) -> None:
        """Compensating actions must run in DT -> OFT -> Task -> Schema order.

        Compensating actions are appended in registration order (schema, then DT, then
        task, then OFT — each registered just before its forward call so partial-success
        failures still schedule rollback). The handler must not unwind that list by
        reverse-append: it walks explicit dependency tiers instead.

        Rationale (matches ``_recreate_append_only_feature_view_atomically``): restore
        the running pipeline (DT, then OFT, then refresh task) first so a
        partially-recovered FV is queryable as soon as possible; run snapshot
        ``ALTER DROP COLUMN`` last because nothing downstream depends on it. Within the
        pipeline tier, OFT rollback in HYBRID_TABLE mode runs ``CREATE ... FROM`` the
        current source DT, so the DT rollback must run before the OFT rollback so the OFT
        inherits the correct column shape. The refresh task body is column-agnostic
        (``SELECT *``), so its rollback follows OFT.

        OFT is the final forward step here, so failing it exercises every compensating
        action. Each mock emits marker SQL on rollback so order is visible in
        ``fs._session.sql`` call history.
        """
        from snowflake.ml.feature_store.feature_view import OnlineConfig

        fs, old_fv, new_fv, new_schema = self._setup()

        old_fv._online_config = OnlineConfig(enable=True)
        new_fv._online_config = OnlineConfig(enable=True)

        dt_calls = {"n": 0}
        oft_calls = {"n": 0}
        task_calls = {"n": 0}

        def _dt_side_effect(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            dt_calls["n"] += 1
            marker = "FORWARD_DT" if dt_calls["n"] == 1 else "ROLLBACK_DT"
            fs._session.sql(f"SELECT '{marker}'").collect()

        def _oft_side_effect(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            oft_calls["n"] += 1
            marker = "FORWARD_OFT" if oft_calls["n"] == 1 else "ROLLBACK_OFT"
            fs._session.sql(f"SELECT '{marker}'").collect()
            if oft_calls["n"] == 1:
                raise RuntimeError("OFT recreate failed")

        def _task_side_effect(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            task_calls["n"] += 1
            marker = "FORWARD_TASK" if task_calls["n"] == 1 else "ROLLBACK_TASK"
            fs._session.sql(f"SELECT '{marker}'").collect()

        fs._create_dynamic_table = MagicMock(side_effect=_dt_side_effect)
        fs._create_online_feature_table = MagicMock(side_effect=_oft_side_effect)
        fs._create_scheduled_refresh_task = MagicMock(side_effect=_task_side_effect)

        with self.assertRaisesRegex(Exception, "OFT recreate failed"):
            fs._recreate_append_only_feature_view_atomically(
                old_feature_view=old_fv,
                new_feature_view=new_fv,
                feature_view_name=SqlIdentifier("SNAP_FV$V1"),
                fully_qualified_name="TEST_DB.TEST_SCHEMA.SNAP_FV$V1",
                version="V1",
                new_schema=new_schema,
            )

        executed = self._executed_sql(fs)
        forward_oft_idx = next((i for i, s in enumerate(executed) if "FORWARD_OFT" in s), -1)
        rollback_dt_idx = next((i for i, s in enumerate(executed) if "ROLLBACK_DT" in s), -1)
        rollback_oft_idx = next((i for i, s in enumerate(executed) if "ROLLBACK_OFT" in s), -1)
        rollback_task_idx = next((i for i, s in enumerate(executed) if "ROLLBACK_TASK" in s), -1)
        schema_drop_idx = next((i for i, s in enumerate(executed) if 'DROP COLUMN "NEW_COL"' in s), -1)

        self.assertGreaterEqual(forward_oft_idx, 0, f"Expected forward OFT marker, got {executed}")
        self.assertGreater(
            rollback_dt_idx, forward_oft_idx, f"DT rollback must run after the OFT failure, got {executed}"
        )
        self.assertGreater(rollback_oft_idx, rollback_dt_idx, f"OFT rollback must follow DT rollback, got {executed}")
        self.assertGreater(
            rollback_task_idx, rollback_oft_idx, f"Task rollback must follow OFT rollback, got {executed}"
        )
        self.assertGreater(schema_drop_idx, rollback_task_idx, f"Schema rollback must run last, got {executed}")


if __name__ == "__main__":
    absltest.main()
