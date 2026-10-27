"""Reconstruction of feature views that no longer satisfy the authoring contract.

``_compose_feature_view`` rebuilds a ``FeatureView`` from applied state. Applied
state is *descriptive*: it reports what exists in Snowflake, not what the current
authoring rules would allow someone to create. Re-running construction-time
invariants over it strands any object deployed under an older contract — it can
then be neither read, updated, nor dropped.

The concrete case pinned here is a tiled feature view with no ``refresh_freq``.
Registering one produces a plain VIEW (``EXTERNAL_FEATURE_VIEW``) rather than a
Dynamic Table, and the external branch of ``_compose_feature_view`` hardcodes
``refresh_freq=None`` while still forwarding the persisted tiling metadata, so the
cadence check fired on every single read.

A VIEW-backed feature view is otherwise a perfectly ordinary, supported kind — it
is how a feature view sits over a source that is not a Dynamic Table — so the
non-tiled case is pinned alongside to keep that path demonstrably unaffected.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from absl.testing import absltest

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature import Feature
from snowflake.ml.feature_store.feature_view import FeatureView, FeatureViewStatus
from snowflake.ml.feature_store.metadata_manager import AggregationMetadata
from snowflake.ml.feature_store.spec.enums import FeatureAggregationMethod


def _create_feature_store_with_mocks() -> Any:
    """Create a FeatureStore with mocked dependencies (bypassing __init__)."""
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


def _tiled_agg_metadata() -> AggregationMetadata:
    """Persisted tiling metadata, round-tripped through the wire form."""
    return AggregationMetadata.from_dict(
        AggregationMetadata(
            feature_granularity="1d",
            features=[Feature.sum("AMOUNT", "2d").to_spec()],
            feature_aggregation_method=FeatureAggregationMethod.TILES.value,
        ).to_dict()
    )


class ComposeFeatureViewWithoutRefreshFreqTest(absltest.TestCase):
    """``_compose_feature_view`` must tolerate a missing cadence on a tiled FV."""

    def _compose(
        self,
        *,
        managed: bool,
        tiled: bool,
        target_lag: str | None,
    ) -> Any:
        # ``Any``, not ``FeatureView``: the mock harness is untyped, so the
        # compose call returns ``Any`` and a narrower annotation trips
        # mypy's no-any-return.
        from snowflake.ml.feature_store.feature_store import _FeatureStoreObjTypes
        from snowflake.ml.feature_store.feature_view import _FeatureViewMetadata
        from snowflake.snowpark.types import TimestampType

        fs = _create_feature_store_with_mocks()

        metadata = _FeatureViewMetadata(
            entities=["USER"],
            timestamp_col="EVENT_TIME",
        )

        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: {
            "name": "USER_CLICK_STATS_DECL$V1",
            "text": "CREATE VIEW x AS SELECT 1",
            "comment": "test",
            "target_lag": target_lag,
            "scheduling_state": "ACTIVE",
            "warehouse": "WH_1",
            "refresh_mode": "FULL",
            "refresh_mode_reason": "",
            "owner": "ROLE_1",
            "cluster_by": "",
        }[key]

        entity_row = MagicMock()
        entity_row.__getitem__ = lambda self, key: {
            "NAME": "USER",
            "JOIN_KEYS": '["USER_ID"]',
            "DESC": "",
        }[key]

        mock_df = MagicMock()
        mock_df.columns = ["USER_ID", "EVENT_TIME", "AMOUNT"]
        mock_df.queries = {"queries": ["SELECT 1"]}
        ts_field = MagicMock()
        ts_field.datatype = TimestampType()
        mock_df.schema.__getitem__ = lambda self, key: ts_field

        fs._session.sql.return_value = mock_df

        fs._lookup_feature_view_metadata = MagicMock(return_value=(metadata, "SELECT 1"))
        fs._determine_online_config_from_oft = MagicMock(return_value='{"enable": false}')
        fs._extract_cluster_by_columns = MagicMock(return_value=None)
        fs._fetch_column_descs = MagicMock(return_value={})
        fs._hydrate_postgres_online_service = MagicMock()
        fs._metadata_manager.get_feature_specs.return_value = _tiled_agg_metadata() if tiled else None
        fs._metadata_manager.get_feature_view_source_refs.return_value = None
        fs._metadata_manager.get_feature_descs.return_value = {}
        fs._metadata_manager.get_feature_view_metadata_config.return_value = None

        obj_type = (
            _FeatureStoreObjTypes.MANAGED_FEATURE_VIEW if managed else _FeatureStoreObjTypes.EXTERNAL_FEATURE_VIEW
        )
        return fs._compose_feature_view(mock_row, obj_type, [entity_row])

    def test_external_tiled_fv_without_refresh_freq_reconstructs(self) -> None:
        """The reported bug. A VIEW-backed tiled FV had no readable form at all,
        which is what made it undeletable — ``delete_feature_view`` resolves the
        object before it can issue any DROP."""
        fv = self._compose(managed=False, tiled=True, target_lag=None)

        self.assertTrue(fv.is_tiled)
        self.assertIsNone(fv.refresh_freq)
        self.assertEqual(fv.status, FeatureViewStatus.STATIC)

    def test_managed_tiled_fv_with_null_target_lag_reconstructs(self) -> None:
        """Same tolerance on the Dynamic Table branch, where a null
        ``target_lag`` reaches ``refresh_freq`` verbatim from SHOW DYNAMIC
        TABLES."""
        fv = self._compose(managed=True, tiled=True, target_lag=None)

        self.assertTrue(fv.is_tiled)
        self.assertIsNone(fv.refresh_freq)

    def test_external_non_tiled_fv_still_reconstructs(self) -> None:
        """Guard pin — a VIEW-backed feature view over a non-Dynamic-Table source
        is a first-class kind. It is non-tiled, so it never reaches the relaxed
        check and must behave exactly as it did before."""
        fv = self._compose(managed=False, tiled=False, target_lag=None)

        self.assertFalse(fv.is_tiled)
        self.assertIsNone(fv.refresh_freq)
        self.assertEqual(fv.status, FeatureViewStatus.STATIC)

    def test_managed_tiled_fv_keeps_its_refresh_freq(self) -> None:
        """Leniency is scoped to the missing-cadence case."""
        fv = self._compose(managed=True, tiled=True, target_lag="1 day")

        self.assertTrue(fv.is_tiled)
        self.assertEqual(fv.refresh_freq, "1 day")


class DeleteStrandedTiledFeatureViewTest(absltest.TestCase):
    """``delete_feature_view`` must reach its DROP for an object that no longer
    satisfies the authoring contract. This is the P1 symptom: such an object was
    removable only by ``DROP SCHEMA … CASCADE``."""

    def test_delete_view_backed_tiled_fv_issues_drop_view(self) -> None:
        fs = _create_feature_store_with_mocks()

        entity = Entity(name="USER", join_keys=["USER_ID"])
        mock_df = MagicMock()
        mock_df.columns = ["USER_ID", "EVENT_TIME", "AMOUNT"]
        mock_df.queries = {"queries": ["SELECT 1"]}

        # The stranded object as reconstruction yields it: tiled, VIEW-backed,
        # no cadence.
        stranded = FeatureView._construct_feature_view(
            name="USER_CLICK_STATS_DECL",
            entities=[entity],
            feature_df=mock_df,
            timestamp_col="EVENT_TIME",
            desc="",
            version="V1",
            status=FeatureViewStatus.STATIC,
            feature_descs={},
            refresh_freq=None,
            database="TEST_DB",
            schema="TEST_SCHEMA",
            warehouse=None,
            refresh_mode=None,
            refresh_mode_reason=None,
            initialize="ON_CREATE",
            owner="ROLE_1",
            infer_schema_df=None,
            session=MagicMock(),
            feature_granularity="1d",
            aggregation_specs=[Feature.sum("AMOUNT", "2d").to_spec()],
            feature_aggregation_method=None,
        )

        fs._get_feature_view_impl = MagicMock(return_value=stranded)
        fs._cleanup_stale_feature_view_resources = MagicMock()

        fs.delete_feature_view("USER_CLICK_STATS_DECL", "V1")

        issued = [call.args[0] for call in fs._session.sql.call_args_list if call.args]
        self.assertTrue(
            any(sql.startswith("DROP VIEW IF EXISTS") for sql in issued),
            f"Expected a DROP VIEW for the VIEW-backed feature view, got: {issued}",
        )


if __name__ == "__main__":
    absltest.main()
