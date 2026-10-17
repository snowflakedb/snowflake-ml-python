"""Regression coverage for the secondary-key BatchFV RECREATE_FV loop.

A tiled ``BatchFeatureView`` whose ``aggregation_secondary_keys`` column is
not present in the authored ``BatchSource.columns`` used to loop on
``RECREATE_FV`` every plan: state recovery
(:func:`imperative_executor._serialize_batch_fv_spec`) rebuilds the
aggregation resolution pool from the stamped ``FV_SOURCE_REFS`` columns,
which omitted the secondary key, so ``_build_batch_feature_view_spec`` raised
``Column '<SK>' not found in resolution pool``, ``_serialize`` returned
``None``, and the planner fell back to the minimal spec -> perpetual
``RECREATE_FV`` (never ``NO_CHANGE``).

The fix unions ``aggregation_secondary_keys`` into the recovery resolution
pool.  These tests pin:

* ``_augment_schema_with_secondary_keys`` — the pure schema-union helper.
* ``_serialize_batch_fv_spec`` — the wired recovery path returns a non-``None``
  spec (instead of falling back to the minimal spec) when the secondary key is
  absent from the stamped source columns but present in the materialized DT
  schema.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from snowflake.ml.feature_store.decl import imperative_executor as ie


def _norm(name: Any) -> str:
    return str(name).strip('"').upper()


def _struct(*pairs: tuple[str, Any]) -> Any:
    from snowflake.snowpark.types import StructField, StructType

    return StructType([StructField(n, t) for n, t in pairs])


# ---------------------------------------------------------------------------
# _augment_schema_with_secondary_keys — pure schema-union helper
# ---------------------------------------------------------------------------


def test_augment_adds_missing_secondary_key_from_fallback() -> None:
    from snowflake.snowpark.types import DoubleType, StringType, TimestampType

    raw = _struct(("USER_ID", StringType()), ("AMOUNT", DoubleType()))
    dt = _struct(
        ("USER_ID", StringType()),
        ("OFFER_KEY", StringType()),
        ("TILE_START", TimestampType()),
    )
    out = ie._augment_schema_with_secondary_keys(raw, ["OFFER_KEY"], [dt])
    names = {_norm(f.name) for f in out.fields}
    assert "OFFER_KEY" in names
    # existing columns are preserved
    assert {"USER_ID", "AMOUNT"} <= names


def test_augment_noop_when_secondary_key_already_present() -> None:
    from snowflake.snowpark.types import DoubleType, StringType

    raw = _struct(
        ("USER_ID", StringType()),
        ("OFFER_KEY", StringType()),
        ("AMOUNT", DoubleType()),
    )
    out = ie._augment_schema_with_secondary_keys(raw, ["OFFER_KEY"], [raw])
    names = [_norm(f.name) for f in out.fields]
    # no duplicate OFFER_KEY appended
    assert names.count("OFFER_KEY") == 1
    assert len(names) == 3


def test_augment_noop_when_no_secondary_keys() -> None:
    from snowflake.snowpark.types import StringType

    raw = _struct(("USER_ID", StringType()))
    for sks in (None, []):  # type: ignore[var-annotated]
        out = ie._augment_schema_with_secondary_keys(raw, sks, [raw])
        assert [_norm(f.name) for f in out.fields] == ["USER_ID"]


def test_augment_skips_key_absent_from_all_fallbacks() -> None:
    from snowflake.snowpark.types import StringType

    raw = _struct(("USER_ID", StringType()))
    dt = _struct(("USER_ID", StringType()))
    out = ie._augment_schema_with_secondary_keys(raw, ["OFFER_KEY"], [dt])
    # cannot type it -> not added (recovery falls back, but no crash)
    assert "OFFER_KEY" not in {_norm(f.name) for f in out.fields}


# ---------------------------------------------------------------------------
# _serialize_batch_fv_spec — wired recovery path
# ---------------------------------------------------------------------------


class _FakeDF:
    def __init__(self, schema: Any) -> None:
        self.schema = schema


class _FakeFV:
    """Minimal stand-in for the rehydrated tiled BatchFeatureView.

    ``feature_df`` starts as the materialized (post-aggregation) DT schema —
    which physically contains the secondary key — exactly as
    ``FeatureStore.get_feature_view`` returns it.
    """

    def __init__(self, dt_schema: Any, sks: list[str], source_refs: list[dict[str, Any]]) -> None:
        self.is_tiled = True
        self.is_rollup = False
        self.aggregation_secondary_keys = sks
        self.source_refs = source_refs
        self._feature_df = _FakeDF(dt_schema)
        self.output_schema = dt_schema

    @property
    def feature_df(self) -> Any:
        return self._feature_df


def _make_session(dt_schema: Any) -> Any:
    session = MagicMock()
    tbl = MagicMock()
    tbl.schema = dt_schema
    session.table.return_value = tbl
    session.create_dataframe.side_effect = lambda data, schema: _FakeDF(schema)
    return session


def _resolution_pool_side_effect(captured: dict[str, Any]) -> Any:
    def _bbfvs(
        *,
        feature_view: Any,
        feature_view_name: Any,
        version: Any,
        target_lag: Any,
        offline_materialized_schema: Any,
    ) -> Any:
        names = {_norm(f.name) for f in feature_view.feature_df.schema.fields}
        captured["names"] = names
        # Emulate snowml-core's _resolve_features: the secondary key spec's
        # source_column must be in the resolution pool.
        if "OFFER_KEY" not in names:
            raise ValueError(f"Column 'OFFER_KEY' not found in resolution pool. Available columns: {sorted(names)}")
        spec = MagicMock()
        spec.to_dict.return_value = {"kind": "BatchFeatureView", "spec": {}}
        return spec

    return _bbfvs


@patch.object(ie, "_canonicalize_enums_in_place", lambda obj: None)
@patch.object(ie, "_emulate_quake_postprocess", lambda obj: None)
def test_serialize_unions_secondary_key_into_recovery_pool() -> None:
    from snowflake.snowpark.types import DoubleType, StringType, TimestampType

    # Materialized DT schema (post-aggregation) carries OFFER_KEY.
    dt_schema = _struct(
        ("USER_ID", StringType()),
        ("OFFER_KEY", StringType()),
        ("TILE_START", TimestampType()),
        ("_PARTIAL_SUM_AMOUNT", DoubleType()),
    )
    # Stamped FV_SOURCE_REFS columns OMIT the secondary key (the bug shape).
    stamped_columns = [
        {"name": "USER_ID", "type": "StringType"},
        {"name": "AMOUNT", "type": "DoubleType"},
    ]
    source_refs = [{"name": "SRC1", "source_type": "Batch", "table": "RAW_EVENTS", "columns": stamped_columns}]
    fv = _FakeFV(dt_schema, ["OFFER_KEY"], source_refs)

    fs = MagicMock()
    fs.get_feature_view.return_value = fv
    captured: dict[str, Any] = {}
    fs._build_batch_feature_view_spec.side_effect = _resolution_pool_side_effect(captured)

    session = _make_session(dt_schema)
    row = {
        "name": "CONTENT_EVENT_AGGREGATIONS_FV",
        "version": "V1",
        "database_name": "DB",
        "schema_name": "SC",
        "physical_dt_name": "DT",
    }

    result = ie._serialize_batch_fv_spec(fs, session, row)

    assert result is not None, "recovery should not fall back to the minimal spec"
    assert "OFFER_KEY" in captured["names"], "secondary key must be unioned into the resolution pool"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
