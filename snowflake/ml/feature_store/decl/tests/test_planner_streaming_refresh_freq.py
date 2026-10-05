"""Planner tests: tiled StreamingFeatureView refresh_freq → UPDATE_FV.

A *tiled* ``StreamingFeatureView`` materialises its aggregate as an
offline Dynamic Table whose refresh cadence is ``refresh_freq``.  Because
``refresh_freq`` is an operational field for the streaming kind
(:data:`planner._FV_OPERATIONAL_FIELDS_BY_KIND`), editing only the
cadence must plan as a non-destructive ``UPDATE_FV`` — and a matching
cadence must plan as ``NO_CHANGE`` (no phantom drift).

The applied side mirrors the deployed runtime shape: the OFT
``spec.target_lag_sec`` is stamped ``0`` for every streaming kind while
the DT cadence is recovered onto ``spec.refresh_freq`` by
``state._inject_fv_refresh_freq_from_list_row``.  The planner's
``_refresh_freq_drifted`` helper must therefore compare against
``spec.refresh_freq`` (the DT cadence), NOT the ``target_lag_sec`` OFT
staleness.
"""

from __future__ import annotations

import copy
from typing import Any

from snowflake.ml.feature_store.decl.invariants import _full_spec_hash
from snowflake.ml.feature_store.decl.planner import generate_plan
from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec
from snowflake.ml.feature_store.decl.spec_models import (
    Entity,
    FeatureView,
    FSColumn,
    StreamingSource,
)
from snowflake.ml.feature_store.decl.types import (
    AppliedObject,
    AppliedState,
    PlanOptions,
    SpecBatch,
)
from snowflake.ml.test_utils import pytest_driver

_DB = "DB1"
_SCHEMA = "SC1"
_FV_NAME = "SFV_TILED_PLAN"


def _tiled_streaming_fv(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "kind": "StreamingFeatureView",
        "name": _FV_NAME,
        "version": "V1",
        "database": _DB,
        "schema": _SCHEMA,
        "online": True,
        "entities": ["USER_ID"],
        "timestamp_col": "EVENT_TS",
        "sources": [
            {
                "name": "CLICKSTREAM_EVENTS",
                "source_type": "Stream",
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "VALUE", "type": "FloatType"},
                    {"name": "EVENT_TS", "type": "TimestampType"},
                ],
            }
        ],
        "udf": {
            "name": "compute",
            "engine": "pandas",
            "function_definition": "def compute(df):\n    return df\n",
            "output_columns": [
                {"name": "USER_ID", "type": "StringType"},
                {"name": "VALUE", "type": "FloatType"},
                {"name": "EVENT_TS", "type": "TimestampType"},
            ],
        },
        "feature_granularity_sec": 3600,
        "feature_aggregation_method": "tiles",
        "features": [
            {
                "source_column": {"name": "VALUE", "type": "FloatType"},
                "output_column": {"name": "VALUE_SUM_1H", "type": "FloatType"},
                "function": "sum",
                "window_sec": 3600,
            }
        ],
        "refresh_freq": "5 minutes",
    }
    base.update(overrides)
    return base


def _entity() -> Entity:
    return Entity(
        kind="Entity",
        name="USER",
        database=_DB,
        schema_=_SCHEMA,
        join_keys=[FSColumn(name="USER_ID", type="StringType")],
    )


def _source() -> StreamingSource:
    return StreamingSource(
        kind="StreamingSource",
        name="CLICKSTREAM_EVENTS",
        database=_DB,
        schema_=_SCHEMA,
        columns=[
            FSColumn(name="USER_ID", type="StringType"),
            FSColumn(name="VALUE", type="FloatType"),
            FSColumn(name="EVENT_TS", type="TimestampType"),
        ],
    )


def _applied_streaming_state(local: dict[str, Any]) -> AppliedState:
    """Build applied state mirroring a deployed tiled streaming FV.

    The runtime stamps the OFT ``target_lag_sec`` to ``0``; the DT
    cadence is recovered onto ``spec.refresh_freq`` by the state helper.

    Args:
        local: The authoring-side spec dict for the tiled streaming FV.

    Returns:
        An ``AppliedState`` whose single ``AppliedObject`` mirrors the
        deployed runtime shape (``target_lag_sec: 0`` + injected
        ``refresh_freq``).
    """
    compiled = compile_to_spec(local, _DB, _SCHEMA)
    inner = compiled.setdefault("spec", {})
    inner["target_lag_sec"] = 0
    inner["refresh_freq"] = local["refresh_freq"]
    h = _full_spec_hash(compiled)
    key = f"StreamingFeatureView:{_DB}.{_SCHEMA}:{_FV_NAME}:V1"
    return AppliedState(
        objects={
            key: AppliedObject(
                key=key,
                kind="StreamingFeatureView",
                name=_FV_NAME,
                version="V1",
                content_hash=h,
                spec_payload=copy.deepcopy(compiled),
                columns=[],
                from_specification=True,
            )
        }
    )


def test_planner_no_change_when_tiled_streaming_refresh_freq_matches() -> None:
    local = _tiled_streaming_fv()
    state = _applied_streaming_state(local)
    fv = FeatureView.model_validate(local)
    plan = generate_plan(
        SpecBatch(specs=[_entity(), _source(), fv]),
        state,
        PlanOptions(),
        database=_DB,
        schema=_SCHEMA,
    )
    fv_ops = [op for op in plan.ops if op.name == _FV_NAME]
    assert len(fv_ops) == 1
    assert fv_ops[0].kind.value == "NO_CHANGE", (
        "identical tiled streaming refresh_freq must be NO_CHANGE, not a "
        f"phantom UPDATE_FV; got {fv_ops[0].kind.value} ({fv_ops[0].reason!r})"
    )


def test_planner_update_fv_when_tiled_streaming_refresh_freq_changes() -> None:
    local = _tiled_streaming_fv()
    state = _applied_streaming_state(local)
    changed = copy.deepcopy(local)
    changed["refresh_freq"] = "10 minutes"
    fv = FeatureView.model_validate(changed)
    plan = generate_plan(
        SpecBatch(specs=[_entity(), _source(), fv]),
        state,
        PlanOptions(),
        database=_DB,
        schema=_SCHEMA,
    )
    fv_ops = [op for op in plan.ops if op.name == _FV_NAME]
    assert len(fv_ops) == 1
    assert fv_ops[0].kind.value == "UPDATE_FV", (
        "editing only refresh_freq on a tiled streaming FV must plan as "
        f"UPDATE_FV; got {fv_ops[0].kind.value} ({fv_ops[0].reason!r})"
    )
    assert fv_ops[0].destructive is False


if __name__ == "__main__":
    pytest_driver.main()
