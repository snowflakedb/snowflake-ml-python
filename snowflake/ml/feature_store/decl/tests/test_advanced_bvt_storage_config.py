"""Phase 5 — declarative-side coverage for the BFV ``storage_config`` field.

``storage_config`` is a nested struct (``format`` +
``external_volume?`` + ``base_location?``) that pins the deployed
FV's storage backend — native Snowflake (Dynamic Table) or Iceberg
(Dynamic Iceberg Table).  Snowflake cannot flip a Dynamic Table's
storage format in place, so any edit is *structural* and routes
through ``RECREATE_FV(destructive=True)``.

Authoring shapes::

    storage_config:
      format: snowflake          # default; offline DT

    storage_config:
      format: iceberg            # offline DIT
      external_volume: MY_VOL
      base_location: feature_store/bfv

Validator: ``BATCH_FV_STORAGE_ICEBERG_NO_VOLUME`` fires when
``format: iceberg`` is authored without ``external_volume`` AND the
``FeatureStore.default_iceberg_external_volume`` is unknown to the
spec (i.e. the local YAML has no fallback).  The validator only
fires at authoring time; runtime fallback to the FS default is the
imperative library's responsibility.

Coverage map:

* ``test_storage_config_round_trips_through_model_validate`` —
  spec model accepts both authoring shapes.
* ``test_storage_config_validator_rejects_unknown_format`` —
  typoed ``format`` fails at load time.
* ``test_storage_config_validator_rejects_iceberg_without_volume``
  — Iceberg without ``external_volume`` triggers
  ``BATCH_FV_STORAGE_ICEBERG_NO_VOLUME``.
* ``test_compile_to_spec_threads_storage_config`` — compile
  surfaces ``storage_config`` into the inner ``spec``.
* ``test_hash_changes_when_storage_format_flips`` — flipping
  snowflake → iceberg bumps the structural hash.
* ``test_planner_emits_recreate_fv_on_storage_format_flip`` — the
  diff lands as ``RECREATE_FV(destructive=True)``.
* ``test_executor_create_passes_storage_config_kwarg`` — executor
  builds a ``StorageConfig(format=..., external_volume=...)`` and
  forwards it to the imperative ``FeatureView`` constructor.
* ``test_exporter_recovers_storage_config_from_inner_spec`` —
  exporter copies ``storage_config`` back into the YAML doc.
"""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.exporter import _build_full_fidelity_fv
from snowflake.ml.feature_store.decl.imperative_executor import _build_feature_view
from snowflake.ml.feature_store.decl.invariants import _full_spec_hash, validate_specs
from snowflake.ml.feature_store.decl.planner import generate_plan
from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec
from snowflake.ml.feature_store.decl.spec_models import (
    BatchSource,
    Entity,
    FeatureView,
    FSColumn,
)
from snowflake.ml.feature_store.decl.types import (
    AppliedObject,
    AppliedState,
    PlanOptions,
    SpecBatch,
)
from snowflake.ml.test_utils import pytest_driver


def _minimal_batch_fv_authoring(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "kind": "BatchFeatureView",
        "name": "BFV_SC",
        "version": "V1",
        "database": "DB1",
        "schema": "SC1",
        "online": False,
        "entities": ["USER_ID"],
        "sources": [
            {
                "name": "SRC1",
                "source_type": "Batch",
                "table": "RAW_EVENTS",
                "columns": [{"name": "USER_ID", "type": "StringType"}],
            }
        ],
        "refresh_freq": "5 minutes",
    }
    base.update(overrides)
    return base


def _applied_state_for(local: dict[str, Any]) -> AppliedState:
    compiled = compile_to_spec(local, "DB1", "SC1")
    h = _full_spec_hash(compiled)
    key = "BatchFeatureView:DB1.SC1:BFV_SC:V1"
    return AppliedState(
        objects={
            key: AppliedObject(
                key=key,
                kind="BatchFeatureView",
                name="BFV_SC",
                version="V1",
                content_hash=h,
                spec_payload=copy.deepcopy(compiled),
                columns=[],
                from_specification=True,
            )
        }
    )


@pytest.mark.parametrize(
    "value",
    [
        {"format": "snowflake"},
        {"format": "iceberg", "external_volume": "MY_VOL", "base_location": "fs/bfv"},
    ],
)
def test_storage_config_round_trips_through_model_validate(value: dict[str, Any]) -> None:
    """Each canonical storage_config shape survives spec-model validation."""
    payload = _minimal_batch_fv_authoring(storage_config=value)
    fv = FeatureView.model_validate(payload)
    dumped = fv.model_dump(exclude_none=True).get("storage_config")
    assert dumped is not None
    assert dumped.get("format") == value["format"]
    if "external_volume" in value:
        assert dumped.get("external_volume") == value["external_volume"]
    if "base_location" in value:
        assert dumped.get("base_location") == value["base_location"]


def test_storage_config_validator_rejects_unknown_format() -> None:
    """Unknown ``format`` values raise ``ValidationError`` at load time."""
    payload = _minimal_batch_fv_authoring(storage_config={"format": "parquet"})
    with pytest.raises(ValidationError):
        FeatureView.model_validate(payload)


def test_storage_config_validator_rejects_iceberg_without_volume() -> None:
    """``format: iceberg`` without ``external_volume`` raises a validator error."""
    payload = _minimal_batch_fv_authoring(storage_config={"format": "iceberg"})
    fv = FeatureView.model_validate(payload)

    ent = Entity(
        kind="Entity",
        name="USER",
        join_keys=[FSColumn(name="USER_ID", type="StringType")],
    )
    src = BatchSource(
        kind="BatchSource",
        name="SRC1",
        table="RAW_EVENTS",
        columns=[FSColumn(name="USER_ID", type="StringType")],
    )
    results = validate_specs(
        SpecBatch(specs=[ent, src, fv]),
        AppliedState(objects={}),
        target_database="DB1",
        target_schema="SC1",
    )
    codes = [r.code for r in results]
    assert "BATCH_FV_STORAGE_ICEBERG_NO_VOLUME" in codes


def test_compile_to_spec_threads_storage_config() -> None:
    local = _minimal_batch_fv_authoring(
        storage_config={"format": "iceberg", "external_volume": "MY_VOL", "base_location": "fs/bfv"}
    )
    compiled = compile_to_spec(local, "DB1", "SC1")
    inner = compiled.get("spec", {})
    assert inner.get("storage_config") == {
        "format": "iceberg",
        "external_volume": "MY_VOL",
        "base_location": "fs/bfv",
    }


def test_hash_changes_when_storage_format_flips() -> None:
    a = compile_to_spec(_minimal_batch_fv_authoring(storage_config={"format": "snowflake"}), "DB1", "SC1")
    b = compile_to_spec(
        _minimal_batch_fv_authoring(storage_config={"format": "iceberg", "external_volume": "MY_VOL"}),
        "DB1",
        "SC1",
    )
    assert _full_spec_hash(a) != _full_spec_hash(b)


def test_planner_emits_recreate_fv_on_storage_format_flip() -> None:
    local = _minimal_batch_fv_authoring(storage_config={"format": "snowflake"})
    applied = _applied_state_for(local)
    edited = _minimal_batch_fv_authoring(storage_config={"format": "iceberg", "external_volume": "MY_VOL"})

    ent = Entity(
        kind="Entity",
        name="USER",
        join_keys=[FSColumn(name="USER_ID", type="StringType")],
    )
    src = BatchSource(
        kind="BatchSource",
        name="SRC1",
        table="RAW_EVENTS",
        columns=[FSColumn(name="USER_ID", type="StringType")],
    )
    fv = FeatureView.model_validate(edited)
    plan = generate_plan(
        SpecBatch(specs=[ent, src, fv]),
        applied,
        PlanOptions(),
        database="DB1",
        schema="SC1",
    )
    fv_ops = [op for op in plan.ops if op.name == "BFV_SC"]
    assert len(fv_ops) == 1
    assert fv_ops[0].kind is OpKind.RECREATE_FV
    assert fv_ops[0].destructive is True


def test_executor_create_passes_storage_config_kwarg() -> None:
    """Executor constructs ``StorageConfig`` and forwards as kwarg."""
    payload = _minimal_batch_fv_authoring(
        storage_config={"format": "iceberg", "external_volume": "MY_VOL", "base_location": "fs/bfv"}
    )
    session = MagicMock()
    session.table.return_value = MagicMock()
    fs = MagicMock()
    captured: dict[str, Any] = {}
    captured_sc_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class _FakeFV:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

    class _FakeStorageConfig:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured_sc_calls.append((args, kwargs))
            self.kwargs = kwargs

    class _FakeStorageFormat:
        def __init__(self, value: str) -> None:
            self.value = value

        def __eq__(self, other: object) -> bool:
            return isinstance(other, _FakeStorageFormat) and other.value == self.value

    with patch("snowflake.ml.feature_store.feature_view.FeatureView", _FakeFV,), patch(
        "snowflake.ml.feature_store.feature_view.OnlineConfig",
        MagicMock(),
    ), patch("snowflake.ml.feature_store.feature_view.OnlineStoreType", MagicMock(),), patch(
        "snowflake.ml.feature_store.feature_view.StorageConfig",
        _FakeStorageConfig,
    ), patch(
        "snowflake.ml.feature_store.feature_view.StorageFormat",
        _FakeStorageFormat,
    ):
        _build_feature_view(payload, session, "DB1", "SC1", "WH_DEFAULT", fs=fs)

    assert "storage_config" in captured, f"got kwargs={sorted(captured)}"
    assert len(captured_sc_calls) == 1
    sc_kwargs = captured_sc_calls[0][1]
    assert sc_kwargs.get("external_volume") == "MY_VOL"
    assert sc_kwargs.get("base_location") == "fs/bfv"
    fmt = sc_kwargs.get("format")
    assert isinstance(fmt, _FakeStorageFormat) and fmt.value == "iceberg"


def test_exporter_recovers_storage_config_from_inner_spec() -> None:
    full_spec = {
        "kind": "BatchFeatureView",
        "metadata": {"name": "BFV_SC", "version": "V1", "database": "DB1", "schema": "SC1"},
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "storage_config": {
                "format": "iceberg",
                "external_volume": "MY_VOL",
                "base_location": "fs/bfv",
            },
            "sources": [],
            "features": [],
        },
    }
    doc = _build_full_fidelity_fv(
        full_spec,
        fallback_name="BFV_SC",
        fallback_version="V1",
        fallback_database="DB1",
        fallback_schema="SC1",
    )
    assert doc.get("storage_config") == {
        "format": "iceberg",
        "external_volume": "MY_VOL",
        "base_location": "fs/bfv",
    }


if __name__ == "__main__":
    pytest_driver.main()
