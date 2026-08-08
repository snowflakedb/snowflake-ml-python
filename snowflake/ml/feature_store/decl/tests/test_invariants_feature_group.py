"""Phase 2 RED tests — FG content-hash + cross-FV source validator.

Pin two new helpers in :mod:`decl.invariants`:

* ``fg_content_hash(spec)`` — deterministic, reorder-stable hash over the
  FG identity (``name``, ``version``) + ``desc`` + ``auto_prefix`` + the
  sorted ``(fv_name, fv_version, slice_columns, alias)`` source tuples.
  Output columns (derived) are NOT in the basis.

* ``_check_feature_group_sources(spec, batch_fv_specs, applied_state)`` —
  rejects unresolved sources, duplicate ``(name, version)`` pairs, and
  source FVs that are visibly not online + Postgres in either the local
  batch or applied state.  Sources that are unknown to both views
  soft-pass (the imperative ``register_feature_group`` will raise the
  precondition; we only surface what we can see).

Both tests deliberately fail until the helpers are implemented.
"""

from __future__ import annotations

from typing import Any

import pytest

from snowflake.ml.feature_store.decl.types import AppliedObject, AppliedState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fg(
    name: str = "MY_FG",
    version: str = "V1",
    desc: str = "",
    auto_prefix: bool = True,
    feature_views: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "FeatureGroup",
        "name": name,
        "database": "DB",
        "schema": "SCH",
        "version": version,
        "desc": desc,
        "auto_prefix": auto_prefix,
        "feature_views": feature_views or [{"name": "FV_A", "version": "V1"}],
    }


def _fv_payload(
    name: str = "FV_A",
    version: str = "V1",
    online: bool = True,
    store_type: str = "POSTGRES",
) -> dict[str, Any]:
    return {
        "kind": "StreamingFeatureView",
        "name": name,
        "database": "DB",
        "schema": "SCH",
        "version": version,
        "online": online,
        "online_config": {"store_type": store_type} if store_type else {},
    }


def _applied_with(*payloads: Any) -> AppliedState:
    objs: dict[str, AppliedObject] = {}
    for p in payloads:
        kind = p.get("kind", "")
        name = (p.get("name") or "").upper()
        db = (p.get("database") or "").upper()
        schema = (p.get("schema") or "").upper()
        key = f"{kind}:{db}.{schema}:{name}"
        objs[key] = AppliedObject(
            key=key,
            kind=kind,
            name=p.get("name", ""),
            version=p.get("version"),
            content_hash="x",
            spec_payload=p,
        )
    return AppliedState(objects=objs)


# ---------------------------------------------------------------------------
# fg_content_hash
# ---------------------------------------------------------------------------


class TestFgContentHashStability:
    def test_stable_across_unrelated_key_reorder(self) -> None:
        from snowflake.ml.feature_store.decl.invariants import fg_content_hash

        a = _fg(feature_views=[{"name": "FV_A", "version": "V1"}])
        b = {
            "feature_views": [{"name": "FV_A", "version": "V1"}],
            **{k: v for k, v in a.items() if k != "feature_views"},
        }
        assert fg_content_hash(a) == fg_content_hash(b)

    def test_stable_across_source_reorder(self) -> None:
        from snowflake.ml.feature_store.decl.invariants import fg_content_hash

        a = _fg(
            feature_views=[
                {"name": "FV_A", "version": "V1"},
                {"name": "FV_B", "version": "V1"},
            ]
        )
        b = _fg(
            feature_views=[
                {"name": "FV_B", "version": "V1"},
                {"name": "FV_A", "version": "V1"},
            ]
        )
        assert fg_content_hash(a) == fg_content_hash(b)


class TestFgContentHashSensitivity:
    def test_sensitive_to_desc(self) -> None:
        from snowflake.ml.feature_store.decl.invariants import fg_content_hash

        a = _fg(desc="one")
        b = _fg(desc="two")
        assert fg_content_hash(a) != fg_content_hash(b)

    def test_sensitive_to_auto_prefix(self) -> None:
        from snowflake.ml.feature_store.decl.invariants import fg_content_hash

        a = _fg(auto_prefix=True)
        b = _fg(auto_prefix=False)
        assert fg_content_hash(a) != fg_content_hash(b)

    def test_sensitive_to_fv_version(self) -> None:
        from snowflake.ml.feature_store.decl.invariants import fg_content_hash

        a = _fg(feature_views=[{"name": "FV_A", "version": "V1"}])
        b = _fg(feature_views=[{"name": "FV_A", "version": "V2"}])
        assert fg_content_hash(a) != fg_content_hash(b)

    def test_sensitive_to_slice_columns(self) -> None:
        from snowflake.ml.feature_store.decl.invariants import fg_content_hash

        a = _fg(feature_views=[{"name": "FV_A", "version": "V1"}])
        b = _fg(feature_views=[{"name": "FV_A", "version": "V1", "slice_columns": ["X"]}])
        assert fg_content_hash(a) != fg_content_hash(b)

    def test_sensitive_to_alias(self) -> None:
        from snowflake.ml.feature_store.decl.invariants import fg_content_hash

        a = _fg(feature_views=[{"name": "FV_A", "version": "V1", "alias": ""}])
        b = _fg(feature_views=[{"name": "FV_A", "version": "V1", "alias": "x"}])
        # alias="" is preserved (semantically: "no prefix"), distinct from "x".
        assert fg_content_hash(a) != fg_content_hash(b)

    def test_sensitive_to_fg_version(self) -> None:
        from snowflake.ml.feature_store.decl.invariants import fg_content_hash

        a = _fg(version="V1")
        b = _fg(version="V2")
        assert fg_content_hash(a) != fg_content_hash(b)


class TestFgContentHashIgnoresOutputColumns:
    def test_output_columns_not_in_basis(self) -> None:
        from snowflake.ml.feature_store.decl.invariants import fg_content_hash

        a = _fg()
        b = _fg()
        # Inject a derived field that the basis must ignore.
        b["output_columns"] = ["A", "B", "C"]
        assert fg_content_hash(a) == fg_content_hash(b)


# ---------------------------------------------------------------------------
# _check_feature_group_sources
# ---------------------------------------------------------------------------


class TestCheckFeatureGroupSourcesUnresolved:
    def test_raises_missing_feature_view_when_unknown(self) -> None:
        from snowflake.ml.feature_store.decl.invariants import (
            _check_feature_group_sources,
        )

        spec = _fg(feature_views=[{"name": "GHOST_FV", "version": "V1"}])
        results = _check_feature_group_sources(spec, batch_fv_specs={}, applied_state=AppliedState(objects={}))
        codes = [r.code for r in results]
        assert "MISSING_FEATURE_VIEW" in codes

    def test_resolves_via_batch_fv_specs(self) -> None:
        from snowflake.ml.feature_store.decl.invariants import (
            _check_feature_group_sources,
        )

        spec = _fg(feature_views=[{"name": "FV_A", "version": "V1"}])
        results = _check_feature_group_sources(
            spec,
            batch_fv_specs={"FV_A": _fv_payload("FV_A", online=True, store_type="POSTGRES")},
            applied_state=AppliedState(objects={}),
        )
        # Resolved + online + Postgres → no error.
        assert all(r.code != "MISSING_FEATURE_VIEW" for r in results)

    def test_resolves_via_applied_state(self) -> None:
        from snowflake.ml.feature_store.decl.invariants import (
            _check_feature_group_sources,
        )

        spec = _fg(feature_views=[{"name": "FV_A", "version": "V1"}])
        applied = _applied_with(_fv_payload("FV_A", online=True, store_type="POSTGRES"))
        results = _check_feature_group_sources(spec, batch_fv_specs={}, applied_state=applied)
        assert all(r.code != "MISSING_FEATURE_VIEW" for r in results)


class TestCheckFeatureGroupSourcesDuplicate:
    def test_raises_duplicate_source(self) -> None:
        from snowflake.ml.feature_store.decl.invariants import (
            _check_feature_group_sources,
        )

        spec = _fg(
            feature_views=[
                {"name": "FV_A", "version": "V1"},
                {"name": "FV_A", "version": "V1"},
            ]
        )
        applied = _applied_with(_fv_payload("FV_A", online=True, store_type="POSTGRES"))
        results = _check_feature_group_sources(spec, batch_fv_specs={}, applied_state=applied)
        codes = [r.code for r in results]
        assert "FG_DUPLICATE_SOURCE" in codes


class TestCheckFeatureGroupSourcesNotOnlinePostgres:
    def test_offline_source_in_batch_raises(self) -> None:
        from snowflake.ml.feature_store.decl.invariants import (
            _check_feature_group_sources,
        )

        spec = _fg(feature_views=[{"name": "FV_A", "version": "V1"}])
        results = _check_feature_group_sources(
            spec,
            batch_fv_specs={"FV_A": _fv_payload("FV_A", online=False, store_type="")},
            applied_state=AppliedState(objects={}),
        )
        codes = [r.code for r in results]
        assert "FG_SOURCE_NOT_ONLINE_POSTGRES" in codes

    def test_non_postgres_source_in_applied_raises(self) -> None:
        from snowflake.ml.feature_store.decl.invariants import (
            _check_feature_group_sources,
        )

        spec = _fg(feature_views=[{"name": "FV_A", "version": "V1"}])
        # Applied side declares online but a non-Postgres store type.
        applied = _applied_with(_fv_payload("FV_A", online=True, store_type="DUCKDB"))
        results = _check_feature_group_sources(spec, batch_fv_specs={}, applied_state=applied)
        codes = [r.code for r in results]
        assert "FG_SOURCE_NOT_ONLINE_POSTGRES" in codes

    def test_local_fv_without_store_type_is_soft_pass(self) -> None:
        """The decl shape today doesn't author ``online_config.store_type``;
        fall through silently and let the imperative API raise at apply time
        (TODO: a future ``store_type:`` authoring field would let us be strict).
        """
        from snowflake.ml.feature_store.decl.invariants import (
            _check_feature_group_sources,
        )

        spec = _fg(feature_views=[{"name": "FV_A", "version": "V1"}])
        # Local FV declares ``online: true`` but does not carry online_config.
        local_fv_payload = {
            "kind": "StreamingFeatureView",
            "name": "FV_A",
            "version": "V1",
            "database": "DB",
            "schema": "SCH",
            "online": True,
        }
        results = _check_feature_group_sources(
            spec,
            batch_fv_specs={"FV_A": local_fv_payload},
            applied_state=AppliedState(objects={}),
        )
        codes = [r.code for r in results]
        assert "FG_SOURCE_NOT_ONLINE_POSTGRES" not in codes
        assert "MISSING_FEATURE_VIEW" not in codes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
