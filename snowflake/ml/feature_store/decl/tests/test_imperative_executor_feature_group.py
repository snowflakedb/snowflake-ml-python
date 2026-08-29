"""Phase 4 RED tests — FG write path in ``imperative_executor.execute_plan``.

Three contracts are pinned here:

1. ``_build_feature_group(fs, payload, version)`` hydrates each source ref
   via ``fs.get_feature_view(fv_name, fv_version)`` — never by reconstructing
   ``FeatureView(...)`` from a payload.  Going through ``get_feature_view``
   is the only forward-compatible way to inherit the six advanced BFV fields
   and any future FV authoring fields.  A mock-style assertion pins the
   call shape.

2. ``execute_plan`` dispatches ``CREATE_FG`` / ``DROP_FG`` /
   destructive-``CREATE_FG`` to ``fs.register_feature_group`` /
   ``fs.delete_feature_group`` (delete-then-register pair on the
   destructive variant).

3. The existing ``--allow-recreate`` gate in ``execute_plan`` refuses a
   plan whose only op is a destructive ``CREATE_FG``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.imperative_executor import execute_plan
from snowflake.ml.feature_store.decl.types import Plan, PlanOp, PlanOptions
from snowflake.ml.test_utils import pytest_driver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fg_payload(
    *,
    name: str = "USER_FRAUD_FG",
    version: str = "V1",
    desc: str = "",
    auto_prefix: bool = True,
    feature_views: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "FeatureGroup",
        "name": name,
        "version": version,
        "database": "DB",
        "schema": "SCH",
        "desc": desc,
        "auto_prefix": auto_prefix,
        "feature_views": feature_views or [{"name": "USER_CLICK_STATS", "version": "V1"}],
    }


def _make_fg_op(payload: dict[str, Any], *, destructive: bool = False) -> PlanOp:
    return PlanOp(
        kind=OpKind.CREATE_FG,
        name=payload["name"],
        depends_on=[],
        destructive=destructive,
        reason="test",
        payload=payload,
    )


def _make_drop_fg_op(payload: dict[str, Any]) -> PlanOp:
    return PlanOp(
        kind=OpKind.DROP_FG,
        name=payload["name"],
        depends_on=[],
        destructive=True,
        reason="test",
        payload=payload,
    )


# ---------------------------------------------------------------------------
# _build_feature_group contract
# ---------------------------------------------------------------------------


class TestBuildFeatureGroupContract:
    def test_helper_is_exported(self) -> None:
        from snowflake.ml.feature_store.decl import imperative_executor

        assert hasattr(
            imperative_executor, "_build_feature_group"
        ), "_build_feature_group must exist (Phase 4 deliverable)"

    def test_calls_get_feature_view_for_each_source_ref(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_group,
        )

        fs = MagicMock(name="FeatureStore")

        # Each call to fs.get_feature_view returns a distinct mock FV.
        fv_a = MagicMock(name="FV_A")
        fv_b = MagicMock(name="FV_B")
        fv_b.slice = MagicMock(name="FV_B.slice", return_value=MagicMock(name="FV_B_slice"))
        # fv_a does not need slice — slice_columns is unset.
        fs.get_feature_view = MagicMock(side_effect=[fv_a, fv_b])

        payload = _fg_payload(
            feature_views=[
                {"name": "FV_A", "version": "V1"},
                {
                    "name": "FV_B",
                    "version": "V1",
                    "slice_columns": ["X"],
                    "alias": "b",
                },
            ]
        )

        # Stub the imperative FeatureGroup constructor so we don't depend on
        # snowml-core invariants we don't care about here.
        with patch("snowflake.ml.feature_store.feature_group.FeatureGroup") as mock_fg_cls:
            mock_fg_cls.return_value = MagicMock(name="ReturnedFG")
            fg, version = _build_feature_group(fs, payload, payload["version"])

        # Pin: get_feature_view called once per source ref, with the exact
        # (name, version) authored in the payload.
        assert fs.get_feature_view.call_count == 2
        fs.get_feature_view.assert_any_call("FV_A", "V1")
        fs.get_feature_view.assert_any_call("FV_B", "V1")
        # The returned object is the constructed imperative FeatureGroup.
        assert fg is mock_fg_cls.return_value
        assert version == "V1"

    def test_does_not_construct_featureview_directly(self) -> None:
        """The contract: ``_build_feature_group`` MUST NOT instantiate
        ``FeatureView(...)`` itself — that path silently strips the six
        advanced BFV fields the imperative ``get_feature_view`` knows about
        (warehouse, cluster_by, refresh_mode, initialize, storage_config,
        aggregation_secondary_keys).  Pin the negative.
        """
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_group,
        )

        fs = MagicMock(name="FeatureStore")
        fv_a = MagicMock(name="FV_A")
        fs.get_feature_view = MagicMock(return_value=fv_a)

        payload = _fg_payload(
            feature_views=[{"name": "FV_A", "version": "V1"}],
        )

        with patch("snowflake.ml.feature_store.feature_view.FeatureView") as mock_fv_cls, patch(
            "snowflake.ml.feature_store.feature_group.FeatureGroup"
        ) as mock_fg_cls:
            mock_fg_cls.return_value = MagicMock(name="ReturnedFG")
            _build_feature_group(fs, payload, payload["version"])

        # ``FeatureView(...)`` must never be constructed.
        mock_fv_cls.assert_not_called()

    def test_applies_slice_and_with_name(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_group,
        )

        fs = MagicMock(name="FeatureStore")
        sliced = MagicMock(name="FV_A_slice")
        renamed = MagicMock(name="FV_A_renamed")
        sliced.with_name = MagicMock(return_value=renamed)

        fv_a = MagicMock(name="FV_A")
        fv_a.slice = MagicMock(return_value=sliced)
        fs.get_feature_view = MagicMock(return_value=fv_a)

        payload = _fg_payload(
            feature_views=[
                {
                    "name": "FV_A",
                    "version": "V1",
                    "slice_columns": ["X", "Y"],
                    "alias": "a",
                }
            ]
        )

        with patch("snowflake.ml.feature_store.feature_group.FeatureGroup") as mock_fg_cls:
            captured: dict[str, list[Any]] = {}

            def _capture_features(name: Any, features: Any, *, desc: Any, auto_prefix: Any) -> Any:
                captured["features"] = list(features)
                return MagicMock(name="FG")

            mock_fg_cls.side_effect = _capture_features
            _build_feature_group(fs, payload, payload["version"])

        # slice() then with_name() applied; the FG was constructed with the
        # final renamed slice as the source.
        fv_a.slice.assert_called_once_with(["X", "Y"])
        sliced.with_name.assert_called_once_with("a")
        assert captured["features"] == [renamed]

    def test_alias_empty_string_calls_with_name_explicitly(self) -> None:
        """``alias = ""`` is a valid imperative override of ``auto_prefix``.

        The helper must call ``.with_name("")`` — distinct from leaving the
        ref untouched (which would defer to ``auto_prefix``).
        """
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_group,
        )

        fs = MagicMock(name="FeatureStore")
        fv_a = MagicMock(name="FV_A")
        fv_a.with_name = MagicMock(return_value=MagicMock(name="FV_A_no_prefix"))
        fs.get_feature_view = MagicMock(return_value=fv_a)

        payload = _fg_payload(
            feature_views=[{"name": "FV_A", "version": "V1", "alias": ""}],
        )

        with patch("snowflake.ml.feature_store.feature_group.FeatureGroup"):
            _build_feature_group(fs, payload, payload["version"])

        fv_a.with_name.assert_called_once_with("")

    def test_alias_none_does_not_call_with_name(self) -> None:
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_group,
        )

        fs = MagicMock(name="FeatureStore")
        fv_a = MagicMock(name="FV_A")
        fv_a.with_name = MagicMock()
        fs.get_feature_view = MagicMock(return_value=fv_a)

        payload = _fg_payload(
            feature_views=[{"name": "FV_A", "version": "V1"}],
        )

        with patch("snowflake.ml.feature_store.feature_group.FeatureGroup"):
            _build_feature_group(fs, payload, payload["version"])

        # alias unset → default to FG.auto_prefix; no .with_name() call.
        fv_a.with_name.assert_not_called()


# ---------------------------------------------------------------------------
# execute_plan dispatch
# ---------------------------------------------------------------------------


class TestExecutePlanCreateFG:
    def test_create_fg_calls_register_feature_group(self) -> None:
        payload = _fg_payload()
        plan = Plan(ops=[_make_fg_op(payload, destructive=False)], warnings=[])
        options = PlanOptions(allow_recreate=False)

        fs = MagicMock(name="FeatureStore")
        constructed_fg = MagicMock(name="ConstructedFG")
        with patch(
            "snowflake.ml.feature_store.decl.imperative_executor.assert_feature_store_initialized",
            return_value=fs,
        ), patch(
            "snowflake.ml.feature_store.decl.imperative_executor._build_feature_group",
            return_value=(constructed_fg, "V1"),
        ) as mock_build:
            session = MagicMock(name="session")
            result = execute_plan(plan, session, "DB", "SCH", "WH", options)

        mock_build.assert_called_once()
        fs.register_feature_group.assert_called_once_with(constructed_fg, "V1")
        assert result.status == "applied"
        assert result.ops[0]["status"] == "success"
        assert result.ops[0]["operation"] == OpKind.CREATE_FG.value


class TestExecutePlanDropFG:
    def test_drop_fg_calls_delete_feature_group(self) -> None:
        payload = _fg_payload()
        plan = Plan(ops=[_make_drop_fg_op(payload)], warnings=[])
        # DROP_FG is destructive → must allow recreate.
        options = PlanOptions(allow_recreate=True)

        fs = MagicMock(name="FeatureStore")
        with patch(
            "snowflake.ml.feature_store.decl.imperative_executor.assert_feature_store_initialized",
            return_value=fs,
        ):
            session = MagicMock(name="session")
            result = execute_plan(plan, session, "DB", "SCH", "WH", options)

        fs.delete_feature_group.assert_called_once_with("USER_FRAUD_FG", "V1")
        assert result.status == "applied"


class TestExecutePlanDestructiveCreateFG:
    def test_destructive_create_fg_calls_delete_then_register(self) -> None:
        payload = _fg_payload()
        plan = Plan(ops=[_make_fg_op(payload, destructive=True)], warnings=[])
        options = PlanOptions(allow_recreate=True)

        fs = MagicMock(name="FeatureStore")
        constructed_fg = MagicMock(name="ConstructedFG")
        with patch(
            "snowflake.ml.feature_store.decl.imperative_executor.assert_feature_store_initialized",
            return_value=fs,
        ), patch(
            "snowflake.ml.feature_store.decl.imperative_executor._build_feature_group",
            return_value=(constructed_fg, "V1"),
        ):
            session = MagicMock(name="session")
            execute_plan(plan, session, "DB", "SCH", "WH", options)

        # Delete first, then register (the imperative side has no
        # update_feature_group; this is the only valid recreate path).
        fs.delete_feature_group.assert_called_once_with("USER_FRAUD_FG", "V1")
        fs.register_feature_group.assert_called_once_with(constructed_fg, "V1")

    def test_destructive_create_tolerates_missing_existing_fg(self) -> None:
        """When the OFT was already dropped (or never registered), the delete
        call may raise; ``execute_plan`` must still run the register half.
        """
        payload = _fg_payload()
        plan = Plan(ops=[_make_fg_op(payload, destructive=True)], warnings=[])
        options = PlanOptions(allow_recreate=True)

        fs = MagicMock(name="FeatureStore")
        fs.delete_feature_group.side_effect = RuntimeError("not found")
        constructed_fg = MagicMock(name="ConstructedFG")
        with patch(
            "snowflake.ml.feature_store.decl.imperative_executor.assert_feature_store_initialized",
            return_value=fs,
        ), patch(
            "snowflake.ml.feature_store.decl.imperative_executor._build_feature_group",
            return_value=(constructed_fg, "V1"),
        ):
            session = MagicMock(name="session")
            result = execute_plan(plan, session, "DB", "SCH", "WH", options)

        fs.register_feature_group.assert_called_once_with(constructed_fg, "V1")
        assert result.status == "applied"


# ---------------------------------------------------------------------------
# --allow-recreate gate
# ---------------------------------------------------------------------------


class TestAllowRecreateGateRefusesFG:
    def test_refuses_destructive_create_fg_without_allow_recreate(self) -> None:
        payload = _fg_payload()
        plan = Plan(ops=[_make_fg_op(payload, destructive=True)], warnings=[])
        options = PlanOptions(allow_recreate=False)

        # ``execute_plan`` must refuse before constructing FeatureStore.
        session = MagicMock(name="session")
        with patch("snowflake.ml.feature_store.decl.imperative_executor.assert_feature_store_initialized") as mock_init:
            result = execute_plan(plan, session, "DB", "SCH", "WH", options)

        mock_init.assert_not_called()
        assert result.status == "refused"
        # The refused row carries the FG op kind.
        refused = [r for r in result.ops if r.get("status") == "refused"]
        assert any(r["operation"] == OpKind.CREATE_FG.value for r in refused)

    def test_refuses_drop_fg_without_allow_recreate(self) -> None:
        payload = _fg_payload()
        plan = Plan(ops=[_make_drop_fg_op(payload)], warnings=[])
        options = PlanOptions(allow_recreate=False)

        session = MagicMock(name="session")
        with patch("snowflake.ml.feature_store.decl.imperative_executor.assert_feature_store_initialized") as mock_init:
            result = execute_plan(plan, session, "DB", "SCH", "WH", options)

        mock_init.assert_not_called()
        assert result.status == "refused"


# ---------------------------------------------------------------------------
# Cross-plan non-regression — FG over an advanced-BFV source
# ---------------------------------------------------------------------------


class TestFGHydrationPreservesAdvancedBfvFields:
    def test_get_feature_view_returns_advanced_bfv_fields_through(self) -> None:
        """If the source FV happens to be a BFV with cluster_by /
        storage_config / aggregation_secondary_keys set, ``_build_feature_group``
        must NOT touch those fields — it just hands the FV object straight
        to ``FeatureGroup(...)``.  This is the cross-plan non-regression
        invariant: the FG path is forward-compatible with the advanced BFV
        plan by going through ``fs.get_feature_view`` only.
        """
        from snowflake.ml.feature_store.decl.imperative_executor import (
            _build_feature_group,
        )

        fs = MagicMock(name="FeatureStore")
        # Build a mock FV that carries advanced BFV fields as attributes.
        # We just need to assert they survive the trip.
        bfv = MagicMock(name="BFV_with_advanced_fields")
        bfv.cluster_by = ["USER_ID"]
        bfv.storage_config = {"format": "iceberg"}
        bfv.aggregation_secondary_keys = ["KEYS"]
        bfv.warehouse = "MY_WH"
        bfv.refresh_mode = "INCREMENTAL"
        bfv.initialize = "ON_CREATE"
        fs.get_feature_view = MagicMock(return_value=bfv)

        payload = _fg_payload(
            feature_views=[{"name": "ADV_BFV", "version": "V1"}],
        )

        with patch("snowflake.ml.feature_store.feature_group.FeatureGroup") as mock_fg_cls:
            captured: dict[str, list[Any]] = {}

            def _capture_features(name: Any, features: Any, *, desc: Any, auto_prefix: Any) -> Any:
                captured["features"] = list(features)
                return MagicMock()

            mock_fg_cls.side_effect = _capture_features
            _build_feature_group(fs, payload, payload["version"])

        # Same object identity — no payload-side reconstruction.
        assert captured["features"] == [bfv]
        # The FV object's advanced fields are intact (the helper never
        # mutated them).
        assert bfv.cluster_by == ["USER_ID"]
        assert bfv.storage_config == {"format": "iceberg"}
        assert bfv.aggregation_secondary_keys == ["KEYS"]


if __name__ == "__main__":
    pytest_driver.main()
