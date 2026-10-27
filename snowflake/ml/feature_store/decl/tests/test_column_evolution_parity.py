"""Validator/planner parity for spurious ``COLUMN_REMOVED`` warnings.

A BatchFeatureView the planner reports as ``NO_CHANGE`` must not trip the
validator's column-evolution path. Two causes are exercised here:

1. ``_check_idempotency`` compared the local hash against the *raw*
   ``applied.content_hash``; the planner additionally strips a
   Snowflake-stamped ``refresh_mode`` (INCREMENTAL/FULL) for BatchFeatureViews
   via ``_normalize_applied_bfv_for_hash``.  The validator applies the same
   normalisation so its ``NO_CHANGE`` lines up with the planner's ``NO_CHANGE``
   op — otherwise column evolution runs on a spec the planner treats as
   unchanged.
2. ``_check_fv_column_evolution`` diffed the *raw* applied features, so the
   server-synthesised ``<SK>_KEYS_<window>`` arrays (and query-derived columns
   on a passthrough BFV) looked like removed output columns.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from snowflake.ml.feature_store.decl.invariants import (
    _check_column_evolution,
    _check_idempotency,
    model_to_dict,
    validate_specs,
)
from snowflake.ml.feature_store.decl.planner import generate_plan
from snowflake.ml.feature_store.decl.spec_models import FeatureView
from snowflake.ml.feature_store.decl.tests.test_refresh_mode_normalization import (
    _DB,
    _FV_NAME,
    _SCH,
    _applied_state_incremental,
    _minimal_bfv_authoring,
    _spec_batch,
)
from snowflake.ml.feature_store.decl.types import AppliedObject, PlanOptions
from snowflake.ml.test_utils import pytest_driver

_GOLDEN_DIR = Path(__file__).parent / "golden_specs"
_SK_GOLDEN = _GOLDEN_DIR / "CONTENT_AGG_BATCH_SK.json"


def _load_sk_golden() -> dict[str, Any]:
    """Load the captured SK BatchFeatureView DESCRIBE payload.

    Returns:
        The parsed golden dict (features nested under ``spec.features``, with
        the synthesised ``OFFER_KEY_KEYS_*`` arrays present).
    """
    golden: dict[str, Any] = json.loads(_SK_GOLDEN.read_text())
    return golden


def _sk_applied(golden: dict[str, Any]) -> AppliedObject:
    """Wrap the SK golden as a ``from_specification`` applied object.

    Args:
        golden: The parsed golden payload.

    Returns:
        An ``AppliedObject`` carrying the golden as its deployed spec.  The
        ``content_hash`` is intentionally divergent so ``_check_idempotency``
        does not short-circuit and the column-evolution path is exercised
        directly (defence in depth against a genuine edit to an SK feature view).
    """
    return AppliedObject(
        key="k",
        kind="BatchFeatureView",
        name=golden["metadata"]["name"],
        version="V1",
        content_hash="deadbeef" * 8,
        spec_payload=golden,
        from_specification=True,
    )


def _sk_local(golden: dict[str, Any], *, drop_output: str | None = None) -> dict[str, Any]:
    """Build the operator-authored SK spec: the real aggregations only.

    The operator never authors the synthesised ``OFFER_KEY_KEYS_*`` arrays; they
    are a pure server derivation.  Authored output columns carry a scalar type
    (not the deployed ``ArrayType`` wrapping the server derives), so the diff is
    exercised against a realistic authoring shape.  Optionally drop one real
    aggregation to model a genuine output-column removal.

    Args:
        golden: The parsed golden payload (source of the real aggregations).
        drop_output: Output-column name to omit from the authored list, or None.

    Returns:
        A local spec dict with ``aggregation_secondary_keys`` and the real
        aggregation features.
    """
    real: list[dict[str, Any]] = []
    for feat in golden["spec"]["features"]:
        if not feat.get("function"):
            continue
        out_name = feat.get("output_column", {}).get("name")
        if drop_output is not None and out_name == drop_output:
            continue
        authored = copy.deepcopy(feat)
        element_type = authored.get("output_column", {}).get("element_type") or "DoubleType"
        authored["output_column"] = {"name": out_name, "type": element_type}
        real.append(authored)
    return {
        "kind": "BatchFeatureView",
        "name": golden["metadata"]["name"],
        "version": "V1",
        "aggregation_secondary_keys": ["OFFER_KEY"],
        "features": real,
    }


# ---------------------------------------------------------------------------
# 1. Validator/planner idempotency parity (end-to-end)
# ---------------------------------------------------------------------------


class TestValidatorPlannerIdempotencyParity:
    """A BatchFeatureView the planner calls ``NO_CHANGE`` must not trip the
    validator's column-evolution path."""

    def test_validate_specs_no_change_when_applied_has_incremental(self) -> None:
        """Operator omits ``refresh_mode``; Snowflake stamped INCREMENTAL on the
        DT.  ``validate_specs`` must emit exactly one ``NO_CHANGE`` result for
        the FV and zero ``COLUMN_*`` results — matching the planner, which
        already returns ``NO_CHANGE``."""
        batch = _spec_batch(_minimal_bfv_authoring())
        state = _applied_state_incremental()
        results = validate_specs(batch, state, target_database=_DB, target_schema=_SCH)

        no_change = [r for r in results if r.code == "NO_CHANGE" and _FV_NAME in (r.message or "")]
        assert len(no_change) == 1, (
            "validator must short-circuit to NO_CHANGE for a BatchFeatureView whose "
            "only applied-side difference is a Snowflake-stamped refresh_mode; got "
            f"{[(r.code, r.message) for r in results if _FV_NAME in (r.message or '') or r.object_name == _FV_NAME]!r}"
        )
        column_results = [r for r in results if r.code.startswith("COLUMN_") and r.object_name == _FV_NAME]
        assert column_results == [], (
            "no COLUMN_* warning may be emitted for an object the planner treats as "
            f"NO_CHANGE; got {[(r.code, r.message) for r in column_results]!r}"
        )

    def test_generate_plan_still_no_change(self) -> None:
        """Op-parity pin: the planner's op for the same scenario stays
        ``NO_CHANGE`` (the fix changes diagnostics only, not op selection)."""
        batch = _spec_batch(_minimal_bfv_authoring())
        state = _applied_state_incremental()
        plan = generate_plan(batch, state, PlanOptions(), database=_DB, schema=_SCH)

        fv_ops = [op for op in plan.ops if op.name == _FV_NAME]
        assert len(fv_ops) == 1
        assert fv_ops[0].kind.value == "NO_CHANGE", f"expected NO_CHANGE, got {fv_ops[0].kind.value}"

    def test_check_idempotency_not_up_to_date_when_local_authors_full(self) -> None:
        """The applied-side normalisation must not over-reach: an operator who
        pins ``refresh_mode: FULL`` against a deployed INCREMENTAL is a genuine
        change, so ``_check_idempotency`` must report not-up-to-date (mirroring
        the planner's RECREATE_FV)."""
        local = model_to_dict(FeatureView.model_validate(_minimal_bfv_authoring(refresh_mode="FULL")))
        applied = next(iter(_applied_state_incremental().objects.values()))
        is_up_to_date, _ = _check_idempotency(local, applied, target_database=_DB, target_schema=_SCH)
        assert is_up_to_date is False, "local FULL vs applied INCREMENTAL is a real change, not NO_CHANGE"


# ---------------------------------------------------------------------------
# 2. Secondary-key column diff (defence in depth on a forced idempotency miss)
# ---------------------------------------------------------------------------


class TestSecondaryKeyColumnDiff:
    """``_check_fv_column_evolution`` must canonicalise the applied payload so
    server-derived SK arrays are not reported as removed."""

    def test_sk_arrays_not_reported_removed(self) -> None:
        """The synthesised ``OFFER_KEY_KEYS_*`` arrays exist only on the deployed
        side; they must never surface as ``COLUMN_REMOVED``."""
        golden = _load_sk_golden()
        results = _check_column_evolution(_sk_local(golden), _sk_applied(golden))

        sk_removed = [r for r in results if r.code == "COLUMN_REMOVED" and "OFFER_KEY_KEYS" in r.message]
        assert sk_removed == [], (
            "secondary-key array features are a pure server derivation and must not be "
            f"reported removed; got {[r.message for r in sk_removed]!r}"
        )

    def test_genuine_removed_aggregation_still_warns(self) -> None:
        """A real authored aggregation that the operator drops must still warn —
        the SK-strip must not suppress genuine removals."""
        golden = _load_sk_golden()
        results = _check_column_evolution(_sk_local(golden, drop_output="VIEWS_4WEEKS"), _sk_applied(golden))

        removed = {r.message.split(".")[1].split(":")[0] for r in results if r.code == "COLUMN_REMOVED"}
        assert "VIEWS_4WEEKS" in removed, f"genuine removal of VIEWS_4WEEKS must warn; got removed={removed!r}"


# ---------------------------------------------------------------------------
# 3. Passthrough BatchFeatureView (no authored features / query-derived columns)
# ---------------------------------------------------------------------------


class TestPassthroughBfvColumnDiff:
    """A BFV that authors no ``features`` draws its output columns from the
    query; a deployed-only column is not an operator-actionable removal."""

    def _passthrough_applied(self) -> AppliedObject:
        payload = {
            "kind": "BatchFeatureView",
            "metadata": {"name": "PASSTHROUGH_BFV", "version": "V1"},
            "spec": {
                "features": [
                    {"output_column": {"name": "EVENT_TS", "type": "TimestampType"}},
                    {"output_column": {"name": "METRIC_VAL", "type": "DoubleType"}},
                ],
            },
        }
        return AppliedObject(
            key="k",
            kind="BatchFeatureView",
            name="PASSTHROUGH_BFV",
            version="V1",
            content_hash="deadbeef" * 8,
            spec_payload=payload,
            from_specification=True,
        )

    def test_no_features_authored_no_removed(self) -> None:
        """Local authors no ``features``; the deployed query-derived columns must
        not be reported as removed."""
        local = {"kind": "BatchFeatureView", "name": "PASSTHROUGH_BFV", "version": "V1"}
        results = _check_column_evolution(local, self._passthrough_applied())

        removed = [r for r in results if r.code == "COLUMN_REMOVED"]
        assert removed == [], (
            "a passthrough BFV with no authored features must not report its query-derived "
            f"columns as removed; got {[r.message for r in removed]!r}"
        )

    def _mixed_applied(self) -> AppliedObject:
        """Deployed BFV with one real aggregation plus two query-derived columns."""
        payload = {
            "kind": "BatchFeatureView",
            "metadata": {"name": "PASSTHROUGH_BFV", "version": "V1"},
            "spec": {
                "features": [
                    {
                        "function": "sum",
                        "window_sec": 3600,
                        "source_column": {"name": "AMT", "type": "DoubleType"},
                        "output_column": {"name": "AMT_SUM_1H", "type": "DoubleType"},
                    },
                    {"output_column": {"name": "EVENT_TS", "type": "TimestampType"}},
                    {"output_column": {"name": "METRIC_VAL", "type": "DoubleType"}},
                ],
            },
        }
        return AppliedObject(
            key="k",
            kind="BatchFeatureView",
            name="PASSTHROUGH_BFV",
            version="V1",
            content_hash="deadbeef" * 8,
            spec_payload=payload,
            from_specification=True,
        )

    def test_dropping_all_features_still_warns_real_aggregation(self) -> None:
        """Dropping the whole authored ``features`` list genuinely removes a
        deployed aggregation (``AMT_SUM_1H``), so that must still warn — only
        the query-derived columns are exempt from the no-features skip."""
        local = {"kind": "BatchFeatureView", "name": "PASSTHROUGH_BFV", "version": "V1"}
        results = _check_column_evolution(local, self._mixed_applied())

        removed = {r.message.split(".")[1].split(":")[0] for r in results if r.code == "COLUMN_REMOVED"}
        assert removed == {"AMT_SUM_1H"}, (
            "only the real aggregation may warn as removed when the authored list is " f"dropped; got {removed!r}"
        )


# ---------------------------------------------------------------------------
# 4. Regression guard: genuine removal from an authored list still warns
# ---------------------------------------------------------------------------


class TestGenuineRemovalRegressionGuard:
    """The no-features skip must not mask a real removal from an authored list."""

    def _fv(self, cols: list[tuple[str, str]], version: str = "V1") -> dict[str, Any]:
        return {
            "kind": "StreamingFeatureView",
            "name": "click_features",
            "database": "DB",
            "schema": "SCH",
            "version": version,
            "entities": ["user_id"],
            "sources": [{"name": "clicks", "source_type": "Stream"}],
            "features": [
                {"source_column": {"name": c, "type": t}, "output_column": {"name": c, "type": t}} for c, t in cols
            ],
        }

    def test_authored_column_removal_still_warns(self) -> None:
        applied = AppliedObject(
            key="k",
            kind="StreamingFeatureView",
            name="click_features",
            version="V1",
            content_hash="deadbeef" * 8,
            spec_payload=self._fv([("event", "StringType"), ("count", "IntegerType")]),
            from_specification=False,
        )
        results = _check_column_evolution(self._fv([("event", "StringType")], version="V2"), applied)
        assert any(r.code == "COLUMN_REMOVED" and "count" in r.message for r in results)


if __name__ == "__main__":
    pytest_driver.main()
