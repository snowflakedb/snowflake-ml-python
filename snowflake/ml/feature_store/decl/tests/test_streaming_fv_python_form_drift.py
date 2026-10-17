# Copyright (c) 2024 Snowflake Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression net for the legacy streaming-FV ``spec.features`` RECREATE drift.

Three legacy imperatively-deployed StreamingFeatureViews
(``SK_ALL_AGGREGATIONS``, ``USER_CLICK_STATS_LONG_LOOKBACK``,
``CLICKSTREAM_ALL_AGGREGATIONS``) persistently re-planned as
``RECREATE_FV`` / ``Full-spec change: spec.features changed`` on every
``snow feature init`` + ``snow feature plan`` round-trip against
``JKEW_DB.JKEW_SCHEMA`` — but *only* when ``init`` emitted the default
**python form** (``.py``).  The ``--yaml`` form round-tripped cleanly
(pinned by ``test_golden_spec_round_trip.py``), which is why the drift
hid from every YAML-based regression net.

Root cause: :func:`decl.python_codegen._render_fscolumn` rendered
``name`` / ``type`` / ``length`` / ``precision`` / ``scale`` / ``tz`` /
``default`` but **not** ``element_type``.  Every ``ArrayType`` aggregation
output (``last_n`` / ``last_distinct_n`` / all continuous-secondary-key
outputs) therefore lost its inner element type on the python-form export,
so ``compute_local_spec_hash(local) != _full_spec_hash(applied)`` and the
planner tipped into the destructive branch for each such feature.

This module reproduces and pins the fix at three layers:

* :class:`TestPythonFormStreamingFvRoundTrip` — the load-bearing
  contract: the python-form round-trip hash must equal the applied
  ``DESCRIBE ... TYPE = SPECIFICATION`` hash for each captured golden.
* :class:`TestStreamingFvDriftMatrix` — the cohesive per-hypothesis
  evaluation.  Decomposes the normalized ``spec.features`` diff and
  attributes every entry to exactly one hypothesis (H1 type-promotion /
  H2 nested-length / H3 function_params / H4 element_type / H5 ordering)
  or to the H0 residual bucket, then asserts the round-trip has no drift.
  A future regression surfaces as the precise hypothesis that fired.
* :class:`TestElementTypeCodegenPin` — the tightest unit pin:
  ``_render_fscolumn`` must emit ``element_type`` for ``ArrayType``
  columns, and a change to ``element_type`` must still bump the hash
  (detection-safe canonicalization).

The captured goldens live at ``tests/golden_specs/<NAME>.json`` and are
verbatim ``DESCRIBE ... TYPE = SPECIFICATION`` payloads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from snowflake.ml.feature_store.decl.invariants import (
    _full_spec_hash,
    _normalize_for_full_spec_hash,
    compute_local_spec_hash,
)
from snowflake.ml.feature_store.decl.python_codegen import _render_fscolumn
from snowflake.ml.test_utils import pytest_driver

_GOLDEN_DIR = Path(__file__).parent / "golden_specs"
_LEGACY_FV_NAMES = [
    "SK_ALL_AGGREGATIONS",
    "USER_CLICK_STATS_LONG_LOOKBACK",
    "CLICKSTREAM_ALL_AGGREGATIONS",
]
_LEGACY_FV_PARAMS = [pytest.param(name, id=name) for name in _LEGACY_FV_NAMES]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_golden(name: str) -> dict[str, Any]:
    """Return the captured DESCRIBE payload for *name*.

    Args:
        name: Feature-view name whose golden fixture to load.

    Returns:
        Parsed golden spec dict.
    """
    path = _GOLDEN_DIR / f"{name}.json"
    assert path.exists(), f"required golden fixture missing: {path}"
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def _python_form_local_spec(name: str, tmp_path: Path) -> dict[str, Any]:
    """Export *name* to python form the way ``snow feature init`` does, then load it.

    Mirrors the default (python form) ``snow feature init`` + ``snow
    feature plan`` path end to end:

    1. Author the full-fidelity FV doc from the captured DESCRIBE payload
       via :func:`exporter._build_full_fidelity_fv`.
    2. Render it to a ``.py`` via
       :func:`python_codegen.spec_dict_to_python_source`.
    3. Load + compile it via :func:`loader.process_file` — the exact path
       the planner consumes for a python-form workspace, which inlines the
       UDF callable's source back to a string via
       ``serializer.spec_to_dict`` / ``inspect.getsource``.

    Args:
        name: Feature-view name whose golden fixture to export.
        tmp_path: pytest temp dir for the generated ``.py``.

    Returns:
        The compiled authoring-format FV spec dict (as the planner sees it).
    """
    from snowflake.ml.feature_store.decl.exporter import _build_full_fidelity_fv
    from snowflake.ml.feature_store.decl.loader import process_file
    from snowflake.ml.feature_store.decl.python_codegen import (
        spec_dict_to_python_source,
    )

    spec_payload = _load_golden(name)
    md = spec_payload["metadata"]
    doc = _build_full_fidelity_fv(
        spec_payload,
        fallback_name=md["name"],
        fallback_version=md["version"],
        fallback_database=md["database"],
        fallback_schema=md["schema"],
    )
    source = spec_dict_to_python_source("StreamingFeatureView", md["name"], doc)
    py_file = tmp_path / f"{name}.py"
    py_file.write_text(source)
    compiled = [d for d in process_file(str(py_file)) if "FeatureView" in d.get("kind", "")]
    assert compiled, f"python-form file {py_file} compiled no FeatureView spec"
    return compiled[0]


# ---------------------------------------------------------------------------
# The load-bearing contract: python-form round-trip must be NO_CHANGE.
# ---------------------------------------------------------------------------


class TestPythonFormStreamingFvRoundTrip:
    """``compute_local_spec_hash(pyform) == _full_spec_hash(applied)``."""

    @pytest.mark.parametrize("name", _LEGACY_FV_PARAMS)
    def test_python_form_roundtrip_is_no_change(self, name: str, tmp_path: Path) -> None:
        """The python-form export must re-plan as ``NO_CHANGE``.

        Args:
            name: Legacy FV name under test.
            tmp_path: pytest temp dir for the export workspace.
        """
        golden = _load_golden(name)
        md = golden["metadata"]
        db, schema = md["database"], md["schema"]
        local_spec = _python_form_local_spec(name, tmp_path)
        local_hash = compute_local_spec_hash(local_spec, db, schema)
        applied_hash = _full_spec_hash(golden)
        assert local_hash == applied_hash, (
            f"python-form round-trip drift for {name!r}: "
            f"local={local_hash} applied={applied_hash}. "
            "The legacy streaming-FV spec.features RECREATE cascade is back "
            "(python form). Run TestStreamingFvDriftMatrix for the attributed cause."
        )


# ---------------------------------------------------------------------------
# Cohesive per-hypothesis evaluation matrix.
# ---------------------------------------------------------------------------

# Widen/narrow type mates the runtime stamps on deploy (H1).
_PROMOTION_PAIRS = {("LongType", "IntegerType"), ("DoubleType", "FloatType")}


def _leaf_diffs(applied: Any, local: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    """Return ``(key_path, applied_value, local_value)`` leaf mismatches.

    Args:
        applied: Applied-side value (from the golden).
        local: Local-side value (from the python-form compile).
        path: Recursion accumulator; leave empty at the top level.

    Returns:
        A list of leaf-level diff tuples; empty when the two agree.
    """
    diffs: list[tuple[str, Any, Any]] = []
    if isinstance(applied, dict) and isinstance(local, dict):
        for key in sorted(set(applied) | set(local)):
            sub = f"{path}.{key}" if path else key
            diffs.extend(_leaf_diffs(applied.get(key), local.get(key), sub))
    elif isinstance(applied, list) and isinstance(local, list):
        if len(applied) != len(local):
            diffs.append((f"{path}[]", applied, local))
        else:
            for idx, (a, b) in enumerate(zip(applied, local)):
                diffs.extend(_leaf_diffs(a, b, f"{path}[{idx}]"))
    else:
        if applied != local:
            diffs.append((path, applied, local))
    return diffs


def _attribute(key_path: str, applied_val: Any, local_val: Any) -> str:
    """Attribute one leaf diff to a hypothesis bucket key.

    Args:
        key_path: Dotted leaf path (e.g. ``output_column.element_type``).
        applied_val: Applied-side leaf value.
        local_val: Local-side leaf value.

    Returns:
        One of ``"H1"`` / ``"H2"`` / ``"H3"`` / ``"H4"`` / ``"H0"``.
        (H5 ordering is detected at the list level, not per leaf.)
    """
    if key_path.endswith(".element_type") or key_path.endswith("element_type"):
        return "H4"
    if key_path.endswith(".type") or key_path.endswith("type"):
        if (applied_val, local_val) in _PROMOTION_PAIRS:
            return "H1"
    if key_path.endswith(".length") or key_path.endswith("length"):
        return "H2"
    if "function_params" in key_path:
        return "H3"
    return "H0"


def _features_drift_report(applied_norm: dict[str, Any], local_norm: dict[str, Any]) -> dict[str, list[str]]:
    """Attribute every ``spec.features`` diff to a hypothesis bucket.

    Pairs features across both normalized specs by ``output_column.name``
    (unpaired features are H0), then attributes every leaf mismatch to
    exactly one hypothesis (H1-H4) or the H0 residual.  H5 (ordering) is
    a whole-list signal: the multiset of feature dicts is equal but the
    order differs.

    Args:
        applied_norm: ``_normalize_for_full_spec_hash`` of the applied golden.
        local_norm: ``_normalize_for_full_spec_hash`` of the compiled local spec.

    Returns:
        Dict mapping ``H0``..``H5`` to human-readable diff descriptions.
    """
    report: dict[str, list[str]] = {h: [] for h in ("H1", "H2", "H3", "H4", "H5", "H0")}
    applied_feats = applied_norm.get("spec", {}).get("features", []) or []
    local_feats = local_norm.get("spec", {}).get("features", []) or []

    def _by_name(feats: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {f.get("output_column", {}).get("name", f"__{i}"): f for i, f in enumerate(feats)}

    a_by_name = _by_name(applied_feats)
    l_by_name = _by_name(local_feats)

    # H5 — ordering: same multiset, different order.
    a_sorted = sorted(applied_feats, key=lambda f: json.dumps(f, sort_keys=True))
    l_sorted = sorted(local_feats, key=lambda f: json.dumps(f, sort_keys=True))
    if applied_feats != local_feats and a_sorted == l_sorted:
        report["H5"].append("feature list order differs but multiset is equal")

    for fname in sorted(set(a_by_name) | set(l_by_name)):
        applied_feat = a_by_name.get(fname)
        local_feat = l_by_name.get(fname)
        if applied_feat is None or local_feat is None:
            report["H0"].append(f"{fname}: only on {'applied' if local_feat is None else 'local'} side")
            continue
        for key_path, av, lv in _leaf_diffs(applied_feat, local_feat):
            bucket = _attribute(key_path, av, lv)
            report[bucket].append(f"{fname}.{key_path}: applied={av!r} local={lv!r}")
    return report


class TestStreamingFvDriftMatrix:
    """Per-FV x per-hypothesis attribution of the python-form drift."""

    @pytest.mark.parametrize("name", _LEGACY_FV_PARAMS)
    def test_drift_is_fully_attributed_and_zero(self, name: str, tmp_path: Path) -> None:
        """After the fix, no drift; on any regression, H0 must stay empty.

        The matrix guarantees a future regression is explainable: every
        ``spec.features`` divergence is attributed to a named hypothesis
        (H1-H5).  A non-empty H0 residual would mean an unmodeled cause —
        the objective trigger to widen the hypothesis set.

        Args:
            name: Legacy FV name under test.
            tmp_path: pytest temp dir for the export workspace.
        """
        from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec

        golden = _load_golden(name)
        md = golden["metadata"]
        db, schema = md["database"], md["schema"]
        local_spec = _python_form_local_spec(name, tmp_path)
        local_compiled = compile_to_spec(local_spec, db, schema)

        applied_norm = _normalize_for_full_spec_hash(golden)
        local_norm = _normalize_for_full_spec_hash(local_compiled)
        report = _features_drift_report(applied_norm, local_norm)

        matrix = " ".join(f"{h}={len(report[h])}" for h in ("H1", "H2", "H3", "H4", "H5", "H0"))
        assert not report[
            "H0"
        ], f"{name}: unmodeled spec.features drift (H0 non-empty) — matrix [{matrix}]:\n" + "\n".join(
            f"  {e}" for e in report["H0"]
        )
        total = sum(len(v) for v in report.values())
        assert total == 0, f"{name}: python-form spec.features drift attributed — matrix [{matrix}]:\n" + "\n".join(
            f"  {h}: {e}" for h in report for e in report[h]
        )


# ---------------------------------------------------------------------------
# Tightest unit pin: _render_fscolumn must round-trip element_type.
# ---------------------------------------------------------------------------


class TestElementTypeCodegenPin:
    """``_render_fscolumn`` must emit ``element_type`` for ArrayType cols."""

    def test_render_fscolumn_emits_element_type(self) -> None:
        """An ``ArrayType`` column keeps ``element_type`` in the codegen."""
        rendered = _render_fscolumn({"name": "LAST_PAGES_1H", "type": "ArrayType", "element_type": "StringType"})
        assert "element_type='StringType'" in rendered, (
            "python_codegen._render_fscolumn dropped element_type; ArrayType "
            "aggregation outputs will lose their inner type on python-form "
            f"export. Rendered: {rendered!r}"
        )

    def test_render_fscolumn_omits_element_type_when_absent(self) -> None:
        """A scalar column (no ``element_type``) must not gain the kwarg."""
        rendered = _render_fscolumn({"name": "SUM_1H", "type": "DoubleType"})
        assert "element_type" not in rendered, f"unexpected element_type in {rendered!r}"

    def test_element_type_change_still_bumps_hash(self) -> None:
        """Detection-safety: a real ``element_type`` change must alter the hash.

        ``element_type`` is server-derived (a pure function of the aggregation
        function + source type) and therefore normalized away **only** for
        *secondary-key* FVs — see ``invariants._strip_secondary_key_derivations``
        and ``test_secondary_key_feature_change_still_bumps_hash`` for the SK
        detection contract.  For a **non-SK** streaming ArrayType aggregation the
        inner type is operator-meaningful and stays hash-significant, so a change
        here must still be detected.  This uses the non-SK golden
        ``CLICKSTREAM_ALL_AGGREGATIONS`` (the SK normalization is a no-op there)
        so it continues to pin the original detection intent where it applies.
        """
        base = _load_golden("CLICKSTREAM_ALL_AGGREGATIONS")
        assert not (
            base["spec"].get("ordered_secondary_key_column_names") or base["spec"].get("aggregation_secondary_keys")
        ), "CLICKSTREAM_ALL_AGGREGATIONS unexpectedly carries a secondary key"
        mutated = json.loads(json.dumps(base))
        feats = mutated["spec"]["features"]
        array_feat = next(f for f in feats if f["output_column"].get("element_type"))
        array_feat["output_column"]["element_type"] = (
            "StringType" if array_feat["output_column"]["element_type"] != "StringType" else "LongType"
        )
        assert _full_spec_hash(base) != _full_spec_hash(mutated), (
            "element_type change did not bump the hash for a non-SK ArrayType FV — "
            "canonicalization is masking a real feature change (detection-unsafe)."
        )

    def test_secondary_key_feature_change_still_bumps_hash(self) -> None:
        """Detection-safety for SK FVs whose ``element_type`` is normalized away.

        For a secondary-key FV, ``_strip_secondary_key_derivations`` drops the
        server-derived aggregation ``output_column`` type wrapping (the SK forces
        every real aggregation to ``ArrayType`` with a function-derived
        ``element_type``), so an ``element_type``-only mutation is intentionally
        NOT hash-significant there.  Detection of a genuine feature change
        instead relies on the retained semantic fields
        (``function`` / ``window_sec`` / ``function_params`` / ``source_column`` /
        ``output_column.name``).  Mutating a real aggregation's ``window_sec`` on
        the SK golden must still bump the hash.
        """
        base = _load_golden("SK_ALL_AGGREGATIONS")
        mutated = json.loads(json.dumps(base))
        feats = mutated["spec"]["features"]
        agg_feat = next(f for f in feats if f.get("function"))
        agg_feat["window_sec"] = int(agg_feat.get("window_sec", 0)) + 12345
        assert _full_spec_hash(base) != _full_spec_hash(mutated), (
            "window_sec change did not bump the hash for an SK aggregation — "
            "SK normalization over-stripped a semantic field (detection-unsafe)."
        )

    def test_secondary_key_output_type_wrapping_is_hash_insignificant(self) -> None:
        """On a real SK aggregation, ``_strip_secondary_key_derivations`` drops the
        server-derived ``output_column`` ``type`` / ``element_type`` / ``length``, so
        rewriting all three must not bump the hash; a genuine change is caught by the
        retained semantic fields instead (see the ``window_sec`` test above).
        """
        base = _load_golden("SK_ALL_AGGREGATIONS")
        mutated = json.loads(json.dumps(base))
        feats = mutated["spec"]["features"]
        agg_feat = next(f for f in feats if f.get("function"))
        oc = agg_feat["output_column"]
        oc["type"] = "StringType"
        oc["element_type"] = "StringType"
        oc["length"] = 1
        assert _full_spec_hash(base) == _full_spec_hash(mutated), (
            "output_column type wrapping bumped the hash for an SK aggregation — "
            "the server-derived type/element_type/length is no longer stripped."
        )


if __name__ == "__main__":
    pytest_driver.main()
