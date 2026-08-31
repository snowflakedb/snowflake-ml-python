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

"""Dedicated regression net for the streaming-FV re-plan drift cascade.

The bug-bash §9 symptom is a re-plan emitting ``RECREATE_FV
USER_CLICK_STATS_DECL`` (destructive) instead of ``NO_CHANGE`` after
step 6's deploy lands cleanly.  Any one of the streaming-specific FV
fields silently dropping out of the round-trip — e.g.
``feature_aggregation_method``, ``feature_granularity_sec``,
``timestamp_field``, the per-feature ``window_sec`` / ``function`` /
``source_column`` triple, or the ``udf`` block — is enough to push
``compute_local_spec_hash(local) != applied.content_hash`` and trip
the planner into the destructive branch.

This module is the *focused* regression net for that recurrence path.
It complements (but does not duplicate) the broader
``test_planner_revalidate_golden_describe.py`` and
``test_planner_revalidate_identical_spec.py`` suites by:

- pinning the captured-DESCRIBE payload's *streaming-specific* fields
  one at a time (so a future regression that drops just one of them
  surfaces as the precise field name in the failure message),
- pinning the local-compile path's mirror of those fields (so a
  spec_compiler regression surfaces equally cleanly),
- pinning ``generate_plan`` end-to-end against the same captured
  payload (so anyone reading this file sees the full re-plan contract
  in one place — the single source of truth for the §9 cascade).

References:
- ``plans/streaming_fv_aggregation_lost.plan.md`` — the previous
  resolution of this bug surface.
- ``plans/step9_apply_status_fix_*.plan.md`` — the orchestrating plan
  for the current pass.
- ``docs/ARCHITECTURE.md`` § "Apply Lifecycle (L1–L7)" + "Apply
  result status surface".
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.invariants import (
    _canonical_spec_for_hash,
    _full_spec_hash,
    compute_local_spec_hash,
    model_to_dict,
)
from snowflake.ml.feature_store.decl.planner import generate_plan
from snowflake.ml.feature_store.decl.tests.test_planner_revalidate_golden_describe import (
    _BUG_BASH_GOLDEN,
    _build_full_applied_state,
    _build_full_batch,
)
from snowflake.ml.feature_store.decl.tests.test_planner_revalidate_identical_spec import (
    _BUG_BASH_DB,
    _BUG_BASH_FV_NAME,
    _BUG_BASH_SCHEMA,
    _bug_bash_fv_model,
)
from snowflake.ml.feature_store.decl.types import PlanOptions
from snowflake.ml.test_utils import pytest_driver

# ---------------------------------------------------------------------------
# Module-level helpers (kept as plain functions, not pytest fixtures, so
# darglint's ``Args:``-required check stays happy and so the test
# signatures match the sibling ``test_planner_revalidate_golden_describe.py``
# style).
# ---------------------------------------------------------------------------


def _load_golden_specification() -> dict[str, Any]:
    """Load the captured DESCRIBE TYPE = SPECIFICATION payload.

    The fixture lives at
    ``snowml/snowflake/ml/feature_store/decl/tests/golden_specs/USER_CLICK_STATS_DECL.json``
    and is a verbatim capture of what the live runtime returns for the
    BUG_BASH §5 FV after step 6 deploy.  It carries the
    Snowflake-stamped top-level keys (``offline_configs``,
    ``online_store_type``, ``metadata.oft_id``, ...) that the
    strip-before-hash contract in :func:`_full_spec_hash` must absorb.

    Returns:
        Parsed golden dict — the BUG_BASH §5 FV's DESCRIBE payload.
    """
    if not _BUG_BASH_GOLDEN.exists():
        pytest.skip(f"golden fixture {_BUG_BASH_GOLDEN} not captured yet")
    golden: dict[str, Any] = json.loads(_BUG_BASH_GOLDEN.read_text())
    return golden


def _build_local_compiled() -> dict[str, Any]:
    """Compile the BUG_BASH §5 FV model through ``compile_to_spec``.

    This is the local side of the round-trip: what the planner derives
    from the YAML-loaded model when computing
    :func:`compute_local_spec_hash`.

    A drift between this and the captured DESCRIBE payload in any field
    that contributes to the hash (i.e. anything outside
    ``_VOLATILE_METADATA_KEYS`` / ``_DERIVED_TOP_LEVEL_KEYS``) is the
    root cause of the §9 cascade.

    Returns:
        Compiled spec dict — same shape as the DESCRIBE payload.
    """
    from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec

    fv = _bug_bash_fv_model()
    return compile_to_spec(model_to_dict(fv), _BUG_BASH_DB, _BUG_BASH_SCHEMA)


# ---------------------------------------------------------------------------
# Streaming-specific field round-trip pins.  Each test names one field —
# a regression that drops that single field surfaces as a precise
# failure message instead of an opaque "RECREATE_FV emitted" diagnostic.
# ---------------------------------------------------------------------------


class TestStreamingFvSpecRoundTrip:
    """Per-field round-trip pins for the streaming-FV spec.

    Every assertion in this class names exactly one streaming-specific
    field of the FV's compiled spec.  A regression that drops a field
    on either the local-compile side or the captured-DESCRIBE side
    surfaces as that field name in the failure output, so the
    follow-up fix can target ``decl/spec_compiler.py``,
    ``decl/imperative_executor._build_feature_view``, or
    ``decl/api.fetch_applied_state`` directly.
    """

    def test_golden_carries_feature_aggregation_method(self) -> None:
        """The captured DESCRIBE payload must carry
        ``spec.feature_aggregation_method = 'tiles'`` — dropping this
        field on the SPECIFICATION fetch side reproduces the
        ``streaming_fv_aggregation_lost`` regression directly.
        """
        golden = _load_golden_specification()
        assert golden.get("spec", {}).get("feature_aggregation_method") == "tiles", (
            "DESCRIBE TYPE = SPECIFICATION fixture is missing "
            "spec.feature_aggregation_method.  Re-capture the golden — "
            "or, if the live runtime really stopped emitting it, fix "
            "decl/api.fetch_applied_state to default it back."
        )

    def test_golden_carries_feature_granularity_sec(self) -> None:
        """The captured DESCRIBE payload must carry
        ``spec.feature_granularity_sec`` — drop reproduces the
        granularity half of the aggregation triple drift.
        """
        golden = _load_golden_specification()
        assert golden.get("spec", {}).get("feature_granularity_sec") == 300, (
            "DESCRIBE TYPE = SPECIFICATION fixture is missing "
            "spec.feature_granularity_sec — half of the aggregation "
            "triple is dropped.  Fix on either the SPECIFICATION fetch "
            "or the local-compile side, whichever stopped emitting it."
        )

    def test_local_compile_emits_feature_aggregation_method(self) -> None:
        """The local-compile path must emit
        ``feature_aggregation_method`` so the round-trip hash collides
        with the captured payload's hash.  Drop reproduces the
        ``streaming_fv_aggregation_lost`` regression on the local side
        (the more common failure mode).
        """
        compiled = _build_local_compiled()
        assert compiled.get("spec", {}).get("feature_aggregation_method") == "tiles", (
            "spec_compiler.compile_to_spec dropped "
            "feature_aggregation_method on the local-compile path; the "
            "compute_local_spec_hash digest will not match the "
            "captured DESCRIBE payload, tipping the planner into the "
            "destructive RECREATE_FV branch."
        )

    def test_local_compile_emits_feature_granularity_sec(self) -> None:
        """Companion to the aggregation-method pin — the granularity
        half of the triple.
        """
        compiled = _build_local_compiled()
        assert compiled.get("spec", {}).get("feature_granularity_sec") == 300, (
            "spec_compiler.compile_to_spec dropped feature_granularity_sec " "on the local-compile path."
        )

    def test_local_compile_emits_per_feature_aggregation_triple(self) -> None:
        """Every entry in ``spec.features`` must carry the
        ``window_sec`` + ``function`` + ``source_column`` triple — the
        per-feature aggregation parameters that distinguish a streaming
        FV from a batch FV.

        A regression that drops any of these on local-compile would
        produce ``RECREATE_FV`` per feature — the hash diff would be
        spectacular but the failure mode would also be tagged
        ``"Full-spec change detected"`` so the operator can't tell
        which field drifted.  This pin names them all explicitly.
        """
        compiled = _build_local_compiled()
        features = compiled.get("spec", {}).get("features", [])
        assert features, "local-compiled spec has no features"
        for feature in features:
            for key in ("window_sec", "function", "source_column"):
                assert feature.get(key) is not None, f"feature {feature.get('output_column')!r} missing {key!r}"

    def test_local_compile_emits_timestamp_field(self) -> None:
        """``timestamp_field`` is required on every streaming FV — the
        runtime won't accept the FV without it.  A drop here would
        reproduce a different cascade (CREATE_FV failure on apply
        rather than a re-plan drift), but it lives in the same
        spec_compiler code path so we pin it alongside the aggregation
        triple.
        """
        compiled = _build_local_compiled()
        assert (
            compiled.get("spec", {}).get("timestamp_field") == "TIMESTAMP"
        ), "spec_compiler dropped timestamp_field on the local-compile path"

    def test_local_compile_emits_udf_block(self) -> None:
        """The ``udf`` block (file + name + engine + output_columns) must
        survive the local compile — without it the FV has no
        transform and apply fails.  Pinned here because the §9 cascade
        listed UDF source as one of the four "Full-spec change
        detected" causes.
        """
        compiled = _build_local_compiled()
        udf = compiled.get("spec", {}).get("udf")
        assert udf, "spec_compiler dropped the entire udf block"
        for key in ("name", "engine", "output_columns"):
            assert udf.get(key) is not None, f"udf block missing {key!r}"


# ---------------------------------------------------------------------------
# Hash collision pin: the local + captured payloads must produce
# identical full-spec hashes after the strip-before-hash contract runs.
# This is the single test that, if it fails, definitively confirms the
# planner will tip into RECREATE_FV.
# ---------------------------------------------------------------------------


class TestStreamingFvHashCollision:
    """``compute_local_spec_hash(local)`` must equal
    ``_full_spec_hash(golden)`` — when they disagree the planner emits
    ``RECREATE_FV`` regardless of any per-field test below.
    """

    def test_local_hash_matches_captured_describe_hash(self) -> None:
        """The hash collision is the actual planner contract.

        ``planner._diff_feature_view`` calls ``compute_local_spec_hash``
        on the local model and compares it to ``applied.content_hash``
        (which is ``_full_spec_hash(golden)`` in the live env).  When
        the two digests differ, the planner emits a ``RECREATE_FV`` op
        with reason ``"Full-spec change detected"`` — the §9 cascade.

        This test pins the collision against the captured payload.  If
        it fails, the cascade is back, and the failure message names
        which payload diverged so the fix can be targeted.
        """
        fv = _bug_bash_fv_model()
        local_hash = compute_local_spec_hash(
            model_to_dict(fv),
            _BUG_BASH_DB,
            _BUG_BASH_SCHEMA,
        )
        applied_hash = _full_spec_hash(_load_golden_specification())
        assert local_hash == applied_hash, (
            "Hash collision broken — compute_local_spec_hash(local) != "
            f"_full_spec_hash(golden).  local={local_hash} "
            f"applied={applied_hash}.  The planner will tip into "
            "RECREATE_FV destructive on next re-plan (BUG_BASH §9 "
            "cascade is back).  Run the diagnostic test below to see "
            "the structured diff between local-compiled and golden "
            "payloads, then fix the side that's non-canonical."
        )


# ---------------------------------------------------------------------------
# End-to-end re-plan pin: generate_plan returns NO_CHANGE for the FV
# when applied_state is built off the captured payload.  Same shape
# as the existing ``test_planner_revalidate_golden_describe.py`` —
# kept here so future contributors only have to look in one place
# to see the streaming-FV drift contract end-to-end.
# ---------------------------------------------------------------------------


class TestStreamingFvReplanIsNoChange:
    """``generate_plan`` against the captured payload must emit a
    ``NO_CHANGE`` op for ``USER_CLICK_STATS_DECL``.  This is the
    direct contract for BUG_BASH §9.
    """

    def test_replan_against_captured_specification_is_no_change(self) -> None:
        """Re-plan after step 6's deploy must produce ``NO_CHANGE`` for
        the FV.  ``RECREATE_FV`` here means the §9 cascade is back.

        The applied state is built via the same helper the broader
        golden-describe suite uses, so the failure mode is consistent
        across all three regression nets:

        * ``test_planner_revalidate_golden_describe.py`` — multi-rule
          coverage (validator + planner + divergent-hash fallthrough).
        * ``test_planner_revalidate_identical_spec.py`` — synthetic-
          payload coverage (compile_to_spec produces matching hash).
        * ``test_streaming_fv_replan_drift.py`` (this file) — focused
          per-field coverage for the streaming-specific recurrence.
        """
        applied_state = _build_full_applied_state()
        batch = _build_full_batch()

        plan = generate_plan(
            batch,
            applied_state,
            PlanOptions(),
            database=_BUG_BASH_DB,
            schema=_BUG_BASH_SCHEMA,
        )

        fv_ops = [op for op in plan.ops if op.name == _BUG_BASH_FV_NAME]
        assert len(fv_ops) == 1, (
            f"expected exactly one op for {_BUG_BASH_FV_NAME}; got "
            f"{[(op.kind.value, op.reason) for op in fv_ops]!r}"
        )
        op = fv_ops[0]
        assert op.kind == OpKind.NO_CHANGE, (
            f"BUG_BASH §9 cascade re-introduced — re-plan emitted "
            f"kind={op.kind.value!r} reason={op.reason!r} for "
            f"{_BUG_BASH_FV_NAME} when the captured DESCRIBE payload "
            f"was used as applied_state.  Run the diagnostic test below "
            f"(`pytest -s -k diagnose`) to see the field-level diff "
            f"between local-compiled and captured payloads."
        )
        assert not getattr(op, "destructive", False), (
            f"NO_CHANGE op must not be destructive; got destructive=True " f"for {_BUG_BASH_FV_NAME}."
        )


# ---------------------------------------------------------------------------
# Diagnostic helper: pytest -s -k diagnose to print the structured diff
# between the local-compiled spec and the captured DESCRIBE payload.
# ---------------------------------------------------------------------------


def _payload_diff(local: dict[str, Any], deployed: dict[str, Any]) -> list[str]:
    """Return human-readable per-field diff lines between two specs.

    Both payloads are passed through :func:`_canonical_spec_for_hash` so
    the walk matches what :func:`_full_spec_hash` hashes.

    Args:
        local: Spec dict from ``compile_to_spec`` (local side).
        deployed: Spec dict from the captured DESCRIBE payload.

    Returns:
        List of diff lines; empty when the two specs agree on every
        hash-relevant field.
    """
    return _walk_canonical_payload(
        _canonical_spec_for_hash(local),
        _canonical_spec_for_hash(deployed),
        path="",
    )


def _walk_canonical_payload(local: Any, deployed: Any, *, path: str) -> list[str]:
    """Depth-first walk of two already-canonical spec fragments.

    Args:
        local: Canonical local value at *path*.
        deployed: Canonical deployed value at *path*.
        path: Dotted path for the current node; empty at the spec root.

    Returns:
        Diff lines for this node and its descendants.
    """
    diffs: list[str] = []
    if isinstance(local, dict) and isinstance(deployed, dict):
        local_keys = set(local.keys())
        dep_keys = set(deployed.keys())
        for key in sorted(local_keys | dep_keys):
            sub_path = f"{path}.{key}" if path else key
            if key not in deployed:
                diffs.append(f"{sub_path}: only in local = {local[key]!r}")
            elif key not in local:
                diffs.append(f"{sub_path}: only in deployed = {deployed[key]!r}")
            else:
                diffs.extend(_walk_canonical_payload(local[key], deployed[key], path=sub_path))
    elif isinstance(local, list) and isinstance(deployed, list):
        if len(local) != len(deployed):
            diffs.append(f"{path}: list length differs local={len(local)} " f"deployed={len(deployed)}")
        for idx in range(min(len(local), len(deployed))):
            diffs.extend(_walk_canonical_payload(local[idx], deployed[idx], path=f"{path}[{idx}]"))
    else:
        if local != deployed:
            diffs.append(f"{path}: local={local!r} != deployed={deployed!r}")
    return diffs


def test_diagnose_local_vs_captured_payload_diff() -> None:
    """Emit the structured diff between local-compiled and captured
    payloads on stderr.  Run with ``pytest -s -k diagnose`` to surface
    the diff in the terminal.

    This is the diagnostic helper the plan asks for — when a future
    regression re-introduces the §9 cascade, run this test and the
    failing fields show up directly.  The test itself is documentary
    (always passes) so it doesn't add CI noise; it only adds value
    when explicitly inspected.

    Writes to ``sys.stderr`` directly (instead of ``print``) so the
    flake8-print check stays happy and so the output arrives on the
    same stream the rest of the planner diagnostics use.
    """
    golden = _load_golden_specification()
    compiled = _build_local_compiled()
    diffs = _payload_diff(compiled, golden)
    if diffs:
        sys.stderr.write(f"\n=== local-compiled vs captured DESCRIBE diff ({len(diffs)} entries) ===\n")
        for line in diffs:
            sys.stderr.write(f"  {line}\n")
        sys.stderr.write("=== end diff ===\n")
    else:
        sys.stderr.write("\n=== local-compiled and captured DESCRIBE agree on every hash-relevant field ===\n")
    # Test is documentary; assert nothing.  When a regression fires,
    # ``test_local_hash_matches_captured_describe_hash`` and
    # ``test_replan_against_captured_specification_is_no_change`` both
    # fail and direct the operator here.


def _hash_aligned_fixture() -> dict[str, Any]:
    """Minimal streaming-FV payload for hash-aligned ``_payload_diff`` pins.

    Returns:
        A compiled-shaped spec dict with volatile metadata and derived
        top-level keys that hashing must ignore.
    """
    return {
        "kind": "StreamingFeatureView",
        "metadata": {
            "name": "USER_CLICK_STATS_DECL",
            "version": "v1",
            "database": "DB",
            "schema": "SC",
            "client_version": "1.38.0",
        },
        "offline_configs": {"ignored": True},
        "spec": {
            "timestamp_field": "TS",
            "features": [],
            "udf": {
                "name": "transform",
                "output_columns": [{"name": "x", "type": "STRING"}],
            },
        },
    }


class TestPayloadDiffMatchesHashContract:
    """``_payload_diff`` must report exactly the fields ``_full_spec_hash`` uses."""

    def test_metadata_version_drift_is_reported(self) -> None:
        """``metadata.version`` contributes to the hash; the diagnostic must name it.

        Skipping the entire ``metadata`` dict (the previous walker) hid
        this drift and printed "payloads agree" while the hash test failed.
        """
        local = _hash_aligned_fixture()
        deployed = _hash_aligned_fixture()
        deployed["metadata"]["version"] = "v2"
        assert _full_spec_hash(local) != _full_spec_hash(deployed)
        diffs = _payload_diff(local, deployed)
        assert any("metadata.version" in line for line in diffs), diffs

    def test_runtime_stamped_keys_are_not_reported(self) -> None:
        """``target_lag_sec`` and default ``length`` are stripped before hashing.

        They must not appear in the diagnostic, or a healthy round-trip
        looks like drift during an incident.
        """
        local = _hash_aligned_fixture()
        deployed = _hash_aligned_fixture()
        deployed["metadata"]["client_version"] = "9.9.9"
        deployed["spec"]["target_lag_sec"] = 0
        deployed["spec"]["udf"]["output_columns"][0]["length"] = 16777216
        assert _full_spec_hash(local) == _full_spec_hash(deployed)
        assert _payload_diff(local, deployed) == []


if __name__ == "__main__":
    pytest_driver.main()
