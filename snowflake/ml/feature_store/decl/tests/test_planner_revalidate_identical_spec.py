"""Red tests for the planner re-validation defect (BUG_BASH §9 / §11 / §14).

A re-plan against an identical, just-deployed FV spec must return
``Status: planned`` with every row ``NO_CHANGE`` and zero
``COLUMN_ADDED`` warnings.  In the live BUG_BASH run today it instead
returns ``Status: validation_failed`` with::

    errors:   ["VERSION_CONFLICT USER_CLICK_STATS_DECL: version 'V1' is
                not greater than the currently deployed 'V1'."]
    warnings: ["...COLUMN_ADDED USER_CLICK_STATS_DECL.HAS_CONVERSION_24H ...",
               "...COLUMN_ADDED USER_CLICK_STATS_DECL.TOTAL_ENGAGEMENT_1H ..."]

Three orthogonal bugs cascade into that symptom — see
``plans/planner-revalidate-identical-spec-fix_*.plan.md`` for the
hypothesis tree.  Each test class below pins one rung of that cascade
so the fix's red→green progression is visible per-bug:

- :class:`TestColumnEvolutionNestedFeatures` — H1 (the column-evolution
  reader looks at ``applied.spec_payload['features']`` instead of the
  nested ``applied.spec_payload['spec']['features']`` that
  ``from_specification=True`` AppliedObjects actually carry).
- :class:`TestVersionConflictSemantics` — H3 (``_check_versions`` fires
  ``VERSION_CONFLICT`` on ``version <= applied_version``; for an
  identical spec where ``version == applied_version`` and content has
  not changed, the verdict must be ``NO_CHANGE``).
- :class:`TestReplanIdenticalSpecEndToEnd` — H2 (the round-trip
  contract: loading the BUG_BASH YAML, compiling it, treating the
  result as the just-deployed payload, and re-validating must produce
  zero non-``NO_CHANGE`` ops, zero errors, and no ``COLUMN_ADDED``
  warnings).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from snowflake.ml.feature_store.decl.enums import OpKind
from snowflake.ml.feature_store.decl.invariants import (
    _check_destructive,
    _check_fv_column_evolution,
    _check_versions,
    _full_spec_hash,
    model_to_dict,
    spec_key,
    structural_fingerprint_hash,
    validate_specs,
)
from snowflake.ml.feature_store.decl.loader import load_specs
from snowflake.ml.feature_store.decl.planner import generate_plan
from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec
from snowflake.ml.feature_store.decl.spec_models import FeatureView
from snowflake.ml.feature_store.decl.types import (
    AppliedObject,
    AppliedState,
    ObjectKind,
    PlanOptions,
    SpecBatch,
    ValidationResult,
)

# ---------------------------------------------------------------------------
# Shared fixtures — the BUG_BASH §5 USER_CLICK_STATS_DECL spec, byte-for-byte
# matching scripts/verify_bug_bash.sh.
# ---------------------------------------------------------------------------


_BUG_BASH_DB = "JKEW_DB"
_BUG_BASH_SCHEMA = "JKEW_SCHEMA"
_BUG_BASH_FV_NAME = "USER_CLICK_STATS_DECL"
_BUG_BASH_FV_VERSION = "V1"


def _bug_bash_fv_yaml() -> str:
    """The verbatim BUG_BASH §5 FV YAML (matches scripts/verify_bug_bash.sh)."""
    return (
        "kind: StreamingFeatureView\n"
        f"name: {_BUG_BASH_FV_NAME}\n"
        f"version: {_BUG_BASH_FV_VERSION}\n"
        "online: true\n"
        "entities:\n"
        "  - USER_ID\n"
        "timestamp_col: TIMESTAMP\n"
        "feature_granularity_sec: 300\n"
        "feature_aggregation_method: tiles\n"
        "sources:\n"
        "  - name: CLICKSTREAM_EVENTS\n"
        "    columns:\n"
        "      - name: USER_ID\n"
        "        type: StringType\n"
        "      - name: SESSION_ID\n"
        "        type: StringType\n"
        "      - name: PAGE_URL\n"
        "        type: StringType\n"
        "      - name: EVENT_TYPE\n"
        "        type: StringType\n"
        "      - name: TIMESTAMP\n"
        "        type: TimestampType\n"
        "      - name: TIME_ON_PAGE_SECONDS\n"
        "        type: DoubleType\n"
        "    source_type: Stream\n"
        "features:\n"
        "  - output_column:\n"
        "      name: TOTAL_ENGAGEMENT_1H\n"
        "      type: DoubleType\n"
        "    window_sec: 3600\n"
        "    function: sum\n"
        "    source_column:\n"
        "      name: ENGAGEMENT_SCORE\n"
        "      type: DoubleType\n"
        "  - output_column:\n"
        "      name: HAS_CONVERSION_24H\n"
        "      type: BooleanType\n"
        "    window_sec: 86400\n"
        "    function: max\n"
        "    source_column:\n"
        "      name: IS_CONVERSION\n"
        "      type: BooleanType\n"
        "udf:\n"
        f"  name: compute_engagement_metrics\n"
        "  engine: pandas\n"
        "  output_columns:\n"
        "    - name: USER_ID\n"
        "      type: StringType\n"
        "    - name: TIMESTAMP\n"
        "      type: TimestampType\n"
        "    - name: IS_CONVERSION\n"
        "      type: BooleanType\n"
        "    - name: ENGAGEMENT_SCORE\n"
        "      type: DoubleType\n"
        f"  file: {_BUG_BASH_FV_NAME}.py\n"
    )


def _bug_bash_entity_yaml() -> str:
    """The verbatim BUG_BASH §5 USER_ID entity YAML."""
    return (
        "kind: Entity\n"
        "name: USER_ID\n"
        'description: "Bug-bash user identifier."\n'
        "join_keys:\n"
        "  - name: USER_ID\n"
        "    type: StringType\n"
    )


def _bug_bash_source_yaml() -> str:
    """The verbatim BUG_BASH §5 CLICKSTREAM_EVENTS source YAML."""
    return (
        "kind: StreamingSource\n"
        "name: CLICKSTREAM_EVENTS\n"
        "type: REST\n"
        "columns:\n"
        "  - name: USER_ID\n"
        "    type: StringType\n"
        "  - name: SESSION_ID\n"
        "    type: StringType\n"
        "  - name: PAGE_URL\n"
        "    type: StringType\n"
        "  - name: EVENT_TYPE\n"
        "    type: StringType\n"
        "  - name: TIMESTAMP\n"
        "    type: TimestampType\n"
        "  - name: TIME_ON_PAGE_SECONDS\n"
        "    type: DoubleType\n"
    )


def _bug_bash_udf_py() -> str:
    """The verbatim BUG_BASH §5 UDF body (matches scripts/verify_bug_bash.sh)."""
    return (
        "def compute_engagement_metrics(clickstream: pd.DataFrame) -> pd.DataFrame:\n"
        '    """Compute engagement metrics from click-stream events."""\n'
        "    df = clickstream.copy()\n"
        "\n"
        '    conversion_events = ["purchase", "signup", "subscribe"]\n'
        '    df["IS_CONVERSION"] = df["EVENT_TYPE"].isin(conversion_events)\n'
        "\n"
        "    weights = {\n"
        '        "page_view": 1.0,\n'
        '        "click": 2.0,\n'
        '        "form_submit": 5.0,\n'
        '        "purchase": 10.0,\n'
        '        "signup": 8.0,\n'
        "    }\n"
        '    df["ENGAGEMENT_SCORE"] = df["EVENT_TYPE"].map(weights).fillna(1.0)\n'
        "\n"
        '    return df[["USER_ID", "TIMESTAMP", "IS_CONVERSION", "ENGAGEMENT_SCORE"]]\n'
    )


def _write_bug_bash_tree(tmp_path: Path) -> Path:
    """Lay out the BUG_BASH §5 working directory under *tmp_path*.

    Mirrors the tree ``scripts/verify_bug_bash.sh`` step 5 builds:
    ``$DECL_DIR/{entities/USER_ID.yaml, datasources/CLICKSTREAM_EVENTS.yaml,
    feature_views/USER_CLICK_STATS_DECL.{yaml,py}}``.

    Args:
        tmp_path: pytest tmp dir.

    Returns:
        Path to the ``$DECL_DIR`` root (the tree's parent so the loader
        can walk ``<root>/...`` recursively).
    """
    decl_dir = tmp_path / f"{_BUG_BASH_DB}.{_BUG_BASH_SCHEMA}"
    entity_dir = decl_dir / "entities"
    source_dir = decl_dir / "datasources"
    fv_dir = decl_dir / "feature_views"
    entity_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    fv_dir.mkdir(parents=True)
    (entity_dir / "USER_ID.yaml").write_text(_bug_bash_entity_yaml())
    (source_dir / "CLICKSTREAM_EVENTS.yaml").write_text(_bug_bash_source_yaml())
    (fv_dir / f"{_BUG_BASH_FV_NAME}.yaml").write_text(_bug_bash_fv_yaml())
    (fv_dir / f"{_BUG_BASH_FV_NAME}.py").write_text(_bug_bash_udf_py())
    return decl_dir


def _bug_bash_fv_model() -> FeatureView:
    """Build the BUG_BASH FV directly from a model_validate dict.

    The returned model matches the shape produced by the YAML loader
    *after* duration normalization and UDF inlining — it is the
    canonical input to ``compile_to_spec``.

    Returns:
        FeatureView with all bug-bash fields populated.
    """
    return FeatureView.model_validate(
        {
            "kind": "StreamingFeatureView",
            "name": _BUG_BASH_FV_NAME,
            "version": _BUG_BASH_FV_VERSION,
            "database": _BUG_BASH_DB,
            "schema_": _BUG_BASH_SCHEMA,
            "online": True,
            "entities": ["USER_ID"],
            "timestamp_col": "TIMESTAMP",
            "feature_granularity_sec": 300,
            "feature_aggregation_method": "tiles",
            "sources": [
                {
                    "name": "CLICKSTREAM_EVENTS",
                    "source_type": "Stream",
                    "columns": [
                        {"name": "USER_ID", "type": "StringType"},
                        {"name": "SESSION_ID", "type": "StringType"},
                        {"name": "PAGE_URL", "type": "StringType"},
                        {"name": "EVENT_TYPE", "type": "StringType"},
                        {"name": "TIMESTAMP", "type": "TimestampType"},
                        {"name": "TIME_ON_PAGE_SECONDS", "type": "DoubleType"},
                    ],
                }
            ],
            "features": [
                {
                    "output_column": {"name": "TOTAL_ENGAGEMENT_1H", "type": "DoubleType"},
                    "window_sec": 3600,
                    "function": "sum",
                    "source_column": {"name": "ENGAGEMENT_SCORE", "type": "DoubleType"},
                },
                {
                    "output_column": {"name": "HAS_CONVERSION_24H", "type": "BooleanType"},
                    "window_sec": 86400,
                    "function": "max",
                    "source_column": {"name": "IS_CONVERSION", "type": "BooleanType"},
                },
            ],
            "udf": {
                "name": "compute_engagement_metrics",
                "engine": "pandas",
                "function_definition": _bug_bash_udf_py(),
                "output_columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "TIMESTAMP", "type": "TimestampType"},
                    {"name": "IS_CONVERSION", "type": "BooleanType"},
                    {"name": "ENGAGEMENT_SCORE", "type": "DoubleType"},
                ],
            },
        }
    )


def _build_applied_state_from_compiled(
    fv_model: FeatureView,
    *,
    extra_metadata: dict[str, Any] | None = None,
    extra_top_level: dict[str, Any] | None = None,
    use_hash: str | None = None,
) -> tuple[str, AppliedState]:
    """Build an :class:`AppliedState` mirroring what ``apply`` just deployed.

    The deployed FV's ``DESCRIBE ... TYPE = SPECIFICATION`` payload is
    synthesized via :func:`compile_to_spec` (the same compiler the apply
    path used to produce the ``CREATE ONLINE FEATURE TABLE ... FROM
    SPECIFICATION $$ ... $$`` JSON), then optionally enriched with
    metadata Snowflake stamps in (e.g. ``oft_id``).  This is the
    canonical "just-deployed" baseline for testing re-plan idempotency.

    To reproduce the live BUG_BASH §9 cascade locally, callers can pass
    ``use_hash`` to force a divergent ``content_hash`` (mimicking what
    happens when DESCRIBE returns a payload that ``_full_spec_hash``
    does not symmetrize byte-for-byte against ``compile_to_spec``
    output).  When ``use_hash`` is ``None`` the natural hash of the
    enriched payload is used (the "happy path" — tests that should
    stay green even after the fix).

    Args:
        fv_model: The local :class:`FeatureView` model to compile.
        extra_metadata: Optional extra keys to merge into
            ``spec_payload.metadata``.  Use this to simulate Snowflake
            adding fields like ``oft_id`` post-deploy.
        extra_top_level: Optional extra keys to merge into the top-level
            spec_payload.  Use to simulate Snowflake adding fields the
            local compiler does not emit (this is what the live
            BUG_BASH cascade exposes).
        use_hash: Optional content_hash override.  When set, the
            applied state's ``content_hash`` is forced to this value
            regardless of the actual payload — simulating any cause of
            hash divergence between local compile and live DESCRIBE.

    Returns:
        Tuple of ``(spec_key, applied_state)`` so the caller can look
        the FV up in the state's ``objects`` map.
    """
    local_dict = model_to_dict(fv_model)
    deployed_payload = compile_to_spec(local_dict, _BUG_BASH_DB, _BUG_BASH_SCHEMA)
    if extra_metadata:
        deployed_payload.setdefault("metadata", {}).update(extra_metadata)
    if extra_top_level:
        deployed_payload.update(extra_top_level)

    key = spec_key(local_dict, database=_BUG_BASH_DB, schema=_BUG_BASH_SCHEMA)
    applied = AppliedObject(
        key=key,
        kind=local_dict["kind"],
        name=local_dict["name"],
        version=local_dict["version"],
        content_hash=use_hash if use_hash is not None else _full_spec_hash(deployed_payload),
        spec_payload=deployed_payload,
        from_specification=True,
    )
    return key, AppliedState(objects={key: applied})


def _bug_bash_full_applied_state(
    *,
    use_fv_hash: str | None = None,
) -> AppliedState:
    """Build a full :class:`AppliedState` for the BUG_BASH flow.

    Includes one ``AppliedObject`` per kind (Entity, Datasource,
    StreamingFeatureView) so the validator's cross-spec dependency
    checks (``_check_dependencies``, ``_check_source_evolution``)
    resolve against the deployed state instead of treating every
    referenced object as missing.

    The hash strategy mirrors what ``state.fetch_applied_state``
    actually produces in the live env:

    - Entity: ``content_hash = structural_fingerprint_hash(spec_payload)``,
      ``from_specification=False``.
    - Datasource: ``content_hash = structural_fingerprint_hash(spec_payload)``,
      ``from_specification=True`` (recovered from FV ``spec.sources[]``).
    - StreamingFeatureView: ``content_hash = _full_spec_hash(spec_payload)``,
      ``from_specification=True`` (recovered from
      ``DESCRIBE TYPE = SPECIFICATION``).

    Args:
        use_fv_hash: Optional override for the FV's ``content_hash``.
            Pass ``"deadbeef" * 8`` (or any digest that cannot match a
            real one) to simulate the live BUG_BASH §9 hash divergence
            and force ``_check_idempotency`` to MISS.

    Returns:
        AppliedState with all three bug-bash objects populated.
    """
    objects: dict[str, AppliedObject] = {}

    # Entity ----------------------------------------------------------------
    # ``spec_payload`` mirrors what :func:`state._build_entity_object`
    # produces for a deployed tag — including the ``description`` plumbed
    # from the tag ``COMMENT``.  The Entity branch of
    # :func:`_structural_fingerprint` folds description into the hash, so
    # the applied side must carry the same value the local YAML's
    # ``description:`` produces or the round-trip emits a phantom
    # UPDATE_ENTITY.
    entity_payload = {
        "kind": ObjectKind.ENTITY,
        "name": "USER_ID",
        "database": _BUG_BASH_DB,
        "schema": _BUG_BASH_SCHEMA,
        "description": "Bug-bash user identifier.",
        "join_keys": [{"name": "USER_ID", "type": "StringType"}],
    }
    entity_key = f"{ObjectKind.ENTITY}:{_BUG_BASH_DB}.{_BUG_BASH_SCHEMA}:USER_ID"
    objects[entity_key] = AppliedObject(
        key=entity_key,
        kind=ObjectKind.ENTITY,
        name="USER_ID",
        content_hash=structural_fingerprint_hash(entity_payload),
        spec_payload=entity_payload,
        from_specification=False,
        details={"join_keys": ["USER_ID"], "comment": "Bug-bash user identifier."},
    )

    # Datasource ------------------------------------------------------------
    source_columns = [
        {"name": "USER_ID", "type": "StringType"},
        {"name": "SESSION_ID", "type": "StringType"},
        {"name": "PAGE_URL", "type": "StringType"},
        {"name": "EVENT_TYPE", "type": "StringType"},
        {"name": "TIMESTAMP", "type": "TimestampType"},
        {"name": "TIME_ON_PAGE_SECONDS", "type": "DoubleType"},
    ]
    source_payload = {
        "kind": ObjectKind.DATASOURCE,
        "name": "CLICKSTREAM_EVENTS",
        "database": _BUG_BASH_DB,
        "schema": _BUG_BASH_SCHEMA,
        "source_type": "Stream",
        "columns": source_columns,
    }
    source_key = f"{ObjectKind.DATASOURCE}:{_BUG_BASH_DB}.{_BUG_BASH_SCHEMA}:" "CLICKSTREAM_EVENTS"
    objects[source_key] = AppliedObject(
        key=source_key,
        kind=ObjectKind.DATASOURCE,
        name="CLICKSTREAM_EVENTS",
        content_hash=structural_fingerprint_hash(source_payload),
        spec_payload=source_payload,
        from_specification=True,
        details={"source_type": "Stream", "column_count": len(source_columns)},
    )

    # FeatureView -----------------------------------------------------------
    fv_key, fv_state = _build_applied_state_from_compiled(
        _bug_bash_fv_model(),
        use_hash=use_fv_hash,
    )
    objects[fv_key] = fv_state.objects[fv_key]

    return AppliedState(objects=objects)


# ---------------------------------------------------------------------------
# R1 — Column evolution nested-features lookup (H1)
# ---------------------------------------------------------------------------


class TestColumnEvolutionNestedFeatures:
    """``_check_fv_column_evolution`` must read ``features`` from the right level.

    When ``applied.from_specification=True`` the ``spec_payload`` is the
    full ``DESCRIBE TYPE = SPECIFICATION`` payload, so ``features`` lives
    at ``spec_payload['spec']['features']``.  The current implementation
    reads ``spec_payload.get('features', [])`` which yields ``[]`` —
    every current feature column then looks "added" and the validator
    emits a spurious ``COLUMN_ADDED`` warning per output column.
    """

    def _deployed_fv_payload(self) -> dict[str, Any]:
        """Return a fully-nested SPECIFICATION-shape payload.

        Returns:
            Dict mirroring the shape of ``DESCRIBE ... TYPE =
            SPECIFICATION`` for the BUG_BASH FV (features nested under
            ``spec``).
        """
        return {
            "kind": "StreamingFeatureView",
            "metadata": {
                "database": _BUG_BASH_DB,
                "schema": _BUG_BASH_SCHEMA,
                "name": _BUG_BASH_FV_NAME,
                "version": _BUG_BASH_FV_VERSION,
            },
            "spec": {
                "features": [
                    {
                        "output_column": {"name": "TOTAL_ENGAGEMENT_1H", "type": "DoubleType"},
                        "window_sec": 3600,
                        "function": "sum",
                        "source_column": {"name": "ENGAGEMENT_SCORE", "type": "DoubleType"},
                    },
                    {
                        "output_column": {"name": "HAS_CONVERSION_24H", "type": "BooleanType"},
                        "window_sec": 86400,
                        "function": "max",
                        "source_column": {"name": "IS_CONVERSION", "type": "BooleanType"},
                    },
                ],
            },
        }

    def _local_fv_spec(self) -> dict[str, Any]:
        """Return a top-level-features local spec dict matching the deployed FV.

        Returns:
            Dict with ``features`` at the top level (the shape produced
            by ``model_to_dict`` on a loaded ``FeatureView``).
        """
        return model_to_dict(_bug_bash_fv_model())

    def test_no_column_added_when_local_features_match_nested_deployed(self) -> None:
        """Identical features must not emit COLUMN_ADDED, even when the
        deployed payload nests them under ``spec``.

        Currently FAILS: emits COLUMN_ADDED for every output column.
        """
        deployed = self._deployed_fv_payload()
        applied = AppliedObject(
            key="StreamingFeatureView:JKEW_DB.JKEW_SCHEMA:USER_CLICK_STATS_DECL",
            kind="StreamingFeatureView",
            name=_BUG_BASH_FV_NAME,
            version=_BUG_BASH_FV_VERSION,
            spec_payload=deployed,
            from_specification=True,
        )
        local = self._local_fv_spec()

        results: list[ValidationResult] = []
        _check_fv_column_evolution(local, applied, _BUG_BASH_FV_NAME, results)

        column_added = [r for r in results if r.code == "COLUMN_ADDED"]
        assert column_added == [], (
            "expected zero COLUMN_ADDED warnings for an unchanged FV; got " f"{[r.message for r in column_added]!r}"
        )

    def test_column_added_still_fires_when_a_feature_is_genuinely_new(self) -> None:
        """Regression guard: legitimately new columns must still emit
        COLUMN_ADDED — the fix must not silence the rule entirely.
        """
        deployed = self._deployed_fv_payload()
        # Drop one of the deployed features so the local spec genuinely
        # "adds" it on the next plan.
        deployed["spec"]["features"] = deployed["spec"]["features"][:1]
        applied = AppliedObject(
            key="StreamingFeatureView:JKEW_DB.JKEW_SCHEMA:USER_CLICK_STATS_DECL",
            kind="StreamingFeatureView",
            name=_BUG_BASH_FV_NAME,
            version=_BUG_BASH_FV_VERSION,
            spec_payload=deployed,
            from_specification=True,
        )
        local = self._local_fv_spec()

        results: list[ValidationResult] = []
        _check_fv_column_evolution(local, applied, _BUG_BASH_FV_NAME, results)

        assert any(
            "HAS_CONVERSION_24H" in r.message for r in results if r.code == "COLUMN_ADDED"
        ), f"expected COLUMN_ADDED for HAS_CONVERSION_24H; got messages={[r.message for r in results]!r}"


# ---------------------------------------------------------------------------
# R1b — Destructive-change nested-features lookup (H1 — extended surface)
# ---------------------------------------------------------------------------


class TestDestructiveCheckNestedFeatures:
    """``_check_destructive`` reads ``features`` with the same nesting contract.

    When ``applied.from_specification=True`` the deployed features live
    at ``applied.spec_payload['spec']['features']``; without the same
    branching that :class:`TestColumnEvolutionNestedFeatures` pins on
    ``_check_fv_column_evolution``, ``_check_destructive`` silently
    sees zero deployed features and never fires the
    ``DESTRUCTIVE_CHANGE`` error that gates plain ``snow feature apply``
    against destructive function/aggregation edits.  Without this fix
    a UDF source change on a SPECIFICATION-backed FV slips past the
    plain ``snow feature apply`` flow and runs without the
    ``--allow-recreate`` operator opt-in.
    """

    def _deployed_fv_payload(self) -> dict[str, Any]:
        """Return a fully-nested SPECIFICATION-shape payload.

        Returns:
            Dict mirroring the shape of ``DESCRIBE ... TYPE =
            SPECIFICATION`` for the BUG_BASH FV.
        """
        return {
            "kind": "StreamingFeatureView",
            "metadata": {
                "database": _BUG_BASH_DB,
                "schema": _BUG_BASH_SCHEMA,
                "name": _BUG_BASH_FV_NAME,
                "version": _BUG_BASH_FV_VERSION,
            },
            "spec": {
                "features": [
                    {
                        "output_column": {"name": "TOTAL_ENGAGEMENT_1H", "type": "DoubleType"},
                        "window_sec": 3600,
                        "function": "sum",
                        "source_column": {"name": "ENGAGEMENT_SCORE", "type": "DoubleType"},
                    },
                ],
            },
        }

    def test_destructive_change_fires_when_function_edits_for_from_specification(self) -> None:
        """Editing a feature's ``function`` on a SPECIFICATION-backed
        AppliedObject must produce ``DESTRUCTIVE_CHANGE``.

        Currently FAILS: ``_check_destructive`` reads features at the
        wrong nesting level and never fires for ``from_specification=True``.
        """
        deployed = self._deployed_fv_payload()
        applied = AppliedObject(
            key="StreamingFeatureView:JKEW_DB.JKEW_SCHEMA:USER_CLICK_STATS_DECL",
            kind="StreamingFeatureView",
            name=_BUG_BASH_FV_NAME,
            version=_BUG_BASH_FV_VERSION,
            spec_payload=deployed,
            from_specification=True,
        )
        local = {
            "kind": "StreamingFeatureView",
            "name": _BUG_BASH_FV_NAME,
            "version": _BUG_BASH_FV_VERSION,
            "features": [
                {
                    "output_column": {"name": "TOTAL_ENGAGEMENT_1H", "type": "DoubleType"},
                    "window_sec": 3600,
                    "function": "max",  # changed from "sum" -> "max"
                    "source_column": {"name": "ENGAGEMENT_SCORE", "type": "DoubleType"},
                },
            ],
        }

        results = _check_destructive(local, applied)

        codes = [r.code for r in results]
        assert "DESTRUCTIVE_CHANGE" in codes, (
            "DESTRUCTIVE_CHANGE must fire when an output column's "
            f"function changes on a SPECIFICATION-backed FV; got "
            f"results={[(r.code, r.message) for r in results]!r}"
        )

    def test_no_destructive_change_when_function_unchanged(self) -> None:
        """Regression guard: identical functions must not fire DESTRUCTIVE_CHANGE."""
        deployed = self._deployed_fv_payload()
        applied = AppliedObject(
            key="StreamingFeatureView:JKEW_DB.JKEW_SCHEMA:USER_CLICK_STATS_DECL",
            kind="StreamingFeatureView",
            name=_BUG_BASH_FV_NAME,
            version=_BUG_BASH_FV_VERSION,
            spec_payload=deployed,
            from_specification=True,
        )
        local = {
            "kind": "StreamingFeatureView",
            "name": _BUG_BASH_FV_NAME,
            "version": _BUG_BASH_FV_VERSION,
            "features": [
                {
                    "output_column": {"name": "TOTAL_ENGAGEMENT_1H", "type": "DoubleType"},
                    "window_sec": 3600,
                    "function": "sum",  # unchanged
                    "source_column": {"name": "ENGAGEMENT_SCORE", "type": "DoubleType"},
                },
            ],
        }

        results = _check_destructive(local, applied)

        codes = [r.code for r in results]
        assert (
            "DESTRUCTIVE_CHANGE" not in codes
        ), f"DESTRUCTIVE_CHANGE must not fire for an unchanged function; got codes={codes!r}"


# ---------------------------------------------------------------------------
# R2 — VERSION_CONFLICT semantics on identical spec (H3)
# ---------------------------------------------------------------------------


class TestVersionConflictSemantics:
    """``_check_versions`` must not fire VERSION_CONFLICT on an identical spec.

    Today ``_check_versions`` fires whenever ``version <= applied_version``,
    so a re-plan against the just-deployed ``V1`` lands on
    ``V1 <= V1`` → ``VERSION_CONFLICT``.  The correct verdict for an
    *identical* spec at the same version is ``NO_CHANGE`` (handled by
    ``_check_idempotency``); ``VERSION_CONFLICT`` should only fire when
    the local version is strictly *less than* the deployed version, OR
    when the spec has changed but the version didn't bump (i.e. a real
    user error).
    """

    def test_validate_specs_identical_spec_does_not_emit_version_conflict(self) -> None:
        """End-to-end: ``validate_specs`` on a freshly-deployed identical
        spec must not include any VERSION_CONFLICT result.

        Uses a divergent ``content_hash`` to force ``_check_idempotency``
        to MISS — that's the live BUG_BASH §9 condition where
        VERSION_CONFLICT erroneously fires on equal versions.  The fix
        must not emit VERSION_CONFLICT for ``version == applied_version``
        because the verdict for an identical spec is ``NO_CHANGE``,
        not ``VERSION_CONFLICT``.
        """
        fv = _bug_bash_fv_model()
        _, applied_state = _build_applied_state_from_compiled(
            fv,
            use_hash="deadbeef" * 8,
        )

        results = validate_specs(
            SpecBatch(specs=[fv], source_files=[]),
            applied_state,
            target_database=_BUG_BASH_DB,
            target_schema=_BUG_BASH_SCHEMA,
        )

        version_conflicts = [r for r in results if r.code == "VERSION_CONFLICT"]
        assert version_conflicts == [], (
            "VERSION_CONFLICT must not fire on a re-plan of an identical "
            f"just-deployed spec; got {[r.message for r in version_conflicts]!r}"
        )

    def test_check_versions_does_not_fire_when_version_equals_applied(self) -> None:
        """Unit test on ``_check_versions``: ``version == applied_version``
        alone must not produce VERSION_CONFLICT.

        The ``<=`` semantics in the current code are a bug — equal
        versions are *valid* for an identical spec (handled upstream by
        ``_check_idempotency`` returning ``NO_CHANGE``).  Only a
        *strictly lower* local version warrants VERSION_CONFLICT.
        """
        spec = {
            "kind": "StreamingFeatureView",
            "name": _BUG_BASH_FV_NAME,
            "version": _BUG_BASH_FV_VERSION,
        }
        applied = AppliedObject(
            key="StreamingFeatureView:JKEW_DB.JKEW_SCHEMA:USER_CLICK_STATS_DECL",
            kind="StreamingFeatureView",
            name=_BUG_BASH_FV_NAME,
            version=_BUG_BASH_FV_VERSION,
        )

        results = _check_versions(spec, applied, dev_mode=False)

        codes = [r.code for r in results]
        assert "VERSION_CONFLICT" not in codes, (
            "VERSION_CONFLICT must not fire when local and applied versions " f"are equal; got codes={codes!r}"
        )

    def test_check_versions_still_fires_when_version_genuinely_lower(self) -> None:
        """Regression guard: a strictly-lower local version must still
        produce VERSION_CONFLICT — the fix must not eliminate the rule.
        """
        spec = {
            "kind": "StreamingFeatureView",
            "name": _BUG_BASH_FV_NAME,
            "version": "V1",  # local is V1
        }
        applied = AppliedObject(
            key="StreamingFeatureView:JKEW_DB.JKEW_SCHEMA:USER_CLICK_STATS_DECL",
            kind="StreamingFeatureView",
            name=_BUG_BASH_FV_NAME,
            version="V2",  # deployed is V2 (higher)
        )

        results = _check_versions(spec, applied, dev_mode=False)

        codes = [r.code for r in results]
        assert "VERSION_CONFLICT" in codes, (
            "VERSION_CONFLICT must still fire when local version is "
            f"strictly lower than applied; got codes={codes!r}"
        )


# ---------------------------------------------------------------------------
# R3 — End-to-end re-plan idempotency contract (H2)
# ---------------------------------------------------------------------------


class TestReplanIdenticalSpecEndToEnd:
    """End-to-end contract: re-plan against an identical, just-deployed
    spec must produce zero ERROR results and zero ``COLUMN_ADDED``
    warnings — even when the deployed payload's ``content_hash``
    diverges from the local compile (which is exactly what happens in
    the live ``snow feature plan`` cascade in BUG_BASH §9 / §11 / §14).

    This pins the BUG_BASH §9 acceptance criterion — see
    ``plans/planner_revalidate_identical_spec.plan.md`` for the live-env
    repro and ``scripts/verify_bug_bash.sh`` for the bash-level
    invocation that must go green after the fix lands.

    The tests use ``use_hash="bogus_does_not_match..."`` to force the
    idempotency check to MISS — that's the in-vitro reproduction of the
    live divergence and exposes the H1/H3 defects in the validator's
    fall-through path (without needing a captured live DESCRIBE
    fixture).
    """

    # 64-char placeholder hash that cannot collide with any real digest.
    _DIVERGENT_HASH = "deadbeef" * 8

    def test_revalidate_identical_spec_produces_no_errors(self, tmp_path: Path) -> None:
        """``validate_specs`` on the BUG_BASH YAML against a
        hash-divergent just-deployed payload produces zero ERRORRs.

        Reproduces the live BUG_BASH §9 cascade: when the live DESCRIBE
        payload's hash does not match the local compile hash (for any
        reason — extra fields, normalization, etc.), the validator
        falls through to ``_check_versions`` which fires ``VERSION_CONFLICT``
        on equal versions.  The fix must not emit ERRORRs in this case
        because the spec content is genuinely unchanged at the column /
        feature level.

        Args:
            tmp_path: pytest tmp dir used to stage the bug-bash YAML tree.
        """
        decl_dir = _write_bug_bash_tree(tmp_path)
        batch = load_specs([f"{decl_dir}/..."])

        applied_state = _bug_bash_full_applied_state(
            use_fv_hash=self._DIVERGENT_HASH,
        )

        results = validate_specs(
            batch,
            applied_state,
            target_database=_BUG_BASH_DB,
            target_schema=_BUG_BASH_SCHEMA,
        )

        errors = [r for r in results if r.severity == "ERROR"]
        assert errors == [], (
            "Re-plan of an identical just-deployed FV must produce zero "
            f"validation ERRORRs; got {[(r.code, r.message) for r in errors]!r}"
        )

    def test_revalidate_identical_spec_produces_no_column_added_warnings(self, tmp_path: Path) -> None:
        """``validate_specs`` must not emit COLUMN_ADDED for an unchanged
        FV — even when ``_check_idempotency`` misses.

        Reproduces the live BUG_BASH §9 cascade: with idempotency
        missing, ``_check_fv_column_evolution`` runs and reads
        ``applied.spec_payload.get('features', [])``, which returns
        ``[]`` for a ``from_specification=True`` payload (features are
        nested at ``spec.features``), so every current feature looks
        "added".  The fix must read the nested location.

        Args:
            tmp_path: pytest tmp dir used to stage the bug-bash YAML tree.
        """
        decl_dir = _write_bug_bash_tree(tmp_path)
        batch = load_specs([f"{decl_dir}/..."])

        applied_state = _bug_bash_full_applied_state(
            use_fv_hash=self._DIVERGENT_HASH,
        )

        results = validate_specs(
            batch,
            applied_state,
            target_database=_BUG_BASH_DB,
            target_schema=_BUG_BASH_SCHEMA,
        )

        column_added = [r for r in results if r.code == "COLUMN_ADDED"]
        assert column_added == [], (
            "Re-plan of an identical just-deployed FV must produce zero "
            f"COLUMN_ADDED warnings; got {[r.message for r in column_added]!r}"
        )

    def test_replan_identical_spec_is_no_change_when_hashes_match(self, tmp_path: Path) -> None:
        """Sanity baseline: when hashes match, the planner emits only
        NO_CHANGE ops.  This test must stay green (it is the working
        case) and serves as the regression guard for any fix that
        accidentally over-aggressively flags unchanged specs.

        Args:
            tmp_path: pytest tmp dir used to stage the bug-bash YAML tree.
        """
        decl_dir = _write_bug_bash_tree(tmp_path)
        batch = load_specs([f"{decl_dir}/..."])

        applied_state = _bug_bash_full_applied_state()

        plan = generate_plan(
            batch,
            applied_state,
            PlanOptions(),
            database=_BUG_BASH_DB,
            schema=_BUG_BASH_SCHEMA,
        )

        non_no_change = [op for op in plan.ops if op.kind != OpKind.NO_CHANGE]
        assert non_no_change == [], (
            "Re-plan of an identical just-deployed FV (hashes matching) "
            f"must produce only NO_CHANGE ops; got "
            f"{[(op.kind.value, op.name) for op in non_no_change]!r}"
        )


# ---------------------------------------------------------------------------
# R3 supplemental — golden-spec round-trip pin for USER_CLICK_STATS_DECL
# ---------------------------------------------------------------------------


_GOLDEN_DIR = Path(__file__).parent / "golden_specs"
_BUG_BASH_GOLDEN = _GOLDEN_DIR / "USER_CLICK_STATS_DECL.json"


@pytest.mark.skipif(
    not _BUG_BASH_GOLDEN.exists(),
    reason="Golden spec USER_CLICK_STATS_DECL.json not yet captured.",
)
class TestBugBashGoldenSpecRoundTrip:
    """Pin the BUG_BASH §5 FV against a captured DESCRIBE-time payload.

    The golden spec is constructed to match what
    :func:`compile_to_spec` produces for the bug-bash YAML — i.e. the
    canonical "what we just sent to Snowflake" baseline.  If a future
    change to ``compile_to_spec`` or ``_full_spec_hash`` perturbs the
    round-trip, this test fires before the live BUG_BASH script does.
    """

    def test_golden_payload_hash_matches_local_compile_hash(self) -> None:
        """``_full_spec_hash(golden) == _full_spec_hash(compile_to_spec(local))``."""
        golden = json.loads(_BUG_BASH_GOLDEN.read_text())
        local_dict = model_to_dict(_bug_bash_fv_model())
        compiled = compile_to_spec(local_dict, _BUG_BASH_DB, _BUG_BASH_SCHEMA)

        assert _full_spec_hash(golden) == _full_spec_hash(compiled), (
            "Golden USER_CLICK_STATS_DECL hash diverges from local "
            "compile_to_spec output — live re-plan will hit "
            "VERSION_CONFLICT.  Re-capture the golden or fix the "
            "compile_to_spec round-trip."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
