"""Round-trip pins for StreamingFeatureView array ``element_type`` and secondary keys.

These tests reproduce the two proven root causes behind the persistent
``RECREATE_FV`` drift for streaming FVs whose features materialise an array
output column (``last_distinct_n`` / ``last_n`` / secondary-key aggregations):

* **Cause 1 — dropped ``element_type``.** The authoring model
  :class:`~snowflake.ml.feature_store.decl.spec_models.FSColumn` must retain
  the array element type (``ArrayType`` + ``element_type: DoubleType``) through
  the ``model_dump(exclude_none=True)`` compile path so the local-compile
  ``_full_spec_hash`` matches the applied ``DESCRIBE ... TYPE = SPECIFICATION``
  payload (which carries the element type).
* **Cause 2 — secondary-key field-name asymmetry.** The local-compile path
  emits ``spec.aggregation_secondary_keys`` while the applied wire spec uses
  ``spec.ordered_secondary_key_column_names``; the hash normalisation must
  canonicalise the two so an unedited round-trip re-plans as ``NO_CHANGE``, and
  the exporter must emit ``aggregation_secondary_keys`` from the wire key.

The load-bearing assertions go through the Pydantic model
(``FeatureView.model_validate`` -> ``model_to_dict`` -> ``compile_to_spec``)
because the drop happens at ``model_dump`` time — the raw-dict compile path in
``test_object_hashing.py`` bypasses the model and cannot observe it.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from snowflake.ml.feature_store.decl.exporter import _build_full_fidelity_fv
from snowflake.ml.feature_store.decl.invariants import (
    _full_spec_hash,
    compute_local_spec_hash,
    model_to_dict,
)
from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec
from snowflake.ml.feature_store.decl.spec_models import FeatureView

_DB = "JKEW_DB"
_SCHEMA = "JKEW_SCHEMA"


def _array_element_type_authoring() -> dict[str, Any]:
    """Authoring dict for a streaming FV with a ``last_distinct_n`` array feature."""
    return {
        "kind": "StreamingFeatureView",
        "name": "USER_ENGAGEMENT_ARR",
        "database": _DB,
        "schema_": _SCHEMA,
        "version": "V1",
        "online": True,
        "entities": ["USER_ID"],
        "timestamp_col": "TIMESTAMP",
        "feature_granularity_sec": 300,
        "feature_aggregation_method": "continuous",
        "sources": [
            {
                "name": "CLICKSTREAM_EVENTS",
                "source_type": "Stream",
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "ENGAGEMENT_SCORE", "type": "DoubleType"},
                    {"name": "TIMESTAMP", "type": "TimestampType"},
                ],
            }
        ],
        "features": [
            {
                "source_column": {"name": "ENGAGEMENT_SCORE", "type": "DoubleType"},
                "output_column": {
                    "name": "LAST_2_ENGAGEMENT_1H",
                    "type": "ArrayType",
                    "element_type": "DoubleType",
                },
                "function": "last_distinct_n",
                "window_sec": 3600,
                "function_params": {"n": 2},
            }
        ],
    }


def _array_element_type_applied() -> dict[str, Any]:
    """DESCRIBE-shape payload for :func:`_array_element_type_authoring`."""
    return {
        "kind": "StreamingFeatureView",
        "metadata": {
            "database": _DB,
            "schema": _SCHEMA,
            "name": "USER_ENGAGEMENT_ARR",
            "version": "V1",
            "spec_format_version": "1",
            "internal_data_version": "1",
            "client_version": "1.38.0",
        },
        "offline_configs": [],
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "sources": [
                {
                    "name": "CLICKSTREAM_EVENTS",
                    "source_type": "Stream",
                    "columns": [
                        {"name": "USER_ID", "type": "StringType"},
                        {"name": "ENGAGEMENT_SCORE", "type": "DoubleType"},
                        {"name": "TIMESTAMP", "type": "TimestampType"},
                    ],
                }
            ],
            "features": [
                {
                    "source_column": {"name": "ENGAGEMENT_SCORE", "type": "DoubleType"},
                    "output_column": {
                        "name": "LAST_2_ENGAGEMENT_1H",
                        "type": "ArrayType",
                        "element_type": "DoubleType",
                    },
                    "function": "last_distinct_n",
                    "window_sec": 3600,
                    "function_params": {"n": 2},
                }
            ],
            "timestamp_field": "TIMESTAMP",
            "feature_granularity_sec": 300,
            "feature_aggregation_method": "continuous",
            "target_lag_sec": 0,  # runtime-stamped; stripped by _full_spec_hash
        },
        "online_store_type": "postgres",
    }


def _secondary_keys_authoring() -> dict[str, Any]:
    """Authoring dict for a streaming FV with ``aggregation_secondary_keys``."""
    return {
        "kind": "StreamingFeatureView",
        "name": "SK_AGG",
        "database": _DB,
        "schema_": _SCHEMA,
        "version": "V1",
        "online": True,
        "entities": ["USER_ID"],
        "timestamp_col": "TIMESTAMP",
        "feature_granularity_sec": 300,
        "feature_aggregation_method": "continuous",
        "aggregation_secondary_keys": ["AD_ID"],
        "sources": [
            {
                "name": "AD_IMPRESSIONS_STREAM",
                "source_type": "Stream",
                "columns": [
                    {"name": "USER_ID", "type": "StringType"},
                    {"name": "AD_ID", "type": "StringType"},
                    {"name": "WATCH_SECONDS", "type": "DoubleType"},
                    {"name": "TIMESTAMP", "type": "TimestampType"},
                ],
            }
        ],
        "features": [
            {
                "source_column": {"name": "WATCH_SECONDS", "type": "DoubleType"},
                "output_column": {
                    "name": "SUM_WATCH_1H",
                    "type": "ArrayType",
                    "element_type": "DoubleType",
                },
                "function": "sum",
                "window_sec": 3600,
            }
        ],
    }


def _secondary_keys_applied() -> dict[str, Any]:
    """DESCRIBE-shape payload for :func:`_secondary_keys_authoring`.

    The applied wire spec surfaces the secondary key under
    ``ordered_secondary_key_column_names`` (not the authoring
    ``aggregation_secondary_keys``).

    Returns:
        A DESCRIBE-shape spec payload using the wire secondary-key name.
    """
    return {
        "kind": "StreamingFeatureView",
        "metadata": {
            "database": _DB,
            "schema": _SCHEMA,
            "name": "SK_AGG",
            "version": "V1",
            "spec_format_version": "1",
            "internal_data_version": "1",
            "client_version": "1.38.0",
        },
        "offline_configs": [],
        "spec": {
            "ordered_entity_column_names": ["USER_ID"],
            "ordered_secondary_key_column_names": ["AD_ID"],
            "sources": [
                {
                    "name": "AD_IMPRESSIONS_STREAM",
                    "source_type": "Stream",
                    "columns": [
                        {"name": "USER_ID", "type": "StringType"},
                        {"name": "AD_ID", "type": "StringType"},
                        {"name": "WATCH_SECONDS", "type": "DoubleType"},
                        {"name": "TIMESTAMP", "type": "TimestampType"},
                    ],
                }
            ],
            "features": [
                {
                    "source_column": {"name": "WATCH_SECONDS", "type": "DoubleType"},
                    "output_column": {
                        "name": "SUM_WATCH_1H",
                        "type": "ArrayType",
                        "element_type": "DoubleType",
                    },
                    "function": "sum",
                    "window_sec": 3600,
                }
            ],
            "timestamp_field": "TIMESTAMP",
            "feature_granularity_sec": 300,
            "feature_aggregation_method": "continuous",
            "target_lag_sec": 0,  # runtime-stamped; stripped by _full_spec_hash
        },
        "online_store_type": "postgres",
    }


def _secondary_keys_applied_with_derivations() -> dict[str, Any]:
    """:func:`_secondary_keys_applied` plus the SK derivations core stamps on.

    A newer backend returns the secondary key appended to
    ``ordered_entity_column_names`` and one synthesised ``<SK>_KEYS_<window>S``
    array feature per distinct window (sourced from the SK, no aggregation
    ``function``).  Both are pure derivations from
    ``ordered_secondary_key_column_names`` — see the live captures in
    ``golden_specs/CONTENT_AGG_STREAM_SK.json``.

    Returns:
        A DESCRIBE-shape spec payload carrying the SK-derived entity column
        and the synthesised SK-array feature.
    """
    payload = _secondary_keys_applied()
    inner = payload["spec"]
    inner["ordered_entity_column_names"] = ["USER_ID", "AD_ID"]
    inner["features"].append(
        {
            "source_column": {"name": "AD_ID", "type": "StringType"},
            "output_column": {
                "name": "AD_ID_KEYS_3600S",
                "type": "ArrayType",
                "element_type": "StringType",
            },
            "window_sec": 3600,
        }
    )
    return payload


def _local_hash(authoring: dict[str, Any]) -> str:
    """Compile *authoring* the way the planner does (through the Pydantic model)."""
    model = FeatureView.model_validate(authoring)
    model_dict = model_to_dict(model)
    return compute_local_spec_hash(model_dict, _DB, _SCHEMA)


class TestArrayElementTypePreserved:
    """``element_type`` must survive the model -> compile path."""

    def test_model_dump_retains_element_type(self) -> None:
        model = FeatureView.model_validate(_array_element_type_authoring())
        model_dict = model_to_dict(model)
        out_col = model_dict["features"][0]["output_column"]
        assert out_col["type"] == "ArrayType"
        assert out_col["element_type"] == "DoubleType"

    def test_compiled_spec_retains_element_type(self) -> None:
        model = FeatureView.model_validate(_array_element_type_authoring())
        compiled = compile_to_spec(model_to_dict(model), _DB, _SCHEMA)
        out_col = compiled["spec"]["features"][0]["output_column"]
        assert out_col["element_type"] == "DoubleType"

    def test_local_hash_matches_applied(self) -> None:
        assert _local_hash(_array_element_type_authoring()) == _full_spec_hash(
            _array_element_type_applied()
        ), "array element_type streaming FV must re-plan as NO_CHANGE"


class TestSecondaryKeyCanonicalization:
    """Local ``aggregation_secondary_keys`` must hash-match applied wire key."""

    def test_local_hash_matches_applied(self) -> None:
        assert _local_hash(_secondary_keys_authoring()) == _full_spec_hash(
            _secondary_keys_applied()
        ), "secondary-key streaming FV must re-plan as NO_CHANGE"

    def test_wire_key_canonicalized_in_full_spec_hash(self) -> None:
        wire = _secondary_keys_applied()
        authoring_shaped = _secondary_keys_applied()
        inner = authoring_shaped["spec"]
        inner["aggregation_secondary_keys"] = inner.pop("ordered_secondary_key_column_names")
        assert _full_spec_hash(wire) == _full_spec_hash(authoring_shaped)

    def test_secondary_key_change_still_detected(self) -> None:
        base = _secondary_keys_applied()
        changed = _secondary_keys_applied()
        changed["spec"]["ordered_secondary_key_column_names"] = ["OTHER_KEY"]
        assert _full_spec_hash(base) != _full_spec_hash(changed)


class TestExporterEmitsSecondaryKeysFromWireKey:
    """``_build_full_fidelity_fv`` must translate the wire key on export."""

    def test_streaming_fv_emits_aggregation_secondary_keys(self) -> None:
        full_spec = _secondary_keys_applied()
        doc = _build_full_fidelity_fv(
            full_spec,
            fallback_name="SK_AGG",
            fallback_version="V1",
            fallback_database=_DB,
            fallback_schema=_SCHEMA,
        )
        assert doc.get("aggregation_secondary_keys") == ["AD_ID"]

    def test_streaming_fv_retains_array_element_type(self) -> None:
        full_spec = _secondary_keys_applied()
        doc = _build_full_fidelity_fv(
            full_spec,
            fallback_name="SK_AGG",
            fallback_version="V1",
            fallback_database=_DB,
            fallback_schema=_SCHEMA,
        )
        out_col = doc["features"][0]["output_column"]
        assert out_col["element_type"] == "DoubleType"

    def test_synthesized_sk_array_feature_is_not_exported(self) -> None:
        """The ``<SK>_KEYS_<window>S`` derivation must not reach the YAML.

        It carries a ``window_sec`` but no ``function``, so re-emitting it
        makes the freshly exported tree fail ``snow feature plan`` with
        ``FEATURE_INCOMPLETE``.  Core re-synthesises it on the next apply.
        """
        doc = _build_full_fidelity_fv(
            _secondary_keys_applied_with_derivations(),
            fallback_name="SK_AGG",
            fallback_version="V1",
            fallback_database=_DB,
            fallback_schema=_SCHEMA,
        )
        assert [f["output_column"]["name"] for f in doc["features"]] == ["SUM_WATCH_1H"]

    def test_secondary_key_is_not_exported_as_an_entity(self) -> None:
        """The SK stays on ``aggregation_secondary_keys``, never on ``entities``.

        Core appends the SK to the deployed entity list; re-emitting it there
        would make the next apply register it as a second join key.
        """
        doc = _build_full_fidelity_fv(
            _secondary_keys_applied_with_derivations(),
            fallback_name="SK_AGG",
            fallback_version="V1",
            fallback_database=_DB,
            fallback_schema=_SCHEMA,
        )
        assert doc["entities"] == ["USER_ID"]
        assert doc["aggregation_secondary_keys"] == ["AD_ID"]

    def test_export_does_not_mutate_the_input_payload(self) -> None:
        """``_build_full_fidelity_fv`` must leave the caller's payload intact.

        The exporter's ``specification_map`` doubles as the applied-state
        payload on the plan path, so an in-place rewrite of the recovered
        features would corrupt the comparison the planner then makes.
        """
        full_spec = _secondary_keys_applied_with_derivations()
        untouched = copy.deepcopy(full_spec)
        _build_full_fidelity_fv(
            full_spec,
            fallback_name="SK_AGG",
            fallback_version="V1",
            fallback_database=_DB,
            fallback_schema=_SCHEMA,
        )
        assert full_spec == untouched


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
