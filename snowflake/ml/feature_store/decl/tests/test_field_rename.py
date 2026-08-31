"""Tests for the authoring-side rename of ``FeatureView`` fields.

To bring the declarative authoring surface in line with the imperative
``snowflake.ml.feature_store.FeatureView`` constructor, two fields are
renamed:

* ``ordered_entity_column_names`` → ``entities``
* ``timestamp_field`` → ``timestamp_col``

Scope decisions (see the matching plan in
``.cursor/plans/rename_declarative_auth_fields_*.plan.md``):

1. Hard rename — old keys are rejected at load time with a migration
   error.  No Pydantic alias.
2. Authoring-only — only the Pydantic field names, YAML / Python kwargs,
   and the exporter's YAML output use the new names.  The compiled
   SPECIFICATION JSON sent to Snowflake and the ``DESCRIBE … TYPE =
   SPECIFICATION`` recovery shape continue to use the legacy
   ``ordered_entity_column_names`` / ``timestamp_field`` keys because
   that contract is owned by GS.

The translation happens at exactly two seams:

* ``decl.spec_compiler.compile_to_spec`` is the single point that
  translates the authoring keys (``entities`` / ``timestamp_col``) into
  the wire-form keys (``ordered_entity_column_names`` /
  ``timestamp_field``) when building the SPECIFICATION JSON.  Every
  earlier-stage dict (``serializer.spec_to_dict``,
  ``invariants.model_to_dict``, raw ``model_dump``) keeps the authoring
  names.
* ``decl.exporter._build_full_fidelity_fv`` translates the wire-form
  keys recovered from DESCRIBE back into the authoring keys when
  emitting YAML for ``snow feature init``.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from snowflake.ml.feature_store.decl.exporter import _build_full_fidelity_fv
from snowflake.ml.feature_store.decl.serializer import spec_to_dict
from snowflake.ml.feature_store.decl.spec_models import (
    BatchFeatureView,
    FeatureView,
    FSColumn,
    RealtimeFeatureView,
    SourceRef,
    StreamingFeatureView,
)
from snowflake.ml.test_utils import pytest_driver


class TestFeatureViewAcceptsNewKeys:
    """The renamed fields are the canonical authoring entry points."""

    def test_construct_with_entities_kwarg(self) -> None:
        fv = FeatureView(name="fv", entities=["USER_ID"])
        assert fv.entities == ["USER_ID"]

    def test_construct_with_timestamp_col_kwarg(self) -> None:
        fv = FeatureView(name="fv", timestamp_col="TS")
        assert fv.timestamp_col == "TS"

    def test_construct_with_both_kwargs(self) -> None:
        fv = StreamingFeatureView(
            name="fv",
            entities=["USER_ID"],
            timestamp_col="EVENT_TS",
        )
        assert fv.entities == ["USER_ID"]
        assert fv.timestamp_col == "EVENT_TS"

    def test_yaml_dict_with_new_keys_loads(self) -> None:
        """A YAML-shaped dict with the new authoring keys validates."""
        fv = BatchFeatureView.model_validate(
            {
                "name": "BATCH",
                "kind": "BatchFeatureView",
                "entities": ["USER_ID"],
                "timestamp_col": "EVENT_TS",
            }
        )
        assert fv.entities == ["USER_ID"]
        assert fv.timestamp_col == "EVENT_TS"

    def test_old_field_names_no_longer_present(self) -> None:
        """The legacy Pydantic field names are removed from the model."""
        fv = FeatureView(name="fv", entities=["U"], timestamp_col="TS")
        assert not hasattr(fv, "ordered_entity_column_names"), (
            "FeatureView.ordered_entity_column_names must be removed; "
            "the canonical authoring name is now `entities` to match "
            "the imperative FeatureView constructor."
        )
        assert not hasattr(fv, "timestamp_field"), (
            "FeatureView.timestamp_field must be removed; the canonical "
            "authoring name is now `timestamp_col` to match the "
            "imperative FeatureView constructor."
        )


class TestLegacyKeysRejected:
    """A YAML / dict carrying the old names must surface a migration error.

    The error message must:

    * name the legacy field that triggered the failure,
    * point the author at the new name, and
    * carry the stable prefix ``has been renamed to`` so the loader
      (``decl.loader._dict_to_spec``) can detect it and re-raise it
      instead of degrading silently to a bare ``SpecBase``.
    """

    def test_legacy_ordered_entity_column_names_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc:
            FeatureView.model_validate(
                {
                    "name": "fv",
                    "kind": "StreamingFeatureView",
                    "ordered_entity_column_names": ["USER_ID"],
                }
            )
        msg = str(exc.value)
        assert "ordered_entity_column_names" in msg
        assert "entities" in msg
        assert "has been renamed to" in msg

    def test_legacy_timestamp_field_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc:
            FeatureView.model_validate(
                {
                    "name": "fv",
                    "kind": "StreamingFeatureView",
                    "timestamp_field": "TS",
                }
            )
        msg = str(exc.value)
        assert "timestamp_field" in msg
        assert "timestamp_col" in msg
        assert "has been renamed to" in msg

    def test_legacy_keys_rejected_on_subclasses(self) -> None:
        """The reject-legacy validator runs on every FeatureView subclass."""
        for cls in (StreamingFeatureView, BatchFeatureView, RealtimeFeatureView):
            with pytest.raises(ValidationError) as exc:
                cls.model_validate(
                    {
                        "name": "fv",
                        "ordered_entity_column_names": ["U"],
                    }
                )
            assert "has been renamed to" in str(exc.value), (
                f"{cls.__name__} must reject the legacy " "ordered_entity_column_names key with a migration error."
            )

    def test_legacy_keys_rejected_by_loader_dict_to_spec(self) -> None:
        """``_dict_to_spec`` re-raises migration errors instead of degrading.

        Without this propagation the loader silently returns a bare
        ``SpecBase`` and the author sees a confusing "feature view has
        no sources" downstream error instead of the actionable rename
        message.
        """
        from snowflake.ml.feature_store.decl.loader import _dict_to_spec

        with pytest.raises(ValidationError) as exc:
            _dict_to_spec(
                {
                    "kind": "StreamingFeatureView",
                    "name": "fv",
                    "ordered_entity_column_names": ["U"],
                }
            )
        assert "has been renamed to" in str(exc.value)


class TestSerializerPreservesAuthoringKeys:
    """``spec_to_dict`` preserves the new authoring keys.

    The translation from authoring -> wire-form happens inside
    :func:`spec_compiler.compile_to_spec`, not in the serializer.  The
    serializer / ``model_dump`` shape is what other authoring-side code
    (``invariants.model_to_dict``, ``loader`` Python-form round-trip,
    the planner's ``local_authoring`` dict) consumes, and that side
    speaks the new authoring vocabulary.
    """

    def test_entities_preserved_in_serializer_output(self) -> None:
        fv = StreamingFeatureView(
            name="fv",
            entities=["USER_ID"],
            sources=[SourceRef(name="src", source_type="Stream")],
        )
        d = spec_to_dict(fv)
        assert d["entities"] == ["USER_ID"]
        assert "ordered_entity_column_names" not in d, (
            "spec_to_dict must keep the authoring key `entities`; the "
            "wire-form `ordered_entity_column_names` key is only "
            "introduced by spec_compiler.compile_to_spec."
        )

    def test_timestamp_col_preserved_in_serializer_output(self) -> None:
        fv = StreamingFeatureView(
            name="fv",
            entities=["USER_ID"],
            timestamp_col="EVENT_TS",
            sources=[SourceRef(name="src", source_type="Stream")],
        )
        d = spec_to_dict(fv)
        assert d["timestamp_col"] == "EVENT_TS"
        assert "timestamp_field" not in d

    def test_entities_with_entity_objects_still_expanded(self) -> None:
        """Entity objects in ``entities`` expand to join-key name strings.

        The expansion happens in the serializer (so the authoring shape
        carries flat join-key strings); only the *key name* differs from
        the wire-form output.
        """
        from snowflake.ml.feature_store.decl.spec_models import Entity

        entity = Entity(
            name="customer",
            join_keys=[FSColumn(name="customer_id", type="StringType")],
        )
        fv = FeatureView(name="fv", entities=[entity])
        d = spec_to_dict(fv)
        assert d["entities"] == ["customer_id"]


class TestCompilerEmitsWireFormKeys:
    """``compile_to_spec`` is the single authoring -> wire-form seam.

    The compiled inner ``spec`` dict is the contract with GS (it becomes
    the body of the ``FROM SPECIFICATION $$ ... $$`` payload), so the
    wire-form key names (``ordered_entity_column_names`` /
    ``timestamp_field``) MUST appear on the output regardless of how
    the authoring side spelt them.
    """

    def test_compile_translates_entities_to_wire_form(self) -> None:
        from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec

        out = compile_to_spec(
            {
                "kind": "StreamingFeatureView",
                "name": "fv",
                "version": "V1",
                "entities": ["USER_ID"],
                "sources": [],
                "features": [],
            },
            "DB",
            "SCH",
        )
        assert out["spec"]["ordered_entity_column_names"] == ["USER_ID"]
        assert "entities" not in out["spec"]

    def test_compile_translates_timestamp_col_to_wire_form(self) -> None:
        from snowflake.ml.feature_store.decl.spec_compiler import compile_to_spec

        out = compile_to_spec(
            {
                "kind": "StreamingFeatureView",
                "name": "fv",
                "version": "V1",
                "entities": ["USER_ID"],
                "timestamp_col": "EVENT_TS",
                "sources": [],
                "features": [],
            },
            "DB",
            "SCH",
        )
        assert out["spec"]["timestamp_field"] == "EVENT_TS"
        assert "timestamp_col" not in out["spec"]


class TestExporterEmitsAuthoringKeys:
    """``_build_full_fidelity_fv`` translates wire-form keys to authoring keys.

    The exporter consumes the recovered SPECIFICATION dict (wire form) and
    emits a YAML stub for ``snow feature init``.  The emitted YAML must use
    the new authoring keys so a subsequent ``snow feature plan`` against
    the exported tree re-validates cleanly through the Pydantic models.
    """

    def _full_spec(self) -> dict[str, Any]:
        return {
            "kind": "StreamingFeatureView",
            "metadata": {
                "database": "DB",
                "schema": "PUBLIC",
                "name": "user_clicks",
                "version": "v1",
            },
            "offline_configs": [],
            "spec": {
                "ordered_entity_column_names": ["USER_ID"],
                "sources": [
                    {
                        "name": "events",
                        "source_type": "Stream",
                        "columns": [{"name": "USER_ID", "type": "StringType"}],
                    }
                ],
                "features": [],
                "timestamp_field": "EVENT_TS",
                "feature_granularity_sec": 60,
            },
            "online_store_type": "postgres",
        }

    def test_exporter_yaml_uses_entities_key(self) -> None:
        fv_doc = _build_full_fidelity_fv(
            self._full_spec(),
            fallback_name="user_clicks",
            fallback_version="v1",
            fallback_database="DB",
            fallback_schema="PUBLIC",
        )
        assert "entities" in fv_doc, "Exported YAML must use the new authoring key `entities`."
        assert fv_doc["entities"] == ["USER_ID"]
        assert "ordered_entity_column_names" not in fv_doc, (
            "Exported YAML must NOT use the legacy wire-form key — that "
            "key would fail to load through the renamed Pydantic model."
        )

    def test_exporter_yaml_uses_timestamp_col_key(self) -> None:
        fv_doc = _build_full_fidelity_fv(
            self._full_spec(),
            fallback_name="user_clicks",
            fallback_version="v1",
            fallback_database="DB",
            fallback_schema="PUBLIC",
        )
        assert fv_doc["timestamp_col"] == "EVENT_TS"
        assert "timestamp_field" not in fv_doc


if __name__ == "__main__":
    pytest_driver.main()
