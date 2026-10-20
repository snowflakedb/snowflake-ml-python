"""Tests for duplicate spec identity.

``SpecBatch.spec_sources`` stays index-aligned with ``specs``. Two files
that resolve to one ``spec_key`` raise ``DUPLICATE_SPEC_KEY``; two versions,
or one name in two schemas, do not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from snowflake.ml.feature_store.decl.loader import load_from_project, load_specs
from snowflake.ml.test_utils import pytest_driver

# ---------------------------------------------------------------------------
# Fixture bodies
# ---------------------------------------------------------------------------

_ENTITY_PY_TEMPLATE = (
    "from snowflake.ml.feature_store.decl.spec_models import Entity, FSColumn\n"
    "{var} = Entity(\n"
    "    name='{name}',\n"
    "    join_keys=[FSColumn(name='{key}', type='StringType')],\n"
    ")\n"
)

_ENTITY_YAML_TEMPLATE = "kind: Entity\nname: {name}\njoin_keys:\n  - name: {key}\n    type: str\n"

_TWO_ENTITY_PY = (
    "from snowflake.ml.feature_store.decl.spec_models import Entity, FSColumn\n"
    "one = Entity(name='one', join_keys=[FSColumn(name='one_id', type='StringType')])\n"
    "two = Entity(name='two', join_keys=[FSColumn(name='two_id', type='StringType')])\n"
)

# One Entity object bound to two module-level names (alias re-assignment).
_ALIAS_ENTITY_PY = (
    "from snowflake.ml.feature_store.decl.spec_models import Entity, FSColumn\n"
    "alias_a = Entity(name='aliased', join_keys=[FSColumn(name='a_id', type='StringType')])\n"
    "alias_b = alias_a\n"
)

_FV_YAML_WITH_UDF = (
    "kind: StreamingFeatureView\n"
    "name: user_click_stats\n"
    "online: true\n"
    "entities:\n  - customer_id\n"
    "sources:\n  - name: clickstream\n    source_type: Stream\n"
    "udf:\n"
    "  name: compute_engagement_metrics\n"
    "  engine: pandas\n"
    "  output_columns:\n"
    "    - name: customer_id\n"
    "      type: StringType\n"
    "  file: user_click_stats.py\n"
)

_UDF_BODY = (
    "def compute_engagement_metrics(df: pd.DataFrame) -> pd.DataFrame:\n"
    '    """UDF body that references pandas without importing it."""\n'
    "    return df\n"
)

_BATCH_SOURCE_YAML_WITH_QUERY_FILE = "kind: BatchSource\nname: user_events\nquery_file: user_events.sql\n"

_SQL_BODY = "SELECT user_id, event_ts FROM DB.SCH.RAW_EVENTS\n"


def _make_sources_tree(
    project_root: Path,
    *,
    entities: dict[str, str] | None = None,
    datasources: dict[str, str] | None = None,
    feature_views: dict[str, str] | None = None,
) -> None:
    """Create ``<project_root>/sources/{entities,datasources,feature_views}/``.

    Args:
        project_root: Directory to root the ``sources/`` tree under.
        entities: ``{filename: file_body}`` for ``sources/entities/``.
        datasources: ``{filename: file_body}`` for ``sources/datasources/``.
        feature_views: ``{filename: file_body}`` for ``sources/feature_views/``.
    """
    sources = project_root / "sources"
    for sub in ("entities", "datasources", "feature_views"):
        (sources / sub).mkdir(parents=True, exist_ok=True)
    for name, body in (entities or {}).items():
        (sources / "entities" / name).write_text(body)
    for name, body in (datasources or {}).items():
        (sources / "datasources" / name).write_text(body)
    for name, body in (feature_views or {}).items():
        (sources / "feature_views" / name).write_text(body)


# ---------------------------------------------------------------------------
# Cycle 1 — provenance: SpecBatch.spec_sources index-aligned with specs
# ---------------------------------------------------------------------------


class TestLoadSpecsSpecSources:
    def test_spec_sources_index_aligned(self, tmp_path: Path) -> None:
        alpha = tmp_path / "alpha.py"
        alpha.write_text(_ENTITY_PY_TEMPLATE.format(var="alpha", name="alpha", key="a_id"))
        beta = tmp_path / "beta.yaml"
        beta.write_text(_ENTITY_YAML_TEMPLATE.format(name="beta", key="b_id"))

        batch = load_specs([str(alpha), str(beta)])

        assert len(batch.spec_sources) == len(batch.specs)
        by_name = {s.name: src for s, src in zip(batch.specs, batch.spec_sources)}
        assert by_name["alpha"].endswith("alpha.py")
        assert by_name["beta"].endswith("beta.yaml")


class TestLoadFromProjectSpecSources:
    def test_spec_sources_index_aligned(self, tmp_path: Path) -> None:
        _make_sources_tree(
            tmp_path,
            entities={
                "customer.yaml": _ENTITY_YAML_TEMPLATE.format(name="customer", key="customer_id"),
            },
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert len(batch.spec_sources) == len(batch.specs)
        assert batch.spec_sources[0].endswith("customer.yaml")

    def test_py_module_declaring_two_specs_records_file_twice(self, tmp_path: Path) -> None:
        # A single ``.py`` module that instantiates two spec objects must
        # contribute one ``spec_sources`` entry per spec — both pointing at
        # the same file — so the list stays index-aligned with ``specs``.
        _make_sources_tree(tmp_path, entities={"pair.py": _TWO_ENTITY_PY})

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert len(batch.specs) == 2
        assert len(batch.spec_sources) == 2
        by_name = {s.name: src for s, src in zip(batch.specs, batch.spec_sources)}
        assert by_name["one"].endswith("pair.py")
        assert by_name["two"].endswith("pair.py")

    def test_udf_companion_py_contributes_no_spec_source(self, tmp_path: Path) -> None:
        # A ``.py`` claimed by a sibling YAML's ``udf.file:`` is a UDF body,
        # not a spec — it must not appear in ``spec_sources``.
        _make_sources_tree(
            tmp_path,
            feature_views={
                "user_click_stats.yaml": _FV_YAML_WITH_UDF,
                "user_click_stats.py": _UDF_BODY,
            },
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert len(batch.specs) == 1
        assert len(batch.spec_sources) == 1
        assert all(not src.endswith(".py") for src in batch.spec_sources)
        assert batch.spec_sources[0].endswith("user_click_stats.yaml")

    def test_query_companion_sql_contributes_no_spec_source(self, tmp_path: Path) -> None:
        # A ``.sql`` claimed by a BatchSource's ``query_file:`` is a SQL
        # body, not a spec — it must not appear in ``spec_sources``.
        _make_sources_tree(
            tmp_path,
            datasources={
                "user_events.yaml": _BATCH_SOURCE_YAML_WITH_QUERY_FILE,
                "user_events.sql": _SQL_BODY,
            },
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert len(batch.specs) == 1
        assert len(batch.spec_sources) == 1
        assert all(not src.endswith(".sql") for src in batch.spec_sources)
        assert batch.spec_sources[0].endswith("user_events.yaml")


# ---------------------------------------------------------------------------
# Cycle 2 — DUPLICATE_SPEC_KEY: two files that resolve to one object identity
# ---------------------------------------------------------------------------


def _bfv_yaml(
    *,
    name: str = "MY_BFV",
    version: str = "V1",
    refresh_freq: str | None = None,
    schema: str | None = None,
    kind: str = "BatchFeatureView",
) -> str:
    lines = [f"kind: {kind}", f"name: {name}", f"version: {version}"]
    if schema is not None:
        lines.append(f"schema: {schema}")
    lines += ["entities:", "  - USER_ID", "timestamp_col: EVENT_TS"]
    if refresh_freq is not None:
        lines.append(f"refresh_freq: '{refresh_freq}'")
    lines += [
        "sources:",
        "  - name: EV",
        "    source_type: Batch",
        "    table: RAW_EV",
        "features:",
        "  - source_column: {name: M, type: DoubleType}",
        "    output_column: {name: M, type: DoubleType}",
    ]
    return "\n".join(lines) + "\n"


def _bfv_py(*, refresh_freq: str = "5 minutes") -> str:
    return (
        "from snowflake.ml.feature_store.decl.spec_models import (\n"
        "    BatchFeatureView, SourceRef, Feature, FSColumn,\n"
        ")\n"
        "MY_BFV = BatchFeatureView(\n"
        "    name='MY_BFV', version='V1',\n"
        "    entities=['USER_ID'], timestamp_col='EVENT_TS',\n"
        f"    refresh_freq='{refresh_freq}',\n"
        "    sources=[SourceRef(name='EV', source_type='Batch', table='RAW_EV')],\n"
        "    features=[Feature(source_column=FSColumn(name='M', type='DoubleType'),\n"
        "                      output_column=FSColumn(name='M', type='DoubleType'))],\n"
        ")\n"
    )


_BATCH_SOURCE_YAML = "kind: BatchSource\nname: EVENTS\ntable: DB.SC.RAW_EVENTS\n"
_STREAMING_SOURCE_YAML = "kind: StreamingSource\nname: EVENTS\n"


def _validate(batch: object) -> list[Any]:
    from snowflake.ml.feature_store.decl.invariants import validate_specs
    from snowflake.ml.feature_store.decl.types import AppliedState

    return validate_specs(batch, AppliedState(objects={}))  # type: ignore[arg-type]


def _dups(results: list[Any]) -> list[Any]:
    return [r for r in results if r.code == "DUPLICATE_SPEC_KEY"]


class TestDuplicateSpecKeyDetected:
    def test_py_and_yaml_same_bfv_identity_errors(self, tmp_path: Path) -> None:
        # The exact AC2 scenario: a stale ``.py`` export and a fresh ``.yaml``
        # for the same BatchFeatureView name/version, differing only in an
        # operational field. Both load; validate must reject the ambiguity.
        _make_sources_tree(
            tmp_path,
            feature_views={
                "MY_BFV.py": _bfv_py(refresh_freq="5 minutes"),
                "MY_BFV.yaml": _bfv_yaml(refresh_freq="10 minutes"),
            },
        )
        batch = load_from_project(tmp_path, database="DB", schema="SC")

        dups = _dups(_validate(batch))
        assert len(dups) == 1
        finding = dups[0]
        assert finding.severity == "ERROR"
        assert finding.object_name == "MY_BFV"
        assert "MY_BFV.py" in finding.message
        assert "MY_BFV.yaml" in finding.message

    def test_exporter_versioned_stem_collides_with_bare_stem(self, tmp_path: Path) -> None:
        # ``exporter._versioned_stem`` writes ``NAME_V1.yaml``; a hand-authored
        # ``NAME.yaml`` for the same object collides by identity even though
        # the filenames differ. Identity comes from the body, not the stem.
        _make_sources_tree(
            tmp_path,
            feature_views={
                "MY_BFV_V1.yaml": _bfv_yaml(),
                "MY_BFV.yaml": _bfv_yaml(),
            },
        )
        batch = load_from_project(tmp_path, database="DB", schema="SC")

        dups = _dups(_validate(batch))
        assert len(dups) == 1
        assert "MY_BFV_V1.yaml" in dups[0].message
        assert "MY_BFV.yaml" in dups[0].message

    def test_batch_and_streaming_fv_same_identity_collide(self, tmp_path: Path) -> None:
        # A stale batch export and a fresh streaming YAML for one name+version
        # are one physical object — the FV subkind (batch / streaming /
        # realtime) is not part of its identity, exactly like the source kinds.
        _make_sources_tree(
            tmp_path,
            feature_views={
                "USER_FV_batch.yaml": _bfv_yaml(name="USER_FV", kind="BatchFeatureView"),
                "USER_FV_stream.yaml": _bfv_yaml(name="USER_FV", kind="StreamingFeatureView"),
            },
        )
        batch = load_from_project(tmp_path, database="DB", schema="SC")

        dups = _dups(_validate(batch))
        assert len(dups) == 1
        assert dups[0].object_name == "USER_FV"
        assert "USER_FV_batch.yaml" in dups[0].message
        assert "USER_FV_stream.yaml" in dups[0].message

    def test_duplicate_entity_errors(self, tmp_path: Path) -> None:
        _make_sources_tree(
            tmp_path,
            entities={
                "customer.yaml": _ENTITY_YAML_TEMPLATE.format(name="customer", key="customer_id"),
                "customer_dup.yaml": _ENTITY_YAML_TEMPLATE.format(name="customer", key="other_id"),
            },
        )
        batch = load_from_project(tmp_path, database="DB", schema="SC")

        dups = _dups(_validate(batch))
        assert len(dups) == 1
        assert dups[0].object_name == "customer"
        assert "customer.yaml" in dups[0].message
        assert "customer_dup.yaml" in dups[0].message

    def test_duplicate_batch_source_errors(self, tmp_path: Path) -> None:
        _make_sources_tree(
            tmp_path,
            datasources={
                "events.yaml": _BATCH_SOURCE_YAML,
                "events_copy.yaml": _BATCH_SOURCE_YAML,
            },
        )
        batch = load_from_project(tmp_path, database="DB", schema="SC")

        dups = _dups(_validate(batch))
        assert len(dups) == 1
        assert dups[0].object_name == "EVENTS"

    def test_stream_and_batch_source_same_name_collide(self, tmp_path: Path) -> None:
        # StreamingSource and BatchSource fold to one ``Datasource`` spec_key,
        # so a same-named pair is one object identity — a real duplicate.
        _make_sources_tree(
            tmp_path,
            datasources={
                "events_batch.yaml": _BATCH_SOURCE_YAML,
                "events_stream.yaml": _STREAMING_SOURCE_YAML,
            },
        )
        batch = load_from_project(tmp_path, database="DB", schema="SC")

        dups = _dups(_validate(batch))
        assert len(dups) == 1
        assert dups[0].object_name == "EVENTS"

    def test_byte_identical_duplicate_still_errors(self, tmp_path: Path) -> None:
        # Two byte-identical files are still two plan ops for one object —
        # the ambiguity is real regardless of content equality.
        _make_sources_tree(
            tmp_path,
            feature_views={
                "MY_BFV.yaml": _bfv_yaml(refresh_freq="5 minutes"),
                "MY_BFV_COPY.yaml": _bfv_yaml(refresh_freq="5 minutes"),
            },
        )
        batch = load_from_project(tmp_path, database="DB", schema="SC")

        assert len(_dups(_validate(batch))) == 1

    def test_fires_even_when_specs_are_up_to_date(self) -> None:
        # The per-spec validate loop ``continue``s on the idempotency skip, so
        # a batch of duplicates that all hash-match applied state would never
        # reach a per-spec check. The duplicate detection must be batch-level.
        from snowflake.ml.feature_store.decl.invariants import (
            model_to_dict,
            spec_key,
            structural_fingerprint_hash,
            validate_specs,
        )
        from snowflake.ml.feature_store.decl.loader import _dict_to_spec
        from snowflake.ml.feature_store.decl.types import (
            AppliedObject,
            AppliedState,
            SpecBatch,
        )

        ent_dict = {
            "kind": "Entity",
            "name": "user",
            "database": "DB",
            "join_keys": [{"name": "user_id", "type": "StringType"}],
        }
        model = _dict_to_spec(ent_dict)
        normalized = model_to_dict(model)
        key = spec_key(normalized)
        applied = AppliedObject(
            key=key,
            kind="Entity",
            name="user",
            content_hash=structural_fingerprint_hash(normalized),
            spec_payload=normalized,
        )
        batch = SpecBatch(
            specs=[_dict_to_spec(ent_dict), _dict_to_spec(ent_dict)],
            spec_sources=["a/user.yaml", "b/user.yaml"],
        )
        results = validate_specs(batch, AppliedState(objects={key: applied}))

        dups = [r for r in results if r.code == "DUPLICATE_SPEC_KEY"]
        assert len(dups) == 1
        assert "a/user.yaml" in dups[0].message
        assert "b/user.yaml" in dups[0].message

    def test_empty_spec_sources_degrades_filename_to_unknown(self) -> None:
        from snowflake.ml.feature_store.decl.loader import _dict_to_spec
        from snowflake.ml.feature_store.decl.types import SpecBatch

        ent_dict = {
            "kind": "Entity",
            "name": "user",
            "database": "DB",
            "join_keys": [{"name": "user_id", "type": "StringType"}],
        }
        batch = SpecBatch(specs=[_dict_to_spec(ent_dict), _dict_to_spec(ent_dict)], spec_sources=[])

        dups = _dups(_validate(batch))
        assert len(dups) == 1
        assert "<unknown>" in dups[0].message


class TestDuplicateSpecKeyNegatives:
    def test_two_versions_of_one_name_is_not_a_duplicate(self, tmp_path: Path) -> None:
        _make_sources_tree(
            tmp_path,
            feature_views={
                "MY_BFV_V1.yaml": _bfv_yaml(version="V1"),
                "MY_BFV_V2.yaml": _bfv_yaml(version="V2"),
            },
        )
        batch = load_from_project(tmp_path, database="DB", schema="SC")

        assert _dups(_validate(batch)) == []

    def test_batch_and_streaming_fv_different_versions_not_a_duplicate(self, tmp_path: Path) -> None:
        # Collapsing the FV subkind must not erase the version: a batch V1 and
        # a streaming V2 of one name are two distinct objects, not a duplicate.
        _make_sources_tree(
            tmp_path,
            feature_views={
                "USER_FV_v1.yaml": _bfv_yaml(name="USER_FV", version="V1", kind="BatchFeatureView"),
                "USER_FV_v2.yaml": _bfv_yaml(name="USER_FV", version="V2", kind="StreamingFeatureView"),
            },
        )
        batch = load_from_project(tmp_path, database="DB", schema="SC")

        assert _dups(_validate(batch)) == []

    def test_same_name_different_schema_is_not_a_duplicate(self, tmp_path: Path) -> None:
        _make_sources_tree(
            tmp_path,
            feature_views={
                "MY_BFV_A.yaml": _bfv_yaml(schema="SC1"),
                "MY_BFV_B.yaml": _bfv_yaml(schema="SC2"),
            },
        )
        batch = load_from_project(tmp_path, database="DB", schema="SC")

        assert _dups(_validate(batch)) == []

    def test_aliased_python_assignment_is_one_spec(self, tmp_path: Path) -> None:
        # Two module-level names bound to one spec object collapse to a single
        # spec — not a false DUPLICATE_SPEC_KEY naming one file twice.
        _make_sources_tree(tmp_path, entities={"aliased.py": _ALIAS_ENTITY_PY})

        batch = load_from_project(tmp_path, database="DB", schema="SC")

        assert len(batch.specs) == 1
        assert len(batch.spec_sources) == 1
        assert _dups(_validate(batch)) == []


if __name__ == "__main__":
    pytest_driver.main()
