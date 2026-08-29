"""Tests for ``decl/loader.load_from_project`` (Phase 1C).

Covers the project-mode walk over ``<project_root>/sources/{entities,
datasources,feature_views}/`` and the ``database`` / ``schema``
injection rules. Locked decisions D1, D2, D3, D7 from
``plans/MANIFEST_YML_LAYOUT_DECISIONS.md`` are enforced here.

These tests are the regression net for the new entry point. They are
intentionally independent from ``test_loader.py`` so that the legacy
entry points (``expand_input_files`` / ``load_specs``) can stay in
place until Phase 4.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from snowflake.ml.feature_store.decl.errors import SpecLoadError
from snowflake.ml.feature_store.decl.types import SpecBatch
from snowflake.ml.test_utils import pytest_driver

# ---------------------------------------------------------------------------
# Fixture helpers — write a small ``sources/`` tree under ``tmp_path``.
# ---------------------------------------------------------------------------


def _make_sources_tree(
    project_root: Path,
    *,
    entities: dict[str, str] | None = None,
    datasources: dict[str, str] | None = None,
    feature_views: dict[str, str] | None = None,
) -> None:
    """Create ``<project_root>/sources/{entities,datasources,feature_views}/``.

    Each mapping is ``{filename: content}``. Subdirectories that have no
    mapping are still created (empty).

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


_ENTITY_YAML = "kind: Entity\nname: customer\njoin_keys:\n  - name: customer_id\n    type: str\n"
_DATASOURCE_YAML = (
    "kind: StreamingSource\nname: clickstream\ntype: REST\ncolumns:\n  - name: ts\n    type: TimestampType\n"
)
_FV_YAML = (
    "kind: StreamingFeatureView\n"
    "name: user_click_stats\n"
    "online: true\n"
    "entities:\n  - customer_id\n"
    "sources:\n  - name: clickstream\n    source_type: Stream\n"
)


# ---------------------------------------------------------------------------
# load_from_project: happy path
# ---------------------------------------------------------------------------


class TestLoadFromProjectHappyPath:
    def test_returns_spec_batch(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(
            tmp_path,
            entities={"customer.yaml": _ENTITY_YAML},
            datasources={"clickstream.yaml": _DATASOURCE_YAML},
            feature_views={"user_click_stats.yaml": _FV_YAML},
        )

        batch = load_from_project(tmp_path, database="MY_DB", schema="MY_SCHEMA")

        assert isinstance(batch, SpecBatch)
        assert len(batch.specs) == 3

    def test_order_is_entities_then_datasources_then_feature_views(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(
            tmp_path,
            entities={"customer.yaml": _ENTITY_YAML},
            datasources={"clickstream.yaml": _DATASOURCE_YAML},
            feature_views={"user_click_stats.yaml": _FV_YAML},
        )

        batch = load_from_project(tmp_path, database="MY_DB", schema="MY_SCHEMA")

        kinds = [s.kind for s in batch.specs]
        assert kinds[0] == "Entity"
        assert kinds[1] == "StreamingSource"
        assert kinds[2] == "StreamingFeatureView"

    def test_database_injected_from_kwargs(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(
            tmp_path,
            entities={"customer.yaml": _ENTITY_YAML},
            datasources={"clickstream.yaml": _DATASOURCE_YAML},
            feature_views={"user_click_stats.yaml": _FV_YAML},
        )

        batch = load_from_project(tmp_path, database="MY_DB", schema="MY_SCHEMA")

        for spec in batch.specs:
            assert spec.database == "MY_DB", f"{spec.kind}/{spec.name} missing database injection"

    def test_schema_injected_from_kwargs(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(
            tmp_path,
            entities={"customer.yaml": _ENTITY_YAML},
            datasources={"clickstream.yaml": _DATASOURCE_YAML},
            feature_views={"user_click_stats.yaml": _FV_YAML},
        )

        batch = load_from_project(tmp_path, database="MY_DB", schema="MY_SCHEMA")

        for spec in batch.specs:
            assert spec.schema_ == "MY_SCHEMA", f"{spec.kind}/{spec.name} missing schema injection"

    def test_yaml_database_overrides_kwarg(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(
            tmp_path,
            entities={
                "customer.yaml": (
                    "kind: Entity\nname: customer\ndatabase: USER_DB\n"
                    "join_keys:\n  - name: customer_id\n    type: str\n"
                ),
            },
        )

        batch = load_from_project(tmp_path, database="DEFAULT_DB", schema="DEFAULT_SCHEMA")

        assert len(batch.specs) == 1
        assert batch.specs[0].database == "USER_DB"
        assert batch.specs[0].schema_ == "DEFAULT_SCHEMA"

    def test_yaml_schema_overrides_kwarg(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(
            tmp_path,
            entities={
                "customer.yaml": (
                    "kind: Entity\nname: customer\nschema: USER_SCHEMA\n"
                    "join_keys:\n  - name: customer_id\n    type: str\n"
                ),
            },
        )

        batch = load_from_project(tmp_path, database="DEFAULT_DB", schema="DEFAULT_SCHEMA")

        assert len(batch.specs) == 1
        assert batch.specs[0].schema_ == "USER_SCHEMA"
        assert batch.specs[0].database == "DEFAULT_DB"

    def test_lexicographic_ordering_within_subdirectory(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(
            tmp_path,
            entities={
                "zebra.yaml": "kind: Entity\nname: zebra\n",
                "apple.yaml": "kind: Entity\nname: apple\n",
                "mango.yaml": "kind: Entity\nname: mango\n",
            },
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        names = [s.name for s in batch.specs]
        assert names == ["apple", "mango", "zebra"]


# ---------------------------------------------------------------------------
# load_from_project: directory structure rules
# ---------------------------------------------------------------------------


class TestLoadFromProjectStructure:
    def test_missing_sources_raises(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        with pytest.raises(SpecLoadError) as ei:
            load_from_project(tmp_path, database="DB", schema="SCH")

        message = str(ei.value)
        assert "sources" in message
        assert str(tmp_path) in message

    def test_empty_sources_returns_empty_batch(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        (tmp_path / "sources").mkdir()

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert isinstance(batch, SpecBatch)
        assert batch.specs == []
        assert batch.source_files == []

    def test_empty_subdirectories_returns_empty_batch(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(tmp_path)

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert batch.specs == []

    def test_macros_directory_present_raises(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        (tmp_path / "sources" / "macros").mkdir(parents=True)

        with pytest.raises(SpecLoadError) as ei:
            load_from_project(tmp_path, database="DB", schema="SCH")

        assert "sources/macros/" in str(ei.value)
        assert "reserved" in str(ei.value)
        assert "snowflake-ml-feature-store-decl" in str(ei.value)

    def test_macros_directory_with_files_raises(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        macros = tmp_path / "sources" / "macros"
        macros.mkdir(parents=True)
        (macros / "lib.py").write_text("# placeholder\n")

        with pytest.raises(SpecLoadError) as ei:
            load_from_project(tmp_path, database="DB", schema="SCH")

        assert "sources/macros/" in str(ei.value)

    def test_unknown_subdirectory_silently_ignored(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(tmp_path, entities={"customer.yaml": _ENTITY_YAML})

        notes_dir = tmp_path / "sources" / "notes"
        notes_dir.mkdir()
        (notes_dir / "random.yaml").write_text("kind: Entity\nname: SHOULD_NOT_LOAD\n")
        (notes_dir / "README.md").write_text("# notes\n")

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        names = [s.name for s in batch.specs]
        assert names == ["customer"]
        assert "SHOULD_NOT_LOAD" not in names

    def test_manifest_yml_at_project_root_not_loaded(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(tmp_path, entities={"customer.yaml": _ENTITY_YAML})
        (tmp_path / "manifest.yml").write_text(
            "manifest_version: 1\ntype: feature_store\ntargets:\n  DEFAULT:\n    account_identifier: acct\n"
            "    database: DB\n    schema: SCH\n"
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert len(batch.specs) == 1
        assert batch.specs[0].name == "customer"

    def test_files_outside_sources_not_loaded(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(tmp_path, entities={"customer.yaml": _ENTITY_YAML})
        (tmp_path / "scratch.yaml").write_text("kind: Entity\nname: SHOULD_NOT_LOAD\n")
        (tmp_path / "stray.py").write_text(
            "from snowflake.ml.feature_store.decl.spec_models import Entity\nfoo = Entity(name='SHOULD_NOT_LOAD')\n"
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        names = [s.name for s in batch.specs]
        assert names == ["customer"]
        assert "SHOULD_NOT_LOAD" not in names


# ---------------------------------------------------------------------------
# load_from_project: file types
# ---------------------------------------------------------------------------


class TestLoadFromProjectFileTypes:
    def test_yaml_files_loaded(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(tmp_path, entities={"customer.yaml": _ENTITY_YAML})

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert len(batch.specs) == 1

    def test_yml_files_loaded(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(
            tmp_path,
            entities={"customer.yml": "kind: Entity\nname: customer\n"},
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert len(batch.specs) == 1
        assert batch.specs[0].name == "customer"

    def test_json_files_loaded(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(
            tmp_path,
            entities={"customer.json": '{"kind": "Entity", "name": "customer"}'},
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert len(batch.specs) == 1
        assert batch.specs[0].kind == "Entity"

    def test_python_feature_view_loaded(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        py_source = (
            "from snowflake.ml.feature_store.decl.spec_models import (\n"
            "    FeatureView, SourceRef,\n"
            ")\n"
            "my_fv = FeatureView(\n"
            "    name='user_click_stats',\n"
            "    kind='StreamingFeatureView',\n"
            "    online=True,\n"
            "    sources=[SourceRef(name='clickstream', source_type='Stream')],\n"
            "    entities=['customer_id'],\n"
            ")\n"
        )
        _make_sources_tree(tmp_path, feature_views={"user_click_stats.py": py_source})

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert len(batch.specs) == 1
        assert batch.specs[0].name == "user_click_stats"
        assert batch.specs[0].kind == "StreamingFeatureView"

    def test_non_spec_extensions_silently_skipped(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(tmp_path, entities={"customer.yaml": _ENTITY_YAML})
        # Scatter non-spec files alongside the entity.
        (tmp_path / "sources" / "entities" / "README.md").write_text("# entities\n")
        (tmp_path / "sources" / "entities" / "notes.txt").write_text("internal\n")
        (tmp_path / "sources" / "entities" / "snapshot.sql").write_text("SELECT 1;\n")

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert len(batch.specs) == 1
        assert batch.specs[0].name == "customer"


# ---------------------------------------------------------------------------
# load_from_project: determinism
# ---------------------------------------------------------------------------


class TestLoadFromProjectDeterminism:
    def test_repeat_invocation_yields_identical_dumps(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(
            tmp_path,
            entities={
                "zebra.yaml": "kind: Entity\nname: zebra\n",
                "apple.yaml": "kind: Entity\nname: apple\n",
            },
            datasources={"clickstream.yaml": _DATASOURCE_YAML},
            feature_views={"user_click_stats.yaml": _FV_YAML},
        )

        first = load_from_project(tmp_path, database="DB", schema="SCH")
        second = load_from_project(tmp_path, database="DB", schema="SCH")

        first_dumps = [s.model_dump() for s in first.specs]
        second_dumps = [s.model_dump() for s in second.specs]
        assert first_dumps == second_dumps

    def test_filesystem_creation_order_does_not_affect_load_order(self, tmp_path: Path) -> None:
        """Determinism is lexicographic, NOT filesystem-order dependent.

        Two project roots whose entity files are written in opposite
        order must yield the same spec ordering, because the loader
        sorts within each subdirectory.

        Args:
            tmp_path: Pytest temporary directory fixture.
        """
        from snowflake.ml.feature_store.decl.loader import load_from_project

        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        _make_sources_tree(
            root_a,
            entities={
                "apple.yaml": "kind: Entity\nname: apple\n",
                "zebra.yaml": "kind: Entity\nname: zebra\n",
            },
        )
        _make_sources_tree(root_b, entities={})
        # Write files into root_b in reverse order on disk
        (root_b / "sources" / "entities" / "zebra.yaml").write_text("kind: Entity\nname: zebra\n")
        # Touch the mtimes so they don't collide; the loader still sorts by name.
        os.utime(root_b / "sources" / "entities" / "zebra.yaml", (1, 1))
        (root_b / "sources" / "entities" / "apple.yaml").write_text("kind: Entity\nname: apple\n")
        os.utime(root_b / "sources" / "entities" / "apple.yaml", (2, 2))

        batch_a = load_from_project(root_a, database="DB", schema="SCH")
        batch_b = load_from_project(root_b, database="DB", schema="SCH")

        assert [s.name for s in batch_a.specs] == [s.name for s in batch_b.specs] == ["apple", "zebra"]


# ---------------------------------------------------------------------------
# load_from_project: source_files
# ---------------------------------------------------------------------------


class TestLoadFromProjectUdfCompanion:
    """A ``<NAME>.py`` next to a ``<NAME>.yaml`` whose top-level
    ``udf.file:`` basename equals the ``.py`` filename is treated as
    the FeatureView's UDF source body and is NOT loaded as a Python
    spec module. The compiler's ``inline_udf_source`` reads the file
    as text instead. This avoids tripping ``importlib`` on UDF bodies
    that reference modules (e.g. ``pd.DataFrame``) the loader does not
    pre-import.
    """

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

    _UDF_BODY_WITH_PD_ANNOTATION = (
        "def compute_engagement_metrics(df: pd.DataFrame) -> pd.DataFrame:\n"
        '    """UDF body that references pandas without importing it."""\n'
        "    return df\n"
    )

    def test_companion_py_not_imported_when_yaml_udf_file_matches(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(
            tmp_path,
            feature_views={
                "user_click_stats.yaml": self._FV_YAML_WITH_UDF,
                "user_click_stats.py": self._UDF_BODY_WITH_PD_ANNOTATION,
            },
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert len(batch.specs) == 1
        spec = batch.specs[0]
        assert spec.kind == "StreamingFeatureView"
        assert spec.name == "user_click_stats"

    def test_companion_py_path_not_in_source_files(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(
            tmp_path,
            feature_views={
                "user_click_stats.yaml": self._FV_YAML_WITH_UDF,
                "user_click_stats.py": self._UDF_BODY_WITH_PD_ANNOTATION,
            },
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        py_in_sources = [sf for sf in batch.source_files if sf.endswith(".py")]
        assert py_in_sources == []
        yaml_in_sources = [sf for sf in batch.source_files if sf.endswith(".yaml")]
        assert len(yaml_in_sources) == 1

    def test_yaml_without_matching_udf_file_does_not_skip_py(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        # YAML's udf.file references a different file
        yaml_body = self._FV_YAML_WITH_UDF.replace("file: user_click_stats.py", "file: other_udf.py")
        # The .py is a self-contained spec definition (no pandas refs)
        py_spec = (
            "from snowflake.ml.feature_store.decl.spec_models import (\n"
            "    Entity,\n"
            ")\n"
            "my_other = Entity(\n"
            "    kind='Entity',\n"
            "    name='other',\n"
            "    join_keys=[],\n"
            ")\n"
        )
        _make_sources_tree(
            tmp_path,
            feature_views={
                "user_click_stats.yaml": yaml_body,
                "user_click_stats.py": py_spec,
            },
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        # YAML FV + PY entity = 2 specs
        kinds_names = sorted((s.kind, s.name) for s in batch.specs)
        assert ("Entity", "other") in kinds_names
        assert ("StreamingFeatureView", "user_click_stats") in kinds_names

    def test_companion_detection_is_basename_only(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        yaml_body = self._FV_YAML_WITH_UDF.replace("file: user_click_stats.py", "file: ./user_click_stats.py")
        _make_sources_tree(
            tmp_path,
            feature_views={
                "user_click_stats.yaml": yaml_body,
                "user_click_stats.py": self._UDF_BODY_WITH_PD_ANNOTATION,
            },
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")
        assert len(batch.specs) == 1
        assert all(not sf.endswith(".py") for sf in batch.source_files)

    def test_yml_extension_companion(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(
            tmp_path,
            feature_views={
                "user_click_stats.yml": self._FV_YAML_WITH_UDF,
                "user_click_stats.py": self._UDF_BODY_WITH_PD_ANNOTATION,
            },
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")
        assert len(batch.specs) == 1
        assert batch.specs[0].kind == "StreamingFeatureView"


# ---------------------------------------------------------------------------
# load_from_project: BatchSource SQL companion files (Phase 2)
# ---------------------------------------------------------------------------


class TestLoadFromProjectQueryCompanion:
    """A ``<NAME>.sql`` next to a ``<NAME>.yaml`` whose top-level
    ``query_file:`` basename equals the ``.sql`` filename is treated as
    the BatchSource's query body and is NOT loaded as a spec file. The
    compiler's ``inline_query_source`` reads the file as plain text
    (Phase 3); the loader skips it during spec discovery so it never
    reaches ``yaml.safe_load`` / ``json.loads``.

    Mirrors :class:`TestLoadFromProjectUdfCompanion`.
    """

    _BATCH_SOURCE_YAML_WITH_QUERY_FILE = "kind: BatchSource\nname: USER_EVENTS\nquery_file: USER_EVENTS.sql\n"
    _SQL_BODY = "SELECT user_id, ts, kind FROM RAW.EVENTS WHERE ts > '2024-01-01'\n"

    def test_companion_sql_not_loaded_when_yaml_query_file_matches(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(
            tmp_path,
            datasources={
                "USER_EVENTS.yaml": self._BATCH_SOURCE_YAML_WITH_QUERY_FILE,
                "USER_EVENTS.sql": self._SQL_BODY,
            },
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        # YAML BatchSource = 1 spec; the .sql file MUST NOT introduce a second
        # spec or raise.
        assert len(batch.specs) == 1
        spec = batch.specs[0]
        assert spec.kind == "BatchSource"
        assert spec.name == "USER_EVENTS"

    def test_companion_sql_path_not_in_source_files(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(
            tmp_path,
            datasources={
                "USER_EVENTS.yaml": self._BATCH_SOURCE_YAML_WITH_QUERY_FILE,
                "USER_EVENTS.sql": self._SQL_BODY,
            },
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        sql_in_sources = [sf for sf in batch.source_files if sf.endswith(".sql")]
        assert sql_in_sources == []
        yaml_in_sources = [sf for sf in batch.source_files if sf.endswith(".yaml")]
        assert len(yaml_in_sources) == 1

    def test_orphan_sql_silently_skipped(self, tmp_path: Path) -> None:
        # A .sql file with no matching BatchSource YAML must be silently
        # skipped — .sql is not a spec format. Orphan .sql files have no
        # spec semantics at all, whereas orphan .py files are loaded as
        # Python specs (see TestLoadFromProjectUdfCompanion above).
        from snowflake.ml.feature_store.decl.loader import load_from_project

        # Add a SQL file with no matching YAML alongside a real entity YAML
        _make_sources_tree(
            tmp_path,
            entities={"customer.yaml": _ENTITY_YAML},
            datasources={"orphan.sql": "SELECT 1\n"},
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert len(batch.specs) == 1
        assert batch.specs[0].kind == "Entity"
        sql_in_sources = [sf for sf in batch.source_files if sf.endswith(".sql")]
        assert sql_in_sources == []

    def test_yaml_without_matching_query_file_does_not_drag_in_orphan(self, tmp_path: Path) -> None:
        # A BatchSource YAML pointing at a *different* .sql file must not
        # change how an unrelated stray .sql is treated — the stray is
        # silently skipped, the YAML loads without error.
        from snowflake.ml.feature_store.decl.loader import load_from_project

        yaml_body = self._BATCH_SOURCE_YAML_WITH_QUERY_FILE.replace(
            "query_file: USER_EVENTS.sql", "query_file: actual_query.sql"
        )
        _make_sources_tree(
            tmp_path,
            datasources={
                "USER_EVENTS.yaml": yaml_body,
                "actual_query.sql": self._SQL_BODY,
                "stray.sql": "SELECT 99\n",
            },
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        # The BatchSource YAML must load — even though the compiler will
        # fail to resolve actual_query.sql at compile time, that is a
        # Phase 3 concern, not a Phase 2 concern.
        kinds_names = sorted((s.kind, s.name) for s in batch.specs)
        # Loader-level expectation: orphan .sql files do NOT produce spec
        # objects regardless of their basename.
        assert all(k != "Entity" or n != "stray" for k, n in kinds_names)
        assert all(not sf.endswith("stray.sql") for sf in batch.source_files)

    def test_companion_sql_yml_extension(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(
            tmp_path,
            datasources={
                "USER_EVENTS.yml": self._BATCH_SOURCE_YAML_WITH_QUERY_FILE,
                "USER_EVENTS.sql": self._SQL_BODY,
            },
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")
        assert len(batch.specs) == 1
        assert batch.specs[0].kind == "BatchSource"
        assert all(not sf.endswith(".sql") for sf in batch.source_files)


# ---------------------------------------------------------------------------
# load_from_project: source_files
# ---------------------------------------------------------------------------


class TestLoadFromProjectSourceFiles:
    def test_source_files_recorded(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_sources_tree(
            tmp_path,
            entities={"customer.yaml": _ENTITY_YAML},
            datasources={"clickstream.yaml": _DATASOURCE_YAML},
            feature_views={"user_click_stats.yaml": _FV_YAML},
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert len(batch.source_files) == 3
        # Every recorded file is inside the project_root/sources tree
        for sf in batch.source_files:
            assert "sources" in sf
            assert "customer.yaml" in sf or "clickstream.yaml" in sf or "user_click_stats.yaml" in sf


if __name__ == "__main__":
    pytest_driver.main()
