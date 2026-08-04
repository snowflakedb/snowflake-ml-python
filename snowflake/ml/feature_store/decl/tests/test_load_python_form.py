"""End-to-end tests for the Python authoring form.

Pin the contracts decided in Q1-Q9 of
``plans/python_form/python_authoring_form.md``:

* Q1 — ``.py`` spec files instantiate the same Pydantic
  classes the YAML path validates.  No new dataclass module.
* Q2 — Object kind is discriminated by Python class type via
  ``isinstance``; authors never type ``kind="..."``.
* Q3 — ``.py`` spec files live alongside YAML in the four canonical
  subdirs (``sources/{entities,datasources,feature_views,feature_groups}/``).
  Class-based detection wins regardless of which subdir holds the file.
* Q4 — Inline UDF callables are first-class (extracted by the existing
  ``serializer.callable_to_source`` mechanism, byte-equivalent to YAML
  + sidecar ``.py``).
* Q5 — Cross-spec references work both by name-string AND by Python
  object; both produce the same on-wire shape.
* Q8 — YAML ``kind: <X>FeatureView`` strings still map to the
  matching subclass (see ``test_spec_models_fv_subclasses.py``).
* Q9 — Spec-first with UDF fallback for ``.py`` files in spec subdirs:
  try to exec as a spec module first.  Zero spec instances → treated
  as a pure UDF body (silently skipped from spec discovery).  Exec
  failure + sibling YAML claims it via ``udf.file:`` → silently
  skipped.  Otherwise raise ``SpecLoadError``.

These are integration tests against
:func:`snowflake.ml.feature_store.decl.loader.load_from_project`
because that is the only public entrypoint that ties templating, exec,
serialization, and Pydantic validation together.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from snowflake.ml.feature_store.decl.errors import SpecLoadError
from snowflake.ml.feature_store.decl.loader import load_from_project
from snowflake.ml.feature_store.decl.spec_models import (
    BatchFeatureView,
    BatchSource,
    Entity,
    FeatureGroup,
    StreamingFeatureView,
    StreamingSource,
)


def _make_sources_tree(
    project_root: Path,
    *,
    entities: dict[str, str] | None = None,
    datasources: dict[str, str] | None = None,
    feature_views: dict[str, str] | None = None,
    feature_groups: dict[str, str] | None = None,
) -> None:
    """Create the canonical four-subdir spec tree under ``project_root``.

    Args:
        project_root: Directory to root the ``sources/`` tree under.
        entities: ``{filename: file_body}`` for ``sources/entities/``.
        datasources: ``{filename: file_body}`` for ``sources/datasources/``.
        feature_views: ``{filename: file_body}`` for ``sources/feature_views/``.
        feature_groups: ``{filename: file_body}`` for ``sources/feature_groups/``.
    """
    sources = project_root / "sources"
    for sub in ("entities", "datasources", "feature_views", "feature_groups"):
        (sources / sub).mkdir(parents=True, exist_ok=True)

    for name, body in (entities or {}).items():
        (sources / "entities" / name).write_text(body)
    for name, body in (datasources or {}).items():
        (sources / "datasources" / name).write_text(body)
    for name, body in (feature_views or {}).items():
        (sources / "feature_views" / name).write_text(body)
    for name, body in (feature_groups or {}).items():
        (sources / "feature_groups" / name).write_text(body)


class TestPythonFormScenario01_SingleEntity:
    """Scenario 1: one ``.py`` declaring one ``Entity`` instance."""

    def test_single_entity_py(self, tmp_path: Path) -> None:
        py = (
            "from snowflake.ml.feature_store.decl.spec_models import (\n"
            "    Entity, FSColumn\n"
            ")\n"
            "customer = Entity(\n"
            "    name='customer',\n"
            "    join_keys=[FSColumn(name='customer_id', type='StringType')],\n"
            ")\n"
        )
        _make_sources_tree(tmp_path, entities={"customer.py": py})

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert len(batch.specs) == 1
        ent = batch.specs[0]
        assert isinstance(ent, Entity)
        assert ent.name == "customer"
        assert ent.kind == "Entity"
        assert len(ent.join_keys) == 1
        assert ent.join_keys[0].name == "customer_id"


class TestPythonFormScenario02_Sources:
    """Scenario 2: ``.py`` files declaring a ``BatchSource`` and a
    ``StreamingSource``."""

    def test_batch_source_py(self, tmp_path: Path) -> None:
        py = (
            "from snowflake.ml.feature_store.decl.spec_models import BatchSource\n"
            "events = BatchSource(\n"
            "    name='events_batch',\n"
            "    table='DB.SCH.RAW_EVENTS',\n"
            ")\n"
        )
        _make_sources_tree(tmp_path, datasources={"events_batch.py": py})

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert len(batch.specs) == 1
        ds = batch.specs[0]
        assert isinstance(ds, BatchSource)
        assert ds.kind == "BatchSource"
        assert ds.name == "events_batch"
        assert ds.table == "DB.SCH.RAW_EVENTS"

    def test_streaming_source_py(self, tmp_path: Path) -> None:
        py = (
            "from snowflake.ml.feature_store.decl.spec_models import StreamingSource, FSColumn\n"
            "clicks = StreamingSource(\n"
            "    name='clicks',\n"
            "    type='REST',\n"
            "    columns=[FSColumn(name='ts', type='TimestampType')],\n"
            ")\n"
        )
        _make_sources_tree(tmp_path, datasources={"clicks.py": py})

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert len(batch.specs) == 1
        ds = batch.specs[0]
        assert isinstance(ds, StreamingSource)
        assert ds.kind == "StreamingSource"
        assert ds.name == "clicks"


class TestPythonFormScenario03_InlineUdfCallable:
    """Scenario 3 (Q4): a ``StreamingFeatureView`` with an inline UDF
    *callable* (a Python ``def`` in the same file).  The serializer
    extracts the source via ``callable_to_source`` and the resulting
    compiled dict carries ``udf.function_definition`` as a plain
    string ready for SQL emission.
    """

    def test_streaming_fv_with_inline_callable(self, tmp_path: Path) -> None:
        py = (
            "from snowflake.ml.feature_store.decl.spec_models import (\n"
            "    UDF, FSColumn, SourceRef, StreamingFeatureView,\n"
            ")\n"
            "\n"
            "def compute_engagement(df):\n"
            "    df['engagement'] = df['clicks'] * 2\n"
            "    return df\n"
            "\n"
            "user_clicks = StreamingFeatureView(\n"
            "    name='user_clicks',\n"
            "    online=True,\n"
            "    entities=['customer_id'],\n"
            "    sources=[SourceRef(name='clicks', source_type='Stream')],\n"
            "    udf=UDF(\n"
            "        name='compute_engagement',\n"
            "        engine='pandas',\n"
            "        function_definition=compute_engagement,\n"
            "        output_columns=[FSColumn(name='engagement', type='IntegerType')],\n"
            "    ),\n"
            ")\n"
        )
        _make_sources_tree(tmp_path, feature_views={"user_clicks.py": py})

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert len(batch.specs) == 1
        fv = batch.specs[0]
        assert isinstance(fv, StreamingFeatureView)
        assert fv.kind == "StreamingFeatureView"
        assert fv.udf is not None
        # The inline callable was extracted to source text by callable_to_source.
        assert isinstance(fv.udf.function_definition, str)
        assert "def compute_engagement" in fv.udf.function_definition
        assert "df['engagement']" in fv.udf.function_definition


class TestPythonFormScenario04_PythonObjectSourceRef:
    """Scenario 4 (Q5): a ``BatchFeatureView`` referencing a
    ``BatchSource`` by *Python object*.  The serializer collapses the
    object to a name-string ``SourceRef`` (the same shape YAML emits).
    """

    def test_batch_fv_references_batch_source_by_object(self, tmp_path: Path) -> None:
        ds_py = (
            "from snowflake.ml.feature_store.decl.spec_models import BatchSource\n"
            "raw_events = BatchSource(\n"
            "    name='raw_events',\n"
            "    table='DB.SCH.RAW_EVENTS',\n"
            ")\n"
        )
        fv_py = (
            "from snowflake.ml.feature_store.decl.spec_models import BatchFeatureView\n"
            "from raw_events import raw_events\n"
            "fv = BatchFeatureView(\n"
            "    name='daily_events',\n"
            "    online=False,\n"
            "    entities=['customer_id'],\n"
            "    sources=[raw_events],\n"
            ")\n"
        )
        _make_sources_tree(
            tmp_path,
            datasources={"raw_events.py": ds_py},
            feature_views={"daily_events.py": fv_py},
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        kinds = sorted((s.kind, s.name) for s in batch.specs)
        assert ("BatchSource", "raw_events") in kinds
        assert ("BatchFeatureView", "daily_events") in kinds

        fv = next(s for s in batch.specs if s.kind == "BatchFeatureView")
        assert isinstance(fv, BatchFeatureView)
        assert len(fv.sources) == 1
        # After serialization, the Python-object ref is collapsed to a
        # name-string SourceRef dict (FeatureView.sources is list[Any]
        # so the on-wire shape passes through Pydantic untouched).
        src = fv.sources[0]
        src_name = src["name"] if isinstance(src, dict) else src.name
        src_type = src["source_type"] if isinstance(src, dict) else src.source_type
        assert src_name == "raw_events"
        assert src_type == "Batch"


class TestPythonFormScenario05_FeatureGroupCrossObjectRefs:
    """Scenario 5: a ``FeatureGroup`` whose ``feature_views=[fv1, fv2]``
    are *Python objects* imported from sibling files (cross-file
    Python-object refs)."""

    def test_feature_group_with_python_object_fv_refs(self, tmp_path: Path) -> None:
        # Source FVs must carry an explicit ``version`` so the
        # FeatureGroup's ``FeatureViewRef.version`` field (required —
        # see ``spec_models.FeatureViewRef``) gets populated when the
        # Python objects are collapsed at validation time.
        fv1_py = (
            "from snowflake.ml.feature_store.decl.spec_models import BatchFeatureView\n"
            "fv_clicks = BatchFeatureView(\n"
            "    name='fv_clicks', version='V1', online=False,\n"
            "    entities=['customer_id'],\n"
            ")\n"
        )
        fv2_py = (
            "from snowflake.ml.feature_store.decl.spec_models import BatchFeatureView\n"
            "fv_amounts = BatchFeatureView(\n"
            "    name='fv_amounts', version='V1', online=False,\n"
            "    entities=['customer_id'],\n"
            ")\n"
        )
        fg_py = (
            "from snowflake.ml.feature_store.decl.spec_models import FeatureGroup\n"
            "from fv_clicks import fv_clicks\n"
            "from fv_amounts import fv_amounts\n"
            "fg = FeatureGroup(\n"
            "    name='customer_fg', version='V1',\n"
            "    entities=['customer_id'],\n"
            "    feature_views=[fv_clicks, fv_amounts],\n"
            ")\n"
        )
        _make_sources_tree(
            tmp_path,
            feature_views={"fv_clicks.py": fv1_py, "fv_amounts.py": fv2_py},
            feature_groups={"customer_fg.py": fg_py},
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        fg = next(s for s in batch.specs if s.kind == "FeatureGroup")
        assert isinstance(fg, FeatureGroup)
        # FeatureView Python objects are collapsed to name-only ref dicts.
        names = [fv_ref.name for fv_ref in fg.feature_views]
        assert names == ["fv_clicks", "fv_amounts"]


class TestPythonFormScenario06_MultipleKindsInOneFile:
    """Scenario 6: a single ``.py`` declaring multiple kinds
    (``Entity`` + ``BatchSource`` + ``BatchFeatureView``) at module
    level.  All three are returned in the batch."""

    def test_multiple_kinds_in_one_py(self, tmp_path: Path) -> None:
        py = (
            "from snowflake.ml.feature_store.decl.spec_models import (\n"
            "    BatchFeatureView, BatchSource, Entity, FSColumn,\n"
            ")\n"
            "customer = Entity(\n"
            "    name='customer',\n"
            "    join_keys=[FSColumn(name='customer_id', type='StringType')],\n"
            ")\n"
            "src = BatchSource(\n"
            "    name='src_t', table='DB.SCH.T',\n"
            ")\n"
            "fv = BatchFeatureView(\n"
            "    name='daily_fv', online=False,\n"
            "    entities=['customer_id'],\n"
            "    sources=[src],\n"
            ")\n"
        )
        # Drop it in feature_views/ to also exercise Q3 (loose subdir)
        _make_sources_tree(tmp_path, feature_views={"mixed.py": py})

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        kinds = sorted((s.kind, s.name) for s in batch.specs)
        assert ("Entity", "customer") in kinds
        assert ("BatchSource", "src_t") in kinds
        assert ("BatchFeatureView", "daily_fv") in kinds
        assert len(batch.specs) == 3


class TestPythonFormScenario07_LooseSubdir:
    """Scenario 7 (Q3): a ``.py`` in ``sources/feature_views/`` that
    declares an ``Entity`` instance.  Class-based detection wins —
    the Entity is loaded regardless of which subdir holds the file.
    """

    def test_entity_in_feature_views_subdir_is_loaded(self, tmp_path: Path) -> None:
        py = (
            "from snowflake.ml.feature_store.decl.spec_models import Entity, FSColumn\n"
            "merchant = Entity(\n"
            "    name='merchant',\n"
            "    join_keys=[FSColumn(name='merchant_id', type='StringType')],\n"
            ")\n"
        )
        # Note: file is under feature_views/ NOT entities/
        _make_sources_tree(tmp_path, feature_views={"merchant.py": py})

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert len(batch.specs) == 1
        assert isinstance(batch.specs[0], Entity)
        assert batch.specs[0].name == "merchant"


class TestPythonFormScenario08_TemplatingBeforeExec:
    """Scenario 8: the existing Jinja2 template substitution runs
    BEFORE the Python file is exec'd.  ``name='USER_{{ env }}'`` with
    ``template_vars={'env': 'PROD'}`` yields a loaded name
    ``'USER_PROD'``."""

    def test_template_vars_substituted_before_exec(self, tmp_path: Path) -> None:
        py = (
            "{# Jinja preamble to opt into templating #}\n"
            "from snowflake.ml.feature_store.decl.spec_models import Entity, FSColumn\n"
            "ent = Entity(\n"
            "    name='USER_{{ env }}',\n"
            "    join_keys=[FSColumn(name='user_id', type='StringType')],\n"
            ")\n"
        )
        _make_sources_tree(tmp_path, entities={"user.py": py})

        batch = load_from_project(
            tmp_path,
            database="DB",
            schema="SCH",
            template_vars={"env": "PROD"},
        )

        assert len(batch.specs) == 1
        ent = batch.specs[0]
        assert isinstance(ent, Entity)
        assert ent.name == "USER_PROD"


class TestPythonFormScenario09_SpecFirstWinsOverCompanion:
    """Scenario 9 (Q9): a ``.py`` declaring an ``Entity`` PLUS a sibling
    YAML whose ``udf.file:`` basename matches.  The spec module wins —
    the Entity is loaded.  (The YAML's ``udf.file:`` reference still
    inlines the .py source as UDF text downstream; that is the user's
    mistake to catch, not the loader's.)
    """

    def test_spec_wins_over_udf_file_companion(self, tmp_path: Path) -> None:
        py = (
            "from snowflake.ml.feature_store.decl.spec_models import Entity, FSColumn\n"
            "weirdly_named = Entity(\n"
            "    name='weirdly_named',\n"
            "    join_keys=[FSColumn(name='id', type='StringType')],\n"
            ")\n"
        )
        yaml_body = (
            "kind: StreamingFeatureView\n"
            "name: companion_fv\n"
            "online: true\n"
            "entities:\n  - id\n"
            "sources:\n  - name: src\n    source_type: Stream\n"
            "udf:\n"
            "  name: f\n"
            "  engine: pandas\n"
            "  output_columns:\n"
            "    - name: id\n"
            "      type: StringType\n"
            "  file: weirdly_named.py\n"
        )
        _make_sources_tree(
            tmp_path,
            feature_views={
                "weirdly_named.py": py,
                "companion_fv.yaml": yaml_body,
            },
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        kinds = sorted((s.kind, s.name) for s in batch.specs)
        # The Entity from the .py MUST be in the batch (Q9 spec-first).
        assert ("Entity", "weirdly_named") in kinds
        # The companion YAML still loads as a StreamingFeatureView.
        assert ("StreamingFeatureView", "companion_fv") in kinds


class TestPythonFormScenario10_UdfBodyFallback:
    """Scenario 10 (Q9): a ``.py`` that exec's cleanly but declares ZERO
    spec instances (pure UDF body) is silently skipped from spec
    discovery.  The companion YAML's ``udf.file:`` mechanism still
    inlines it as UDF text downstream.
    """

    _FV_YAML_WITH_UDF = (
        "kind: StreamingFeatureView\n"
        "name: udf_fv\n"
        "online: true\n"
        "entities:\n  - id\n"
        "sources:\n  - name: src\n    source_type: Stream\n"
        "udf:\n"
        "  name: pure_udf\n"
        "  engine: pandas\n"
        "  output_columns:\n"
        "    - name: id\n"
        "      type: StringType\n"
        "  file: pure_udf.py\n"
    )

    _PURE_UDF_BODY = "def pure_udf(df):\n" "    df['x'] = 1\n" "    return df\n"

    def test_pure_udf_py_silently_skipped(self, tmp_path: Path) -> None:
        _make_sources_tree(
            tmp_path,
            feature_views={
                "pure_udf.py": self._PURE_UDF_BODY,
                "udf_fv.yaml": self._FV_YAML_WITH_UDF,
            },
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        # Only the YAML feature view should be loaded as a spec.
        assert len(batch.specs) == 1
        assert batch.specs[0].kind == "StreamingFeatureView"
        assert batch.specs[0].name == "udf_fv"

    def test_pure_udf_py_not_in_source_files(self, tmp_path: Path) -> None:
        _make_sources_tree(
            tmp_path,
            feature_views={
                "pure_udf.py": self._PURE_UDF_BODY,
                "udf_fv.yaml": self._FV_YAML_WITH_UDF,
            },
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        py_in_sources = [sf for sf in batch.source_files if sf.endswith(".py")]
        assert py_in_sources == []


class TestPythonFormScenario11_SpecLoadErrorOnExecFailure:
    """Scenario 11 (Q9): a ``.py`` that fails to exec (e.g. raises
    ``NameError`` at module load time) AND is NOT a UDF companion
    raises ``SpecLoadError`` whose message names the file.
    """

    def test_namerror_in_non_companion_py_raises_spec_load_error(self, tmp_path: Path) -> None:
        py = "from snowflake.ml.feature_store.decl.spec_models import Entity\n" "ent = Entity(name=undefined_symbol)\n"
        _make_sources_tree(tmp_path, entities={"bad.py": py})

        with pytest.raises(SpecLoadError) as excinfo:
            load_from_project(tmp_path, database="DB", schema="SCH")

        assert "bad.py" in str(excinfo.value)

    def test_namerror_in_companion_py_does_not_raise(self, tmp_path: Path) -> None:
        # Twin of the test above: the same exec failure IS silently
        # skipped when a sibling YAML's ``udf.file:`` claims the file
        # as a UDF body (Q9 fallback).
        py = "def f(df: pd.DataFrame) -> pd.DataFrame:\n" "    return df\n"
        yaml_body = (
            "kind: StreamingFeatureView\n"
            "name: companion_fv\n"
            "online: true\n"
            "entities:\n  - id\n"
            "sources:\n  - name: src\n    source_type: Stream\n"
            "udf:\n"
            "  name: f\n"
            "  engine: pandas\n"
            "  output_columns:\n"
            "    - name: id\n"
            "      type: StringType\n"
            "  file: udf_with_pd.py\n"
        )
        _make_sources_tree(
            tmp_path,
            feature_views={
                "udf_with_pd.py": py,
                "companion_fv.yaml": yaml_body,
            },
        )

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert len(batch.specs) == 1
        assert batch.specs[0].name == "companion_fv"


class TestPythonFormScenario12_NoKindRequired:
    """Scenario 12 (Q2): authors never type ``kind="..."``.  The class
    IS the kind.  Each subclass auto-sets ``kind`` to its class name.
    """

    def test_entity_no_kind_arg(self, tmp_path: Path) -> None:
        py = (
            "from snowflake.ml.feature_store.decl.spec_models import Entity, FSColumn\n"
            "x = Entity(name='X', join_keys=[FSColumn(name='X', type='StringType')])\n"
        )
        _make_sources_tree(tmp_path, entities={"x.py": py})

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert len(batch.specs) == 1
        assert batch.specs[0].kind == "Entity"

    def test_batch_fv_no_kind_arg(self, tmp_path: Path) -> None:
        py = (
            "from snowflake.ml.feature_store.decl.spec_models import BatchFeatureView\n"
            "fv = BatchFeatureView(\n"
            "    name='nokind_fv', online=False,\n"
            "    entities=['id'],\n"
            ")\n"
        )
        _make_sources_tree(tmp_path, feature_views={"nokind_fv.py": py})

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert len(batch.specs) == 1
        assert batch.specs[0].kind == "BatchFeatureView"

    def test_streaming_fv_no_kind_arg(self, tmp_path: Path) -> None:
        py = (
            "from snowflake.ml.feature_store.decl.spec_models import StreamingFeatureView\n"
            "fv = StreamingFeatureView(\n"
            "    name='sfv', online=True,\n"
            "    entities=['id'],\n"
            ")\n"
        )
        _make_sources_tree(tmp_path, feature_views={"sfv.py": py})

        batch = load_from_project(tmp_path, database="DB", schema="SCH")

        assert len(batch.specs) == 1
        assert batch.specs[0].kind == "StreamingFeatureView"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
