"""End-to-end templating tests against the committed ``templated_project`` fixture.

Exercises the full ``decl_api.load_manifest → resolve_target → load_project``
pipeline against ``decl/tests/fixtures/templated_project/`` to verify that
Jinja2 templating works uniformly across all three spec kinds (entities,
datasources, feature_views) AND across both kinds of companion sidecars
(``.py`` UDF bodies, ``.sql`` query bodies).

These tests pin the DCM-style template precedence contract end-to-end:
``manifest.templating.defaults < configurations[target.templating_config] <
runtime_vars``.  Two targets (``DEV`` and ``PROD``) share the same
on-disk sources tree but resolve different ``row_limit`` values; the
``.sql`` sidecar's ``LIMIT {{ row_limit }}`` clause is what makes the
two configurations visibly diverge.

The negative tests cover hypothesis H7 — a missing variable in any
rendered file (top-level YAML or sidecar) MUST surface as a
``SpecLoadError`` whose message names the offending file so operators
can act on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from snowflake.ml.feature_store.decl import api as decl_api
from snowflake.ml.feature_store.decl.errors import SpecLoadError
from snowflake.ml.feature_store.decl.loader import load_from_project
from snowflake.ml.feature_store.decl.types import SpecBatch
from snowflake.ml.test_utils import pytest_driver

_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "templated_project"


# ---------------------------------------------------------------------------
# Fixture loaders
# ---------------------------------------------------------------------------


def _resolve(target_name: str) -> Any:
    manifest = decl_api.load_manifest(_FIXTURE_ROOT)
    return manifest, decl_api.resolve_target(manifest, target_name)


def _load(target_name: str, runtime_vars: dict[str, Any] | None = None) -> SpecBatch:
    _, target = _resolve(target_name)
    return decl_api.load_project(_FIXTURE_ROOT, target=target, runtime_vars=runtime_vars)


def _get_spec(batch: SpecBatch, kind: str, name: str) -> Any:
    for spec in batch.specs:
        if getattr(spec, "kind", "") == kind and getattr(spec, "name", "") == name:
            return spec
    raise AssertionError(f"No {kind}/{name} in batch.specs: {[(s.kind, s.name) for s in batch.specs]}")


# ---------------------------------------------------------------------------
# End-to-end templating across YAML spec kinds and sidecars
# ---------------------------------------------------------------------------


class TestTemplatedProjectFixture:
    """Smoke check: the committed fixture round-trips cleanly through
    ``load_manifest → resolve_target → load_project`` for both targets.
    """

    def test_manifest_declares_two_targets_and_configs(self) -> None:
        manifest = decl_api.load_manifest(_FIXTURE_ROOT)
        assert set(manifest.targets.keys()) == {"DEV", "PROD"}
        assert set(manifest.templating.configurations.keys()) == {"DEV", "PROD"}
        assert manifest.templating.defaults.get("row_limit") == 1
        assert manifest.templating.configurations["DEV"]["row_limit"] == 10
        assert manifest.templating.configurations["PROD"]["row_limit"] == 100

    def test_dev_target_loads_three_specs(self) -> None:
        batch = _load("DEV")
        kinds_names = sorted((s.kind, s.name) for s in batch.specs)
        assert kinds_names == [
            ("BatchFeatureView", "USER_FEATURES"),
            ("BatchSource", "USER_EVENTS"),
            ("Entity", "USER"),
        ]


class TestTemplatedProjectYamlRendering:
    """Each top-level YAML spec MUST pick up the merged template
    variables.  Pins H1 across all three spec kinds in a real project.
    """

    def test_entity_yaml_description_rendered_for_dev(self) -> None:
        batch = _load("DEV")
        entity = _get_spec(batch, "Entity", "USER")
        assert entity.description == "env=_DEV"

    def test_entity_yaml_description_rendered_for_prod(self) -> None:
        batch = _load("PROD")
        entity = _get_spec(batch, "Entity", "USER")
        assert entity.description == "env=_PROD"

    def test_feature_view_yaml_description_rendered_for_dev(self) -> None:
        batch = _load("DEV")
        fv = _get_spec(batch, "BatchFeatureView", "USER_FEATURES")
        assert fv.description == "env=_DEV"

    def test_feature_view_yaml_description_rendered_for_prod(self) -> None:
        batch = _load("PROD")
        fv = _get_spec(batch, "BatchFeatureView", "USER_FEATURES")
        assert fv.description == "env=_PROD"


class TestTemplatedProjectSqlSidecarRendering:
    """The ``.sql`` sidecar referenced by ``USER_EVENTS.yaml``
    (``query_file: USER_EVENTS.sql``) MUST pick up the merged template
    variables — specifically ``row_limit`` — so the two configurations
    produce visibly different rendered queries.

    Pins H4 end-to-end through the public ``decl_api.load_project``
    surface.
    """

    def test_sql_sidecar_renders_limit_10_for_dev(self) -> None:
        batch = _load("DEV")
        ds = _get_spec(batch, "BatchSource", "USER_EVENTS")
        assert "LIMIT 10" in ds.query
        assert "{{" not in ds.query

    def test_sql_sidecar_renders_limit_100_for_prod(self) -> None:
        batch = _load("PROD")
        ds = _get_spec(batch, "BatchSource", "USER_EVENTS")
        assert "LIMIT 100" in ds.query
        assert "{{" not in ds.query

    def test_two_configurations_render_sidecar_differently(self) -> None:
        dev_batch = _load("DEV")
        prod_batch = _load("PROD")
        dev_ds = _get_spec(dev_batch, "BatchSource", "USER_EVENTS")
        prod_ds = _get_spec(prod_batch, "BatchSource", "USER_EVENTS")
        assert dev_ds.query != prod_ds.query
        assert "LIMIT 10" in dev_ds.query
        assert "LIMIT 100" in prod_ds.query

    def test_runtime_variable_overrides_configuration(self) -> None:
        batch = _load("PROD", runtime_vars={"row_limit": 7})
        ds = _get_spec(batch, "BatchSource", "USER_EVENTS")
        assert "LIMIT 7" in ds.query
        # The PROD-level row_limit=100 MUST NOT survive the override.
        assert "LIMIT 100" not in ds.query

    def test_query_carries_post_render_text_only(self) -> None:
        # After templating + whitespace normalisation the inlined query
        # has no Jinja markers, no source-file newlines, no double
        # spaces. This is what the planner hashes and what gets stored
        # on the deployed Dynamic Table.
        batch = _load("DEV")
        ds = _get_spec(batch, "BatchSource", "USER_EVENTS")
        assert "{{" not in ds.query
        assert "\n" not in ds.query
        assert "  " not in ds.query


# ---------------------------------------------------------------------------
# UDF sidecar templating end-to-end (separate tmp_path project)
# ---------------------------------------------------------------------------


_UDF_FV_YAML = (
    "kind: StreamingFeatureView\n"
    "name: USER_CLICK_FEATURES\n"
    "online: true\n"
    "entities:\n  - USER_ID\n"
    "sources:\n  - name: CLICKSTREAM\n    source_type: Stream\n"
    "udf:\n"
    "  name: compute_engagement\n"
    "  engine: pandas\n"
    "  output_columns:\n"
    "    - name: USER_ID\n"
    "      type: StringType\n"
    "  file: USER_CLICK_FEATURES.py\n"
)
_UDF_BODY_WITH_TEMPLATE = (
    "def compute_engagement(df):\n"
    '    """UDF compiled for env={{ env_suffix }}."""\n'
    "    threshold = {{ engagement_threshold }}\n"
    "    return df[df['clicks'] > threshold]\n"
)
_STREAM_SOURCE_YAML = (
    "kind: StreamingSource\nname: CLICKSTREAM\ntype: REST\n"
    "columns:\n  - name: USER_ID\n    type: StringType\n"
    "  - name: TS\n    type: TimestampType\n"
    "  - name: CLICKS\n    type: LongType\n"
)
_ENTITY_YAML = "kind: Entity\nname: USER\njoin_keys:\n  - name: USER_ID\n    type: StringType\n"


def _make_udf_project(root: Path, *, manifest_body: str) -> None:
    (root / "manifest.yml").write_text(manifest_body)
    sources = root / "sources"
    (sources / "entities").mkdir(parents=True, exist_ok=True)
    (sources / "datasources").mkdir(parents=True, exist_ok=True)
    (sources / "feature_views").mkdir(parents=True, exist_ok=True)
    (sources / "entities" / "USER.yaml").write_text(_ENTITY_YAML)
    (sources / "datasources" / "CLICKSTREAM.yaml").write_text(_STREAM_SOURCE_YAML)
    (sources / "feature_views" / "USER_CLICK_FEATURES.yaml").write_text(_UDF_FV_YAML)
    (sources / "feature_views" / "USER_CLICK_FEATURES.py").write_text(_UDF_BODY_WITH_TEMPLATE)


class TestTemplatedProjectUdfSidecarRendering:
    """A ``.py`` UDF sidecar referenced by a sibling YAML via
    ``udf.file:`` MUST pick up the merged template variables.

    Pins H3 end-to-end through ``decl_api.load_project``.
    """

    _MANIFEST_BODY = (
        "manifest_version: 1\n"
        "type: feature_store\n"
        "default_target: DEV\n"
        "targets:\n"
        "  DEV:\n"
        "    account_identifier: ORG-ACCOUNT\n"
        "    database: TPL_DB\n"
        "    schema: TPL_SC\n"
        "    templating_config: DEV\n"
        "templating:\n"
        "  defaults:\n"
        "    env_suffix: _BASE\n"
        "    engagement_threshold: 0\n"
        "  configurations:\n"
        "    DEV:\n"
        "      env_suffix: _DEV\n"
        "      engagement_threshold: 5\n"
    )

    def test_udf_sidecar_renders_through_load_project(self, tmp_path: Path) -> None:
        _make_udf_project(tmp_path, manifest_body=self._MANIFEST_BODY)
        manifest = decl_api.load_manifest(tmp_path)
        target = decl_api.resolve_target(manifest, "DEV")
        batch = decl_api.load_project(tmp_path, target=target)

        fv = _get_spec(batch, "StreamingFeatureView", "USER_CLICK_FEATURES")
        assert fv.udf is not None
        source = fv.udf.function_definition
        assert isinstance(source, str)
        assert "UDF compiled for env=_DEV" in source
        assert "threshold = 5" in source
        assert "{{" not in source

    def test_udf_sidecar_runtime_override_wins(self, tmp_path: Path) -> None:
        _make_udf_project(tmp_path, manifest_body=self._MANIFEST_BODY)
        manifest = decl_api.load_manifest(tmp_path)
        target = decl_api.resolve_target(manifest, "DEV")
        batch = decl_api.load_project(
            tmp_path,
            target=target,
            runtime_vars={"engagement_threshold": 99},
        )

        fv = _get_spec(batch, "StreamingFeatureView", "USER_CLICK_FEATURES")
        source = fv.udf.function_definition
        assert "threshold = 99" in source
        assert "threshold = 5" not in source


# ---------------------------------------------------------------------------
# Missing-variable diagnostics (H7 extended to sidecars)
# ---------------------------------------------------------------------------


class TestMissingTemplateVariableDiagnostics:
    """Missing template variables MUST surface a ``SpecLoadError`` whose
    message names the offending file — whether the placeholder lives in
    a top-level YAML, a ``.py`` UDF sidecar, or a ``.sql`` query
    sidecar.  This is the operator-facing contract that lets the bug
    bash negative case point at the right file when a configuration is
    missing a variable.
    """

    _MANIFEST_NO_ROW_LIMIT = (
        "manifest_version: 1\n"
        "type: feature_store\n"
        "default_target: DEV\n"
        "targets:\n"
        "  DEV:\n"
        "    account_identifier: ORG-ACCOUNT\n"
        "    database: TPL_DB\n"
        "    schema: TPL_SC\n"
        "    templating_config: DEV\n"
        "templating:\n"
        "  defaults:\n"
        "    env_suffix: _BASE\n"
        "  configurations:\n"
        "    DEV:\n"
        "      env_suffix: _DEV\n"
    )

    def test_missing_variable_in_sql_sidecar_names_filepath(self, tmp_path: Path) -> None:
        # Re-author the templated_project files into tmp_path with a
        # manifest that does NOT define ``row_limit``. The .sql sidecar
        # still uses ``LIMIT {{ row_limit }}`` so the load MUST fail
        # naming the .sql file.
        (tmp_path / "manifest.yml").write_text(self._MANIFEST_NO_ROW_LIMIT)
        sources = tmp_path / "sources"
        (sources / "entities").mkdir(parents=True)
        (sources / "datasources").mkdir(parents=True)
        (sources / "feature_views").mkdir(parents=True)
        (sources / "entities" / "USER.yaml").write_text(_ENTITY_YAML)
        (sources / "datasources" / "USER_EVENTS.yaml").write_text(
            "kind: BatchSource\nname: USER_EVENTS\nquery_file: USER_EVENTS.sql\n"
            "columns:\n  - name: USER_ID\n    type: StringType\n"
        )
        (sources / "datasources" / "USER_EVENTS.sql").write_text("SELECT * FROM RAW_EVENTS LIMIT {{ row_limit }}\n")

        manifest = decl_api.load_manifest(tmp_path)
        target = decl_api.resolve_target(manifest, "DEV")

        with pytest.raises(SpecLoadError) as excinfo:
            decl_api.load_project(tmp_path, target=target)

        msg = str(excinfo.value)
        assert "row_limit" in msg
        assert "USER_EVENTS.sql" in msg

    def test_missing_variable_in_udf_sidecar_names_filepath(self, tmp_path: Path) -> None:
        # Same shape for the UDF sidecar: the template references
        # ``engagement_threshold`` but the manifest only defines
        # ``env_suffix``. The .py sidecar's filepath MUST appear in the
        # error so the operator can fix it.
        manifest_body = (
            "manifest_version: 1\n"
            "type: feature_store\n"
            "default_target: DEV\n"
            "targets:\n"
            "  DEV:\n"
            "    account_identifier: ORG-ACCOUNT\n"
            "    database: TPL_DB\n"
            "    schema: TPL_SC\n"
            "    templating_config: DEV\n"
            "templating:\n"
            "  defaults:\n"
            "    env_suffix: _BASE\n"
            "  configurations:\n"
            "    DEV:\n"
            "      env_suffix: _DEV\n"
        )
        _make_udf_project(tmp_path, manifest_body=manifest_body)

        manifest = decl_api.load_manifest(tmp_path)
        target = decl_api.resolve_target(manifest, "DEV")

        with pytest.raises(SpecLoadError) as excinfo:
            decl_api.load_project(tmp_path, target=target)

        msg = str(excinfo.value)
        assert "engagement_threshold" in msg
        assert "USER_CLICK_FEATURES.py" in msg


# ---------------------------------------------------------------------------
# load_from_project: template_vars kwarg threads to compiler
# ---------------------------------------------------------------------------


class TestLoadFromProjectThreadsTemplateVarsToCompiler:
    """``load_from_project(..., template_vars=...)`` MUST thread the
    merged template vars all the way down to the compiler so sidecar
    files render with the same context as the top-level YAML.  This is
    the structural piece that closes the gap exposed by H4 — even when
    callers bypass :func:`decl_api.load_project` and reach for
    :func:`loader.load_from_project` directly, sidecar templating MUST
    still work.
    """

    def test_load_from_project_renders_sql_sidecar(self, tmp_path: Path) -> None:
        sources = tmp_path / "sources"
        (sources / "entities").mkdir(parents=True)
        (sources / "datasources").mkdir(parents=True)
        (sources / "feature_views").mkdir(parents=True)
        (sources / "entities" / "USER.yaml").write_text(_ENTITY_YAML)
        (sources / "datasources" / "EVENTS.yaml").write_text(
            "kind: BatchSource\nname: EVENTS\nquery_file: EVENTS.sql\n"
            "columns:\n  - name: USER_ID\n    type: StringType\n"
        )
        (sources / "datasources" / "EVENTS.sql").write_text("SELECT * FROM RAW LIMIT {{ row_limit }}\n")

        batch = load_from_project(
            tmp_path,
            database="DB",
            schema="SCH",
            template_vars={"row_limit": 17},
        )

        ds = _get_spec(batch, "BatchSource", "EVENTS")
        assert "LIMIT 17" in ds.query


if __name__ == "__main__":
    pytest_driver.main()
