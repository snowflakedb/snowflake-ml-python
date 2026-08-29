"""Phase 1 RED tests — ``load_from_project`` walks ``sources/feature_groups/``.

Pins the contract that a project tree containing
``<root>/sources/feature_groups/<NAME>.yaml`` is loaded as a fourth canonical
sub-directory in deterministic order (after ``feature_views``), yielding
``FeatureGroup`` Pydantic models.
"""

from __future__ import annotations

from pathlib import Path

from snowflake.ml.feature_store.decl.spec_models import FeatureGroup
from snowflake.ml.feature_store.decl.types import SpecBatch
from snowflake.ml.test_utils import pytest_driver

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
_FG_YAML = (
    "kind: FeatureGroup\n"
    "name: USER_FRAUD_FG\n"
    "version: V1\n"
    "desc: Combined user signals.\n"
    "auto_prefix: true\n"
    "feature_views:\n"
    "  - name: user_click_stats\n"
    "    version: V1\n"
)


def _make_project_with_fg(project_root: Path) -> None:
    """Lay out a minimal project tree with a ``feature_groups/`` subdir."""
    sources = project_root / "sources"
    for sub in ("entities", "datasources", "feature_views", "feature_groups"):
        (sources / sub).mkdir(parents=True, exist_ok=True)
    (sources / "entities" / "customer.yaml").write_text(_ENTITY_YAML)
    (sources / "datasources" / "clickstream.yaml").write_text(_DATASOURCE_YAML)
    (sources / "feature_views" / "user_click_stats.yaml").write_text(_FV_YAML)
    (sources / "feature_groups" / "USER_FRAUD_FG.yaml").write_text(_FG_YAML)


# ---------------------------------------------------------------------------
# load_from_project — feature_groups/ walk
# ---------------------------------------------------------------------------


class TestLoadFromProjectFeatureGroups:
    def test_includes_feature_group(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_project_with_fg(tmp_path)
        batch = load_from_project(tmp_path, database="MY_DB", schema="MY_SCHEMA")

        assert isinstance(batch, SpecBatch)
        kinds = [s.kind for s in batch.specs]
        assert "FeatureGroup" in kinds

    def test_feature_group_appears_after_feature_view(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_project_with_fg(tmp_path)
        batch = load_from_project(tmp_path, database="MY_DB", schema="MY_SCHEMA")
        kinds = [s.kind for s in batch.specs]
        # FG sub-directory is walked AFTER feature_views so the file order
        # already reflects topological order before any further sorting.
        assert kinds.index("StreamingFeatureView") < kinds.index("FeatureGroup")

    def test_feature_group_parses_to_pydantic_model(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_project_with_fg(tmp_path)
        batch = load_from_project(tmp_path, database="MY_DB", schema="MY_SCHEMA")

        fg_specs = [s for s in batch.specs if s.kind == "FeatureGroup"]
        assert len(fg_specs) == 1
        fg = fg_specs[0]
        assert isinstance(fg, FeatureGroup)
        assert fg.name == "USER_FRAUD_FG"
        assert fg.version == "V1"
        assert fg.desc == "Combined user signals."
        assert fg.auto_prefix is True
        assert len(fg.feature_views) == 1
        assert fg.feature_views[0].name == "user_click_stats"
        assert fg.feature_views[0].version == "V1"

    def test_missing_feature_groups_subdir_is_silently_ok(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        # Lay out the legacy three subdirs only; no feature_groups/.
        sources = tmp_path / "sources"
        for sub in ("entities", "datasources", "feature_views"):
            (sources / sub).mkdir(parents=True, exist_ok=True)
        (sources / "entities" / "customer.yaml").write_text(_ENTITY_YAML)
        (sources / "datasources" / "clickstream.yaml").write_text(_DATASOURCE_YAML)
        (sources / "feature_views" / "user_click_stats.yaml").write_text(_FV_YAML)

        batch = load_from_project(tmp_path, database="MY_DB", schema="MY_SCHEMA")
        kinds = [s.kind for s in batch.specs]
        assert "FeatureGroup" not in kinds
        assert "Entity" in kinds


class TestLoadFromProjectFeatureGroupsInjection:
    def test_database_schema_injected_from_kwargs(self, tmp_path: Path) -> None:
        from snowflake.ml.feature_store.decl.loader import load_from_project

        _make_project_with_fg(tmp_path)
        batch = load_from_project(tmp_path, database="MY_DB", schema="MY_SCHEMA")

        fg_specs = [s for s in batch.specs if s.kind == "FeatureGroup"]
        fg = fg_specs[0]
        assert fg.database == "MY_DB"
        assert fg.schema_ == "MY_SCHEMA"


if __name__ == "__main__":
    pytest_driver.main()
