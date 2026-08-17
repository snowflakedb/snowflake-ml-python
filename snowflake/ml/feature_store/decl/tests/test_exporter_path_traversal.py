"""Tests — exporter sidecar writes reject path traversal (CWE-22).

``_extract_udf_to_py_file`` and ``_extract_query_to_sql_file`` derive the
sidecar filename from ``file_stem``, which originates from
attacker-influenceable recovered metadata (``metadata.name`` / SHOW output).
A crafted stem such as ``../../etc/crontab`` must be rejected with
``ValueError`` rather than writing content outside the export directory.
Legitimate identifier-shaped stems must still round-trip unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from snowflake.ml.feature_store.decl.exporter import (
    _extract_query_to_sql_file,
    _extract_udf_to_py_file,
    export_specs,
)

# Stems that must be rejected: parent-dir escapes, embedded separators, and
# anything outside the ``[A-Za-z0-9_-]`` allowlist.
_TRAVERSAL_STEMS = [
    "../../evil",
    "../escape",
    "a/b",
    "..",
    ".",
    "",
    "foo.bar",
    "with space",
]

# Stems that are legitimate SQL-identifier-shaped names and must keep working.
_VALID_STEMS = ["USER_CLICKS", "my-fv_1", "ABC123"]

_UDF_SOURCE = "def compute(df):\n    return df\n"
_QUERY_SOURCE = "SELECT * FROM events\n"


class TestExtractUdfPathTraversal:
    @pytest.mark.parametrize("stem", _TRAVERSAL_STEMS)
    def test_rejects_traversal_stem(self, tmp_path: Path, stem: str) -> None:
        fv_dir = tmp_path / "feature_views"
        fv_dir.mkdir()
        fv_doc: dict[str, Any] = {"udf": {"engine": "pandas", "function_definition": _UDF_SOURCE}}

        with pytest.raises(ValueError):
            _extract_udf_to_py_file(fv_doc, fv_dir, stem)

        # Nothing was written anywhere under the sandbox, and the udf block
        # was left untouched (not rewritten to a ``file`` reference).
        assert list(tmp_path.rglob("*.py")) == []
        assert fv_doc["udf"].get("function_definition") == _UDF_SOURCE
        assert "file" not in fv_doc["udf"]


class TestExtractUdfValidStem:
    @pytest.mark.parametrize("stem", _VALID_STEMS)
    def test_writes_and_rewrites_udf(self, tmp_path: Path, stem: str) -> None:
        fv_dir = tmp_path / "feature_views"
        fv_dir.mkdir()
        fv_doc: dict[str, Any] = {"udf": {"engine": "pandas", "function_definition": _UDF_SOURCE}}

        result = _extract_udf_to_py_file(fv_doc, fv_dir, stem)

        py_path = fv_dir / f"{stem}.py"
        assert py_path.exists()
        assert py_path.read_text() == _UDF_SOURCE
        assert result is not None and Path(result).name == f"{stem}.py"
        assert "function_definition" not in fv_doc["udf"]
        assert fv_doc["udf"]["file"] == f"{stem}.py"


class TestExtractQueryPathTraversal:
    @pytest.mark.parametrize("stem", _TRAVERSAL_STEMS)
    def test_rejects_traversal_stem(self, tmp_path: Path, stem: str) -> None:
        ds_dir = tmp_path / "datasources"
        ds_dir.mkdir()
        ds_doc: dict[str, Any] = {"query": _QUERY_SOURCE}

        with pytest.raises(ValueError):
            _extract_query_to_sql_file(ds_doc, ds_dir, stem)

        assert list(tmp_path.rglob("*.sql")) == []
        assert ds_doc.get("query") == _QUERY_SOURCE
        assert "query_file" not in ds_doc


class TestExtractQueryValidStem:
    @pytest.mark.parametrize("stem", _VALID_STEMS)
    def test_writes_and_rewrites_query(self, tmp_path: Path, stem: str) -> None:
        ds_dir = tmp_path / "datasources"
        ds_dir.mkdir()
        ds_doc: dict[str, Any] = {"query": _QUERY_SOURCE}

        result = _extract_query_to_sql_file(ds_doc, ds_dir, stem)

        sql_path = ds_dir / f"{stem}.sql"
        assert sql_path.exists()
        assert sql_path.read_text() == _QUERY_SOURCE
        assert result is not None and Path(result).name == f"{stem}.sql"
        assert "query" not in ds_doc
        assert ds_doc["query_file"] == f"{stem}.sql"


class TestExportSpecsRejectsTraversalName:
    def test_export_specs_raises_on_malicious_metadata_name(self, tmp_path: Path) -> None:
        full_spec: dict[str, Any] = {
            "kind": "StreamingFeatureView",
            "metadata": {
                "name": "../escape",
                "version": "V1",
                "database": "DB",
                "schema": "SCH",
            },
            "spec": {
                "ordered_entity_column_names": ["USER_ID"],
                "udf": {"engine": "pandas", "function_definition": _UDF_SOURCE},
            },
        }

        with pytest.raises(ValueError):
            export_specs(
                show_rows=[{"name": "MALICIOUS$V1$ONLINE", "database_name": "DB", "schema_name": "SCH"}],
                describe_rows_by_oft={},
                output_dir=str(tmp_path),
                database="DB",
                schema="SCH",
                specification_map={"MALICIOUS$V1$ONLINE": full_spec},
                entity_rows=[],
                layout="sources",
            )

        # The crafted name must not have written a sidecar outside the tree.
        assert list(tmp_path.rglob("escape.py")) == []
