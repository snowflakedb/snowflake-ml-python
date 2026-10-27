"""Tests — exporter sidecar writes are path-traversal-safe (CWE-22).

``_extract_udf_to_py_file`` (FV UDF body) stays on the strict
``_safe_sidecar_path`` allowlist and *rejects* a crafted ``file_stem`` such as
``../../etc/crontab`` with ``ValueError`` — its stem doubles as an exported FV
identity and must not be silently rewritten.

``_extract_query_to_sql_file`` (datasource query body) instead *encodes* the
stem via ``_encode_source_stem`` (output alphabet ``[A-Za-z0-9_]``), so a name
carrying ``.`` / ``-`` / ``/`` / ``$`` becomes a writable, traversal-proof
filename rather than raising — a datasource's identity lives in its ``name``
field, so the filename is pure transport.  The encoded output contains no path
separators or ``..`` sequences, so content can never escape the export dir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from snowflake.ml.feature_store.decl.exporter import (
    _encode_source_stem,
    _extract_query_to_sql_file,
    _extract_udf_to_py_file,
    export_specs,
)
from snowflake.ml.test_utils import pytest_driver

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
    # The datasource query sidecar ENCODES rather than rejects: a crafted stem
    # becomes a safe, in-directory filename.  ``""`` is the one degenerate case
    # that still raises (an empty stem has no valid encoding).
    @pytest.mark.parametrize("stem", [s for s in _TRAVERSAL_STEMS if s != ""])
    def test_encodes_unsafe_stem_without_escaping(self, tmp_path: Path, stem: str) -> None:
        ds_dir = tmp_path / "datasources"
        ds_dir.mkdir()
        ds_doc: dict[str, Any] = {"query": _QUERY_SOURCE}

        result = _extract_query_to_sql_file(ds_doc, ds_dir, stem)

        # Written, encoded, and strictly inside ds_dir (no traversal).
        assert result is not None
        written = Path(result)
        assert written.parent == ds_dir, f"sidecar escaped export dir: {written}"
        expected_name = f"{_encode_source_stem(stem)}.sql"
        assert written.name == expected_name
        assert written.read_text() == _QUERY_SOURCE
        assert ds_doc["query_file"] == expected_name
        # No file landed outside the datasources dir.
        assert [p for p in tmp_path.rglob("*.sql") if p.parent != ds_dir] == []

    def test_empty_stem_still_raises(self, tmp_path: Path) -> None:
        ds_dir = tmp_path / "datasources"
        ds_dir.mkdir()
        ds_doc: dict[str, Any] = {"query": _QUERY_SOURCE}

        with pytest.raises(ValueError):
            _extract_query_to_sql_file(ds_doc, ds_dir, "")
        assert ds_doc.get("query") == _QUERY_SOURCE
        assert "query_file" not in ds_doc


class TestExtractQueryValidStem:
    @pytest.mark.parametrize("stem", _VALID_STEMS)
    def test_writes_and_rewrites_query(self, tmp_path: Path, stem: str) -> None:
        ds_dir = tmp_path / "datasources"
        ds_dir.mkdir()
        ds_doc: dict[str, Any] = {"query": _QUERY_SOURCE}

        result = _extract_query_to_sql_file(ds_doc, ds_dir, stem)

        # ``-`` is encoded (``my-fv_1`` -> ``my_2Dfv_1``); all-``[A-Za-z0-9_]``
        # stems pass through unchanged.
        expected_name = f"{_encode_source_stem(stem)}.sql"
        sql_path = ds_dir / expected_name
        assert sql_path.exists()
        assert sql_path.read_text() == _QUERY_SOURCE
        assert result is not None and Path(result).name == expected_name
        assert "query" not in ds_doc
        assert ds_doc["query_file"] == expected_name


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
                # A stream source is required or the source-less export guard
                # would warn-and-skip this FV before the malicious stem reaches
                # the sidecar-write path that must reject it.
                "sources": [{"name": "USER_EVENTS", "source_type": "Stream"}],
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


if __name__ == "__main__":
    pytest_driver.main()
