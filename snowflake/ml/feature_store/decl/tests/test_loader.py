"""Tests for decl/loader.py — expand_input_files, process_file, load_python_file, load_specs."""

import os
import tempfile

import pytest

from snowflake.ml.feature_store.decl.loader import (
    _is_query_companion_sql,
    _is_udf_companion_py,
    expand_input_files,
    load_python_file,
    load_specs,
    process_file,
)

# ---------------------------------------------------------------------------
# expand_input_files
# ---------------------------------------------------------------------------


class TestExpandInputFiles:
    def test_py_file_included(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "entity.py")
            open(path, "w").close()
            result = expand_input_files([path])
        assert path in result

    def test_yaml_file_included(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "spec.yaml")
            open(path, "w").close()
            result = expand_input_files([path])
        assert path in result

    def test_yml_file_included(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "spec.yml")
            open(path, "w").close()
            result = expand_input_files([path])
        assert path in result

    def test_json_file_included(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "spec.json")
            open(path, "w").close()
            result = expand_input_files([path])
        assert path in result

    def test_recursive_glob_expands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "entities")
            os.makedirs(subdir)
            py_path = os.path.join(subdir, "customer.py")
            open(py_path, "w").close()
            result = expand_input_files([tmpdir + "/..."])
        assert py_path in result

    def test_recursive_glob_dotslash(self) -> None:
        """./... expands current directory recursively."""
        orig_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                py_path = os.path.join(tmpdir, "spec.py")
                open(py_path, "w").close()
                result = expand_input_files(["./..."])
            finally:
                os.chdir(orig_dir)
        assert any("spec.py" in r for r in result)

    def test_unsupported_extensions_excluded_from_glob(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            py_path = os.path.join(tmpdir, "spec.py")
            txt_path = os.path.join(tmpdir, "readme.txt")
            open(py_path, "w").close()
            open(txt_path, "w").close()
            result = expand_input_files([tmpdir + "/..."])
        assert py_path in result
        assert txt_path not in result

    def test_bare_directory_expands_recursively(self) -> None:
        """A bare directory path (no '/...') is auto-expanded to a recursive walk.

        Previously, ``glob.glob`` returned the directory path itself and the
        loader silently dropped it via the bare ``except Exception: pass`` in
        :func:`load_specs`, producing an empty SpecBatch and a misleading
        "no changes" plan.  The fix rewrites bare directories to ``<dir>/...``
        so the recursive-walk branch picks up every spec file inside.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            entities_dir = os.path.join(tmpdir, "entities")
            fv_dir = os.path.join(tmpdir, "feature_views")
            os.makedirs(entities_dir)
            os.makedirs(fv_dir)
            ent_path = os.path.join(entities_dir, "customer.yaml")
            fv_path = os.path.join(fv_dir, "stats.yaml")
            open(ent_path, "w").close()
            open(fv_path, "w").close()
            result = expand_input_files([tmpdir])
        assert ent_path in result, result
        assert fv_path in result, result

    def test_bare_directory_with_trailing_slash_expands_recursively(self) -> None:
        """Trailing-slash directory paths are also auto-expanded recursively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = os.path.join(tmpdir, "feature_views")
            os.makedirs(sub)
            fv_path = os.path.join(sub, "fv.yaml")
            open(fv_path, "w").close()
            result = expand_input_files([tmpdir + "/"])
        assert fv_path in result, result

    def test_specific_file_not_expanded(self) -> None:
        """A path pointing to a real file is loaded as-is, not as a directory walk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "only.yaml")
            open(path, "w").close()
            other = os.path.join(tmpdir, "other.yaml")
            open(other, "w").close()
            result = expand_input_files([path])
        assert path in result
        assert other not in result


# ---------------------------------------------------------------------------
# process_file: YAML
# ---------------------------------------------------------------------------


class TestProcessFileYaml:
    def _write_yaml(self, tmpdir: str, content: str, name: str = "spec.yaml") -> str:
        path = os.path.join(tmpdir, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_simple_yaml_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_yaml(
                tmpdir,
                "kind: Entity\nname: customer\njoin_keys:\n  - name: customer_id\n    type: str\n",
            )
            results = process_file(path)
        assert len(results) == 1
        assert results[0]["kind"] == "Entity"
        assert results[0]["name"] == "customer"

    def test_yaml_type_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_yaml(
                tmpdir,
                "kind: Entity\nname: e\njoin_keys:\n  - name: id\n    type: str\n",
            )
            results = process_file(path)
        assert results[0]["join_keys"][0]["type"] == "StringType"

    def test_yaml_duration_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_yaml(
                tmpdir,
                "kind: StreamingFeatureView\nname: fv\nfeature_granularity: 5m\n",
            )
            results = process_file(path)
        assert results[0]["feature_granularity_sec"] == 300

    def test_yml_extension_also_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_yaml(
                tmpdir,
                "kind: Entity\nname: e\n",
                name="spec.yml",
            )
            results = process_file(path)
        assert results[0]["kind"] == "Entity"

    def test_template_yaml_with_config_rendered(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_yaml(
                tmpdir,
                'kind: Entity\nname: "{{ entity_name }}"\n',
            )
            results = process_file(path, config_source='{"entity_name": "customer"}')
        assert results[0]["name"] == "customer"

    def test_template_yaml_without_config_raises(self) -> None:
        import pytest

        from snowflake.ml.feature_store.decl.errors import SpecLoadError

        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_yaml(tmpdir, "kind: Entity\nname: {{ entity_name }}\n")
            with pytest.raises(SpecLoadError):
                process_file(path)


# ---------------------------------------------------------------------------
# process_file: JSON
# ---------------------------------------------------------------------------


class TestProcessFileJson:
    def test_simple_json_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "spec.json")
            with open(path, "w") as f:
                f.write('{"kind": "Entity", "name": "customer"}')
            results = process_file(path)
        assert len(results) == 1
        assert results[0]["kind"] == "Entity"

    def test_json_type_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "spec.json")
            with open(path, "w") as f:
                f.write('{"kind": "Entity", "join_keys": [{"name": "id", "type": "int"}]}')
            results = process_file(path)
        assert results[0]["join_keys"][0]["type"] == "LongType"


# ---------------------------------------------------------------------------
# load_python_file
# ---------------------------------------------------------------------------


class TestLoadPythonFile:
    def test_loads_pydantic_spec_instance(self) -> None:
        code = """
from snowflake.ml.feature_store.decl.spec_models import Entity, FSColumn

customer_entity = Entity(
    name="customer",
    join_keys=[FSColumn(name="customer_id", type="StringType")],
)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "customer_entity.py")
            with open(path, "w") as f:
                f.write(code)
            objects = load_python_file(path)
        assert len(objects) == 1
        name, obj = objects[0]
        assert name == "customer_entity"
        assert obj.name == "customer"

    def test_imports_not_included(self) -> None:
        """Imported specs should not be returned as locally defined."""
        entity_code = """
from snowflake.ml.feature_store.decl.spec_models import Entity, FSColumn
customer_entity = Entity(name="customer", join_keys=[FSColumn(name="id", type="StringType")])
"""
        fv_code = """
import sys
from snowflake.ml.feature_store.decl.spec_models import Entity, FSColumn, FeatureView, SourceRef
from customer_entity import customer_entity

my_fv = FeatureView(
    name="my_fv",
    kind="StreamingFeatureView",
    sources=[SourceRef(name="src", source_type="Stream")],
    entities=["customer_id"],
)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            entity_path = os.path.join(tmpdir, "customer_entity.py")
            fv_path = os.path.join(tmpdir, "my_fv.py")
            with open(entity_path, "w") as f:
                f.write(entity_code)
            with open(fv_path, "w") as f:
                f.write(fv_code)
            # load_python_file manages sys.path internally — no manual management needed
            objects = load_python_file(fv_path)

        names = [name for name, _ in objects]
        assert "my_fv" in names
        assert "customer_entity" not in names

    def test_sys_path_not_polluted_after_load(self) -> None:
        """load_python_file must not permanently mutate sys.path."""
        import sys as _sys

        code = """
from snowflake.ml.feature_store.decl.spec_models import Entity, FSColumn
my_entity = Entity(name="e", join_keys=[FSColumn(name="id", type="StringType")])
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "my_entity.py")
            with open(path, "w") as f:
                f.write(code)
            path_before = list(_sys.path)
            load_python_file(path)
            path_after = list(_sys.path)
        assert path_after == path_before, f"sys.path was polluted: added {set(path_after) - set(path_before)}"


# ---------------------------------------------------------------------------
# process_file: Python
# ---------------------------------------------------------------------------


class TestProcessFilePython:
    def test_python_entity_loaded_and_compiled(self) -> None:
        code = """
from snowflake.ml.feature_store.decl.spec_models import Entity, FSColumn

my_entity = Entity(
    name="user",
    join_keys=[FSColumn(name="user_id", type="str")],
)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "my_entity.py")
            with open(path, "w") as f:
                f.write(code)
            results = process_file(path)

        assert len(results) == 1
        # compile_spec should have normalized the type
        assert results[0]["join_keys"][0]["type"] == "StringType"


# ---------------------------------------------------------------------------
# load_specs
# ---------------------------------------------------------------------------


class TestLoadSpecs:
    def test_returns_spec_batch(self) -> None:
        from snowflake.ml.feature_store.decl.types import SpecBatch

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "entity.yaml")
            with open(path, "w") as f:
                f.write("kind: Entity\nname: customer\n")
            batch = load_specs([path])

        assert isinstance(batch, SpecBatch)
        assert len(batch.specs) == 1
        assert len(batch.source_files) == 1

    def test_multiple_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = os.path.join(tmpdir, "e1.yaml")
            p2 = os.path.join(tmpdir, "e2.yaml")
            with open(p1, "w") as f:
                f.write("kind: Entity\nname: user\n")
            with open(p2, "w") as f:
                f.write("kind: Entity\nname: customer\n")
            batch = load_specs([p1, p2])

        assert len(batch.specs) == 2

    def test_source_files_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "e.yaml")
            with open(path, "w") as f:
                f.write("kind: Entity\nname: e\n")
            batch = load_specs([path])

        assert path in batch.source_files

    def test_warns_on_invalid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bad = os.path.join(tmpdir, "bad.yaml")
            with open(bad, "w") as f:
                f.write("invalid: [unclosed")
            with pytest.warns(UserWarning, match=bad.replace("\\", "\\\\")):
                batch = load_specs([bad])
        assert batch.specs == []


# ---------------------------------------------------------------------------
# _is_query_companion_sql — mirrors _is_udf_companion_py for SQL sidecars.
# ---------------------------------------------------------------------------


class TestIsQueryCompanionSql:
    """A ``<NAME>.sql`` file is treated as a sidecar for a sibling
    ``BatchSource`` YAML when that YAML's top-level ``query_file:`` basename
    equals the ``.sql`` filename. The compiler's ``inline_query_source``
    reads the file as plain text (Phase 3); the loader must skip it so it
    is not silently routed through ``yaml.safe_load`` / ``json.loads``.
    """

    def _write(self, dirpath: str, name: str, content: str) -> str:
        path = os.path.join(dirpath, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_companion_sql_matched_by_sibling_yaml_query_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = self._write(tmpdir, "events.sql", "SELECT * FROM EVENTS\n")
            self._write(
                tmpdir,
                "events.yaml",
                "kind: BatchSource\nname: events\nquery_file: events.sql\n",
            )
            assert _is_query_companion_sql(sql_path) is True

    def test_companion_sql_matched_by_sibling_yml_query_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = self._write(tmpdir, "events.sql", "SELECT * FROM EVENTS\n")
            self._write(
                tmpdir,
                "events.yml",
                "kind: BatchSource\nname: events\nquery_file: events.sql\n",
            )
            assert _is_query_companion_sql(sql_path) is True

    def test_companion_detection_is_basename_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = self._write(tmpdir, "events.sql", "SELECT 1\n")
            self._write(
                tmpdir,
                "events.yaml",
                "kind: BatchSource\nname: events\nquery_file: ./events.sql\n",
            )
            assert _is_query_companion_sql(sql_path) is True

    def test_orphan_sql_without_yaml_is_not_companion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = self._write(tmpdir, "stray.sql", "SELECT 1\n")
            assert _is_query_companion_sql(sql_path) is False

    def test_yaml_referencing_different_sql_is_not_companion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = self._write(tmpdir, "events.sql", "SELECT 1\n")
            self._write(
                tmpdir,
                "events.yaml",
                "kind: BatchSource\nname: events\nquery_file: other.sql\n",
            )
            assert _is_query_companion_sql(sql_path) is False

    def test_yaml_without_query_file_is_not_companion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = self._write(tmpdir, "events.sql", "SELECT 1\n")
            self._write(
                tmpdir,
                "events.yaml",
                "kind: BatchSource\nname: events\ntable: EVENTS\n",
            )
            assert _is_query_companion_sql(sql_path) is False

    def test_unparsable_yaml_is_not_companion(self) -> None:
        """Jinja2 placeholders or malformed YAML must not trip the helper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = self._write(tmpdir, "events.sql", "SELECT 1\n")
            self._write(
                tmpdir,
                "events.yaml",
                "kind: BatchSource\nname: {{ source_name }}\nquery_file: events.sql\n",
            )
            assert _is_query_companion_sql(sql_path) is False

    def test_helper_is_symmetric_with_udf_companion_py(self) -> None:
        """The two helpers share an identical contract — same input shape,
        same return type, same behaviour for orphans / mismatches. Pin both
        in one place so the symmetry is obvious."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sql_path = self._write(tmpdir, "events.sql", "SELECT 1\n")
            py_path = self._write(tmpdir, "compute.py", "def fn(df):\n    return df\n")
            self._write(
                tmpdir,
                "events.yaml",
                "kind: BatchSource\nname: events\nquery_file: events.sql\n",
            )
            self._write(
                tmpdir,
                "compute.yaml",
                (
                    "kind: StreamingFeatureView\nname: compute\n"
                    "entities:\n  - id\n"
                    "udf:\n  name: fn\n  file: compute.py\n"
                ),
            )
            assert _is_query_companion_sql(sql_path) is True
            assert _is_udf_companion_py(py_path) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
