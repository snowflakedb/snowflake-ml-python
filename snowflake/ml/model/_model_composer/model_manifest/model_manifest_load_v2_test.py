import os
import pathlib
import tempfile

import yaml
from absl.testing import absltest

from snowflake.ml.model._model_composer.model_manifest import (
    model_manifest,
    model_manifest_v2_schema,
)


class ModelManifestLoadV2Test(absltest.TestCase):
    def test_load_v2_minimal(self) -> None:
        raw = {
            "manifest_version": model_manifest_v2_schema.MODEL_MANIFEST_VERSION_V2,
            "model_extension_spec": model_manifest_v2_schema.DEFAULT_MODEL_EXTENSION_SPEC_PATH,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(
                os.path.join(tmpdir, model_manifest.ModelManifest.MANIFEST_FILE_REL_PATH), "w", encoding="utf-8"
            ) as f:
                yaml.safe_dump(raw, f)

            loaded = model_manifest.ModelManifest(pathlib.Path(tmpdir)).load_v2()

            self.assertEqual(loaded["manifest_version"], "2.0")
            self.assertEqual(loaded["model_extension_spec"], "model/model.yaml")

    def test_load_v2_with_lineage_sources(self) -> None:
        raw = {
            "manifest_version": "2.0",
            "model_extension_spec": "model/model.yaml",
            "lineage_sources": [{"type": "DATASET", "entity": "db.schema.ds", "version": "v1"}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(
                os.path.join(tmpdir, model_manifest.ModelManifest.MANIFEST_FILE_REL_PATH), "w", encoding="utf-8"
            ) as f:
                yaml.safe_dump(raw, f)

            loaded = model_manifest.ModelManifest(pathlib.Path(tmpdir)).load_v2()

            self.assertEqual(len(loaded["lineage_sources"]), 1)

    def test_load_v2_rejects_numeric_version(self) -> None:
        # Unquoted YAML 2.0 loads as float; the server requires a string version.
        raw = {
            "manifest_version": 2.0,
            "model_extension_spec": model_manifest_v2_schema.DEFAULT_MODEL_EXTENSION_SPEC_PATH,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(
                os.path.join(tmpdir, model_manifest.ModelManifest.MANIFEST_FILE_REL_PATH), "w", encoding="utf-8"
            ) as f:
                yaml.safe_dump(raw, f)

            with self.assertRaisesRegex(ValueError, r"manifest_version must be a string"):
                model_manifest.ModelManifest(pathlib.Path(tmpdir)).load_v2()

    def test_load_v2_rejects_v1(self) -> None:
        raw = {"manifest_version": "1.0", "model_extension_spec": "model/model.yaml"}
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(
                os.path.join(tmpdir, model_manifest.ModelManifest.MANIFEST_FILE_REL_PATH), "w", encoding="utf-8"
            ) as f:
                yaml.safe_dump(raw, f)

            with self.assertRaisesRegex(ValueError, r"MANIFEST version is 1\.0; use ModelManifest\.load\(\)"):
                model_manifest.ModelManifest(pathlib.Path(tmpdir)).load_v2()

    def test_load_v2_rejects_missing_version(self) -> None:
        raw = {"model_extension_spec": "model/model.yaml"}
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(
                os.path.join(tmpdir, model_manifest.ModelManifest.MANIFEST_FILE_REL_PATH), "w", encoding="utf-8"
            ) as f:
                yaml.safe_dump(raw, f)

            with self.assertRaisesRegex(ValueError, "Unable to get the version of the MANIFEST file"):
                model_manifest.ModelManifest(pathlib.Path(tmpdir)).load_v2()

    def test_load_v2_rejects_missing_extension_spec(self) -> None:
        raw = {"manifest_version": "2.0"}
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(
                os.path.join(tmpdir, model_manifest.ModelManifest.MANIFEST_FILE_REL_PATH), "w", encoding="utf-8"
            ) as f:
                yaml.safe_dump(raw, f)

            with self.assertRaisesRegex(ValueError, "model_extension_spec must be a non-empty string"):
                model_manifest.ModelManifest(pathlib.Path(tmpdir)).load_v2()

    def test_load_v2_rejects_blank_extension_spec(self) -> None:
        raw = {"manifest_version": "2.0", "model_extension_spec": "  "}
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(
                os.path.join(tmpdir, model_manifest.ModelManifest.MANIFEST_FILE_REL_PATH), "w", encoding="utf-8"
            ) as f:
                yaml.safe_dump(raw, f)

            with self.assertRaisesRegex(ValueError, "model_extension_spec must be a non-empty string"):
                model_manifest.ModelManifest(pathlib.Path(tmpdir)).load_v2()

    def test_load_v2_rejects_non_list_lineage_sources(self) -> None:
        raw = {
            "manifest_version": "2.0",
            "model_extension_spec": "model/model.yaml",
            "lineage_sources": {"type": "DATASET"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(
                os.path.join(tmpdir, model_manifest.ModelManifest.MANIFEST_FILE_REL_PATH), "w", encoding="utf-8"
            ) as f:
                yaml.safe_dump(raw, f)

            with self.assertRaisesRegex(ValueError, "lineage_sources must be a list"):
                model_manifest.ModelManifest(pathlib.Path(tmpdir)).load_v2()

    def test_load_rejects_v2_envelope(self) -> None:
        raw = {
            "manifest_version": "2.0",
            "model_extension_spec": "model/model.yaml",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(
                os.path.join(tmpdir, model_manifest.ModelManifest.MANIFEST_FILE_REL_PATH), "w", encoding="utf-8"
            ) as f:
                yaml.safe_dump(raw, f)

            with self.assertRaisesRegex(
                ValueError,
                r"MANIFEST version is 2\.0; ModelManifest\.load\(\) reads version 1\.0 manifests\.",
            ):
                model_manifest.ModelManifest(pathlib.Path(tmpdir)).load()

    def test_load_rejects_version_other_than_v1(self) -> None:
        raw = {"manifest_version": "3.0", "methods": [], "runtimes": {}}
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(
                os.path.join(tmpdir, model_manifest.ModelManifest.MANIFEST_FILE_REL_PATH), "w", encoding="utf-8"
            ) as f:
                yaml.safe_dump(raw, f)

            with self.assertRaisesRegex(ValueError, r"MANIFEST version is 3\.0; ModelManifest\.load\(\) reads version"):
                model_manifest.ModelManifest(pathlib.Path(tmpdir)).load()

    def test_get_manifest_version_and_dual_read_branch(self) -> None:
        v1 = {"manifest_version": "1.0", "runtimes": {}, "methods": []}
        v2 = {
            "manifest_version": "2.0",
            "model_extension_spec": "model/custom_spec.yaml",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, model_manifest.ModelManifest.MANIFEST_FILE_REL_PATH)
            mm = model_manifest.ModelManifest(pathlib.Path(tmpdir))

            with open(manifest_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(v1, f)
            self.assertEqual(mm.get_manifest_version(), "1.0")
            self.assertEqual(mm.load()["methods"], [])

            with open(manifest_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(v2, f)
            self.assertEqual(mm.get_manifest_version(), "2.0")
            self.assertEqual(mm.load_v2()["model_extension_spec"], "model/custom_spec.yaml")

    def test_get_manifest_version_uses_provided_mapping(self) -> None:
        mm = model_manifest.ModelManifest(pathlib.Path("/unused"))

        self.assertEqual(mm.get_manifest_version({"manifest_version": "2.0"}), "2.0")

    def test_get_manifest_version_rejects_empty_string(self) -> None:
        mm = model_manifest.ModelManifest(pathlib.Path("/unused"))

        with self.assertRaisesRegex(ValueError, "Unable to get the version of the MANIFEST file."):
            mm.get_manifest_version({"manifest_version": ""})

    def test_load_rejects_non_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(
                os.path.join(tmpdir, model_manifest.ModelManifest.MANIFEST_FILE_REL_PATH), "w", encoding="utf-8"
            ) as f:
                yaml.safe_dump(["not", "a", "manifest"], f)

            with self.assertRaisesRegex(ValueError, "Read ill-formatted model MANIFEST, should be a dict"):
                model_manifest.ModelManifest(pathlib.Path(tmpdir)).load()

    def test_load_v2_rejects_non_string_extension_spec(self) -> None:
        raw = {"manifest_version": "2.0", "model_extension_spec": 1}
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(
                os.path.join(tmpdir, model_manifest.ModelManifest.MANIFEST_FILE_REL_PATH), "w", encoding="utf-8"
            ) as f:
                yaml.safe_dump(raw, f)

            with self.assertRaisesRegex(ValueError, "manifest: model_extension_spec must be a non-empty string."):
                model_manifest.ModelManifest(pathlib.Path(tmpdir)).load_v2()

    def test_load_v2_accepts_empty_lineage_sources(self) -> None:
        raw = {
            "manifest_version": "2.0",
            "model_extension_spec": "model/model.yaml",
            "lineage_sources": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(
                os.path.join(tmpdir, model_manifest.ModelManifest.MANIFEST_FILE_REL_PATH), "w", encoding="utf-8"
            ) as f:
                yaml.safe_dump(raw, f)

            loaded = model_manifest.ModelManifest(pathlib.Path(tmpdir)).load_v2()

            self.assertEqual(loaded["lineage_sources"], [])


if __name__ == "__main__":
    absltest.main()
