import os
import pathlib
import tempfile
from unittest import mock

import importlib_resources
import yaml
from absl.testing import absltest

from snowflake.ml._internal import env_utils
from snowflake.ml.model import model_signature, type_hints
from snowflake.ml.model._model_composer.model_manifest import model_manifest
from snowflake.ml.model._packager.model_meta import model_blob_meta, model_meta

_DUMMY_SIG = {
    "predict": model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input_1"),
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input_2", shape=(-1,)),
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input_3", shape=(-1,)),
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input_4", shape=(-1,)),
        ],
        outputs=[
            model_signature.FeatureSpec(name="output_1", dtype=model_signature.DataType.FLOAT),
            model_signature.FeatureSpec(name="output_2", dtype=model_signature.DataType.FLOAT, shape=(2, 2)),
            model_signature.FeatureSpec(name="output_3", dtype=model_signature.DataType.FLOAT, shape=(2, 2)),
            model_signature.FeatureSpec(name="output_4", dtype=model_signature.DataType.FLOAT, shape=(2, 2)),
        ],
    )
}


_DUMMY_BLOB = model_blob_meta.ModelBlobMeta(
    name="model1", model_type="custom", path="mock_path", handler_version="version_0"
)


class ModelManifestTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock.MagicMock()
        self.mock_to_use_released_snowml = mock.patch.object(
            env_utils,
            "get_matched_package_versions_in_information_schema_with_active_session",
            return_value={env_utils.SNOWPARK_ML_PKG_NAME: [""]},
        )
        self.mock_to_use_local_snowml = mock.patch.object(
            env_utils,
            "get_matched_package_versions_in_information_schema_with_active_session",
            return_value={env_utils.SNOWPARK_ML_PKG_NAME: []},
        )

    def test_model_manifest_1(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with self.mock_to_use_released_snowml:
                with model_meta.create_model_metadata(
                    model_dir_path=tmpdir,
                    name="model1",
                    model_type="custom",
                    signatures=_DUMMY_SIG,
                    python_version="3.8",
                ) as meta:
                    meta.models["model1"] = _DUMMY_BLOB

                mm.save(self.m_session, meta, pathlib.PurePosixPath("model.zip"))
                with open(os.path.join(workspace, "MANIFEST.yml"), encoding="utf-8") as f:
                    self.assertEqual(
                        (
                            importlib_resources.files("snowflake.ml.model._model_composer.model_manifest")
                            .joinpath("fixtures")
                            .joinpath("MANIFEST_1.yml")
                            .read_text()
                        ),
                        f.read(),
                    )
                with open(pathlib.Path(workspace, "functions", "predict.py"), encoding="utf-8") as f:
                    self.assertEqual(
                        (
                            importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                            .joinpath("fixtures")
                            .joinpath("function_1.py")
                            .read_text()
                        ),
                        f.read(),
                    )

    def test_model_manifest_2(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with self.mock_to_use_local_snowml:
                with model_meta.create_model_metadata(
                    model_dir_path=tmpdir,
                    name="model1",
                    model_type="custom",
                    signatures={"__call__": _DUMMY_SIG["predict"]},
                    python_version="3.8",
                ) as meta:
                    meta.models["model1"] = _DUMMY_BLOB

                mm.save(
                    self.m_session,
                    meta,
                    pathlib.PurePosixPath("model.zip"),
                    options=type_hints.BaseModelSaveOption(
                        method_options={"__call__": type_hints.ModelMethodSaveOptions(max_batch_size=10)}
                    ),
                )
                with open(os.path.join(workspace, "MANIFEST.yml"), encoding="utf-8") as f:
                    self.assertEqual(
                        (
                            importlib_resources.files("snowflake.ml.model._model_composer.model_manifest")
                            .joinpath("fixtures")
                            .joinpath("MANIFEST_2.yml")
                            .read_text()
                        ),
                        f.read(),
                    )
                with open(pathlib.Path(workspace, "functions", "__call__.py"), encoding="utf-8") as f:
                    self.assertEqual(
                        (
                            importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                            .joinpath("fixtures")
                            .joinpath("function_2.py")
                            .read_text()
                        ),
                        f.read(),
                    )

    def test_model_manifest_mix(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with self.mock_to_use_local_snowml:
                with model_meta.create_model_metadata(
                    model_dir_path=tmpdir,
                    name="model1",
                    model_type="custom",
                    signatures={"predict": _DUMMY_SIG["predict"], "__call__": _DUMMY_SIG["predict"]},
                    python_version="3.8",
                ) as meta:
                    meta.models["model1"] = _DUMMY_BLOB

                mm.save(
                    self.m_session,
                    meta,
                    pathlib.PurePosixPath("model.zip"),
                    options=type_hints.BaseModelSaveOption(
                        method_options={
                            "predict": type_hints.ModelMethodSaveOptions(case_sensitive=True),
                            "__call__": type_hints.ModelMethodSaveOptions(max_batch_size=10),
                        }
                    ),
                )
                with open(os.path.join(workspace, "MANIFEST.yml"), encoding="utf-8") as f:
                    self.assertEqual(
                        (
                            importlib_resources.files("snowflake.ml.model._model_composer.model_manifest")
                            .joinpath("fixtures")
                            .joinpath("MANIFEST_3.yml")
                            .read_text()
                        ),
                        f.read(),
                    )
                with open(pathlib.Path(workspace, "functions", "predict.py"), encoding="utf-8") as f:
                    self.assertEqual(
                        (
                            importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                            .joinpath("fixtures")
                            .joinpath("function_1.py")
                            .read_text()
                        ),
                        f.read(),
                    )
                with open(pathlib.Path(workspace, "functions", "__call__.py"), encoding="utf-8") as f:
                    self.assertEqual(
                        (
                            importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                            .joinpath("fixtures")
                            .joinpath("function_2.py")
                            .read_text()
                        ),
                        f.read(),
                    )

    def test_model_manifest_bad(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with self.mock_to_use_local_snowml:
                with model_meta.create_model_metadata(
                    model_dir_path=tmpdir,
                    name="model1",
                    model_type="custom",
                    signatures={"predict": _DUMMY_SIG["predict"], "PREDICT": _DUMMY_SIG["predict"]},
                ) as meta:
                    meta.models["model1"] = _DUMMY_BLOB

                with self.assertRaisesRegex(
                    ValueError, "Found duplicate method named resolved as PREDICT in the model."
                ):
                    mm.save(
                        self.m_session,
                        meta,
                        pathlib.PurePosixPath("model.zip"),
                    )

    def test_model_manifest_table_function(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with self.mock_to_use_local_snowml:
                with model_meta.create_model_metadata(
                    model_dir_path=tmpdir,
                    name="model1",
                    model_type="custom",
                    signatures={"predict": _DUMMY_SIG["predict"]},
                    python_version="3.8",
                ) as meta:
                    meta.models["model1"] = _DUMMY_BLOB

                mm.save(
                    self.m_session,
                    meta,
                    pathlib.PurePosixPath("model.zip"),
                    options=type_hints.BaseModelSaveOption(
                        method_options={"predict": type_hints.ModelMethodSaveOptions(function_type="TABLE_FUNCTION")}
                    ),
                )
                with open(os.path.join(workspace, "MANIFEST.yml"), encoding="utf-8") as f:
                    self.assertEqual(
                        (
                            importlib_resources.files("snowflake.ml.model._model_composer.model_manifest")
                            .joinpath("fixtures")
                            .joinpath("MANIFEST_4.yml")
                            .read_text()
                        ),
                        f.read(),
                    )
                with open(pathlib.Path(workspace, "functions", "predict.py"), encoding="utf-8") as f:
                    self.assertEqual(
                        (
                            importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                            .joinpath("fixtures")
                            .joinpath("function_3.py")
                            .read_text()
                        ),
                        f.read(),
                    )

    def test_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "MANIFEST.yml"), "w", encoding="utf-8") as f:
                yaml.safe_dump({}, f)

            mm = model_manifest.ModelManifest(pathlib.Path(tmpdir))

            with self.assertRaisesRegex(ValueError, "Unable to get the version of the MANIFEST file."):
                mm.load()

        raw_input = {
            "manifest_version": "1.0",
            "runtimes": {
                "python_runtime": {
                    "language": "PYTHON",
                    "version": "3.8",
                    "imports": ["model.zip", "runtimes/python_runtime/snowflake-ml-python.zip"],
                    "dependencies": {"conda": "runtimes/python_runtime/env/conda.yml"},
                }
            },
            "methods": [
                {
                    "name": "predict",
                    "runtime": "python_runtime",
                    "type": "FUNCTION",
                    "handler": "functions.predict.infer",
                    "inputs": [{"name": "input", "type": "FLOAT"}],
                    "outputs": [{"type": "OBJECT"}],
                },
                {
                    "name": "__CALL__",
                    "runtime": "python_runtime",
                    "type": "FUNCTION",
                    "handler": "functions.__call__.infer",
                    "inputs": [{"name": "INPUT", "type": "FLOAT"}],
                    "outputs": [{"type": "OBJECT"}],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "MANIFEST.yml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(raw_input, f)

            mm = model_manifest.ModelManifest(pathlib.Path(tmpdir))

            self.assertDictEqual(raw_input, mm.load())

        raw_input["user_data"] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "MANIFEST.yml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(raw_input, f)

            mm = model_manifest.ModelManifest(pathlib.Path(tmpdir))

            self.assertDictEqual(raw_input, mm.load())

        raw_input["user_data"] = {"description": "Hello"}

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "MANIFEST.yml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(raw_input, f)

            mm = model_manifest.ModelManifest(pathlib.Path(tmpdir))

            self.assertDictEqual(raw_input, mm.load())


if __name__ == "__main__":
    absltest.main()
