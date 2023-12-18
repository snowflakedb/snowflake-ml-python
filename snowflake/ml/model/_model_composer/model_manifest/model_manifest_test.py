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
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
        ],
        outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
    )
}


_DUMMY_BLOB = model_blob_meta.ModelBlobMeta(
    name="model1", model_type="custom", path="mock_path", handler_version="version_0"
)


class ModelManifestTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock.MagicMock()

    def test_model_manifest_1(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                with mock.patch.object(env_utils, "validate_requirements_in_information_schema", return_value=[""]):
                    mm.save(self.m_session, meta, pathlib.PurePosixPath("model.zip"))
                with open(os.path.join(workspace, "MANIFEST.yml"), encoding="utf-8") as f:
                    loaded_manifest = yaml.safe_load(f)
                self.assertDictEqual(
                    loaded_manifest,
                    {
                        "manifest_version": "1.0",
                        "runtimes": {
                            "python_runtime": {
                                "language": "PYTHON",
                                "version": meta.env.python_version,
                                "imports": ["model.zip"],
                                "dependencies": {"conda": "runtimes/python_runtime/env/conda.yml"},
                            }
                        },
                        "methods": [
                            {
                                "name": "PREDICT",
                                "runtime": "python_runtime",
                                "type": "FUNCTION",
                                "handler": "functions.predict.infer",
                                "inputs": [{"name": "INPUT", "type": "FLOAT"}],
                                "outputs": [{"type": "OBJECT"}],
                            }
                        ],
                    },
                )
                with open(pathlib.Path(workspace, "functions", "predict.py"), encoding="utf-8") as f:
                    self.assertEqual(
                        (
                            importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                            .joinpath("fixtures")  # type: ignore[no-untyped-call]
                            .joinpath("function_fixture_1.py_fixture")
                            .read_text()
                        ),
                        f.read(),
                    )

    def test_model_manifest_2(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures={"__call__": _DUMMY_SIG["predict"]},
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                with mock.patch.object(env_utils, "validate_requirements_in_information_schema", return_value=[""]):
                    mm.save(
                        self.m_session,
                        meta,
                        pathlib.PurePosixPath("model.zip"),
                        options=type_hints.BaseModelSaveOption(
                            method_options={"__call__": type_hints.ModelMethodSaveOptions(max_batch_size=10)}
                        ),
                    )
                with open(os.path.join(workspace, "MANIFEST.yml"), encoding="utf-8") as f:
                    loaded_manifest = yaml.safe_load(f)
                self.assertDictEqual(
                    loaded_manifest,
                    {
                        "manifest_version": "1.0",
                        "runtimes": {
                            "python_runtime": {
                                "language": "PYTHON",
                                "version": meta.env.python_version,
                                "imports": ["model.zip"],
                                "dependencies": {"conda": "runtimes/python_runtime/env/conda.yml"},
                            }
                        },
                        "methods": [
                            {
                                "name": "__CALL__",
                                "runtime": "python_runtime",
                                "type": "FUNCTION",
                                "handler": "functions.__call__.infer",
                                "inputs": [{"name": "INPUT", "type": "FLOAT"}],
                                "outputs": [{"type": "OBJECT"}],
                            }
                        ],
                    },
                )
                with open(pathlib.Path(workspace, "functions", "__call__.py"), encoding="utf-8") as f:
                    self.assertEqual(
                        (
                            importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                            .joinpath("fixtures")  # type: ignore[no-untyped-call]
                            .joinpath("function_fixture_2.py_fixture")
                            .read_text()
                        ),
                        f.read(),
                    )

    def test_model_manifest_mix(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures={"predict": _DUMMY_SIG["predict"], "__call__": _DUMMY_SIG["predict"]},
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                with mock.patch.object(env_utils, "validate_requirements_in_information_schema", return_value=None):
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
                    loaded_manifest = yaml.safe_load(f)
                self.assertDictEqual(
                    loaded_manifest,
                    {
                        "manifest_version": "1.0",
                        "runtimes": {
                            "python_runtime": {
                                "language": "PYTHON",
                                "version": meta.env.python_version,
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
                    },
                )
                with open(pathlib.Path(workspace, "functions", "predict.py"), encoding="utf-8") as f:
                    self.assertEqual(
                        (
                            importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                            .joinpath("fixtures")  # type: ignore[no-untyped-call]
                            .joinpath("function_fixture_1.py_fixture")
                            .read_text()
                        ),
                        f.read(),
                    )
                with open(pathlib.Path(workspace, "functions", "__call__.py"), encoding="utf-8") as f:
                    self.assertEqual(
                        (
                            importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                            .joinpath("fixtures")  # type: ignore[no-untyped-call]
                            .joinpath("function_fixture_2.py_fixture")
                            .read_text()
                        ),
                        f.read(),
                    )

    def test_model_manifest_bad(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            mm = model_manifest.ModelManifest(pathlib.Path(workspace))
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures={"predict": _DUMMY_SIG["predict"], "PREDICT": _DUMMY_SIG["predict"]},
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                with mock.patch.object(env_utils, "validate_requirements_in_information_schema", return_value=[""]):
                    with self.assertRaisesRegex(
                        ValueError, "Found duplicate method named resolved as PREDICT in the model."
                    ):
                        mm.save(
                            self.m_session,
                            meta,
                            pathlib.PurePosixPath("model.zip"),
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
