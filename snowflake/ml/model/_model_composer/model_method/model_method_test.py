import pathlib
import tempfile

import importlib_resources
from absl.testing import absltest

from snowflake.ml.model import model_signature, type_hints
from snowflake.ml.model._model_composer import model_method as model_method_pkg
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.ml.model._model_composer.model_method import (
    function_generator,
    model_method,
)
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta,
    model_meta_schema,
)

_DUMMY_SIG = {
    "predict": model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
            model_signature.FeatureSpec(dtype=model_signature.DataType.STRING, name="name"),
        ],
        outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
    )
}

_DUMMY_BLOB = model_blob_meta.ModelBlobMeta(
    name="model1", model_type="custom", path="mock_path", handler_version="version_0"
)


class ModelMethodTest(absltest.TestCase):
    def test_model_method(self) -> None:
        fg = function_generator.FunctionGenerator(pathlib.PurePosixPath("@a.b.c/abc/model"))

        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
            mm = model_method.ModelMethod(
                meta,
                "predict",
                "python_runtime",
                fg,
            )
            method_dict = mm.save(pathlib.Path(workspace))
            with open(pathlib.Path(workspace, "functions", "predict.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files(model_method_pkg)
                        .joinpath("fixtures")
                        .joinpath("function_1.py")
                        .read_text()
                    ),
                    f.read(),
                )
            self.assertDictEqual(
                method_dict,
                {
                    "name": "PREDICT",
                    "runtime": "python_runtime",
                    "type": "FUNCTION",
                    "handler": "functions.predict.infer",
                    "inputs": [{"name": "INPUT", "type": "FLOAT"}, {"name": "NAME", "type": "STRING"}],
                    "outputs": [{"type": "OBJECT"}],
                },
            )

        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures={"__call__": _DUMMY_SIG["predict"]},
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
            mm = model_method.ModelMethod(
                meta,
                "__call__",
                "python_runtime",
                fg,
            )
            method_dict = mm.save(
                pathlib.Path(workspace), function_generator.FunctionGenerateOptions(max_batch_size=10)
            )
            with open(pathlib.Path(workspace, "functions", "__call__.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files(model_method_pkg)
                        .joinpath("fixtures")
                        .joinpath("function_2.py")
                        .read_text()
                    ),
                    f.read(),
                )
            self.assertDictEqual(
                method_dict,
                {
                    "name": "__CALL__",
                    "runtime": "python_runtime",
                    "type": "FUNCTION",
                    "handler": "functions.__call__.infer",
                    "inputs": [{"name": "INPUT", "type": "FLOAT"}, {"name": "NAME", "type": "STRING"}],
                    "outputs": [{"type": "OBJECT"}],
                },
            )

        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
            with self.assertRaisesRegex(
                ValueError, "Your target method идентификатор cannot be resolved as valid SQL identifier."
            ):
                mm = model_method.ModelMethod(
                    meta,
                    "идентификатор",
                    "python_runtime",
                    fg,
                )

        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
            with self.assertRaisesRegex(
                ValueError, "Target method predict_proba is not available in the signatures of the model."
            ):
                mm = model_method.ModelMethod(
                    meta,
                    "predict_proba",
                    "python_runtime",
                    fg,
                )

        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
            mm = model_method.ModelMethod(
                meta,
                "predict",
                "python_runtime",
                fg,
                options=model_method.ModelMethodOptions(case_sensitive=True),
            )
            method_dict = mm.save(pathlib.Path(workspace))
            with open(pathlib.Path(workspace, "functions", "predict.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files(model_method_pkg)
                        .joinpath("fixtures")
                        .joinpath("function_1.py")
                        .read_text()
                    ),
                    f.read(),
                )
            self.assertDictEqual(
                method_dict,
                {
                    "name": "predict",
                    "runtime": "python_runtime",
                    "type": "FUNCTION",
                    "handler": "functions.predict.infer",
                    "inputs": [{"name": "input", "type": "FLOAT"}, {"name": "name", "type": "STRING"}],
                    "outputs": [{"type": "OBJECT"}],
                },
            )

        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            mm = model_method.ModelMethod(
                meta,
                "predict",
                "python_runtime",
                fg,
                options=model_method.ModelMethodOptions(
                    function_type=model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value
                ),
            )
            method_dict = mm.save(
                pathlib.Path(workspace),
            )
            with open(pathlib.Path(workspace, "functions", "predict.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files(model_method_pkg)
                        .joinpath("fixtures")
                        .joinpath("function_3.py")
                        .read_text()
                    ),
                    f.read(),
                )
            self.assertDictEqual(
                method_dict,
                {
                    "name": "PREDICT",
                    "runtime": "python_runtime",
                    "type": "TABLE_FUNCTION",
                    "handler": "functions.predict.infer",
                    "inputs": [{"name": "INPUT", "type": "FLOAT"}, {"name": "NAME", "type": "STRING"}],
                    "outputs": [{"name": "OUTPUT", "type": "FLOAT"}],
                },
            )

        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as tmpdir:
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                function_properties={"predict": {model_meta_schema.FunctionProperties.PARTITIONED.value: True}},
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            mm = model_method.ModelMethod(
                meta,
                "predict",
                "python_runtime",
                fg,
                is_partitioned_function=True,
                options=model_method.ModelMethodOptions(
                    function_type=model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value
                ),
            )
            method_dict = mm.save(
                pathlib.Path(workspace),
            )
            with open(pathlib.Path(workspace, "functions", "predict.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files(model_method_pkg)
                        .joinpath("fixtures")
                        .joinpath("function_4.py")
                        .read_text()
                    ),
                    f.read(),
                )
            self.assertDictEqual(
                method_dict,
                {
                    "name": "PREDICT",
                    "runtime": "python_runtime",
                    "type": "TABLE_FUNCTION",
                    "handler": "functions.predict.infer",
                    "inputs": [{"name": "INPUT", "type": "FLOAT"}, {"name": "NAME", "type": "STRING"}],
                    "outputs": [{"name": "OUTPUT", "type": "FLOAT"}],
                },
            )


class ModelMethodOptionsTest(absltest.TestCase):
    def test_get_model_method_options(self) -> None:
        options: type_hints.ModelSaveOption = {
            "function_type": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value,
        }
        method_options = model_method.get_model_method_options_from_options(options=options, target_method="test")
        self.assertEqual(
            method_options["function_type"], model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value
        )

        # method option overrides global.
        options = {
            "function_type": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value,
            "method_options": {
                "test": {"function_type": model_manifest_schema.ModelMethodFunctionTypes.FUNCTION.value}
            },
        }
        method_options = model_method.get_model_method_options_from_options(options=options, target_method="test")
        self.assertEqual(method_options["function_type"], model_manifest_schema.ModelMethodFunctionTypes.FUNCTION.value)

        # explain methods should default to table function.
        method_options = model_method.get_model_method_options_from_options(
            options={"enable_explainability": True}, target_method="explain"
        )
        self.assertEqual(
            method_options["function_type"], model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value
        )


if __name__ == "__main__":
    absltest.main()
