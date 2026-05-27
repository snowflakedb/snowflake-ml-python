import datetime
import pathlib
import tempfile
from typing import Any, Callable, Sequence

import importlib_resources
import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml._internal import platform_capabilities
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
from snowflake.ml.model.volatility import Volatility

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


class ModelMethodTest(parameterized.TestCase):
    def test_model_method(self) -> None:
        fg = function_generator.FunctionGenerator(pathlib.PurePosixPath("@a.b.c/abc/model"))

        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
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

        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
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

        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
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

        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
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

        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
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

        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
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

        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
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

        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model5", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model5"] = _DUMMY_BLOB
            mm = model_method.ModelMethod(
                meta,
                "predict",
                "python_runtime",
                fg,
                wide_input=True,
            )
            method_dict = mm.save(pathlib.Path(workspace))
            with open(pathlib.Path(workspace, "functions", "predict.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files(model_method_pkg)
                        .joinpath("fixtures")
                        .joinpath("function_5.py")
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

    def test_model_method_model_init_once_true(self) -> None:
        """model_init_once=True selects the init_once template regardless of model type."""
        fg = function_generator.FunctionGenerator(pathlib.PurePosixPath("@a.b.c/abc/model"))

        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
            mm = model_method.ModelMethod(
                meta,
                "predict",
                "python_runtime",
                fg,
                options=model_method.ModelMethodOptions(model_init_once=True),
            )
            mm.save(pathlib.Path(workspace))
            with open(pathlib.Path(workspace, "functions", "predict.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files(model_method_pkg)
                        .joinpath("fixtures")
                        .joinpath("function_1_init_once.py")
                        .read_text()
                    ),
                    f.read(),
                )

    def test_model_method_table_function_model_init_once(self) -> None:
        """model_init_once=True with TABLE_FUNCTION uses infer_table_function_init_once template."""
        fg = function_generator.FunctionGenerator(pathlib.PurePosixPath("@a.b.c/abc/model"))

        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
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
                    function_type=model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value,
                    model_init_once=True,
                ),
            )
            mm.save(pathlib.Path(workspace))
            with open(pathlib.Path(workspace, "functions", "predict.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files(model_method_pkg)
                        .joinpath("fixtures")
                        .joinpath("function_3_init_once.py")
                        .read_text()
                    ),
                    f.read(),
                )

    def test_model_method_partitioned_table_function_model_init_once(self) -> None:
        """model_init_once=True with partitioned TABLE_FUNCTION uses infer_partitioned_init_once template."""
        fg = function_generator.FunctionGenerator(pathlib.PurePosixPath("@a.b.c/abc/model"))

        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
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
                    function_type=model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value,
                    model_init_once=True,
                ),
            )
            mm.save(pathlib.Path(workspace))
            with open(pathlib.Path(workspace, "functions", "predict.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files(model_method_pkg)
                        .joinpath("fixtures")
                        .joinpath("function_4_init_once.py")
                        .read_text()
                    ),
                    f.read(),
                )

    def test_model_method_model_init_once_false(self) -> None:
        """model_init_once=False uses the plain inference template regardless of model type."""
        fg = function_generator.FunctionGenerator(pathlib.PurePosixPath("@a.b.c/abc/model"))

        for opts in (
            model_method.ModelMethodOptions(model_init_once=False),
            model_method.ModelMethodOptions(),  # absent — defaults to False
        ):
            with (
                tempfile.TemporaryDirectory() as workspace,
                tempfile.TemporaryDirectory() as tmpdir,
                platform_capabilities.PlatformCapabilities.mock_features(),
            ):
                with model_meta.create_model_metadata(
                    model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
                ) as meta:
                    meta.models["model1"] = _DUMMY_BLOB
                mm = model_method.ModelMethod(
                    meta,
                    "predict",
                    "python_runtime",
                    fg,
                    options=opts,
                )
                mm.save(pathlib.Path(workspace))
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

    def test_model_method_model_init_once_via_options_dict(self) -> None:
        """model_init_once=True in a raw options dict flows through
        get_model_method_options_from_options and produces the init_once template."""
        fg = function_generator.FunctionGenerator(pathlib.PurePosixPath("@a.b.c/abc/model"))

        raw_options: type_hints.ModelSaveOption = {
            "enable_explainability": False,
            "model_init_once": True,
        }

        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            resolved_options = model_method.get_model_method_options_from_options(
                raw_options, "predict", model_type="custom"
            )
            mm = model_method.ModelMethod(
                meta,
                "predict",
                "python_runtime",
                fg,
                options=resolved_options,
            )
            mm.save(pathlib.Path(workspace))
            with open(pathlib.Path(workspace, "functions", "predict.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files(model_method_pkg)
                        .joinpath("fixtures")
                        .joinpath("function_1_init_once.py")
                        .read_text()
                    ),
                    f.read(),
                )

    def test_model_method_with_volatility_default(self) -> None:
        """Test that ModelMethod.save() omits volatility when not enabled."""
        fg = function_generator.FunctionGenerator(pathlib.PurePosixPath("@a.b.c/abc/model"))

        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(
                {platform_capabilities.SET_MODULE_FUNCTIONS_VOLATILITY_FROM_MANIFEST: False}
            ),
        ):
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

            # Verify volatility is omitted by default
            self.assertNotIn("volatility", method_dict)

    @parameterized.parameters(  # type: ignore[misc]
        (Volatility.IMMUTABLE,),
        (Volatility.VOLATILE,),
    )
    def test_model_method_with_volatility_explicit(self, volatility: Volatility) -> None:
        """Test that ModelMethod.save() includes explicit volatility values in the method dict."""
        fg = function_generator.FunctionGenerator(pathlib.PurePosixPath("@a.b.c/abc/model"))

        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(
                {platform_capabilities.SET_MODULE_FUNCTIONS_VOLATILITY_FROM_MANIFEST: True}
            ),
        ):
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            mm = model_method.ModelMethod(
                meta,
                "predict",
                "python_runtime",
                fg,
                options=model_method.ModelMethodOptions(volatility=volatility),
            )
            method_dict = mm.save(pathlib.Path(workspace))

            # Verify volatility is included
            self.assertIn("volatility", method_dict)
            self.assertEqual(method_dict.get("volatility"), volatility.name)

    @parameterized.parameters(  # type: ignore[misc]
        (Volatility.IMMUTABLE,),
        (Volatility.VOLATILE,),
    )
    def test_model_method_with_complete_dict_structure(self, volatility: Volatility) -> None:
        """Test that volatility is included alongside all other expected fields."""
        fg = function_generator.FunctionGenerator(pathlib.PurePosixPath("@a.b.c/abc/model"))

        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(
                {platform_capabilities.SET_MODULE_FUNCTIONS_VOLATILITY_FROM_MANIFEST: True}
            ),
        ):
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
                    volatility=volatility,
                    function_type=model_manifest_schema.ModelMethodFunctionTypes.FUNCTION.value,
                ),
            )
            method_dict = mm.save(pathlib.Path(workspace))

            # Verify the complete structure includes volatility
            expected_dict = {
                "name": "PREDICT",
                "runtime": "python_runtime",
                "type": "FUNCTION",
                "handler": "functions.predict.infer",
                "inputs": [{"name": "INPUT", "type": "FLOAT"}, {"name": "NAME", "type": "STRING"}],
                "outputs": [{"type": "OBJECT"}],
                "volatility": volatility.name,
            }
            self.assertDictEqual(method_dict, expected_dict)

    def test_model_method_with_parameters_enabled(self) -> None:
        """Test that ModelMethod.save() includes parameters when capability is enabled."""
        fg = function_generator.FunctionGenerator(pathlib.PurePosixPath("@a.b.c/abc/model"))

        sig_with_params = {
            "predict": model_signature.ModelSignature(
                inputs=[
                    model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
                ],
                outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
                params=[
                    model_signature.ParamSpec(
                        name="threshold", dtype=model_signature.DataType.FLOAT, default_value=0.5
                    ),
                    model_signature.ParamSpec(name="max_iter", dtype=model_signature.DataType.INT64, default_value=100),
                ],
            )
        }

        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=sig_with_params
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            mm = model_method.ModelMethod(
                meta,
                "predict",
                "python_runtime",
                fg,
            )
            method_dict = mm.save(pathlib.Path(workspace))

            # Verify params are included in method_dict
            self.assertIn("params", method_dict)
            self.assertEqual(len(method_dict["params"]), 2)
            self.assertEqual(method_dict["params"][0]["name"], "THRESHOLD")
            self.assertEqual(method_dict["params"][0]["type"], "FLOAT")
            self.assertEqual(method_dict["params"][0]["default"], "0.5")
            self.assertEqual(method_dict["params"][1]["name"], "MAX_ITER")
            self.assertEqual(method_dict["params"][1]["type"], "BIGINT")
            self.assertEqual(method_dict["params"][1]["default"], "100")

            # Verify generated code matches expected fixture (same structure regardless of params)
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

    def test_model_method_with_parameters_case_sensitive(self) -> None:
        """Test that parameters respect case_sensitive option."""
        fg = function_generator.FunctionGenerator(pathlib.PurePosixPath("@a.b.c/abc/model"))

        sig_with_params = {
            "predict": model_signature.ModelSignature(
                inputs=[
                    model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
                ],
                outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
                params=[
                    model_signature.ParamSpec(name="myParam", dtype=model_signature.DataType.FLOAT, default_value=0.5),
                ],
            )
        }

        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=sig_with_params
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

            # Verify parameter name is case-sensitive
            self.assertIn("params", method_dict)
            self.assertEqual(method_dict["params"][0]["name"], "myParam")

    def test_model_method_with_array_parameter(self) -> None:
        """Test that array parameters are correctly typed as ARRAY and default is JSON formatted."""
        fg = function_generator.FunctionGenerator(pathlib.PurePosixPath("@a.b.c/abc/model"))

        sig_with_array_param = {
            "predict": model_signature.ModelSignature(
                inputs=[
                    model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
                ],
                outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
                params=[
                    model_signature.ParamSpec(
                        name="stop", dtype=model_signature.DataType.STRING, default_value=[], shape=(-1,)
                    ),
                    model_signature.ParamSpec(
                        name="ids", dtype=model_signature.DataType.INT64, default_value=[1, 2, 3], shape=(-1,)
                    ),
                ],
            )
        }

        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=sig_with_array_param
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            mm = model_method.ModelMethod(
                meta,
                "predict",
                "python_runtime",
                fg,
            )
            method_dict = mm.save(pathlib.Path(workspace))

            # Verify array params are correctly typed and formatted
            self.assertIn("params", method_dict)
            self.assertEqual(len(method_dict["params"]), 2)

            # First param: stop (empty array)
            self.assertEqual(method_dict["params"][0]["name"], "STOP")
            self.assertEqual(method_dict["params"][0]["type"], "ARRAY")
            self.assertEqual(method_dict["params"][0]["default"], "[]")

            # Second param: ids (array with values)
            self.assertEqual(method_dict["params"][1]["name"], "IDS")
            self.assertEqual(method_dict["params"][1]["type"], "ARRAY")
            self.assertEqual(method_dict["params"][1]["default"], "[1, 2, 3]")

    def test_model_method_with_dict_parameter(self) -> None:
        """Test that ParamGroupSpec produces type=OBJECT with OBJECT_CONSTRUCT_KEEP_NULL default."""
        fg = function_generator.FunctionGenerator(pathlib.PurePosixPath("@a.b.c/abc/model"))

        sig_with_dict_param = {
            "predict": model_signature.ModelSignature(
                inputs=[
                    model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
                ],
                outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
                params=[
                    model_signature.ParamGroupSpec(
                        name="config",
                        specs=[
                            model_signature.ParamSpec(
                                name="temperature", dtype=model_signature.DataType.FLOAT, default_value=1.0
                            ),
                            model_signature.ParamSpec(
                                name="top_k", dtype=model_signature.DataType.INT32, default_value=50
                            ),
                        ],
                    ),
                ],
            )
        }

        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=sig_with_dict_param
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            mm = model_method.ModelMethod(
                meta,
                "predict",
                "python_runtime",
                fg,
            )
            method_dict = mm.save(pathlib.Path(workspace))

            self.assertIn("params", method_dict)
            self.assertEqual(len(method_dict["params"]), 1)
            self.assertEqual(method_dict["params"][0]["name"], "CONFIG")
            self.assertEqual(method_dict["params"][0]["type"], "OBJECT")
            self.assertEqual(
                method_dict["params"][0]["default"],
                "OBJECT_CONSTRUCT_KEEP_NULL('temperature', 1.0, 'top_k', 50)",
            )

    def test_model_method_with_nested_dict_parameter(self) -> None:
        """Test ParamGroupSpec with nested ParamSpec (shaped) and nested ParamGroupSpec (shaped)."""
        fg = function_generator.FunctionGenerator(pathlib.PurePosixPath("@a.b.c/abc/model"))

        sig_with_nested = {
            "predict": model_signature.ModelSignature(
                inputs=[
                    model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
                ],
                outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
                params=[
                    model_signature.ParamGroupSpec(
                        name="config",
                        specs=[
                            model_signature.ParamSpec(
                                name="temperature", dtype=model_signature.DataType.FLOAT, default_value=1.0
                            ),
                            model_signature.ParamSpec(
                                name="nested_list",
                                dtype=model_signature.DataType.INT64,
                                default_value=[[1, 2], [3, 4]],
                                shape=(2, 2),
                            ),
                            model_signature.ParamGroupSpec(
                                name="nested_dict",
                                specs=[
                                    model_signature.ParamSpec(
                                        name="a", dtype=model_signature.DataType.INT64, default_value=1
                                    ),
                                    model_signature.ParamSpec(
                                        name="b", dtype=model_signature.DataType.INT64, default_value=2
                                    ),
                                ],
                                shape=(2,),
                            ),
                        ],
                    ),
                ],
            )
        }

        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as tmpdir,
            platform_capabilities.PlatformCapabilities.mock_features(),
        ):
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=sig_with_nested
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB

            mm = model_method.ModelMethod(
                meta,
                "predict",
                "python_runtime",
                fg,
            )
            method_dict = mm.save(pathlib.Path(workspace))

            self.assertIn("params", method_dict)
            self.assertEqual(len(method_dict["params"]), 1)
            self.assertEqual(method_dict["params"][0]["name"], "CONFIG")
            self.assertEqual(method_dict["params"][0]["type"], "OBJECT")

            default_sql = method_dict["params"][0]["default"]
            self.assertEqual(
                default_sql,
                "OBJECT_CONSTRUCT_KEEP_NULL("
                "'temperature', 1.0, "
                "'nested_list', [[1, 2], [3, 4]], "
                "'nested_dict', [{'a': 1, 'b': 2}, {'a': 1, 'b': 2}]"
                ")",
            )


class FormatParamDefaultValueTest(absltest.TestCase):
    """Tests for ModelMethod._format_param_default_value() helper."""

    def test_format_none_value(self) -> None:
        """Test that None is formatted as 'NULL' for SQL compatibility."""
        self.assertEqual(model_method.ModelMethod._format_param_default_value(None), "NULL")

    def test_format_empty_list(self) -> None:
        """Test that empty list is formatted as JSON '[]'."""
        self.assertEqual(model_method.ModelMethod._format_param_default_value([]), "[]")

    def test_format_list_with_values(self) -> None:
        """Test that list with values is formatted using Python's str() representation.

        This uses single quotes for strings, which SQL interprets as string literals.
        """
        self.assertEqual(model_method.ModelMethod._format_param_default_value([1, 2, 3]), "[1, 2, 3]")
        self.assertEqual(model_method.ModelMethod._format_param_default_value(["a", "b"]), "['a', 'b']")
        self.assertEqual(model_method.ModelMethod._format_param_default_value([1.5, 2.5]), "[1.5, 2.5]")

    def test_format_scalar_values(self) -> None:
        """Test that numeric values are formatted via str()."""
        self.assertEqual(model_method.ModelMethod._format_param_default_value(0.5), "0.5")
        self.assertEqual(model_method.ModelMethod._format_param_default_value(100), "100")

    def test_format_boolean_values(self) -> None:
        """Test that boolean values are formatted as lowercase SQL booleans."""
        self.assertEqual(model_method.ModelMethod._format_param_default_value(True), "true")
        self.assertEqual(model_method.ModelMethod._format_param_default_value(False), "false")

    def test_format_string_value(self) -> None:
        """Test that string values are formatted as single-quoted SQL literals."""
        self.assertEqual(model_method.ModelMethod._format_param_default_value("hello"), "'hello'")
        self.assertEqual(model_method.ModelMethod._format_param_default_value("default"), "'default'")
        # Test escaping single quotes (uses backslash escaping)
        self.assertEqual(model_method.ModelMethod._format_param_default_value("it's"), "'it\\'s'")

    def test_format_bytes_value(self) -> None:
        """Test that bytes values are formatted as SQL hex literals."""
        self.assertEqual(model_method.ModelMethod._format_param_default_value(b"hello"), "X'68656c6c6f'")
        self.assertEqual(model_method.ModelMethod._format_param_default_value(b""), "X''")

    def test_format_datetime_value(self) -> None:
        """Test that datetime values are formatted as SQL timestamp literals."""
        import datetime

        dt = datetime.datetime(2024, 1, 1, 12, 0, 0)
        self.assertEqual(
            model_method.ModelMethod._format_param_default_value(dt), "'2024-01-01 12:00:00'::TIMESTAMP_NTZ"
        )


class ParamDtypeCastTest(parameterized.TestCase):
    """Tests for the scalar-param dtype cast logic embedded in the inference templates.

    The wrapper templates (infer_function.py_template and its 5 siblings) build a
    `param_dtype_map` at load time and apply it to each scalar param's runtime value
    before passing it to the model. These helpers mirror that logic on real ParamSpec
    objects so regressions in either the membership rule or the per-value cast are
    caught here, separately from the fixture-equality checks above.
    """

    @staticmethod
    def _build_param_dtype_map(
        specs: Sequence[model_signature.BaseParamSpec],
    ) -> dict[str, Callable[[Any], Any]]:
        """Mirror of the load-time loop in the inference templates."""
        param_dtype_map: dict[str, Callable[[Any], Any]] = {}
        for param_spec in specs:
            if param_spec.shape is None:
                # BaseParamSpec doesn't declare `dtype`, but both concrete subclasses
                # (ParamSpec, ParamGroupSpec) do — same duck-type access the templates use.
                np_type = param_spec.dtype._numpy_type  # type: ignore[attr-defined]
                if callable(np_type):
                    param_dtype_map[param_spec.name] = np_type
        return param_dtype_map

    @staticmethod
    def _extract_method_param(
        col: str,
        series: pd.Series,
        param_defaults: dict[str, Any],
        param_dtype_map: dict[str, Callable[[Any], Any]],
    ) -> Any:
        """Mirror of the per-request extraction branch in the inference templates."""
        val = series.iloc[0]
        if val is None or (not isinstance(val, (list, np.ndarray, dict)) and pd.isna(val)):
            return param_defaults[col]
        if col in param_dtype_map:
            return param_dtype_map[col](val)
        return val

    def test_int64_widens_narrower_numpy_int(self) -> None:
        """np.int16(3) arriving for an INT64 param must be re-widened to np.int64.

        This is the original MLflow failure mode: Snowflake's column inference
        can narrow a BIGINT constant column to a smaller numpy width, and MLflow's
        enforce_param_datatype rejects np.int16 for a `long` param.
        """
        specs = [model_signature.ParamSpec("repeat", model_signature.DataType.INT64, default_value=1)]
        m = self._build_param_dtype_map(specs)
        self.assertIn("repeat", m)
        cast = m["repeat"](np.int16(3))
        self.assertIsInstance(cast, np.int64)
        self.assertEqual(cast, 3)

    def test_int64_widens_python_int(self) -> None:
        specs = [model_signature.ParamSpec("repeat", model_signature.DataType.INT64, default_value=1)]
        m = self._build_param_dtype_map(specs)
        cast = m["repeat"](3)
        self.assertIsInstance(cast, np.int64)
        self.assertEqual(cast, 3)

    def test_int64_idempotent_when_already_correct(self) -> None:
        specs = [model_signature.ParamSpec("repeat", model_signature.DataType.INT64, default_value=1)]
        m = self._build_param_dtype_map(specs)
        cast = m["repeat"](np.int64(7))
        self.assertIsInstance(cast, np.int64)
        self.assertEqual(cast, 7)

    def test_double_widens_float32(self) -> None:
        specs = [model_signature.ParamSpec("temperature", model_signature.DataType.DOUBLE, default_value=1.0)]
        m = self._build_param_dtype_map(specs)
        cast = m["temperature"](np.float32(0.5))
        self.assertIsInstance(cast, np.float64)
        self.assertAlmostEqual(float(cast), 0.5)

    def test_float_narrows_float64(self) -> None:
        specs = [model_signature.ParamSpec("temperature", model_signature.DataType.FLOAT, default_value=1.0)]
        m = self._build_param_dtype_map(specs)
        cast = m["temperature"](np.float64(0.5))
        self.assertIsInstance(cast, np.float32)
        self.assertAlmostEqual(float(cast), 0.5)

    def test_bool_param_in_map(self) -> None:
        specs = [model_signature.ParamSpec("flag", model_signature.DataType.BOOL, default_value=False)]
        m = self._build_param_dtype_map(specs)
        cast = m["flag"](True)
        self.assertIsInstance(cast, np.bool_)
        self.assertTrue(bool(cast))

    def test_string_param_in_map(self) -> None:
        specs = [model_signature.ParamSpec("name", model_signature.DataType.STRING, default_value="x")]
        m = self._build_param_dtype_map(specs)
        cast = m["name"]("hello")
        self.assertIsInstance(cast, np.str_)
        self.assertEqual(str(cast), "hello")

    def test_timestamp_ntz_scalar_skipped(self) -> None:
        """TIMESTAMP_NTZ's _numpy_type is the string 'datetime64[ns]', not a callable.

        The `callable(np_type)` guard intentionally excludes it, so the per-request
        branch falls through and the value passes to the model unchanged.
        """
        specs = [
            model_signature.ParamSpec(
                "when",
                model_signature.DataType.TIMESTAMP_NTZ,
                default_value=datetime.datetime(2025, 1, 1),
            )
        ]
        self.assertNotIn("when", self._build_param_dtype_map(specs))

    def test_array_param_skipped(self) -> None:
        """shape != None ⇒ the param flows as a Python list, not a scalar, so we don't cast."""
        specs = [
            model_signature.ParamSpec(
                "xs",
                model_signature.DataType.FLOAT,
                default_value=[],
                shape=(-1,),
            )
        ]
        self.assertNotIn("xs", self._build_param_dtype_map(specs))

    def test_param_group_spec_scalar_in_map_is_noop_on_dict(self) -> None:
        """ParamGroupSpec.dtype is always OBJECT (np.object_), which is a callable.

        np.object_ on a dict is effectively a no-op, so including it costs nothing
        and keeps the membership rule uniform across BaseParamSpec subclasses.
        """
        group = model_signature.ParamGroupSpec(
            name="opts",
            specs=[model_signature.ParamSpec("a", model_signature.DataType.INT64, default_value=1)],
        )
        m = self._build_param_dtype_map([group])
        self.assertIn("opts", m)
        val = {"a": 1}
        self.assertEqual(m["opts"](val), val)

    def test_param_group_spec_with_shape_skipped(self) -> None:
        group = model_signature.ParamGroupSpec(
            name="opts",
            specs=[model_signature.ParamSpec("a", model_signature.DataType.INT64, default_value=1)],
            shape=(-1,),
        )
        self.assertNotIn("opts", self._build_param_dtype_map([group]))

    def test_per_request_null_value_uses_default(self) -> None:
        """The null branch is unchanged: a Series of NULLs returns the spec default."""
        specs = [model_signature.ParamSpec("repeat", model_signature.DataType.INT64, default_value=1)]
        m = self._build_param_dtype_map(specs)
        defaults = {spec.name: spec.default_value for spec in specs}
        series = pd.Series([None, None], dtype=object)
        self.assertEqual(self._extract_method_param("repeat", series, defaults, m), 1)

    def test_per_request_cast_applied_to_extracted_scalar(self) -> None:
        """End-to-end: an Arrow-style int16 column becomes np.int64 by the time it reaches the model."""
        specs = [model_signature.ParamSpec("repeat", model_signature.DataType.INT64, default_value=1)]
        m = self._build_param_dtype_map(specs)
        defaults = {spec.name: spec.default_value for spec in specs}
        series = pd.Series([3, 3, 3], dtype=np.int16)
        out = self._extract_method_param("repeat", series, defaults, m)
        self.assertIsInstance(out, np.int64)
        self.assertEqual(out, 3)

    def test_per_request_no_cast_for_array_param(self) -> None:
        """Array params take the else-branch and the value (a list) is forwarded as-is."""
        specs = [
            model_signature.ParamSpec(
                "xs",
                model_signature.DataType.FLOAT,
                default_value=[],
                shape=(-1,),
            )
        ]
        m = self._build_param_dtype_map(specs)
        defaults = {spec.name: spec.default_value for spec in specs}
        series = pd.Series([[0.1, 0.2, 0.3]], dtype=object)
        out = self._extract_method_param("xs", series, defaults, m)
        self.assertEqual(out, [0.1, 0.2, 0.3])

    def test_mixed_specs_split_correctly(self) -> None:
        """A representative mix: only the scalar non-timestamp params end up in the map."""
        specs = [
            model_signature.ParamSpec("repeat", model_signature.DataType.INT64, default_value=1),
            model_signature.ParamSpec("temperature", model_signature.DataType.DOUBLE, default_value=1.0),
            model_signature.ParamSpec(
                "when",
                model_signature.DataType.TIMESTAMP_NTZ,
                default_value=datetime.datetime(2025, 1, 1),
            ),
            model_signature.ParamSpec(
                "rates",
                model_signature.DataType.FLOAT,
                default_value=[],
                shape=(-1,),
            ),
        ]
        self.assertEqual(set(self._build_param_dtype_map(specs)), {"repeat", "temperature"})


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

        # Prophet models should automatically default to table function.
        method_options = model_method.get_model_method_options_from_options(
            options={}, target_method="predict", model_type="prophet"
        )
        self.assertEqual(
            method_options["function_type"], model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value
        )

    def test_get_model_method_options_with_volatility(self) -> None:
        """Test that get_model_method_options_from_options properly handles volatility."""
        # Test unset volatility
        options: type_hints.ModelSaveOption = {}
        method_options = model_method.get_model_method_options_from_options(options=options, target_method="test")
        self.assertNotIn("volatility", method_options)

        # Test default global volatility
        options = {"volatility": Volatility.IMMUTABLE}
        method_options = model_method.get_model_method_options_from_options(options=options, target_method="test")
        self.assertEqual(method_options.get("volatility"), Volatility.IMMUTABLE)

        # Test explicit volatility in method_options
        options = {"method_options": {"test": {"volatility": Volatility.IMMUTABLE}}}
        method_options = model_method.get_model_method_options_from_options(options=options, target_method="test")
        self.assertEqual(method_options.get("volatility"), Volatility.IMMUTABLE)

        # Test method level overrides global
        options = {"volatility": Volatility.VOLATILE, "method_options": {"test": {"volatility": Volatility.IMMUTABLE}}}
        method_options = model_method.get_model_method_options_from_options(options=options, target_method="test")
        self.assertEqual(method_options.get("volatility"), Volatility.IMMUTABLE)


if __name__ == "__main__":
    absltest.main()
