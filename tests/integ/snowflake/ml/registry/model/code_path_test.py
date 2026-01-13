"""Integration tests for CodePath functionality in model registry."""

import os
import tempfile

import pandas as pd
from absl.testing import absltest

from snowflake.ml.model import CodePath, custom_model, model_signature
from tests.integ.snowflake.ml.registry.model import registry_model_test_base


class CodePathIntegTest(registry_model_test_base.RegistryModelTestBase):
    """Integration tests for CodePath functionality."""

    def test_codepath_basic(self) -> None:
        """Test CodePath(root) without filter works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create module
            module_dir = os.path.join(tmpdir, "my_utils")
            os.makedirs(module_dir, exist_ok=True)
            with open(os.path.join(module_dir, "__init__.py"), "w") as f:
                f.write("from my_utils.helpers import double\n")
            with open(os.path.join(module_dir, "helpers.py"), "w") as f:
                f.write("def double(x):\n    return x * 2\n")

            class DoubleModel(custom_model.CustomModel):
                @custom_model.inference_api
                def predict(self, X: pd.DataFrame) -> pd.DataFrame:
                    from my_utils import double

                    return pd.DataFrame({"output": X["input"].apply(double)})

            model = DoubleModel(custom_model.ModelContext())
            test_input = pd.DataFrame({"input": [1, 2, 3, 4, 5]})

            self._test_registry_model(
                model=model,
                prediction_assert_fns={
                    "predict": (
                        test_input,
                        lambda res: self.assertEqual(list(res["output"]), [2, 4, 6, 8, 10]),
                    )
                },
                signatures={
                    "predict": model_signature.ModelSignature(
                        inputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="input")],
                        outputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="output")],
                    )
                },
                code_paths=[CodePath(module_dir)],
            )

    def test_codepath_multiple_helpers(self) -> None:
        """Test CodePath with multiple helper files in a module."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create module with multiple helper files
            module_dir = os.path.join(tmpdir, "math_ops")
            os.makedirs(module_dir, exist_ok=True)

            with open(os.path.join(module_dir, "__init__.py"), "w") as f:
                f.write("from math_ops.adder import add_ten\n")
                f.write("from math_ops.multiplier import multiply_by_two\n")
                f.write("from math_ops.subtractor import subtract_five\n")

            with open(os.path.join(module_dir, "adder.py"), "w") as f:
                f.write("def add_ten(x):\n    return x + 10\n")

            with open(os.path.join(module_dir, "multiplier.py"), "w") as f:
                f.write("def multiply_by_two(x):\n    return x * 2\n")

            with open(os.path.join(module_dir, "subtractor.py"), "w") as f:
                f.write("def subtract_five(x):\n    return x - 5\n")

            class MultiOpModel(custom_model.CustomModel):
                @custom_model.inference_api
                def predict(self, X: pd.DataFrame) -> pd.DataFrame:
                    from math_ops import add_ten, multiply_by_two, subtract_five

                    # Apply: (x + 10) * 2 - 5
                    return pd.DataFrame(
                        {"output": X["input"].apply(lambda x: subtract_five(multiply_by_two(add_ten(x))))}
                    )

            model = MultiOpModel(custom_model.ModelContext())
            test_input = pd.DataFrame({"input": [1, 2, 3]})
            # (1+10)*2-5=17, (2+10)*2-5=19, (3+10)*2-5=21

            self._test_registry_model(
                model=model,
                prediction_assert_fns={
                    "predict": (
                        test_input,
                        lambda res: self.assertEqual(list(res["output"]), [17, 19, 21]),
                    )
                },
                signatures={
                    "predict": model_signature.ModelSignature(
                        inputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="input")],
                        outputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="output")],
                    )
                },
                code_paths=[CodePath(module_dir)],
            )

    def test_codepath_with_complex_signature(self) -> None:
        """Test CodePath with complicated signature with multiple params."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create module
            module_dir = os.path.join(tmpdir, "calculator")
            os.makedirs(module_dir, exist_ok=True)

            with open(os.path.join(module_dir, "__init__.py"), "w") as f:
                f.write("from calculator.ops import weighted_sum\n")

            with open(os.path.join(module_dir, "ops.py"), "w") as f:
                f.write("def weighted_sum(a, b, weight_a, weight_b):\n")
                f.write("    return a * weight_a + b * weight_b\n")

            class WeightedSumModel(custom_model.CustomModel):
                @custom_model.inference_api
                def predict(self, X: pd.DataFrame) -> pd.DataFrame:
                    from calculator import weighted_sum

                    results = []
                    for _, row in X.iterrows():
                        results.append(weighted_sum(row["a"], row["b"], row["weight_a"], row["weight_b"]))
                    return pd.DataFrame({"result": results})

            model = WeightedSumModel(custom_model.ModelContext())
            test_input = pd.DataFrame(
                {
                    "a": [1.0, 2.0, 3.0],
                    "b": [4.0, 5.0, 6.0],
                    "weight_a": [0.5, 0.5, 0.5],
                    "weight_b": [0.5, 0.5, 0.5],
                }
            )
            # 1*0.5 + 4*0.5 = 2.5, 2*0.5 + 5*0.5 = 3.5, 3*0.5 + 6*0.5 = 4.5

            self._test_registry_model(
                model=model,
                prediction_assert_fns={
                    "predict": (
                        test_input,
                        lambda res: self.assertEqual(list(res["result"]), [2.5, 3.5, 4.5]),
                    )
                },
                signatures={
                    "predict": model_signature.ModelSignature(
                        inputs=[
                            model_signature.FeatureSpec(dtype=model_signature.DataType.DOUBLE, name="a"),
                            model_signature.FeatureSpec(dtype=model_signature.DataType.DOUBLE, name="b"),
                            model_signature.FeatureSpec(dtype=model_signature.DataType.DOUBLE, name="weight_a"),
                            model_signature.FeatureSpec(dtype=model_signature.DataType.DOUBLE, name="weight_b"),
                        ],
                        outputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.DOUBLE, name="result")],
                    )
                },
                code_paths=[CodePath(module_dir)],
            )

    def test_codepath_with_filter(self) -> None:
        """Test CodePath(root, filter='subdir') selects subdirectory correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure: project/src/math_utils/
            project_dir = os.path.join(tmpdir, "project", "src")
            os.makedirs(project_dir, exist_ok=True)

            # Create the math_utils module inside src
            math_utils_dir = os.path.join(project_dir, "math_utils")
            os.makedirs(math_utils_dir, exist_ok=True)

            with open(os.path.join(math_utils_dir, "__init__.py"), "w") as f:
                f.write("from math_utils.core import triple\n")

            with open(os.path.join(math_utils_dir, "core.py"), "w") as f:
                f.write("def triple(x):\n    return x * 3\n")

            # Create another module that should NOT be included
            other_utils_dir = os.path.join(project_dir, "other_utils")
            os.makedirs(other_utils_dir, exist_ok=True)
            with open(os.path.join(other_utils_dir, "__init__.py"), "w") as f:
                f.write("# This should not be included\n")

            class TripleModel(custom_model.CustomModel):
                @custom_model.inference_api
                def predict(self, X: pd.DataFrame) -> pd.DataFrame:
                    from math_utils import triple

                    return pd.DataFrame({"output": X["input"].apply(triple)})

            model = TripleModel(custom_model.ModelContext())
            test_input = pd.DataFrame({"input": [1, 2, 3]})

            self._test_registry_model(
                model=model,
                prediction_assert_fns={
                    "predict": (
                        test_input,
                        lambda res: self.assertEqual(list(res["output"]), [3, 6, 9]),
                    )
                },
                signatures={
                    "predict": model_signature.ModelSignature(
                        inputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="input")],
                        outputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="output")],
                    )
                },
                code_paths=[CodePath(project_dir, filter="math_utils")],
            )

    def test_codepath_recursive_subfolders(self) -> None:
        """Test CodePath correctly handles recursive sub-folders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create deeply nested module structure
            module_dir = os.path.join(tmpdir, "deep_module")
            level1_dir = os.path.join(module_dir, "level1")
            level2_dir = os.path.join(level1_dir, "level2")
            level3_dir = os.path.join(level2_dir, "level3")
            os.makedirs(level3_dir, exist_ok=True)

            # Create __init__.py at each level
            with open(os.path.join(module_dir, "__init__.py"), "w") as f:
                f.write("from deep_module.level1 import func_level1\n")
                f.write("from deep_module.level1.level2 import func_level2\n")
                f.write("from deep_module.level1.level2.level3 import func_level3\n")

            with open(os.path.join(level1_dir, "__init__.py"), "w") as f:
                f.write("def func_level1(x):\n    return x + 1\n")

            with open(os.path.join(level2_dir, "__init__.py"), "w") as f:
                f.write("def func_level2(x):\n    return x + 10\n")

            with open(os.path.join(level3_dir, "__init__.py"), "w") as f:
                f.write("def func_level3(x):\n    return x + 100\n")

            class DeepModel(custom_model.CustomModel):
                @custom_model.inference_api
                def predict(self, X: pd.DataFrame) -> pd.DataFrame:
                    from deep_module import func_level1, func_level2, func_level3

                    # Apply: x + 1 + 10 + 100 = x + 111
                    return pd.DataFrame(
                        {"output": X["input"].apply(lambda x: func_level1(x) + func_level2(x) + func_level3(x) - 2 * x)}
                    )

            model = DeepModel(custom_model.ModelContext())
            test_input = pd.DataFrame({"input": [0, 1, 2]})
            # 0+111=111, 1+111=112, 2+111=113

            self._test_registry_model(
                model=model,
                prediction_assert_fns={
                    "predict": (
                        test_input,
                        lambda res: self.assertEqual(list(res["output"]), [111, 112, 113]),
                    )
                },
                signatures={
                    "predict": model_signature.ModelSignature(
                        inputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="input")],
                        outputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="output")],
                    )
                },
                code_paths=[CodePath(module_dir)],
            )

    def test_codepath_mixed_string_and_codepath(self) -> None:
        """Test mixing string paths and CodePath objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two separate modules
            adder_dir = os.path.join(tmpdir, "adder")
            os.makedirs(adder_dir, exist_ok=True)
            with open(os.path.join(adder_dir, "__init__.py"), "w") as f:
                f.write("def add_one(x):\n    return x + 1\n")

            multiplier_dir = os.path.join(tmpdir, "multiplier")
            os.makedirs(multiplier_dir, exist_ok=True)
            with open(os.path.join(multiplier_dir, "__init__.py"), "w") as f:
                f.write("def times_two(x):\n    return x * 2\n")

            class CombinedModel(custom_model.CustomModel):
                @custom_model.inference_api
                def predict(self, X: pd.DataFrame) -> pd.DataFrame:
                    from adder import add_one
                    from multiplier import times_two

                    return pd.DataFrame({"output": X["input"].apply(lambda x: times_two(add_one(x)))})

            model = CombinedModel(custom_model.ModelContext())
            test_input = pd.DataFrame({"input": [1, 2, 3]})
            # (1+1)*2=4, (2+1)*2=6, (3+1)*2=8

            self._test_registry_model(
                model=model,
                prediction_assert_fns={
                    "predict": (
                        test_input,
                        lambda res: self.assertEqual(list(res["output"]), [4, 6, 8]),
                    )
                },
                signatures={
                    "predict": model_signature.ModelSignature(
                        inputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="input")],
                        outputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="output")],
                    )
                },
                # Mix string path and CodePath
                code_paths=[adder_dir, CodePath(multiplier_dir)],
            )

    def test_codepath_destination_conflict_raises_error(self) -> None:
        """Test that destination path conflicts raise an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two different directories with the same basename
            dir1 = os.path.join(tmpdir, "project1", "utils")
            dir2 = os.path.join(tmpdir, "project2", "utils")
            os.makedirs(dir1, exist_ok=True)
            os.makedirs(dir2, exist_ok=True)

            # Create __init__.py in each
            with open(os.path.join(dir1, "__init__.py"), "w") as f:
                f.write("VERSION = 1\n")
            with open(os.path.join(dir2, "__init__.py"), "w") as f:
                f.write("VERSION = 2\n")

            class DummyModel(custom_model.CustomModel):
                @custom_model.inference_api
                def predict(self, X: pd.DataFrame) -> pd.DataFrame:
                    return X

            model = DummyModel(custom_model.ModelContext())
            sig = model_signature.ModelSignature(
                inputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="input")],
                outputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.INT64, name="input")],
            )

            # Both paths will try to copy to code/utils/ - should conflict
            with self.assertRaises(ValueError) as ctx:
                self.registry.log_model(
                    model,
                    model_name=f"model_{self._run_id}_conflict",
                    version_name="v1",
                    signatures={"predict": sig},
                    code_paths=[CodePath(dir1), CodePath(dir2)],
                )

            self.assertIn("conflict", str(ctx.exception).lower())

    def test_codepath_invalid_filter_escape_raises_error(self) -> None:
        """Test that filter escaping root raises appropriate error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory
            test_dir = os.path.join(tmpdir, "project")
            os.makedirs(test_dir, exist_ok=True)

            with self.assertRaises(ValueError) as ctx:
                CodePath(test_dir, filter="../escape")._resolve()

            self.assertIn("escapes root", str(ctx.exception).lower())


if __name__ == "__main__":
    absltest.main()
