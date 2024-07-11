import pathlib
import tempfile

import importlib_resources
from absl.testing import absltest

from snowflake.ml._internal.exceptions import exceptions as snowml_exceptions
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.ml.model._model_composer.model_method import function_generator


class FunctionGeneratorTest(absltest.TestCase):
    def test_function_generator(self) -> None:
        fg = function_generator.FunctionGenerator(pathlib.PurePosixPath("@a.b.c/abc/model.zip"))
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate standard function.
            fg.generate(
                pathlib.Path(tmpdir, "handler.py"),
                "predict",
                model_manifest_schema.ModelMethodFunctionTypes.FUNCTION.value,
            )
            with open(pathlib.Path(tmpdir, "handler.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                        .joinpath("fixtures")
                        .joinpath("function_1.py")
                        .read_text()
                    ),
                    f.read(),
                )

            # Generate function with `__call__` and `max_batch_size`.
            fg.generate(
                pathlib.Path(tmpdir, "another_handler.py"),
                "__call__",
                model_manifest_schema.ModelMethodFunctionTypes.FUNCTION.value,
                options=function_generator.FunctionGenerateOptions(
                    max_batch_size=10,
                ),
            )
            with open(pathlib.Path(tmpdir, "another_handler.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                        .joinpath("fixtures")
                        .joinpath("function_2.py")
                        .read_text()
                    ),
                    f.read(),
                )

            # Generate table function.
            fg.generate(
                pathlib.Path(tmpdir, "table_function_handler.py"),
                "predict",
                model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value,
            )
            with open(pathlib.Path(tmpdir, "table_function_handler.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                        .joinpath("fixtures")
                        .joinpath("function_3.py")
                        .read_text()
                    ),
                    f.read(),
                )

            # Generate partitioned function.
            fg.generate(
                pathlib.Path(tmpdir, "partitioned_function_handler.py"),
                "predict",
                model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value,
                is_partitioned_function=True,
            )
            with open(pathlib.Path(tmpdir, "partitioned_function_handler.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                        .joinpath("fixtures")
                        .joinpath("function_4.py")
                        .read_text()
                    ),
                    f.read(),
                )

            # Generating partitioned function that is not a table function raises exception.
            with self.assertRaises(snowml_exceptions.SnowflakeMLException):
                fg.generate(
                    pathlib.Path(tmpdir, "partitioned_function_handler.py"),
                    "predict",
                    model_manifest_schema.ModelMethodFunctionTypes.FUNCTION.value,
                    is_partitioned_function=True,
                )


if __name__ == "__main__":
    absltest.main()
