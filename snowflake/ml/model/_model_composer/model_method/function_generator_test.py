import pathlib
import tempfile

import importlib_resources
from absl.testing import absltest

from snowflake.ml.model._model_composer.model_method import function_generator


class FunctionGeneratorTest(absltest.TestCase):
    def test_function_generator(self) -> None:
        fg = function_generator.FunctionGenerator(pathlib.PurePosixPath("@a.b.c/abc/model.zip"))
        with tempfile.TemporaryDirectory() as tmpdir:
            fg.generate(
                pathlib.Path(tmpdir, "handler.py"),
                "predict",
            )
            with open(pathlib.Path(tmpdir, "handler.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                        .joinpath("fixtures")  # type: ignore[no-untyped-call]
                        .joinpath("function_1.py")
                        .read_text()
                    ),
                    f.read(),
                )
            fg.generate(
                pathlib.Path(tmpdir, "another_handler.py"),
                "__call__",
                options=function_generator.FunctionGenerateOptions(max_batch_size=10),
            )
            with open(pathlib.Path(tmpdir, "another_handler.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._model_composer.model_method")
                        .joinpath("fixtures")  # type: ignore[no-untyped-call]
                        .joinpath("function_2.py")
                        .read_text()
                    ),
                    f.read(),
                )


if __name__ == "__main__":
    absltest.main()
