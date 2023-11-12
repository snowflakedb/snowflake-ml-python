import pathlib
import tempfile

import importlib_resources
from absl.testing import absltest

from snowflake.ml.model._module_model.module_method import handler_generator


class HandlerGeneratorTest(absltest.TestCase):
    def test_handler_generator(self) -> None:
        hg = handler_generator.HandlerGenerator(pathlib.PurePosixPath("@a.b.c/abc/model.zip"))
        with tempfile.TemporaryDirectory() as tmpdir:
            hg.generate(
                pathlib.Path(tmpdir, "handler.py"),
                "predict",
            )
            with open(pathlib.Path(tmpdir, "handler.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._module_model.module_method")
                        .joinpath("fixtures")  # type: ignore[no-untyped-call]
                        .joinpath("handler_fixture_1.py_fixture")
                        .read_text()
                    ),
                    f.read(),
                )
            hg.generate(
                pathlib.Path(tmpdir, "another_handler.py"),
                "__call__",
                options=handler_generator.HandlerGenerateOptions(max_batch_size=10),
            )
            with open(pathlib.Path(tmpdir, "another_handler.py"), encoding="utf-8") as f:
                self.assertEqual(
                    (
                        importlib_resources.files("snowflake.ml.model._module_model.module_method")
                        .joinpath("fixtures")  # type: ignore[no-untyped-call]
                        .joinpath("handler_fixture_2.py_fixture")
                        .read_text()
                    ),
                    f.read(),
                )


if __name__ == "__main__":
    absltest.main()
