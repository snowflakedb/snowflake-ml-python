import ast
import inspect
import os
import subprocess
import sys
import tempfile
import warnings
from unittest import mock

from absl.testing import absltest

from snowflake import cortex
from snowflake.cortex import _util


class PackageVisibilityTest(absltest.TestCase):
    """Ensure that the functions in this package are visible externally."""

    def setUp(self) -> None:
        # The per-function deprecation warning fires once per process; reset so each test observes it.
        _util._emitted_deprecations.clear()

    def test_package_import_does_not_warn(self) -> None:
        # Importing the package must not warn: it would wrongly flag users of non-deprecated APIs such
        # as Finetune. The deprecation warning is emitted per deprecated function, on use.
        script = """
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import snowflake.cortex

# Only our own cortex deprecation matters here; unrelated dependency warnings are not this test's concern.
deprecations = [
    w
    for w in caught
    if issubclass(w.category, DeprecationWarning) and "snowflake.cortex." in str(w.message)
]
print(len(deprecations))
"""
        result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
        self.assertEqual("0", result.stdout.strip())

    def test_package_import_does_not_load_snowpark(self) -> None:
        script = """
import sys

import snowflake.cortex

print("snowflake.snowpark" in sys.modules)
snowflake.cortex.complete
print("snowflake.snowpark" in sys.modules)
"""
        result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
        self.assertEqual(["False", "True"], result.stdout.split())

    def test_lazy_imports_match_all(self) -> None:
        self.assertEqual(set(cortex.__all__), set(cortex._LAZY_IMPORTS))

    def test_all_names_resolve(self) -> None:
        for name in cortex.__all__:
            with self.subTest(name=name):
                self.assertIsNotNone(getattr(cortex, name))

    def test_type_checking_imports_match_all(self) -> None:
        tree = ast.parse(inspect.getsource(cortex))
        type_checking_block = next(
            node
            for node in tree.body
            if isinstance(node, ast.If) and isinstance(node.test, ast.Attribute) and node.test.attr == "TYPE_CHECKING"
        )
        imported = {
            alias.asname or alias.name
            for node in type_checking_block.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        self.assertEqual(set(cortex.__all__), imported)

    def test_dir_lists_only_public_api(self) -> None:
        self.assertEqual(set(cortex.__all__), {name for name in dir(cortex) if not name.startswith("_")})

    def test_unknown_name_raises_attribute_error(self) -> None:
        self.assertFalse(hasattr(cortex, "does_not_exist"))

    def test_other_distributions_can_add_subpackages(self) -> None:
        with tempfile.TemporaryDirectory() as site_dir:
            subpackage_dir = os.path.join(site_dir, "snowflake", "cortex", "fake_ext")
            os.makedirs(subpackage_dir)
            with open(os.path.join(subpackage_dir, "__init__.py"), "w") as f:
                f.write("VALUE = 'ok'\n")

            script = """
import sys

import snowflake.cortex.fake_ext

print(snowflake.cortex.fake_ext.VALUE)
print("snowflake.snowpark" in sys.modules)
"""
            env = {**os.environ, "PYTHONPATH": os.pathsep.join([*sys.path, site_dir])}
            result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True, env=env)
        self.assertEqual(["ok", "False"], result.stdout.split())

    @mock.patch("snowflake.cortex._complete._complete_impl", autospec=True)
    def test_deprecated_function_emits_one_warning_on_use(self, mock_complete_impl: mock.Mock) -> None:
        mock_complete_impl.return_value = "response"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual("response", cortex.complete("model", "prompt"))

        deprecations = [
            warning
            for warning in caught
            if issubclass(warning.category, DeprecationWarning)
            and "snowflake.cortex.complete is deprecated" in str(warning.message)
        ]
        self.assertEqual(len(deprecations), 1)
        message = str(deprecations[0].message)
        # Points at the specific snowpark.functions replacement and its API doc URL.
        self.assertIn("Snowflake AI Functions", message)
        self.assertIn("snowflake.snowpark.functions.ai_complete", message)
        self.assertIn("snowflake.snowpark.functions.ai_complete", message.split("API:")[-1])
        self.assertIn("aisql-migrate-legacy-functions", message)

    def test_finetune_does_not_warn(self) -> None:
        # Finetune is NOT being deprecated, so using it must not emit a deprecation warning.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cortex.Finetune(session=mock.MagicMock())

        deprecations = [
            warning
            for warning in caught
            if issubclass(warning.category, DeprecationWarning) and "snowflake.cortex." in str(warning.message)
        ]
        self.assertEqual(len(deprecations), 0)

    def test_classify_text_visible(self) -> None:
        self.assertTrue(callable(cortex.ClassifyText))
        self.assertTrue(callable(cortex.classify_text))

    def test_candidate_visible(self) -> None:
        self.assertTrue(callable(cortex.Candidate))

    def test_complete_visible(self) -> None:
        self.assertTrue(callable(cortex.Complete))
        self.assertTrue(callable(cortex.complete))
        self.assertTrue(callable(cortex.CompleteOptions))

    def test_extract_answer_visible(self) -> None:
        self.assertTrue(callable(cortex.ExtractAnswer))
        self.assertTrue(callable(cortex.extract_answer))

    def test_evaluate_visible(self) -> None:
        self.assertTrue(callable(cortex.evaluate))
        self.assertTrue(callable(cortex.EvaluationJob))
        self.assertTrue(callable(cortex.EvaluationMetric))
        self.assertTrue(callable(cortex.EvaluationResult))

    def test_embed_text_768_visible(self) -> None:
        self.assertTrue(callable(cortex.EmbedText768))
        self.assertTrue(callable(cortex.embed_text_768))

    def test_embed_text_1024_visible(self) -> None:
        self.assertTrue(callable(cortex.EmbedText1024))
        self.assertTrue(callable(cortex.embed_text_1024))

    def test_sentiment_visible(self) -> None:
        self.assertTrue(callable(cortex.Sentiment))
        self.assertTrue(callable(cortex.sentiment))

    def test_summarize_visible(self) -> None:
        self.assertTrue(callable(cortex.Summarize))
        self.assertTrue(callable(cortex.summarize))

    def test_translate_visible(self) -> None:
        self.assertTrue(callable(cortex.Translate))
        self.assertTrue(callable(cortex.translate))

    def test_finetune_visible(self) -> None:
        self.assertTrue(callable(cortex.Finetune))
        self.assertTrue(callable(cortex.FinetuneJob))
        self.assertTrue(callable(cortex.FinetuneStatus))

    def test_optimization_job_visible(self) -> None:
        self.assertTrue(callable(cortex.OptimizationJob))
        self.assertTrue(callable(cortex.OptimizationResult))

    def test_optimize_prompt_visible(self) -> None:
        self.assertTrue(callable(cortex.optimize_prompt))


if __name__ == "__main__":
    absltest.main()
