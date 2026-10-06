import subprocess
import sys
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

    def test_complete_visible(self) -> None:
        self.assertTrue(callable(cortex.Complete))
        self.assertTrue(callable(cortex.complete))
        self.assertTrue(callable(cortex.CompleteOptions))

    def test_extract_answer_visible(self) -> None:
        self.assertTrue(callable(cortex.ExtractAnswer))
        self.assertTrue(callable(cortex.extract_answer))

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


if __name__ == "__main__":
    absltest.main()
