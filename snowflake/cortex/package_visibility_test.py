from absl.testing import absltest

from snowflake import cortex


class PackageVisibilityTest(absltest.TestCase):
    """Ensure that the functions in this package are visible externally."""

    def test_complete_visible(self) -> None:
        self.assertTrue(callable(cortex.Complete))
        self.assertTrue(callable(cortex.CompleteOptions))

    def test_extract_answer_visible(self) -> None:
        self.assertTrue(callable(cortex.ExtractAnswer))

    def test_sentiment_visible(self) -> None:
        self.assertTrue(callable(cortex.Sentiment))

    def test_summarize_visible(self) -> None:
        self.assertTrue(callable(cortex.Summarize))

    def test_translate_visible(self) -> None:
        self.assertTrue(callable(cortex.Translate))


if __name__ == "__main__":
    absltest.main()
