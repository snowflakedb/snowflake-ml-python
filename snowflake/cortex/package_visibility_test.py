from absl.testing import absltest

from snowflake import cortex


class PackageVisibilityTest(absltest.TestCase):
    """Ensure that the functions in this package are visible externally."""

    def test_classify_text_visible(self) -> None:
        self.assertTrue(callable(cortex.ClassifyText))

    def test_complete_visible(self) -> None:
        self.assertTrue(callable(cortex.Complete))
        self.assertTrue(callable(cortex.CompleteOptions))

    def test_extract_answer_visible(self) -> None:
        self.assertTrue(callable(cortex.ExtractAnswer))

    def test_embed_text_768_visible(self) -> None:
        self.assertTrue(callable(cortex.EmbedText768))

    def test_embed_text_1024_visible(self) -> None:
        self.assertTrue(callable(cortex.EmbedText1024))

    def test_sentiment_visible(self) -> None:
        self.assertTrue(callable(cortex.Sentiment))

    def test_summarize_visible(self) -> None:
        self.assertTrue(callable(cortex.Summarize))

    def test_translate_visible(self) -> None:
        self.assertTrue(callable(cortex.Translate))

    def test_finetune_visible(self) -> None:
        self.assertTrue(callable(cortex.Finetune))
        self.assertTrue(callable(cortex.FinetuneJob))
        self.assertTrue(callable(cortex.FinetuneStatus))


if __name__ == "__main__":
    absltest.main()
