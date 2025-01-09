from absl.testing import absltest

from snowflake import cortex


class PackageVisibilityTest(absltest.TestCase):
    """Ensure that the functions in this package are visible externally."""

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
