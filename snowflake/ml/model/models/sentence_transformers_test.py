import os
import tempfile

from absl.testing import absltest

from snowflake.ml.model.models import huggingface


class SentenceTransformerTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls._original_hf_home = os.getenv("HF_HOME", None)
        os.environ["HF_HOME"] = cls.cache_dir.name

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._original_hf_home:
            os.environ["HF_HOME"] = cls._original_hf_home
        else:
            del os.environ["HF_HOME"]
        cls.cache_dir.cleanup()

    def test_remote_logging_basic(self) -> None:
        model = huggingface.SentenceTransformer(
            model="sentence-transformers/all-MiniLM-L6-v2",
        )
        self.assertIsInstance(model, huggingface.TransformersPipeline)
        self.assertEqual(model.model, "sentence-transformers/all-MiniLM-L6-v2")
        self.assertIsNone(model.task)
        self.assertIsNotNone(model.compute_pool_for_log)

    def test_remote_logging_skips_snapshot_download(self) -> None:
        with absltest.mock.patch("huggingface_hub.snapshot_download") as mock_snapshot:
            model = huggingface.SentenceTransformer(
                model="sentence-transformers/all-MiniLM-L6-v2",
                compute_pool_for_log="test_pool",
            )
            mock_snapshot.assert_not_called()
            self.assertIsNone(model.repo_snapshot_dir)
            self.assertEqual(model.compute_pool_for_log, "test_pool")

    def test_local_mode_downloads_snapshot(self) -> None:
        fake_snapshot_dir = "/tmp/fake_snapshot"
        with absltest.mock.patch("huggingface_hub.snapshot_download", return_value=fake_snapshot_dir) as mock_snapshot:
            model = huggingface.SentenceTransformer(
                model="sentence-transformers/all-MiniLM-L6-v2",
                compute_pool_for_log=None,
            )
            mock_snapshot.assert_called_once()
            self.assertEqual(model.repo_snapshot_dir, fake_snapshot_dir)

    def test_requires_task_is_false(self) -> None:
        self.assertFalse(huggingface.SentenceTransformer._requires_task)

    def test_task_is_none(self) -> None:
        model = huggingface.SentenceTransformer(
            model="sentence-transformers/all-MiniLM-L6-v2",
        )
        self.assertIsNone(model.task)

    def test_model_requires_model_arg(self) -> None:
        with self.assertRaises(RuntimeError):
            huggingface.SentenceTransformer(
                model=None,  # type: ignore[arg-type]
            )

    def test_trust_remote_code_defaults_to_false(self) -> None:
        model = huggingface.SentenceTransformer(
            model="sentence-transformers/all-MiniLM-L6-v2",
        )
        self.assertFalse(model.trust_remote_code)

    def test_optional_params_forwarded(self) -> None:
        model = huggingface.SentenceTransformer(
            model="sentence-transformers/all-MiniLM-L6-v2",
            revision="main",
            trust_remote_code=True,
        )
        self.assertEqual(model.revision, "main")
        self.assertTrue(model.trust_remote_code)

    def test_secret_parsing(self) -> None:
        model = huggingface.SentenceTransformer(
            model="sentence-transformers/all-MiniLM-L6-v2",
            token_or_secret="my_db.my_schema.my_secret",
        )
        self.assertIsNotNone(model.secret_identifier)

    def test_has_chat_template_none_for_sentence_transformer(self) -> None:
        model = huggingface.SentenceTransformer(
            model="sentence-transformers/all-MiniLM-L6-v2",
        )
        self.assertIsNone(model.has_chat_template)

    def test_existing_transformers_pipeline_still_requires_task(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Impossible to instantiate a pipeline without a task"):
            huggingface.TransformersPipeline(
                task=None,
                model="some-model",
            )


if __name__ == "__main__":
    absltest.main()
