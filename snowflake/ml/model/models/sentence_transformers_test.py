import json
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
                lazy_upload=False,
            )
            mock_snapshot.assert_called_once()
            self.assertEqual(model.repo_snapshot_dir, fake_snapshot_dir)

    def test_lazy_upload_lists_repo_files_without_snapshot_download(self) -> None:
        """Lazy upload lists repo files and avoids downloading the full snapshot."""
        repo_files = ["modules.json", "1_Pooling/config.json", "model.safetensors"]
        repo_file_sizes = {
            "modules.json": 120,
            "1_Pooling/config.json": 200,
            "model.safetensors": 1000,
        }
        modules = [
            {"idx": 0, "path": "", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
        ]

        def fake_download(
            *,
            repo_id: str,
            filename: str,
            revision: object,
            token: object,
            local_dir: str,
            **kwargs: object,
        ) -> str:
            del repo_id, revision, token, kwargs
            if filename == "modules.json":
                with open(os.path.join(local_dir, "modules.json"), "w") as f:
                    json.dump(modules, f)
                return os.path.join(local_dir, "modules.json")
            if filename == "1_Pooling/config.json":
                pooling_dir = os.path.join(local_dir, "1_Pooling")
                os.makedirs(pooling_dir, exist_ok=True)
                config_path = os.path.join(pooling_dir, "config.json")
                with open(config_path, "w") as f:
                    json.dump({"word_embedding_dimension": 384}, f)
                return config_path
            raise AssertionError(f"Unexpected download: {filename}")

        with absltest.mock.patch("huggingface_hub.snapshot_download") as mock_snapshot_download, absltest.mock.patch(
            "huggingface_hub.hf_hub_download",
            side_effect=fake_download,
        ) as mock_hf_hub_download, absltest.mock.patch("huggingface_hub.HfApi") as mock_hf_api:
            mock_hf_api.return_value.model_info.return_value.siblings = [
                absltest.mock.Mock(rfilename=filename, size=repo_file_sizes[filename]) for filename in repo_files
            ]

            model = huggingface.SentenceTransformer(
                model="sentence-transformers/all-MiniLM-L6-v2",
                compute_pool_for_log=None,
                lazy_upload=True,
            )

            mock_snapshot_download.assert_not_called()
            mock_hf_api.return_value.model_info.assert_called_once_with(
                repo_id="sentence-transformers/all-MiniLM-L6-v2",
                revision=None,
                token=None,
                files_metadata=True,
            )
            self.assertEqual(model._lazy_repo_files, repo_files)
            self.assertEqual(model._lazy_file_sizes, repo_file_sizes)
            self.assertEqual(
                model._lazy_download_kwargs,
                {"repo_id": "sentence-transformers/all-MiniLM-L6-v2", "revision": None},
            )
            downloaded_filenames = {call.kwargs["filename"] for call in mock_hf_hub_download.call_args_list}
            self.assertEqual(downloaded_filenames, {"modules.json", "1_Pooling/config.json"})

    def test_lazy_upload_false_uses_snapshot_download(self) -> None:
        """Opting out of lazy upload restores eager snapshot download."""
        with absltest.mock.patch(
            "huggingface_hub.snapshot_download", return_value="/some/path"
        ) as mock_snapshot_download:
            model = huggingface.SentenceTransformer(
                model="sentence-transformers/all-MiniLM-L6-v2",
                compute_pool_for_log=None,
                lazy_upload=False,
            )

            mock_snapshot_download.assert_called_once()
            self.assertIsNone(model._lazy_repo_files)
            self.assertEqual(model.repo_snapshot_dir, "/some/path")

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
