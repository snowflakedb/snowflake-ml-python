import json
import os
import tempfile

from absl.testing import absltest

from snowflake.ml.model.models import huggingface


class TransformersPipelineTest(absltest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.cache_dir = tempfile.TemporaryDirectory()
        self._original_hf_home = os.getenv("HF_HOME", None)
        os.environ["HF_HOME"] = self.cache_dir.name

    @classmethod
    def tearDownClass(self) -> None:
        if self._original_hf_home:
            os.environ["HF_HOME"] = self._original_hf_home
        else:
            del os.environ["HF_HOME"]
        self.cache_dir.cleanup()

    def test_remote_logging_basic(self) -> None:
        """Test that remote logging creates a pipeline correctly without requiring transformers."""
        pipeline = huggingface.TransformersPipeline(
            task="text-generation",
            model="facebook/opt-125m",
            # compute_pool_for_log uses default value (DEFAULT_CPU_COMPUTE_POOL)
        )
        self.assertEqual(pipeline.model, "facebook/opt-125m")
        self.assertEqual(pipeline.task, "text-generation")

    def test_infers_task_from_hub_for_remote_logging(self) -> None:
        """When task is omitted, resolve it from Hub pipeline_tag."""
        with absltest.mock.patch("huggingface_hub.HfApi") as mock_hf_api:
            mock_hf_api.return_value.model_info.return_value.pipeline_tag = "text-generation"

            pipeline = huggingface.TransformersPipeline(
                model="facebook/opt-125m",
            )

            mock_hf_api.return_value.model_info.assert_called_once_with(
                repo_id="facebook/opt-125m",
                revision=None,
                token=None,
            )
            self.assertEqual(pipeline.task, "text-generation")
            self.assertEqual(pipeline.model, "facebook/opt-125m")

    def test_infers_task_raises_when_pipeline_tag_missing(self) -> None:
        """Fail clearly when Hub metadata has no pipeline_tag."""
        with absltest.mock.patch("huggingface_hub.HfApi") as mock_hf_api:
            mock_hf_api.return_value.model_info.return_value.pipeline_tag = None

            with self.assertRaisesRegex(
                RuntimeError,
                "Unable to determine the HuggingFace pipeline task",
            ):
                huggingface.TransformersPipeline(model="some-org/untagged-model")

    def test_lazy_upload_infers_task_from_same_model_info_call(self) -> None:
        """Lazy upload reuses model_info for both file listing and task inference."""
        repo_files = ["config.json", "model.safetensors", "tokenizer_config.json"]
        repo_file_sizes = {
            "config.json": 570,
            "model.safetensors": 1000,
            "tokenizer_config.json": 48,
        }

        with absltest.mock.patch("huggingface_hub.snapshot_download") as mock_snapshot_download, absltest.mock.patch(
            "huggingface_hub.hf_hub_download"
        ) as mock_hf_hub_download, absltest.mock.patch("huggingface_hub.HfApi") as mock_hf_api:
            mock_info = mock_hf_api.return_value.model_info.return_value
            mock_info.pipeline_tag = "text-generation"
            mock_info.siblings = [
                absltest.mock.Mock(rfilename=filename, size=repo_file_sizes[filename]) for filename in repo_files
            ]

            pipeline = huggingface.TransformersPipeline(
                model="facebook/opt-125m",
                compute_pool_for_log=None,
                lazy_upload=True,
            )

            mock_snapshot_download.assert_not_called()
            mock_hf_api.return_value.model_info.assert_called_once_with(
                repo_id="facebook/opt-125m",
                revision=None,
                token=None,
                files_metadata=True,
            )
            self.assertEqual(pipeline.task, "text-generation")
            self.assertEqual(pipeline._lazy_repo_files, repo_files)
            downloaded_filenames = {call.kwargs["filename"] for call in mock_hf_hub_download.call_args_list}
            self.assertEqual(downloaded_filenames, {"tokenizer_config.json"})

    def test_explicit_task_skips_hub_inference(self) -> None:
        """An explicit task must not trigger a Hub lookup in remote mode."""
        with absltest.mock.patch("huggingface_hub.HfApi") as mock_hf_api:
            pipeline = huggingface.TransformersPipeline(
                task="fill-mask",
                model="google-bert/bert-base-uncased",
            )
            mock_hf_api.assert_not_called()
            self.assertEqual(pipeline.task, "fill-mask")

    def test_wrapper(self) -> None:
        huggingface.TransformersPipeline(
            task="text-generation",
            model="facebook/opt-125m",
            compute_pool_for_log=None,
        )
        huggingface.TransformersPipeline(
            task="fill-mask",
            model="google-bert/bert-base-uncased",
            compute_pool_for_log=None,
        )

        huggingface.TransformersPipeline(
            task="new-task",
            model="lysandre/test-dynamic-pipeline",
            trust_remote_code=True,
            compute_pool_for_log=None,
        )
        huggingface.TransformersPipeline(
            task="new-task",
            model="lysandre/test-dynamic-pipeline",
            trust_remote_code=True,
            download_snapshot=True,
        )

        # Model is required; task may be omitted and inferred from Hub metadata.
        with self.assertRaises(TypeError):
            huggingface.TransformersPipeline(task="text-generation")  # type: ignore[call-arg]
        with self.assertRaisesRegex(
            RuntimeError,
            "Impossible to instantiate a pipeline without a model being specified.",
        ):
            huggingface.TransformersPipeline(task="text-generation", model=None)  # type: ignore[arg-type]
        with self.assertRaisesRegex(
            RuntimeError,
            "Impossible to use non-string model as input for class TransformersPipeline.",
        ):
            huggingface.TransformersPipeline(task="text-generation", model=1)  # type: ignore[arg-type]

        with self.assertRaisesRegex(
            ValueError,
            "You cannot use both `pipeline\\(... device_map=..., model_kwargs",
        ):
            huggingface.TransformersPipeline(
                task="text-generation",
                model="facebook/opt-125m",
                device_map="auto",
                model_kwargs={"device_map": "auto"},
            )

        with self.assertWarnsRegex(
            UserWarning,
            "Both `device` and `device_map` are specified.",
        ):
            huggingface.TransformersPipeline(
                task="text-generation",
                model="facebook/opt-125m",
                device_map="auto",
                device=0,
            )

    def test_lazy_upload_lists_repo_files_without_snapshot_download(self) -> None:
        """Lazy upload lists repo files and avoids downloading the full snapshot."""
        repo_files = ["config.json", "model.safetensors", "tokenizer_config.json"]
        repo_file_sizes = {
            "config.json": 570,
            "model.safetensors": 1000,
            "tokenizer_config.json": 48,
        }

        with absltest.mock.patch("huggingface_hub.snapshot_download") as mock_snapshot_download, absltest.mock.patch(
            "huggingface_hub.hf_hub_download"
        ) as mock_hf_hub_download, absltest.mock.patch("huggingface_hub.HfApi") as mock_hf_api:
            mock_hf_api.return_value.model_info.return_value.siblings = [
                absltest.mock.Mock(rfilename=filename, size=repo_file_sizes[filename]) for filename in repo_files
            ]

            pipeline = huggingface.TransformersPipeline(
                task="text-generation",
                model="facebook/opt-125m",
                compute_pool_for_log=None,
                lazy_upload=True,
            )

            mock_snapshot_download.assert_not_called()
            mock_hf_api.return_value.model_info.assert_called_once_with(
                repo_id="facebook/opt-125m",
                revision=None,
                token=None,
                files_metadata=True,
            )
            self.assertEqual(pipeline._lazy_repo_files, repo_files)
            self.assertEqual(pipeline._lazy_file_sizes, repo_file_sizes)
            self.assertEqual(
                pipeline._lazy_download_kwargs,
                {"repo_id": "facebook/opt-125m", "revision": None},
            )
            downloaded_filenames = {call.kwargs["filename"] for call in mock_hf_hub_download.call_args_list}
            self.assertEqual(downloaded_filenames, {"tokenizer_config.json"})

    def test_lazy_upload_skips_chat_template_metadata_for_non_chat_tasks(self) -> None:
        """Lazy upload should not download chat-template metadata for unrelated tasks."""
        repo_files = ["config.json", "model.safetensors", "tokenizer_config.json"]

        with absltest.mock.patch("huggingface_hub.snapshot_download") as mock_snapshot_download, absltest.mock.patch(
            "huggingface_hub.hf_hub_download"
        ) as mock_hf_hub_download, absltest.mock.patch("huggingface_hub.HfApi") as mock_hf_api:
            mock_hf_api.return_value.model_info.return_value.siblings = [
                absltest.mock.Mock(rfilename=filename, size=100) for filename in repo_files
            ]

            huggingface.TransformersPipeline(
                task="text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                compute_pool_for_log=None,
                lazy_upload=True,
            )

            mock_snapshot_download.assert_not_called()
            mock_hf_hub_download.assert_not_called()

    def test_lazy_upload_detects_chat_template_from_metadata_download(self) -> None:
        """Lazy upload still detects chat templates from metadata files."""
        with tempfile.TemporaryDirectory() as repo_dir:
            config_path = os.path.join(repo_dir, "tokenizer_config.json")
            with open(config_path, "w") as f:
                json.dump({"chat_template": "{% for msg in messages %}{{ msg }}{% endfor %}"}, f)

            with absltest.mock.patch(
                "huggingface_hub.snapshot_download"
            ) as mock_snapshot_download, absltest.mock.patch(
                "huggingface_hub.hf_hub_download",
                return_value=config_path,
            ), absltest.mock.patch(
                "huggingface_hub.HfApi"
            ) as mock_hf_api, absltest.mock.patch(
                "snowflake.ml.model.models.huggingface.tempfile.mkdtemp",
                return_value=repo_dir,
            ):
                mock_hf_api.return_value.model_info.return_value.siblings = [
                    absltest.mock.Mock(rfilename="config.json", size=100),
                    absltest.mock.Mock(rfilename="tokenizer_config.json", size=200),
                ]

                pipeline = huggingface.TransformersPipeline(
                    task="text-generation",
                    model="some-model/with-chat-template",
                    compute_pool_for_log=None,
                    lazy_upload=True,
                )

                mock_snapshot_download.assert_not_called()
                self.assertTrue(pipeline.has_chat_template)

    def test_lazy_upload_false_uses_snapshot_download(self) -> None:
        """Opting out of lazy upload restores eager snapshot download."""
        with absltest.mock.patch(
            "huggingface_hub.snapshot_download", return_value="/some/path"
        ) as mock_snapshot_download:
            pipeline = huggingface.TransformersPipeline(
                task="text-generation",
                model="facebook/opt-125m",
                compute_pool_for_log=None,
                lazy_upload=False,
            )

            mock_snapshot_download.assert_called_once()
            self.assertIsNone(pipeline._lazy_repo_files)
            self.assertEqual(pipeline.repo_snapshot_dir, "/some/path")

    def test_remote_logging_skips_snapshot_download(self) -> None:
        """Test that when compute_pool_for_log is set, we don't download from huggingface_hub."""
        with absltest.mock.patch("huggingface_hub.snapshot_download") as mock_snapshot_download:

            # Create a pipeline with compute_pool_for_log set (remote logging mode)
            pipeline = huggingface.TransformersPipeline(
                task="text-generation",
                model="facebook/opt-125m",
                compute_pool_for_log="test_compute_pool",
            )

            # Verify that snapshot_download was NOT called
            mock_snapshot_download.assert_not_called()

            # Verify the pipeline was created correctly
            self.assertEqual(pipeline.model, "facebook/opt-125m")
            self.assertEqual(pipeline.task, "text-generation")
            self.assertEqual(pipeline.compute_pool_for_log, "test_compute_pool")
            self.assertIsNone(pipeline.repo_snapshot_dir)

    def test_detect_chat_template_with_jinja_file(self) -> None:
        """Test _detect_chat_template returns True when chat_template.jinja file exists."""
        with tempfile.TemporaryDirectory() as repo_dir:
            jinja_path = os.path.join(repo_dir, "chat_template.jinja")
            with open(jinja_path, "w") as f:
                f.write("{{ messages }}")

            self.assertTrue(huggingface.TransformersPipeline._detect_chat_template(repo_dir))

    def test_detect_chat_template_in_tokenizer_config(self) -> None:
        """Test _detect_chat_template returns True when tokenizer_config.json contains chat_template."""
        with tempfile.TemporaryDirectory() as repo_dir:
            config_path = os.path.join(repo_dir, "tokenizer_config.json")
            with open(config_path, "w") as f:
                json.dump({"chat_template": "{% for msg in messages %}{{ msg }}{% endfor %}"}, f)

            self.assertTrue(huggingface.TransformersPipeline._detect_chat_template(repo_dir))

    def test_detect_chat_template_none_value_in_tokenizer_config(self) -> None:
        """Test _detect_chat_template returns False when chat_template is explicitly None."""
        with tempfile.TemporaryDirectory() as repo_dir:
            config_path = os.path.join(repo_dir, "tokenizer_config.json")
            with open(config_path, "w") as f:
                json.dump({"chat_template": None}, f)

            self.assertFalse(huggingface.TransformersPipeline._detect_chat_template(repo_dir))

    def test_detect_chat_template_missing_from_tokenizer_config(self) -> None:
        """Test _detect_chat_template returns False when tokenizer_config.json has no chat_template key."""
        with tempfile.TemporaryDirectory() as repo_dir:
            config_path = os.path.join(repo_dir, "tokenizer_config.json")
            with open(config_path, "w") as f:
                json.dump({"model_max_length": 512, "padding_side": "right"}, f)

            self.assertFalse(huggingface.TransformersPipeline._detect_chat_template(repo_dir))

    def test_detect_chat_template_no_config_files(self) -> None:
        """Test _detect_chat_template returns False when no config files exist."""
        with tempfile.TemporaryDirectory() as repo_dir:
            self.assertFalse(huggingface.TransformersPipeline._detect_chat_template(repo_dir))

    def test_detect_chat_template_invalid_json(self) -> None:
        """Test _detect_chat_template returns False when tokenizer_config.json is invalid JSON."""
        with tempfile.TemporaryDirectory() as repo_dir:
            config_path = os.path.join(repo_dir, "tokenizer_config.json")
            with open(config_path, "w") as f:
                f.write("not valid json {{{")

            self.assertFalse(huggingface.TransformersPipeline._detect_chat_template(repo_dir))

    def test_detect_chat_template_jinja_takes_priority(self) -> None:
        """Test _detect_chat_template returns True from jinja file even if tokenizer_config has no template."""
        with tempfile.TemporaryDirectory() as repo_dir:
            config_path = os.path.join(repo_dir, "tokenizer_config.json")
            with open(config_path, "w") as f:
                json.dump({"model_max_length": 512}, f)

            jinja_path = os.path.join(repo_dir, "chat_template.jinja")
            with open(jinja_path, "w") as f:
                f.write("{{ messages }}")

            self.assertTrue(huggingface.TransformersPipeline._detect_chat_template(repo_dir))

    def test_detect_chat_template_set_for_text_generation_snapshot_download(self) -> None:
        """Test that has_chat_template is set when using SNAPSHOT_DOWNLOAD mode with text-generation task."""
        with tempfile.TemporaryDirectory() as repo_dir:
            config_path = os.path.join(repo_dir, "tokenizer_config.json")
            with open(config_path, "w") as f:
                json.dump({"chat_template": "{% for msg in messages %}{{ msg }}{% endfor %}"}, f)

            with absltest.mock.patch("huggingface_hub.snapshot_download", return_value=repo_dir):
                pipeline = huggingface.TransformersPipeline(
                    task="text-generation",
                    model="some-model/with-chat-template",
                    compute_pool_for_log=None,
                    lazy_upload=False,
                )
                self.assertTrue(pipeline.has_chat_template)

    def test_detect_chat_template_false_for_non_text_generation_task(self) -> None:
        """Test that has_chat_template is False when the task is not text-generation."""
        with absltest.mock.patch("huggingface_hub.snapshot_download", return_value="/some/path"):
            pipeline = huggingface.TransformersPipeline(
                task="fill-mask",
                model="some-model/fill-mask",
                compute_pool_for_log=None,
                lazy_upload=False,
            )
            self.assertFalse(pipeline.has_chat_template)

    def test_detect_chat_template_false_for_remote_logging_mode(self) -> None:
        """Test that has_chat_template is False when using remote logging mode."""
        pipeline = huggingface.TransformersPipeline(
            task="text-generation",
            model="some-model/text-gen",
            compute_pool_for_log="some_compute_pool",
        )
        self.assertFalse(pipeline.has_chat_template)

    def test_detect_chat_template_for_image_text_to_text_task(self) -> None:
        """Test that has_chat_template is detected for image-text-to-text task."""
        with tempfile.TemporaryDirectory() as repo_dir:
            config_path = os.path.join(repo_dir, "tokenizer_config.json")
            with open(config_path, "w") as f:
                json.dump({"chat_template": "{% for msg in messages %}{{ msg }}{% endfor %}"}, f)

            with absltest.mock.patch("huggingface_hub.snapshot_download", return_value=repo_dir):
                pipeline = huggingface.TransformersPipeline(
                    task="image-text-to-text",
                    model="some-model/vision-llm",
                    compute_pool_for_log=None,
                    lazy_upload=False,
                )
                self.assertTrue(pipeline.has_chat_template)

    def test_detect_chat_template_for_video_text_to_text_task(self) -> None:
        """Test that has_chat_template is detected for video-text-to-text task."""
        with tempfile.TemporaryDirectory() as repo_dir:
            config_path = os.path.join(repo_dir, "tokenizer_config.json")
            with open(config_path, "w") as f:
                json.dump({"chat_template": "{% for msg in messages %}{{ msg }}{% endfor %}"}, f)

            with absltest.mock.patch("huggingface_hub.snapshot_download", return_value=repo_dir):
                pipeline = huggingface.TransformersPipeline(
                    task="video-text-to-text",
                    model="some-model/video-llm",
                    compute_pool_for_log=None,
                    lazy_upload=False,
                )
                self.assertTrue(pipeline.has_chat_template)

    def test_detect_chat_template_for_audio_text_to_text_task(self) -> None:
        """Test that has_chat_template is detected for audio-text-to-text task."""
        with tempfile.TemporaryDirectory() as repo_dir:
            jinja_path = os.path.join(repo_dir, "chat_template.jinja")
            with open(jinja_path, "w") as f:
                f.write("{{ messages }}")

            with absltest.mock.patch("huggingface_hub.snapshot_download", return_value=repo_dir):
                pipeline = huggingface.TransformersPipeline(
                    task="audio-text-to-text",
                    model="some-model/audio-llm",
                    compute_pool_for_log=None,
                    lazy_upload=False,
                )
                self.assertTrue(pipeline.has_chat_template)

    def test_detect_chat_template_false_for_multimodal_without_template(self) -> None:
        """Test that has_chat_template is False for multimodal tasks when no template exists."""
        with tempfile.TemporaryDirectory() as repo_dir:
            config_path = os.path.join(repo_dir, "tokenizer_config.json")
            with open(config_path, "w") as f:
                json.dump({"model_max_length": 2048}, f)

            with absltest.mock.patch("huggingface_hub.snapshot_download", return_value=repo_dir):
                pipeline = huggingface.TransformersPipeline(
                    task="image-text-to-text",
                    model="some-model/vision-no-template",
                    compute_pool_for_log=None,
                    lazy_upload=False,
                )
                self.assertFalse(pipeline.has_chat_template)


if __name__ == "__main__":
    absltest.main()
