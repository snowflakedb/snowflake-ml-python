import json
import os
import tempfile

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import inference_engine
from snowflake.ml.model._client.ops import service_ops
from snowflake.ml.model.models import huggingface, huggingface_pipeline
from snowflake.snowpark import async_job, session


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

        # Both task and model are required positional arguments
        with self.assertRaises(TypeError):
            huggingface.TransformersPipeline(model="facebook/opt-125m")  # type: ignore[call-arg]
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
                )
                self.assertTrue(pipeline.has_chat_template)

    def test_detect_chat_template_false_for_non_text_generation_task(self) -> None:
        """Test that has_chat_template is False when the task is not text-generation."""
        with absltest.mock.patch("huggingface_hub.snapshot_download", return_value="/some/path"):
            pipeline = huggingface.TransformersPipeline(
                task="fill-mask",
                model="some-model/fill-mask",
                compute_pool_for_log=None,
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
                )
                self.assertFalse(pipeline.has_chat_template)

    def test_create_service(self) -> None:
        """Test the create_service function with mocked ServiceOperator."""

        # Mock session
        mock_session = absltest.mock.Mock(spec=session.Session)
        mock_session.get_current_database = absltest.mock.Mock(return_value="test_db")
        mock_session.get_current_schema = absltest.mock.Mock(return_value="test_schema")
        mock_session.get_current_warehouse = absltest.mock.Mock(return_value="test_warehouse")

        # Mock the ServiceOperator
        mock_service_operator = absltest.mock.Mock(spec=service_ops.ServiceOperator)
        mock_create_service = absltest.mock.Mock(return_value="test_service_id")
        mock_service_operator.create_service = mock_create_service

        model_ref = huggingface_pipeline.HuggingFacePipelineModel(
            model="openai-community/gpt2",
            task="text-generation",
            trust_remote_code=True,
            download_snapshot=False,
        )

        # Patch the ServiceOperator constructor to return our mock
        with absltest.mock.patch(
            "snowflake.ml.model._client.ops.service_ops.ServiceOperator",
            return_value=mock_service_operator,
        ):
            # Call the create_service function
            result = model_ref.log_model_and_create_service(
                session=mock_session,
                model_name="test_model",
                version_name="v1",
                service_name="test_service",
                service_compute_pool="test_compute_pool",
                image_repo="test_repo",
                ingress_enabled=True,
                min_instances=1,
                max_instances=2,
                num_workers=4,
                max_batch_rows=100,
                cpu_requests="1",
                memory_requests="4Gi",
                gpu_requests="1",
                force_rebuild=True,
                block=True,
            )

            # Check that the result is correct
            self.assertEqual(result, "test_service_id")

            # Verify specific parameters
            args = mock_create_service.call_args.kwargs
            self.assertEqual(args["database_name"], sql_identifier.SqlIdentifier("test_db"))
            self.assertEqual(args["schema_name"], sql_identifier.SqlIdentifier("test_schema"))
            self.assertEqual(args["model_name"], sql_identifier.SqlIdentifier("test_model"))
            self.assertEqual(args["version_name"], sql_identifier.SqlIdentifier("v1"))
            self.assertEqual(args["service_name"], sql_identifier.SqlIdentifier("test_service"))
            self.assertIsNone(args["service_database_name"])
            self.assertIsNone(args["service_schema_name"])
            self.assertEqual(
                args["hf_model_args"],
                service_ops.HFModelArgs(
                    hf_model_name="openai-community/gpt2",
                    hf_task="text-generation",
                    hf_tokenizer=None,
                    hf_revision=None,
                    hf_token=None,
                    hf_trust_remote_code=True,
                    hf_model_kwargs={},
                    pip_requirements=None,
                    conda_dependencies=None,
                    comment=None,
                    warehouse="test_warehouse",
                ),
            )
            self.assertTrue(args["block"])

        # Patch the ServiceOperator constructor to return our mock
        with absltest.mock.patch(
            "snowflake.ml.model._client.ops.service_ops.ServiceOperator",
            return_value=mock_service_operator,
        ):
            # Call the create_service function
            result = model_ref.log_model_and_create_service(
                session=mock_session,
                model_name="test_model",
                version_name="v1",
                service_name="test_service_db.test_service_schema.test_service",
                service_compute_pool="test_compute_pool",
                image_repo="test_repo",
                ingress_enabled=True,
                min_instances=1,
                max_instances=2,
                num_workers=4,
                max_batch_rows=100,
                cpu_requests="1",
                memory_requests="4Gi",
                gpu_requests="1",
                force_rebuild=True,
                block=True,
            )

            # Check that the result is correct
            self.assertEqual(result, "test_service_id")

            # Verify specific parameters
            args = mock_create_service.call_args.kwargs
            self.assertEqual(args["database_name"], sql_identifier.SqlIdentifier("test_db"))
            self.assertEqual(args["schema_name"], sql_identifier.SqlIdentifier("test_schema"))
            self.assertEqual(args["model_name"], sql_identifier.SqlIdentifier("test_model"))
            self.assertEqual(args["version_name"], sql_identifier.SqlIdentifier("v1"))
            self.assertEqual(args["service_name"], sql_identifier.SqlIdentifier("test_service"))
            self.assertEqual(args["service_database_name"], sql_identifier.SqlIdentifier("test_service_db"))
            self.assertEqual(args["service_schema_name"], sql_identifier.SqlIdentifier("test_service_schema"))
            self.assertEqual(
                args["hf_model_args"],
                service_ops.HFModelArgs(
                    hf_model_name="openai-community/gpt2",
                    hf_task="text-generation",
                    hf_tokenizer=None,
                    hf_revision=None,
                    hf_token=None,
                    hf_trust_remote_code=True,
                    hf_model_kwargs={},
                    pip_requirements=None,
                    conda_dependencies=None,
                    comment=None,
                    warehouse="test_warehouse",
                ),
            )
            self.assertTrue(args["block"])

    def test_create_service_async(self) -> None:
        """Test the create_service function in async mode with mocked ServiceOperator."""

        # Mock session
        mock_session = absltest.mock.Mock(spec=session.Session)
        mock_session.get_current_database = absltest.mock.Mock(return_value="test_db")
        mock_session.get_current_schema = absltest.mock.Mock(return_value="test_schema")

        # Mock async job
        mock_async_job = absltest.mock.Mock(spec=async_job.AsyncJob)

        # Mock the ServiceOperator
        mock_service_operator = absltest.mock.Mock(spec=service_ops.ServiceOperator)
        mock_create_service = absltest.mock.Mock(return_value=mock_async_job)
        mock_service_operator.create_service = mock_create_service

        model_ref = huggingface_pipeline.HuggingFacePipelineModel(
            model="openai-community/gpt2",
            task="text-generation",
            trust_remote_code=True,
            download_snapshot=False,
        )

        # Patch the ServiceOperator constructor to return our mock
        with absltest.mock.patch(
            "snowflake.ml.model._client.ops.service_ops.ServiceOperator",
            return_value=mock_service_operator,
        ):
            # Call the create_service function in non-blocking mode
            result = model_ref.log_model_and_create_service(
                session=mock_session,
                model_name="test_model",
                service_name="test_service",
                service_compute_pool="test_compute_pool",
                image_repo="test_repo",
                block=False,  # Non-blocking mode
            )

            # Check that the result is an AsyncJob
            self.assertIs(result, mock_async_job)

            # Check that create_service was called with block=False
            args = mock_create_service.call_args.kwargs
            self.assertFalse(args["block"])

    def test_create_service_with_inference_engine_options(self) -> None:
        """Test the create_service function with inference engine options."""
        # Mock session
        mock_session = absltest.mock.Mock(spec=session.Session)
        mock_session.get_current_database = absltest.mock.Mock(return_value="test_db")
        mock_session.get_current_schema = absltest.mock.Mock(return_value="test_schema")
        mock_session.get_current_warehouse = absltest.mock.Mock(return_value="test_warehouse")

        # Mock the ServiceOperator
        mock_service_operator = absltest.mock.Mock(spec=service_ops.ServiceOperator)
        mock_create_service = absltest.mock.Mock(return_value="test_service_id")
        mock_service_operator.create_service = mock_create_service

        model_ref = huggingface_pipeline.HuggingFacePipelineModel(
            model="openai-community/gpt2",
            task="text-generation",
            trust_remote_code=True,
            download_snapshot=False,
        )

        # Patch the ServiceOperator constructor to return our mock
        with absltest.mock.patch(
            "snowflake.ml.model._client.ops.service_ops.ServiceOperator",
            return_value=mock_service_operator,
        ):
            result = model_ref.log_model_and_create_service(
                session=mock_session,
                model_name="test_model",
                service_name="test_service",
                service_compute_pool="test_compute_pool",
                block=True,
                inference_engine_options={
                    "engine": inference_engine.InferenceEngine.VLLM,
                    "engine_args_override": ["--max_tokens=100", "--temperature=0.7"],
                },
            )
            self.assertEqual(result, "test_service_id")
            args = mock_create_service.call_args.kwargs
            self.assertEqual(
                args["inference_engine_args"],
                service_ops.InferenceEngineArgs(
                    inference_engine=inference_engine.InferenceEngine.VLLM,
                    inference_engine_args_override=["--max_tokens=100", "--temperature=0.7"],
                ),
            )

    def test_create_service_with_inference_engine_options_invalid_task(self) -> None:
        """Test the create_service function with inference engine options."""
        # Mock session
        mock_session = absltest.mock.Mock(spec=session.Session)
        mock_session.get_current_database = absltest.mock.Mock(return_value="test_db")
        mock_session.get_current_schema = absltest.mock.Mock(return_value="test_schema")
        mock_session.get_current_warehouse = absltest.mock.Mock(return_value="test_warehouse")

        # Mock the ServiceOperator
        mock_service_operator = absltest.mock.Mock(spec=service_ops.ServiceOperator)
        mock_create_service = absltest.mock.Mock(return_value="test_service_id")
        mock_service_operator.create_service = mock_create_service

        model_ref = huggingface_pipeline.HuggingFacePipelineModel(
            model="google-bert/bert-base-uncased",
            task="fill-mask",
            trust_remote_code=True,
            download_snapshot=False,
        )

        # Patch the ServiceOperator constructor to return our mock
        with absltest.mock.patch(
            "snowflake.ml.model._client.ops.service_ops.ServiceOperator",
            return_value=mock_service_operator,
        ):
            with self.assertRaises(ValueError) as cm:
                model_ref.log_model_and_create_service(
                    session=mock_session,
                    model_name="test_model",
                    service_name="test_service",
                    service_compute_pool="test_compute_pool",
                    block=True,
                    inference_engine_options={
                        "inference_engine": inference_engine.InferenceEngine.VLLM,
                        "inference_engine_args_override": ["--max_tokens=100", "--temperature=0.7"],
                    },
                )
                self.assertEqual(
                    str(cm.exception),
                    "Currently, InferenceEngine using inference_engine_options is only supported for "
                    "HuggingFace text-generation models.",
                )


if __name__ == "__main__":
    absltest.main()
