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

    def test_remote_logging_does_not_call_from_pretrained(self) -> None:
        """Test that verifies the correct behavior: AutoConfig.from_pretrained is not called
        even when using remote logging.

        When using remote logging, we should not be accessing HuggingFace hosts.
        The model artifacts should be downloaded during deployment on the compute pool,
        not during the TransformersPipeline constructor.
        """
        with absltest.mock.patch("transformers.AutoConfig.from_pretrained") as mock_from_pretrained:
            mock_config = absltest.mock.Mock()
            mock_config._commit_hash = "fake_commit_hash"
            mock_config.custom_pipelines = {}
            mock_from_pretrained.return_value = mock_config

            # Using the default compute pool (DEFAULT_CPU_COMPUTE_POOL) should NOT
            # call AutoConfig.from_pretrained, but due to the bug, it does.
            huggingface.TransformersPipeline(
                task="text-generation",
                model="facebook/opt-125m",
                # compute_pool_for_log uses default value (DEFAULT_CPU_COMPUTE_POOL)
            )

            # BUG: This assertion passes because AutoConfig.from_pretrained IS called
            # even though we're using a compute pool. This should NOT happen.
            mock_from_pretrained.assert_not_called()

    def test_wrapper(self) -> None:

        from transformers import testing_utils

        is_hf_hub_available = False
        try:
            import huggingface_hub as _hf_hub  # noqa: F401

            is_hf_hub_available = True

        except ImportError:
            pass

        with self.assertWarnsRegex(
            UserWarning,
            "Using a pipeline without specifying a model name and revision in production is not recommended.",
        ):
            huggingface.TransformersPipeline(task="text-generation", compute_pool_for_log=None)
        huggingface.TransformersPipeline(
            task="text-generation",
            model="facebook/opt-125m",
            compute_pool_for_log=None,
        )
        # without task - task is inferred when using CONFIG_ONLY mode (no huggingface_hub)
        # but when huggingface_hub is available, task must be explicitly provided
        huggingface.TransformersPipeline(
            task="text-generation",
            model="facebook/opt-125m",
            compute_pool_for_log=None,
        )
        huggingface.TransformersPipeline(
            task="fill-mask",
            model=testing_utils.DUMMY_UNKNOWN_IDENTIFIER,
            compute_pool_for_log=None,
        )

        with absltest.mock.patch("transformers.AutoConfig.from_pretrained") as mock_from_pretrained:
            mock_config = absltest.mock.Mock()
            mock_config._commit_hash = "fake_commit_hash"
            mock_config.custom_pipelines = {}
            mock_from_pretrained.return_value = mock_config
            huggingface.TransformersPipeline(
                task="fill-mask",
                model=testing_utils.SMALL_MODEL_IDENTIFIER,
                token=testing_utils.TOKEN,
                compute_pool_for_log=None,
            )
            if not is_hf_hub_available:
                mock_from_pretrained.assert_called_once_with(
                    testing_utils.SMALL_MODEL_IDENTIFIER,
                    _from_pipeline="fill-mask",
                    revision=None,
                    token=testing_utils.TOKEN,
                    trust_remote_code=None,
                    _commit_hash=None,
                )

        if not is_hf_hub_available:
            with self.assertRaisesRegex(
                ValueError,
                "Loading this pipeline requires you to execute the code in the pipeline file",
            ):
                huggingface.TransformersPipeline(
                    model="lysandre/test-dynamic-pipeline",
                    compute_pool_for_log=None,
                )

            with self.assertRaisesRegex(
                RuntimeError,
                "Impossible to use non-string config as input for class TransformersPipeline.",
            ):
                huggingface.TransformersPipeline(
                    task="text-generation",
                    model="facebook/opt-125m",
                    config=1,
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
        with self.assertRaisesRegex(
            RuntimeError,
            "Impossible to instantiate a pipeline without either a task or a model being specified.",
        ):
            huggingface.TransformersPipeline()
        with self.assertRaisesRegex(
            RuntimeError,
            "Impossible to instantiate a pipeline without either a task or a model being specified.",
        ):
            huggingface.TransformersPipeline(config="facebook/opt-125m", compute_pool_for_log=None)
        with self.assertRaisesRegex(
            RuntimeError,
            "Impossible to instantiate a pipeline with tokenizer specified but not the model as the provided",
        ):
            huggingface.TransformersPipeline(task="text-generation", tokenizer="tokenizer", compute_pool_for_log=None)
        with self.assertRaisesRegex(
            RuntimeError,
            "Impossible to instantiate a pipeline with feature_extractor specified but not the model as the provided",
        ):
            huggingface.TransformersPipeline(task="text-generation", feature_extractor="feature_extractor")
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

    def test_remote_logging_skips_config_and_snapshot_download(self) -> None:
        """Test that when compute_pool_for_log is set, we don't download from huggingface_hub or load config."""
        with absltest.mock.patch(
            "transformers.AutoConfig.from_pretrained"
        ) as mock_from_pretrained, absltest.mock.patch("huggingface_hub.snapshot_download") as mock_snapshot_download:

            # Create a pipeline with compute_pool_for_log set (remote logging mode)
            pipeline = huggingface.TransformersPipeline(
                task="text-generation",
                model="facebook/opt-125m",
                compute_pool_for_log="test_compute_pool",
            )

            # Verify that AutoConfig.from_pretrained was NOT called
            mock_from_pretrained.assert_not_called()

            # Verify that snapshot_download was NOT called
            mock_snapshot_download.assert_not_called()

            # Verify the pipeline was created correctly
            self.assertEqual(pipeline.model, "facebook/opt-125m")
            self.assertEqual(pipeline.task, "text-generation")
            self.assertEqual(pipeline.compute_pool_for_log, "test_compute_pool")
            self.assertIsNone(pipeline.repo_snapshot_dir)

    def test_remote_logging_with_default_model_skips_config(self) -> None:
        """Test that remote logging with default model also skips config lookup."""
        with absltest.mock.patch(
            "transformers.AutoConfig.from_pretrained"
        ) as mock_from_pretrained, absltest.mock.patch("huggingface_hub.snapshot_download") as mock_snapshot_download:

            # Create a pipeline with only task (model will be defaulted) and compute_pool_for_log set
            with self.assertWarnsRegex(
                UserWarning,
                "Using a pipeline without specifying a model name and revision in production is not recommended.",
            ):
                pipeline = huggingface.TransformersPipeline(
                    task="text-generation",
                    compute_pool_for_log="test_compute_pool",
                )

            # Verify that AutoConfig.from_pretrained was NOT called
            mock_from_pretrained.assert_not_called()

            # Verify that snapshot_download was NOT called
            mock_snapshot_download.assert_not_called()

            # Verify the pipeline was created with a default model
            self.assertIsNotNone(pipeline.model)
            self.assertEqual(pipeline.task, "text-generation")
            self.assertEqual(pipeline.compute_pool_for_log, "test_compute_pool")

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
