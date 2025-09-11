import os
import tempfile

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.ops import service_ops
from snowflake.ml.model.models import huggingface_pipeline
from snowflake.snowpark import async_job, session


class HuggingFacePipelineTest(absltest.TestCase):
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

    def test_wrapper(self) -> None:
        from transformers import testing_utils

        with self.assertWarnsRegex(
            UserWarning,
            "Using a pipeline without specifying a model name and revision in production is not recommended.",
        ):
            huggingface_pipeline.HuggingFacePipelineModel(task="text-generation", download_snapshot=False)

        huggingface_pipeline.HuggingFacePipelineModel(task="text-generation", model="facebook/opt-125m")

        huggingface_pipeline.HuggingFacePipelineModel(model="facebook/opt-125m")

        huggingface_pipeline.HuggingFacePipelineModel(model=testing_utils.DUMMY_UNKNOWN_IDENTIFIER)

        with absltest.mock.patch("transformers.AutoConfig.from_pretrained") as mock_from_pretrained:
            mock_config = absltest.mock.Mock()
            mock_config._commit_hash = "fake_commit_hash"
            mock_config.custom_pipelines = {}
            mock_from_pretrained.return_value = mock_config
            huggingface_pipeline.HuggingFacePipelineModel(
                task="fill-mask",
                model=testing_utils.SMALL_MODEL_IDENTIFIER,
                token=testing_utils.TOKEN,
                download_snapshot=False,
            )
            mock_from_pretrained.assert_called_once_with(
                testing_utils.SMALL_MODEL_IDENTIFIER,
                _from_pipeline="fill-mask",
                revision=None,
                token=testing_utils.TOKEN,
                trust_remote_code=None,
                _commit_hash=None,
            )

        with self.assertRaisesRegex(
            ValueError,
            "Loading this pipeline requires you to execute the code in the pipeline file",
        ):
            huggingface_pipeline.HuggingFacePipelineModel(model="lysandre/test-dynamic-pipeline")

        huggingface_pipeline.HuggingFacePipelineModel(model="lysandre/test-dynamic-pipeline", trust_remote_code=True)

        with self.assertRaisesRegex(
            RuntimeError,
            "Impossible to instantiate a pipeline without either a task or a model being specified.",
        ):
            huggingface_pipeline.HuggingFacePipelineModel()

        with self.assertRaisesRegex(
            RuntimeError,
            "Impossible to instantiate a pipeline without either a task or a model being specified.",
        ):
            huggingface_pipeline.HuggingFacePipelineModel(config="facebook/opt-125m")

        with self.assertRaisesRegex(
            RuntimeError,
            "Impossible to instantiate a pipeline with tokenizer specified but not the model as the provided",
        ):
            huggingface_pipeline.HuggingFacePipelineModel(task="text-generation", tokenizer="tokenizer")

        with self.assertRaisesRegex(
            RuntimeError,
            "Impossible to instantiate a pipeline with feature_extractor specified but not the model as the provided",
        ):
            huggingface_pipeline.HuggingFacePipelineModel(task="text-generation", feature_extractor="feature_extractor")

        with self.assertRaisesRegex(
            ValueError,
            "`token` and `use_auth_token` are both specified. Please set only the argument `token`.",
        ):
            huggingface_pipeline.HuggingFacePipelineModel(
                task="text-generation", model="facebook/opt-125m", token="token", model_kwargs={"use_auth_token": True}
            )

        with self.assertRaisesRegex(
            RuntimeError,
            "Impossible to use non-string model as input for HuggingFacePipelineModel.",
        ):
            huggingface_pipeline.HuggingFacePipelineModel(task="text-generation", model=1)  # type: ignore[arg-type]

        with self.assertRaisesRegex(
            RuntimeError,
            "Impossible to use non-string config as input for HuggingFacePipelineModel.",
        ):
            huggingface_pipeline.HuggingFacePipelineModel(task="text-generation", model="facebook/opt-125m", config=1)

        with self.assertRaisesRegex(
            ValueError,
            "You cannot use both `pipeline\\(... device_map=..., model_kwargs",
        ):
            huggingface_pipeline.HuggingFacePipelineModel(
                task="text-generation",
                model="facebook/opt-125m",
                device_map="auto",
                model_kwargs={"device_map": "auto"},
            )

        with self.assertWarnsRegex(
            UserWarning,
            "Both `device` and `device_map` are specified.",
        ):
            huggingface_pipeline.HuggingFacePipelineModel(
                task="text-generation", model="facebook/opt-125m", device_map="auto", device=0
            )

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
            result = model_ref.create_service(
                session=mock_session,
                model_name="test_model",
                version_name="v1",
                service_name="test_service",
                service_compute_pool="test_compute_pool",
                image_repo="test_repo",
                ingress_enabled=True,
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
            result = model_ref.create_service(
                session=mock_session,
                model_name="test_model",
                version_name="v1",
                service_name="test_service_db.test_service_schema.test_service",
                service_compute_pool="test_compute_pool",
                image_repo="test_repo",
                ingress_enabled=True,
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
        )

        # Patch the ServiceOperator constructor to return our mock
        with absltest.mock.patch(
            "snowflake.ml.model._client.ops.service_ops.ServiceOperator",
            return_value=mock_service_operator,
        ):
            # Call the create_service function in non-blocking mode
            result = model_ref.create_service(
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


if __name__ == "__main__":
    absltest.main()
