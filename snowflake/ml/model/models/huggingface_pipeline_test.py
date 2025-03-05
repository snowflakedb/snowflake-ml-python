import os
import tempfile

from absl.testing import absltest

from snowflake.ml.model.models import huggingface_pipeline


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
            huggingface_pipeline.HuggingFacePipelineModel(task="text-generation")

        huggingface_pipeline.HuggingFacePipelineModel(task="text-generation", model="gpt2")

        huggingface_pipeline.HuggingFacePipelineModel(task="text-generation", config="gpt2")

        huggingface_pipeline.HuggingFacePipelineModel(model="gpt2")

        huggingface_pipeline.HuggingFacePipelineModel(model=testing_utils.DUMMY_UNKNOWN_IDENTIFIER)

        with absltest.mock.patch("transformers.AutoConfig.from_pretrained") as mock_from_pretrained:
            mock_config = absltest.mock.Mock()
            mock_config._commit_hash = "fake_commit_hash"
            mock_config.custom_pipelines = {}
            mock_from_pretrained.return_value = mock_config
            huggingface_pipeline.HuggingFacePipelineModel(
                task="fill-mask", model=testing_utils.SMALL_MODEL_IDENTIFIER, token=testing_utils.TOKEN
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
            huggingface_pipeline.HuggingFacePipelineModel(config="gpt2")

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
                task="text-generation", model="gpt2", token="token", model_kwargs={"use_auth_token": True}
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
            huggingface_pipeline.HuggingFacePipelineModel(task="text-generation", model="gpt2", config=1)

        with self.assertRaisesRegex(
            ValueError,
            "You cannot use both `pipeline\\(... device_map=..., model_kwargs",
        ):
            huggingface_pipeline.HuggingFacePipelineModel(
                task="text-generation", model="gpt2", device_map="auto", model_kwargs={"device_map": "auto"}
            )

        with self.assertWarnsRegex(
            UserWarning,
            "Both `device` and `device_map` are specified.",
        ):
            huggingface_pipeline.HuggingFacePipelineModel(
                task="text-generation", model="gpt2", device_map="auto", device=0
            )


if __name__ == "__main__":
    absltest.main()
