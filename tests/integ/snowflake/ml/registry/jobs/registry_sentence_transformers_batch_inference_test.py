import os
import random
import tempfile
from typing import Optional

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model.batch import JobSpec, OutputSpec
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base

MODEL_NAMES = ["intfloat/e5-base-v2"]
SENTENCE_TRANSFORMERS_CACHE_DIR = "SENTENCE_TRANSFORMERS_HOME"


class TestRegistrySentenceTransformerBatchInferenceInteg(
    registry_batch_inference_test_base.RegistryBatchInferenceTestBase
):
    @classmethod
    def setUpClass(self) -> None:
        self.cache_dir = tempfile.TemporaryDirectory()
        self._original_cache_dir = os.getenv(SENTENCE_TRANSFORMERS_CACHE_DIR, None)
        os.environ[SENTENCE_TRANSFORMERS_CACHE_DIR] = self.cache_dir.name

    @classmethod
    def tearDownClass(self) -> None:
        if self._original_cache_dir:
            os.environ[SENTENCE_TRANSFORMERS_CACHE_DIR] = self._original_cache_dir
        self.cache_dir.cleanup()

    @parameterized.parameters(  # type: ignore[misc]
        {"pip_requirements": ["sentence-transformers"], "gpu_requests": None},
        {"pip_requirements": None, "gpu_requests": "1"},
    )
    def test_sentence_transformers(
        self,
        gpu_requests: Optional[str],
        pip_requirements: Optional[list[str]],
    ) -> None:
        import sentence_transformers

        # Sample Data
        sentences = pd.DataFrame(
            {
                "SENTENCES": [
                    "Why don't scientists trust atoms? Because they make up everything.",
                    "I told my wife she should embrace her mistakes. She gave me a hug.",
                    "Im reading a book on anti-gravity. Its impossible to put down!",
                    "Did you hear about the mathematician who's afraid of negative numbers?",
                    "Parallel lines have so much in common. It's a shame they'll never meet.",
                ]
            }
        )
        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        model = sentence_transformers.SentenceTransformer(random.choice(MODEL_NAMES))

        # Generate expected predictions using the original model
        model_output = model.encode(sentences["SENTENCES"].tolist())
        # Convert numpy arrays to lists of floats
        model_output_normalized = [embedding.tolist() for embedding in model_output]
        model_output_df = pd.DataFrame({"output_feature_0": model_output_normalized})

        # Prepare input data and expected predictions using common function
        input_df, expected_predictions = self._prepare_batch_inference_data(sentences, model_output_df)

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=sentences,
            options={"cuda_version": model_env.DEFAULT_CUDA_VERSION},
            pip_requirements=pip_requirements,
            X=input_df,
            output_spec=OutputSpec(stage_location=output_stage_location),
            job_spec=JobSpec(
                job_name=job_name,
                num_workers=1,
                replicas=1,
                gpu_requests=gpu_requests,
                function_name="encode",
            ),
            expected_predictions=expected_predictions,
        )


if __name__ == "__main__":
    absltest.main()
