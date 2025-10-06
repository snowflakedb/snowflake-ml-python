import os
import random
import tempfile
import uuid
from typing import Optional

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model._packager.model_env import model_env
from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)

MODEL_NAMES = ["intfloat/e5-base-v2"]
SENTENCE_TRANSFORMERS_CACHE_DIR = "SENTENCE_TRANSFORMERS_HOME"


class TestRegistrySentenceTransformerBatchInferenceInteg(
    registry_model_deployment_test_base.RegistryModelDeploymentTestBase
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
                    "Why don’t scientists trust atoms? Because they make up everything.",
                    "I told my wife she should embrace her mistakes. She gave me a hug.",
                    "Im reading a book on anti-gravity. Its impossible to put down!",
                    "Did you hear about the mathematician who’s afraid of negative numbers?",
                    "Parallel lines have so much in common. It’s a shame they’ll never meet.",
                ]
            }
        )
        sp_df = self.session.create_dataframe(sentences)
        name = f"{str(uuid.uuid4()).replace('-', '_').upper()}"
        output_stage_location = f"@{self._test_db}.{self._test_schema}.{self._test_stage}/{name}/output/"

        model = sentence_transformers.SentenceTransformer(random.choice(MODEL_NAMES))

        self._test_registry_batch_inference(
            model=model,
            sample_input_data=sentences,
            options={"cuda_version": model_env.DEFAULT_CUDA_VERSION},
            pip_requirements=pip_requirements,
            input_spec=sp_df,
            output_stage_location=output_stage_location,
            num_workers=1,
            service_name=f"batch_inference_{name}",
            replicas=1,
            gpu_requests=gpu_requests,
        )


if __name__ == "__main__":
    absltest.main()
