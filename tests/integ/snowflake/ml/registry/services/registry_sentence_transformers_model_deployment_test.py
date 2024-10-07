import os
import random
import tempfile
from typing import List, Optional

import pandas as pd
from absl.testing import absltest, parameterized

from tests.integ.snowflake.ml.registry.services import (
    registry_model_deployment_test_base,
)

MODEL_NAMES = ["intfloat/e5-base-v2"]  # cant load models in parallel
SENTENCE_TRANSFORMERS_CACHE_DIR = "SENTENCE_TRANSFORMERS_HOME"


class TestRegistrySentenceTransformerDeploymentModelInteg(
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

    @parameterized.product(  # type: ignore[misc]
        gpu_requests=[None, "1"],
        pip_requirements=[None, ["sentence-transformers"]],
    )
    def test_sentence_transformers(
        self,
        gpu_requests: str,
        pip_requirements: Optional[List[str]],
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
        model = sentence_transformers.SentenceTransformer(random.choice(MODEL_NAMES))
        embeddings = pd.DataFrame(model.encode(sentences["SENTENCES"].to_list(), batch_size=sentences.shape[0]))

        self._test_registry_model_deployment(
            model=model,
            sample_input_data=sentences,
            prediction_assert_fns={
                "encode": (
                    sentences,
                    lambda res: pd.testing.assert_frame_equal(
                        pd.DataFrame(res["output_feature_0"].to_list()),
                        embeddings,
                        rtol=1e-2,
                        atol=1e-3,
                        check_dtype=False,
                    ),
                ),
            },
            options={"cuda_version": "11.8"} if gpu_requests else {},
            gpu_requests=gpu_requests,
            pip_requirements=pip_requirements,
        )


if __name__ == "__main__":
    absltest.main()
