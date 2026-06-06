import os
import random
import tempfile
from typing import Optional

import pandas as pd
from absl.testing import absltest, parameterized
from packaging import version as pkg_version

from snowflake.ml.model._packager.model_env import model_env
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
        pip_requirements=[None, ["sentence-transformers", "torch==2.6.0"]],
    )
    # TODO: Remove torch dependency when the version is available in go/conda
    def test_sentence_transformers(
        self,
        pip_requirements: Optional[list[str]],
    ) -> None:
        if pip_requirements is None:
            import sentence_transformers
            import transformers

            if pkg_version.parse(sentence_transformers.__version__) >= pkg_version.parse("5.3.0") and pkg_version.parse(
                transformers.__version__
            ) >= pkg_version.parse("5.0.0"):
                self.skipTest(
                    f"sentence-transformers {sentence_transformers.__version__} with transformers "
                    f"{transformers.__version__}: model serving container image build fails because pinned "
                    "transformers>=5.0.0 conflicts with sentence-transformers requiring transformers<5.0.0."
                )
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
        truncate_dim = 128
        model = sentence_transformers.SentenceTransformer(random.choice(MODEL_NAMES))
        embeddings = pd.DataFrame(
            model.encode(
                sentences["SENTENCES"].to_list(),
                batch_size=sentences.shape[0],
                truncate_dim=truncate_dim,
            )
        )

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
            options={"cuda_version": model_env.DEFAULT_CUDA_VERSION},
            pip_requirements=pip_requirements,
            rest_inference_formats=[registry_model_deployment_test_base.RestInferencePayloadFormat.DATAFRAME_RECORDS],
            params={"truncate_dim": truncate_dim},
        )


if __name__ == "__main__":
    absltest.main()
