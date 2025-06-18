import os
import random
import tempfile

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model._signatures import snowpark_handler
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import dataframe_utils

MODEL_NAMES = ["intfloat/e5-base-v2"]  # cant load models in parallel
SENTENCE_TRANSFORMERS_CACHE_DIR = "SENTENCE_TRANSFORMERS_HOME"


class TestRegistrySentenceTransformerModelInteg(registry_model_test_base.RegistryModelTestBase):
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
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_sentence_transformers(
        self,
        registry_test_fn: str,
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
        embeddings = pd.DataFrame({"output_feature_0": model.encode(sentences["SENTENCES"].tolist()).tolist()})
        getattr(self, registry_test_fn)(
            model=model,
            sample_input_data=sentences,
            prediction_assert_fns={
                "encode": (
                    sentences,
                    lambda res: res.equals(embeddings),
                ),
            },
        )

    def test_sentence_transformers_sp(self) -> None:
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
        sentences_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(self.session, sentences)
        model = sentence_transformers.SentenceTransformer(random.choice(MODEL_NAMES))
        embeddings = pd.DataFrame({"output_feature_0": model.encode(sentences["SENTENCES"].tolist()).tolist()})
        y_df_expected = pd.concat([sentences_sp.to_pandas(), embeddings], axis=1)

        self._test_registry_model(
            model=model,
            sample_input_data=sentences_sp,
            prediction_assert_fns={
                "encode": (
                    sentences_sp,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, atol=1e-6),
                ),
            },
        )


if __name__ == "__main__":
    absltest.main()
