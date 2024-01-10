import numpy as np
import pandas as pd
from absl.testing.absltest import TestCase, main
from sklearn.datasets import load_digits
from sklearn.manifold import (
    MDS as SKMDS,
    TSNE as SKTSNE,
    SpectralEmbedding as SKSpectralEmbedding,
)

from snowflake.ml.modeling.manifold import MDS, TSNE, SpectralEmbedding
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session


class FitTransformTest(TestCase):
    def _load_data(self):
        X, _ = load_digits(return_X_y=True)
        self._input_df_pandas = pd.DataFrame(X)[:100]
        self._input_df_pandas.columns = [str(c) for c in self._input_df_pandas.columns]
        self._input_df = self._session.create_dataframe(self._input_df_pandas)
        self._input_cols = self._input_df.columns
        self._output_cols = [str(c) for c in range(100)]

    def setUp(self):
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        self._load_data()

    def tearDown(self):
        self._session.close()

    def testMDS(self):
        sk_embedding = SKMDS(n_components=2, normalized_stress="auto", random_state=2024)

        embedding = MDS(
            input_cols=self._input_cols,
            output_cols=self._output_cols,
            n_components=2,
            normalized_stress="auto",
            random_state=2024,
        )
        sk_X_transformed = sk_embedding.fit_transform(self._input_df_pandas)
        X_transformed = embedding.fit_transform(self._input_df)
        np.testing.assert_allclose(sk_X_transformed, X_transformed, rtol=1.0e-1, atol=1.0e-2)

    def testSpectralEmbedding(self):
        sk_embedding = SKSpectralEmbedding(n_components=2, random_state=2024)
        sk_X_transformed = sk_embedding.fit_transform(self._input_df_pandas)

        embedding = SpectralEmbedding(
            input_cols=self._input_cols, output_cols=self._output_cols, n_components=2, random_state=2024
        )
        X_transformed = embedding.fit_transform(self._input_df)
        np.testing.assert_allclose(sk_X_transformed, X_transformed, rtol=1.0e-1, atol=1.0e-2)

    def testTSNE(self):
        sk_embedding = SKTSNE(n_components=2, random_state=2024, n_jobs=1)
        sk_X_transformed = sk_embedding.fit_transform(self._input_df_pandas)

        embedding = TSNE(
            input_cols=self._input_cols,
            output_cols=self._output_cols,
            n_components=2,
            random_state=2024,
            n_jobs=1,
        )
        X_transformed = embedding.fit_transform(self._input_df)
        np.testing.assert_allclose(sk_X_transformed.shape, X_transformed.shape, rtol=1.0e-1, atol=1.0e-2)


if __name__ == "__main__":
    main()
