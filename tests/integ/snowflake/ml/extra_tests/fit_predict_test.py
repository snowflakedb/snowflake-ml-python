import numpy as np
import pandas as pd
from absl.testing.absltest import TestCase, main
from sklearn.cluster import (
    DBSCAN as SKDBSCAN,
    OPTICS as SKOPTICS,
    AgglomerativeClustering as SKAgglomerativeClustering,
)

from snowflake.ml.modeling.cluster import DBSCAN, OPTICS, AgglomerativeClustering
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session


class FitPredictTest(TestCase):
    def setUp(self):
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()

    def tearDown(self):
        self._session.close()

    def test_aggolomerative(self):
        sample_data = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        pd_df = pd.DataFrame(sample_data)
        pd_df.columns = [str(c) for c in pd_df.columns]
        sp_df = self._session.create_dataframe(pd_df)
        agg = AgglomerativeClustering(input_cols=sp_df.columns)
        sk_agg = SKAgglomerativeClustering()

        return_label = agg.fit_predict(sp_df)
        sk_label = sk_agg.fit_predict(sample_data)

        np.testing.assert_allclose(return_label, sk_label, rtol=1.0e-1, atol=1.0e-2)

    def test_dbscan(self):
        sample_data = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
        pd_df = pd.DataFrame(sample_data)
        pd_df.columns = [str(c) for c in pd_df.columns]
        sp_df = self._session.create_dataframe(pd_df)
        dbs = DBSCAN(input_cols=sp_df.columns, eps=3, min_samples=2)
        sk_dbs = SKDBSCAN(eps=3, min_samples=2)

        return_label = dbs.fit_predict(sp_df)
        sk_label = sk_dbs.fit_predict(sample_data)

        np.testing.assert_allclose(return_label, sk_label, rtol=1.0e-1, atol=1.0e-2)

    def test_optics(self):
        sample_data = np.array([[1, 2], [2, 5], [3, 6], [8, 7], [8, 8], [7, 3]])
        pd_df = pd.DataFrame(sample_data)
        pd_df.columns = [str(c) for c in pd_df.columns]
        sp_df = self._session.create_dataframe(pd_df)
        opt = OPTICS(input_cols=sp_df.columns, min_samples=2)
        sk_opt = SKOPTICS(min_samples=2)

        return_label = opt.fit_predict(sp_df)
        sk_label = sk_opt.fit_predict(sample_data)

        np.testing.assert_allclose(return_label, sk_label, rtol=1.0e-1, atol=1.0e-2)


if __name__ == "__main__":
    main()
