from typing import List

from absl.testing import absltest

from snowflake import snowpark
from snowflake.cortex import EmbedText768, EmbedText1024
from snowflake.ml._internal.utils import snowflake_env
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session, functions
from tests.integ.snowflake.ml.test_utils import test_env_utils

_TEXT = "Text to embed"


@absltest.skipUnless(
    test_env_utils.get_current_snowflake_cloud_type() == snowflake_env.SnowflakeCloudType.AWS,
    "Embed text only available in AWS",
)
class EmbedTextTest(absltest.TestCase):
    def setUp(self) -> None:
        self._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    def test_embed_text_768(self) -> None:
        df_in = self._session.create_dataframe([snowpark.Row(model="e5-base-v2", text=_TEXT)])
        df_out = df_in.select(EmbedText768(functions.col("model"), functions.col("text")))
        res = df_out.collect()[0][0]
        self.assertIsInstance(res, List)
        self.assertEqual(len(res), 768)
        # Check a subset.
        for first, second in zip(res[:4], [-0.0174, -0.04528, -0.02869, 0.0189]):
            self.assertAlmostEqual(first, second, delta=0.01)

    def test_embed_text_1024(self) -> None:
        df_in = self._session.create_dataframe([snowpark.Row(model="multilingual-e5-large", text=_TEXT)])
        df_out = df_in.select(EmbedText1024(functions.col("model"), functions.col("text")))
        res = df_out.collect()[0][0]
        self.assertIsInstance(res, List)
        self.assertEqual(len(res), 1024)
        # Check a subset.
        for first, second in zip(res[:4], [0.0253, 0.0085, 0.0143, -0.0387]):
            self.assertAlmostEqual(first, second, delta=0.01)


if __name__ == "__main__":
    absltest.main()
