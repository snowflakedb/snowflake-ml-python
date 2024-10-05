from typing import List

from absl.testing import absltest

from snowflake import snowpark
from snowflake.cortex import EmbedText768, EmbedText1024
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session, functions

_TEXT = "Text to embed"


class EmbedTextTest(absltest.TestCase):
    def setUp(self) -> None:
        self._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    def text_embed_text_768(self) -> None:
        df_in = self._session.create_dataframe([snowpark.Row(model="e5-base-v2", text=_TEXT)])
        df_out = df_in.select(EmbedText768(functions.col("model"), functions.col("text")))
        res = df_out.collect()[0][0]
        self.assertIsInstance(res, List)
        self.assertEqual(len(res), 768)
        # Check a subset.
        self.assertEqual(res[:4], [-0.001, 0.002, -0.003, 0.004])

    def text_embed_text_1024(self) -> None:
        df_in = self._session.create_dataframe([snowpark.Row(model="multilingual-e5-large", text=_TEXT)])
        df_out = df_in.select(EmbedText1024(functions.col("model"), functions.col("text")))
        res = df_out.collect()[0][0]
        self.assertIsInstance(res, List)
        self.assertEqual(len(res), 1024)
        # Check a subset.
        self.assertEqual(res[:4], [-0.001, 0.002, -0.003, 0.004])


if __name__ == "__main__":
    absltest.main()
