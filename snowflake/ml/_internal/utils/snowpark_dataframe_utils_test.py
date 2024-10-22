from typing import cast

from absl.testing import absltest

from snowflake import snowpark
from snowflake.ml._internal.utils import snowpark_dataframe_utils
from snowflake.ml.test_utils import mock_data_frame


class IsSingleQuerySnowparkDataframeTest(absltest.TestCase):
    """Testing is_single_query_snowpark_dataframe function."""

    def test_single_query(self) -> None:
        """Test that multiple queries in a dataframe are rejected."""
        df = mock_data_frame.MockDataFrame()
        df.add_query("queries", "SELECT PROMPT, COMPLETION FROM TRAINING")
        self.assertTrue(snowpark_dataframe_utils.is_single_query_snowpark_dataframe(cast(snowpark.DataFrame, df)))

    def test_multiple_queries(self) -> None:
        """Test that multiple queries in a dataframe are rejected."""
        df = mock_data_frame.MockDataFrame()
        df.add_query("queries", "SELECT PROMPT, COMPLETION FROM TRAINING")
        df.add_query("queries", "SELECT PROMPT, COMPLETION FROM VALIDATION")
        self.assertFalse(snowpark_dataframe_utils.is_single_query_snowpark_dataframe(cast(snowpark.DataFrame, df)))

    def test_post_actions(self) -> None:
        """Test that multiple queries in a dataframe are rejected."""
        df = mock_data_frame.MockDataFrame()
        df.add_query("queries", "SELECT PROMPT, COMPLETION FROM TRAINING")
        df.add_query("post_actions", "SELECT PROMPT, COMPLETION FROM VALIDATION")
        self.assertFalse(snowpark_dataframe_utils.is_single_query_snowpark_dataframe(cast(snowpark.DataFrame, df)))


if __name__ == "__main__":
    absltest.main()
