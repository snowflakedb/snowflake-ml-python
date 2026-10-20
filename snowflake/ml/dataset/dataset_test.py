"""Unit tests for dataset.Dataset."""

from collections.abc import Callable
from unittest import mock

from absl.testing import absltest

from snowflake.ml.dataset import dataset
from snowflake.snowpark import Row


def _make_dataset(mock_session: mock.MagicMock) -> dataset.Dataset:
    """Build a Dataset instance without hitting Snowflake, bypassing __init__."""
    ds = dataset.Dataset.__new__(dataset.Dataset)
    ds._session = mock_session
    ds._db = "TESTDB"
    ds._schema = "PUBLIC"
    ds._name = "TESTDS"
    ds._lineage_node_name = "TESTDB.PUBLIC.TESTDS"
    ds._version = None
    ds._reader = None
    return ds


def _make_mock_casted_df() -> mock.MagicMock:
    """Return a minimal mock DataFrame that satisfies create_version's _plan access."""
    mock_query = mock.MagicMock()
    mock_query.sql = "SELECT 1"
    mock_df = mock.MagicMock()
    mock_df.queries = {"queries": ["SELECT 1"]}
    mock_df._plan.post_actions = []
    mock_df._plan.queries = [mock_query]
    return mock_df


def _make_session_sql_side_effect(
    current_user_row: Row,
) -> Callable[..., mock.MagicMock]:
    """Return a session.sql side_effect that serves current_user_row for CURRENT_USER queries."""

    def side_effect(query: str, *args: object, **kwargs: object) -> mock.MagicMock:
        result = mock.MagicMock()
        if "CURRENT_USER" in query.upper():
            result.collect.return_value = [current_user_row]
        else:
            result.collect.return_value = []
        return result

    return side_effect


class DatasetCreateVersionCurrentUserTest(absltest.TestCase):
    """Tests that create_version extracts the dataset owner robustly regardless of column naming."""

    @mock.patch("snowflake.ml.dataset.dataset.snowpark_dataframe_utils.cast_snowpark_dataframe")
    @mock.patch("snowflake.ml.dataset.dataset.Dataset.__init__", return_value=None)
    def _run_create_version(
        self,
        current_user_row: Row,
        mock_ds_init: mock.MagicMock,
        mock_cast: mock.MagicMock,
    ) -> None:
        """Helper that wires up mocks and calls create_version with the given current-user row."""
        mock_session = mock.MagicMock()
        mock_session.sql.side_effect = _make_session_sql_side_effect(current_user_row)

        mock_cast.return_value = _make_mock_casted_df()

        mock_input_df = mock.MagicMock()
        mock_input_df.queries = {"queries": ["SELECT 1"]}

        ds = _make_dataset(mock_session)
        # Should complete without raising KeyError
        ds.create_version("V1", mock_input_df)

    def test_standard_column_name(self) -> None:
        """Accounts that return CURRENT_USER() as the column name work correctly."""
        self._run_create_version(Row(**{"CURRENT_USER()": "ADMIN"}))

    def test_redacted_column_name(self) -> None:
        """Accounts with literal-redaction return a sanitized column name (e.g. LiteralRedacted0).

        These accounts (observed on GCP deployments) caused a KeyError when the code accessed
        the result by column name 'CURRENT_USER()'. Positional access [0] is robust to any
        column naming scheme.
        """
        self._run_create_version(Row(LiteralRedacted0="ADMIN"))


if __name__ == "__main__":
    absltest.main()
