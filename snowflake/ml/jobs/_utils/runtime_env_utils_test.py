from unittest import mock

from absl.testing import absltest, parameterized

from snowflake.ml.jobs._utils import runtime_env_utils


class GetRuntimeImageTest(parameterized.TestCase):
    @parameterized.parameters(  # type: ignore[misc]
        ([["registry.example.com/repo/image:latest"]], "registry.example.com/repo/image:latest"),
        ([["registry.example.com/repo/image:v1.0"]], "registry.example.com/repo/image:v1.0"),
        ([["cre@my_reference"]], "cre@my_reference"),
        ([["CRE@my_reference"]], "CRE@my_reference"),
    )
    @mock.patch("snowflake.ml.jobs._utils.runtime_env_utils.query_helper.run_query", autospec=True)
    def test_returns_image(self, query_result: list[list[str]], expected: str, mock_run_query: mock.MagicMock) -> None:
        mock_run_query.return_value = query_result
        session = mock.MagicMock()
        result = runtime_env_utils.get_runtime_image(session, "pool")
        self.assertEqual(result, expected)

    @parameterized.parameters(  # type: ignore[misc]
        ([["v1.0!invalid"]],),
        ([["tag with spaces"]],),
    )
    @mock.patch("snowflake.ml.jobs._utils.runtime_env_utils.query_helper.run_query", autospec=True)
    def test_raises_on_invalid_image(self, query_result: list[list[str]], mock_run_query: mock.MagicMock) -> None:
        mock_run_query.return_value = query_result
        session = mock.MagicMock()
        with self.assertRaises(ValueError):
            runtime_env_utils.get_runtime_image(session, "pool")


if __name__ == "__main__":
    absltest.main()
