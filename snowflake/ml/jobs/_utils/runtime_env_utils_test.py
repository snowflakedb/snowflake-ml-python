import json
import sys
from unittest import mock

from absl.testing import absltest, parameterized

from snowflake.ml.jobs._utils import feature_flags, runtime_env_utils


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


class ResolveRuntimeImageTest(parameterized.TestCase):
    @parameterized.named_parameters(  # type: ignore[misc]
        ("without_pin", None, {}),
        ("with_pin", "1.7.1", {"runtimeEnvironment": "1.7.1"}),
    )
    @mock.patch("snowflake.ml.jobs._utils.runtime_env_utils.get_runtime_image", autospec=True)
    def test_sends_local_python_version_when_runtime_versions_enabled(
        self,
        runtime_environment: str | None,
        expected_extra: dict[str, str],
        mock_get_runtime_image: mock.MagicMock,
    ) -> None:
        session = mock.MagicMock()
        with mock.patch.object(feature_flags.FeatureFlags.ENABLE_RUNTIME_VERSIONS, "is_enabled", return_value=True):
            result = runtime_env_utils.resolve_runtime_image(session, "pool", runtime_environment)

        expected_selector = {"pythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}", **expected_extra}
        mock_get_runtime_image.assert_called_once_with(session, "pool", json.dumps(expected_selector))
        self.assertEqual(result, mock_get_runtime_image.return_value)

    @parameterized.named_parameters(  # type: ignore[misc]
        ("without_pin", None),
        ("with_pin", "1.7.1"),
    )
    @mock.patch("snowflake.ml.jobs._utils.runtime_env_utils.get_runtime_image", autospec=True)
    def test_passes_runtime_through_when_runtime_versions_disabled(
        self, runtime_environment: str | None, mock_get_runtime_image: mock.MagicMock
    ) -> None:
        session = mock.MagicMock()
        with mock.patch.object(feature_flags.FeatureFlags.ENABLE_RUNTIME_VERSIONS, "is_enabled", return_value=False):
            runtime_env_utils.resolve_runtime_image(session, "pool", runtime_environment)

        mock_get_runtime_image.assert_called_once_with(session, "pool", runtime_environment)


if __name__ == "__main__":
    absltest.main()
