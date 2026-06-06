#!/usr/bin/env python3
"""Tests for update_pinned_snowml_python.py."""

from unittest import mock

import update_pinned_snowml_python
from absl.testing import absltest


class UpdatePinnedSnowmlPythonTest(absltest.TestCase):
    def test_generate_pip_requirements(self) -> None:
        base_content = f"{update_pinned_snowml_python.SNOWML_PYTHON_PIN_PLACEHOLDER}\ntransformers\n"
        result = update_pinned_snowml_python.generate_pip_requirements(base_content, "1.41.0")
        self.assertEqual(result, "snowflake-ml-python==1.41.0\ntransformers\n")

    def test_generate_conda_env(self) -> None:
        base_content = "\n".join(
            [
                "---",
                "dependencies:",
                f"  - {update_pinned_snowml_python.SNOWML_PYTHON_PIN_PLACEHOLDER}",
                "name: snow-env",
                "",
            ]
        )
        result = update_pinned_snowml_python.generate_conda_env(base_content, "1.41.0")
        self.assertIn("  - snowflake-ml-python==1.41.0", result)
        self.assertNotIn(update_pinned_snowml_python.SNOWML_PYTHON_PIN_PLACEHOLDER, result)

    def test_generate_pip_requirements_missing_placeholder_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing __SNOWML_PYTHON_PIN__"):
            update_pinned_snowml_python.generate_pip_requirements("transformers\n", "1.41.0")

    def test_generate_conda_env_missing_placeholder_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing __SNOWML_PYTHON_PIN__"):
            update_pinned_snowml_python.generate_conda_env("dependencies:\n  - pip\n", "1.41.0")

    @mock.patch.object(update_pinned_snowml_python, "get_latest_pypi_version", return_value="1.41.0")
    def test_main_pip_format(self, mock_get_latest: mock.MagicMock) -> None:
        with mock.patch(
            "sys.argv",
            [
                "update_pinned_snowml_python.py",
                "model_container_services_deployment/model_logger/base_requirements.txt",
                "--format",
                "pip",
            ],
        ):
            with mock.patch(
                "builtins.open",
                mock.mock_open(read_data=f"{update_pinned_snowml_python.SNOWML_PYTHON_PIN_PLACEHOLDER}\n"),
            ):
                with mock.patch("sys.stdout") as mock_stdout:
                    update_pinned_snowml_python.main()
                    mock_stdout.write.assert_called_once_with("snowflake-ml-python==1.41.0\n")
        mock_get_latest.assert_called_once_with("snowflake-ml-python")

    @mock.patch.object(update_pinned_snowml_python, "get_latest_pypi_version", return_value="1.41.0")
    def test_main_conda_format(self, mock_get_latest: mock.MagicMock) -> None:
        with mock.patch(
            "sys.argv",
            [
                "update_pinned_snowml_python.py",
                "model_container_services_deployment/ray_orchestrator/base_conda.yml",
                "--format",
                "conda",
            ],
        ):
            with mock.patch(
                "builtins.open",
                mock.mock_open(
                    read_data=f"  - {update_pinned_snowml_python.SNOWML_PYTHON_PIN_PLACEHOLDER}\n",
                ),
            ):
                with mock.patch("sys.stdout") as mock_stdout:
                    update_pinned_snowml_python.main()
                    mock_stdout.write.assert_called_once_with("  - snowflake-ml-python==1.41.0\n")
        mock_get_latest.assert_called_once_with("snowflake-ml-python")


if __name__ == "__main__":
    absltest.main()
