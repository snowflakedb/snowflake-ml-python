from typing import Any
from unittest import mock

from absl.testing import absltest, parameterized

from snowflake.ml.jobs._utils import constants, option_parsing


class PopCommonOptionsTest(parameterized.TestCase):
    def test_defaults(self) -> None:
        kwargs: dict[str, Any] = {}
        options = option_parsing.pop_common_options(kwargs, 1)

        self.assertEqual(options.target_instances, 1)
        self.assertEqual(options.min_instances, 1)
        self.assertIsNone(options.runtime_environment)
        self.assertIsNone(options.artifact_repositories)
        self.assertIsNone(options.env_vars)
        self.assertTrue(options.enable_metrics)
        self.assertIsNone(options.imports)
        self.assertIsNone(options.database)
        self.assertIsNone(options.schema)

    def test_min_instances_defaults_to_target_instances(self) -> None:
        options = option_parsing.pop_common_options({}, 3)
        self.assertEqual(options.min_instances, 3)

    def test_pops_recognized_keys_and_leaves_the_rest(self) -> None:
        kwargs: dict[str, Any] = {
            "min_instances": 2,
            "runtime_environment": "1.7.1",
            "artifact_repositories": ["MY_REPO"],
            "env_vars": {"FOO": "bar"},
            "enable_metrics": False,
            "imports": [("@stage/pkg.zip",)],
            "database": "MY_DB",
            "schema": "MY_SCHEMA",
            "entrypoint": "main.py",
            "bogus": 1,
        }
        options = option_parsing.pop_common_options(kwargs, 4)

        self.assertEqual(options.target_instances, 4)
        self.assertEqual(options.min_instances, 2)
        self.assertEqual(options.runtime_environment, "1.7.1")
        self.assertEqual(options.artifact_repositories, ["MY_REPO"])
        self.assertEqual(options.env_vars, {"FOO": "bar"})
        self.assertFalse(options.enable_metrics)
        self.assertEqual(options.imports, [("@stage/pkg.zip",)])
        self.assertEqual(options.database, "MY_DB")
        self.assertEqual(options.schema, "MY_SCHEMA")
        # Options belonging to the caller are left untouched.
        self.assertEqual(kwargs, {"entrypoint": "main.py", "bogus": 1})

    def test_runtime_environment_defaults_to_env_var(self) -> None:
        with mock.patch.dict("os.environ", {constants.RUNTIME_IMAGE_TAG_ENV_VAR: "1.9.0"}):
            options = option_parsing.pop_common_options({}, 1)
        self.assertEqual(options.runtime_environment, "1.9.0")

    def test_explicit_runtime_environment_beats_env_var(self) -> None:
        with mock.patch.dict("os.environ", {constants.RUNTIME_IMAGE_TAG_ENV_VAR: "1.9.0"}):
            options = option_parsing.pop_common_options({"runtime_environment": "1.7.1"}, 1)
        self.assertEqual(options.runtime_environment, "1.7.1")

    def test_does_not_validate(self) -> None:
        options = option_parsing.pop_common_options({"min_instances": 5, "database": "MY_DB"}, 0)
        self.assertEqual(options.target_instances, 0)
        self.assertEqual(options.min_instances, 5)
        self.assertEqual(options.database, "MY_DB")
        self.assertIsNone(options.schema)


class CheckOptionsTest(parameterized.TestCase):
    @parameterized.parameters(0, -1)  # type: ignore[misc]
    def test_raises_on_invalid_target_instances(self, target_instances: int) -> None:
        options = option_parsing.pop_common_options({}, target_instances)
        with self.assertRaisesRegex(ValueError, "target_instances must be greater than 0."):
            option_parsing.check_instance_counts(options)

    @parameterized.parameters(0, -1, 3)  # type: ignore[misc]
    def test_raises_on_invalid_min_instances(self, min_instances: int) -> None:
        options = option_parsing.pop_common_options({"min_instances": min_instances}, 2)
        with self.assertRaisesRegex(ValueError, "min_instances must be greater than 0"):
            option_parsing.check_instance_counts(options)

    def test_accepts_valid_instance_counts(self) -> None:
        option_parsing.check_instance_counts(option_parsing.pop_common_options({"min_instances": 1}, 2))

    def test_raises_on_database_without_schema(self) -> None:
        options = option_parsing.pop_common_options({"database": "MY_DB"}, 1)
        with self.assertRaisesRegex(ValueError, "Schema must be specified if database is specified."):
            option_parsing.check_database_schema(options)

    def test_schema_without_database_is_allowed(self) -> None:
        option_parsing.check_database_schema(option_parsing.pop_common_options({"schema": "MY_SCHEMA"}, 1))


class WarnUnknownKwargsTest(parameterized.TestCase):
    def test_warns_on_leftovers(self) -> None:
        with self.assertLogs(option_parsing.logger, level="WARNING") as logs:
            option_parsing.warn_unknown_kwargs({"bogus": 1})
        self.assertIn("bogus", logs.output[0])

    def test_silent_when_empty(self) -> None:
        with mock.patch.object(option_parsing.logger, "warning", autospec=True) as mock_warning:
            option_parsing.warn_unknown_kwargs({})
        mock_warning.assert_not_called()


if __name__ == "__main__":
    absltest.main()
