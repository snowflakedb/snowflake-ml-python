import io
from pathlib import PurePath, PurePosixPath
from typing import Any
from unittest.mock import MagicMock, patch

from absl.testing import absltest, parameterized

from snowflake import snowpark
from snowflake.ml.jobs import job_definition
from snowflake.ml.jobs._interop import utils as interop_utils
from snowflake.ml.jobs._utils import arg_protocol, feature_flags, types


def _make_uploaded_payload() -> types.UploadedPayload:
    return types.UploadedPayload(
        stage_path=PurePosixPath("@payload_stage/entry"),
        entrypoint=[PurePath("/mnt/job_stage/app/launcher.py"), PurePath("/mnt/job_stage/app/entry.py")],
    )


class MLJobDefinitionTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.session = MagicMock()
        self.session.get_current_warehouse.return_value = "TEST_WH"
        self.session.get_current_database.return_value = "TEST_DB"
        self.session.get_current_schema.return_value = "TEST_SCHEMA"
        self.uploaded_payload = _make_uploaded_payload()

    def _expected_definition(self, runtime_value: str | None) -> job_definition.MLJobDefinition[[Any], Any]:
        spec_options = types.SpecOptions(
            stage_path=self.uploaded_payload.stage_path.as_posix(),
            args=None,
            enable_metrics=True,
            runtime=runtime_value,
            env_vars={},
            enable_stage_mount_v2=feature_flags.FeatureFlags.ENABLE_STAGE_MOUNT_V2.is_enabled(),
        )
        job_options = types.JobOptions(
            query_warehouse="TEST_WH",
            target_instances=1,
            min_instances=1,
            generate_suffix=False,
        )
        job_def: job_definition.MLJobDefinition[[Any], Any] = job_definition.MLJobDefinition(
            source="entry.py",
            compute_pool="E2E_TEST_POOL",
            stage_name=self.uploaded_payload.stage_path.as_posix(),
            session=self.session,
        )
        # Manually populate attributes to simulate a registered state
        job_def.database = "TEST_DB"
        job_def.schema = "TEST_SCHEMA"
        job_def.name = "entry"
        job_def.runtime_environment = runtime_value
        job_def.generate_suffix = False
        job_def.job_definition_id = "TEST_DB.TEST_SCHEMA.entry"
        job_def.entrypoint_args = [
            v.as_posix() if isinstance(v, PurePath) else v for v in self.uploaded_payload.entrypoint
        ]
        job_def.spec_options = spec_options
        job_def.job_options = job_options
        job_def.arg_protocol = arg_protocol.ArgProtocol.NONE
        job_def.default_args = None
        job_def._is_registered = True
        job_def.payload = MagicMock()

        return job_def

    def test_register_with_runtime_env_passes_through_value(self) -> None:
        with patch(
            "snowflake.ml.jobs.job_definition.payload_utils.JobPayload",
            return_value=MagicMock(upload=MagicMock(return_value=self.uploaded_payload)),
        ), patch("snowflake.ml.jobs.job_definition.payload_utils.get_payload_name", return_value="entry"):
            result: job_definition.MLJobDefinition[[Any], Any] = job_definition.MLJobDefinition.register(
                source="entry.py",
                entrypoint="entry.py",
                compute_pool="E2E_TEST_POOL",
                stage_name="payload_stage",
                session=self.session,
                runtime_environment="test_flag",
                generate_suffix=False,
            )
            expected = self._expected_definition("test_flag")
            self.assertEqual(result.job_options, expected.job_options)
            self.assertEqual(result.spec_options, expected.spec_options)
            self.assertEqual(result.job_definition_id, expected.job_definition_id)
            self.assertEqual(result.runtime_environment, expected.runtime_environment)
            self.assertEqual(result.name, expected.name)
            self.assertEqual(result.generate_suffix, expected.generate_suffix)

    def _create_job_definition_with_arg_protocol(
        self,
        arg_protocol_value: arg_protocol.ArgProtocol,
        default_args: list[Any] | None = None,
    ) -> job_definition.MLJobDefinition[[Any], Any]:
        job_def = job_definition.MLJobDefinition(
            source=lambda x: None,
            compute_pool="TEST_POOL",
            stage_name="@stage/path",
            session=self.session,
            database="TEST_DB",
            schema="TEST_SCHEMA",
            name="test_job",
        )
        # Manually set these because _register() hasn't been called
        job_def.arg_protocol = arg_protocol_value
        job_def.default_args = default_args
        return job_def

    @parameterized.named_parameters(  # type: ignore[misc]
        # NONE protocol
        ("none_with_args", arg_protocol.ArgProtocol.NONE, None, ("arg1", "arg2", 123), {}, ["arg1", "arg2", 123]),
        ("none_empty", arg_protocol.ArgProtocol.NONE, None, (), {}, []),
        # CLI protocol
        ("cli_args_only", arg_protocol.ArgProtocol.CLI, None, ("pos1", "pos2"), {}, ["pos1", "pos2"]),
        (
            "cli_with_kwargs",
            arg_protocol.ArgProtocol.CLI,
            None,
            ("pos1",),
            {"flag": "value"},
            ["pos1", "--flag", "value"],
        ),
        (
            "cli_with_defaults",
            arg_protocol.ArgProtocol.CLI,
            ["--default", "123", "--verbose"],
            ("pos1",),
            {"override": "val"},
            ["pos1", "--default", "123", "--verbose", "None", "--override", "val"],
        ),
        (
            "cli_none_kwarg_no_override",
            arg_protocol.ArgProtocol.CLI,
            ["--delay", "1"],
            (),
            {},
            ["--delay", "1"],
        ),
        (
            "cli_none_kwarg_new_key",
            arg_protocol.ArgProtocol.CLI,
            ["--delay", "1"],
            (),
            {"extra": None},
            ["--delay", "1", "--extra", "None"],
        ),
        (
            "cli_none_kwarg_old_key",
            arg_protocol.ArgProtocol.CLI,
            ["--delay", "1"],
            (),
            {"delay": None},
            ["--delay", "None"],
        ),
        (
            "cli_override_bare_flag",
            arg_protocol.ArgProtocol.CLI,
            ["--verbose"],
            (),
            {"verbose": "0"},
            ["--verbose", "0"],
        ),
        (
            "cli_kwargs_override_defaults",
            arg_protocol.ArgProtocol.CLI,
            ["--key", "default_value"],
            (),
            {"key": "new_value"},
            ["--key", "new_value"],
        ),
        # CLI protocol: edge cases with special characters
        (
            "cli_value_with_spaces",
            arg_protocol.ArgProtocol.CLI,
            None,
            ("hello world",),
            {"delay": "2"},
            ["hello world", "--delay", "2"],
        ),
        (
            "cli_value_with_double_quotes",
            arg_protocol.ArgProtocol.CLI,
            None,
            ('say "hi"',),
            {},
            ['say "hi"'],
        ),
        (
            "cli_value_with_single_quote",
            arg_protocol.ArgProtocol.CLI,
            None,
            ("it's a test",),
            {},
            ["it's a test"],
        ),
        (
            "cli_value_with_mixed_quotes",
            arg_protocol.ArgProtocol.CLI,
            None,
            ('it\'s "quoted"',),
            {},
            ['it\'s "quoted"'],
        ),
        (
            "cli_value_looks_like_flags",
            arg_protocol.ArgProtocol.CLI,
            None,
            ("--other_param foo --cli_injection bar",),
            {},
            ["--other_param foo --cli_injection bar"],
        ),
        # PICKLE protocol
        ("pickle_empty", arg_protocol.ArgProtocol.PICKLE, None, (), {}, []),
        (
            "pickle_basic",
            arg_protocol.ArgProtocol.PICKLE,
            None,
            ("arg1", "arg2"),
            {"key": "value"},
            [["arg1", "arg2"], {"key": "value"}],
        ),
        (
            "pickle_with_runtime_args_1",
            arg_protocol.ArgProtocol.PICKLE,
            None,
            ("bar",),
            {"arg2": 2},
            [["bar"], {"arg2": 2}],
        ),
        (
            "pickle_with_session",
            arg_protocol.ArgProtocol.PICKLE,
            None,
            ("MOCK_SESSION", "arg2"),
            {"key": "value"},
            None,  # Non-JSON-serializable data uses protocol fallback, value verified separately
        ),
    )
    def test_prepare_arguments(
        self,
        protocol: arg_protocol.ArgProtocol,
        default_args: list[Any] | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        expected: Any,
    ) -> None:
        job_def = self._create_job_definition_with_arg_protocol(protocol, default_args)

        # Handle mock session placeholder if used in test data
        actual_args = list(args)
        for i, val in enumerate(actual_args):
            if val == "MOCK_SESSION":
                actual_args[i] = MagicMock(spec=snowpark.Session)

        result = job_def._prepare_arguments(*actual_args, **kwargs)

        if protocol != arg_protocol.ArgProtocol.PICKLE or not result:
            self.assertEqual(result, expected)
        else:
            assert result is not None
            self.assertEqual(len(result), 1)
            self.assertTrue(result[0].startswith("--function_args="))
            encoded_data = result[0].split("=", 1)[1]
            dto = interop_utils.DEFAULT_CODEC.decode(io.StringIO(encoded_data))
            value = dto.value
            self.assertEqual(value, expected)

    def test_prepare_arguments_none_protocol_raises_on_kwargs(self) -> None:
        job_def: job_definition.MLJobDefinition[..., Any] = self._create_job_definition_with_arg_protocol(
            arg_protocol.ArgProtocol.NONE
        )
        with self.assertRaises(ValueError) as ctx:
            job_def._prepare_arguments("arg1", key="value")
        self.assertIn("Keyword arguments are not supported", str(ctx.exception))
        self.assertIn("ArgProtocol.NONE", str(ctx.exception))


if __name__ == "__main__":
    absltest.main()
