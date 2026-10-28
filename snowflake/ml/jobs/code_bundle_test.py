import functools
import pickle
from pathlib import Path, PurePosixPath
from typing import Any, cast
from unittest.mock import MagicMock, call, patch

import yaml
from absl.testing import absltest, parameterized

from snowflake.connector import errors as connector_errors
from snowflake.ml._internal.utils import identifier
from snowflake.ml.jobs import code_bundle
from snowflake.ml.jobs._utils import stage_utils
from snowflake.snowpark import exceptions as sp_exceptions

_RESOLVED_IMAGE = "/db/schema/repo/image_name:2.7.2"


class BuildCodeBundleManifestTest(parameterized.TestCase):
    def test_required_fields_only(self) -> None:
        manifest = yaml.safe_load(
            code_bundle._build_code_bundle_manifest("MY_POOL", "@DB.SCHEMA.RESULT_STAGE", _RESOLVED_IMAGE)
        )

        self.assertEqual(
            manifest,
            {
                "bundle": {
                    "type": "ml",
                    "compute_type": "compute_pool",
                    "language": "python",
                    "compute_options": {
                        "compute_pool": "MY_POOL",
                        "target_instances": 1,
                        "min_instances": 1,
                        "runtime_version": _RESOLVED_IMAGE,
                    },
                    "properties": {
                        "enable_metrics": True,
                        "result_stage": "@DB.SCHEMA.RESULT_STAGE",
                    },
                }
            },
        )

    def test_all_optional_fields(self) -> None:
        manifest = yaml.safe_load(
            code_bundle._build_code_bundle_manifest(
                "MY_POOL",
                "@DB.SCHEMA.RESULT_STAGE",
                _RESOLVED_IMAGE,
                query_warehouse="MY_WH",
                target_instances=3,
                min_instances=2,
                enable_metrics=False,
                artifact_repositories=["DB.SCHEMA.REPO"],
                secrets=["DB.SCHEMA.SECRET"],
                external_access_integrations=["DB.SCHEMA.EAI"],
                env_vars=[{"FOO": "bar"}],
            )
        )

        self.assertEqual(
            manifest["bundle"]["compute_options"],
            {
                "compute_pool": "MY_POOL",
                "query_warehouse": "MY_WH",
                "target_instances": 3,
                "min_instances": 2,
                "runtime_version": _RESOLVED_IMAGE,
            },
        )
        self.assertEqual(
            manifest["bundle"]["properties"],
            {
                "enable_metrics": False,
                "result_stage": "@DB.SCHEMA.RESULT_STAGE",
            },
        )
        self.assertEqual(manifest["bundle"]["artifact_repositories"], ["DB.SCHEMA.REPO"])
        self.assertEqual(manifest["bundle"]["secrets"], ["DB.SCHEMA.SECRET"])
        self.assertEqual(manifest["bundle"]["external_access_integrations"], ["DB.SCHEMA.EAI"])
        self.assertEqual(manifest["bundle"]["env_vars"], [{"FOO": "bar"}])


class EscapeStringLiteralTest(parameterized.TestCase):
    @parameterized.named_parameters(  # type: ignore[misc]
        ("no_special_characters", "payload/train.py", "payload/train.py"),
        ("empty", "", ""),
        ("single_quote", "it's", r"it\'s"),
        ("backslash", r"a\b", r"a\\b"),
        ("newline", "a\nb", r"a\nb"),
        ("carriage_return", "a\rb", r"a\rb"),
        # Only escaping backslashes first keeps this from collapsing into \\\\' .
        ("backslash_then_quote", r"\'", r"\\\'"),
    )
    def test_escapes_sql_literal_characters(self, value: str, expected: str) -> None:
        self.assertEqual(code_bundle._escape_string_literal(value), expected)

    def test_matches_connector_escaping_while_both_exist(self) -> None:
        """The local escaper must agree with the connector's own escaping of bound string values.

        snowflake.connector.converter was removed in connector 5, which is why the escaping is no
        longer taken from there; this pins the replacement to it on the 4.x line we still support.
        """
        try:
            from snowflake.connector import converter
        except ImportError:
            self.skipTest("snowflake.connector.converter was removed in connector 5")

        for value in ("payload/train.py", "", "it's", "a\\b", "a\nb", "a\rb", "\\'", "'\\"):
            with self.subTest(value=value):
                self.assertEqual(
                    code_bundle._escape_string_literal(value),
                    converter.SnowflakeConverter.escape(value),
                )

    def test_module_does_not_import_removed_converter_module(self) -> None:
        """A module-level `from snowflake.connector import converter` breaks import under connector 5."""
        self.assertFalse(hasattr(code_bundle, "converter"))


class MLCodeBundleTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.session = MagicMock()
        self.session.get_current_database.return_value = "TEST_DB"
        self.session.get_current_schema.return_value = "TEST_SCHEMA"
        self.session.get_current_warehouse.return_value = "TEST_WH"
        self.resolved_source = Path("/resolved/my_project")

    def _uploaded_manifest(self) -> dict[str, Any]:
        for upload_call in self.session.file.put_stream.call_args_list:
            if upload_call.kwargs.get("stage_location", "").endswith("code_bundle.yaml"):
                return cast(dict[str, Any], yaml.safe_load(upload_call.args[0].getvalue().decode()))
        raise AssertionError("code_bundle.yaml was never uploaded")

    def _register_patches(self) -> tuple[Any, ...]:
        return (
            patch("snowflake.ml.jobs.code_bundle.payload_utils.resolve_source", return_value=self.resolved_source),
            patch("snowflake.ml.jobs.code_bundle.payload_utils.upload_payloads"),
            patch("snowflake.ml.jobs.code_bundle.payload_utils.upload_imports"),
            patch(
                "snowflake.ml.jobs.code_bundle.runtime_env_utils.resolve_runtime_image",
                return_value=_RESOLVED_IMAGE,
            ),
            patch("snowflake.ml.jobs.code_bundle.query_helper.run_query"),
        )

    def test_register_uploads_then_creates_bundle(self) -> None:
        resolve_source, upload_payloads, upload_imports, resolve_image, run_query = self._register_patches()
        with (
            resolve_source,
            upload_payloads as mock_upload,
            upload_imports as mock_upload_imports,
            resolve_image as mock_resolve_image,
            run_query as mock_run_query,
        ):
            result = code_bundle.register_code_bundle(
                "./my_project",
                "MY_POOL",
                "@RESULT_DB.RESULT_SCHEMA.RESULT_STAGE",
                stage_name="payload_stage",
                name="my_bundle",
                session=self.session,
            )

        expected_stage_path = PurePosixPath("@TEST_DB.TEST_SCHEMA.PAYLOAD_STAGE/my_bundle")
        _, called_stage_path, payload_spec = mock_upload.call_args.args
        self.assertEqual(called_stage_path, expected_stage_path)
        self.assertEqual(payload_spec.source_path, self.resolved_source)
        self.assertEqual(mock_upload.call_args.kwargs, {"overwrite": True})
        mock_upload_imports.assert_called_once_with(None, self.session, expected_stage_path)
        mock_run_query.assert_called_with(
            self.session,
            "CREATE OR REPLACE CODE BUNDLE IDENTIFIER(?) FROM '@TEST_DB.TEST_SCHEMA.PAYLOAD_STAGE/my_bundle'",
            params=["TEST_DB.TEST_SCHEMA.MY_BUNDLE"],
        )
        self.assertEqual(result.name, "TEST_DB.TEST_SCHEMA.MY_BUNDLE")
        self.assertEqual(result._stage_path, "@TEST_DB.TEST_SCHEMA.PAYLOAD_STAGE/my_bundle")

        manifest = self._uploaded_manifest()
        self.assertEqual(manifest["bundle"]["compute_options"]["compute_pool"], "MY_POOL")
        self.assertEqual(manifest["bundle"]["properties"]["result_stage"], "@RESULT_DB.RESULT_SCHEMA.RESULT_STAGE")
        self.assertEqual(manifest["bundle"]["compute_options"]["runtime_version"], _RESOLVED_IMAGE)
        self.assertEqual(manifest["bundle"]["compute_options"]["query_warehouse"], "TEST_WH")
        mock_resolve_image.assert_called_once_with(self.session, "MY_POOL", None)

    def test_register_warns_public_preview(self) -> None:
        resolve_source, upload_payloads, upload_imports, resolve_image, run_query = self._register_patches()
        with (
            resolve_source,
            upload_payloads,
            upload_imports,
            resolve_image,
            run_query,
            self.assertLogs("snowflake.ml.jobs.code_bundle", level="WARNING") as logs,
        ):
            code_bundle.register_code_bundle(
                "./my_project",
                "MY_POOL",
                "@RESULT_STAGE",
                stage_name="payload_stage",
                name="my_bundle",
                session=self.session,
            )

        self.assertEqual(
            logs.output,
            [
                "WARNING:snowflake.ml.jobs.code_bundle:Code bundles are in Public Preview since 2.6.0."
                " Do not use them in production."
            ],
        )

    def test_handle_exposes_name_and_repr(self) -> None:
        bundle = code_bundle.MLCodeBundle("TEST_DB.TEST_SCHEMA.MY_BUNDLE", session=self.session)
        self.assertEqual(bundle.name, "TEST_DB.TEST_SCHEMA.MY_BUNDLE")
        self.assertEqual(repr(bundle), "MLCodeBundle('TEST_DB.TEST_SCHEMA.MY_BUNDLE')")

    def test_handle_pickles_without_session(self) -> None:
        self.session.get_current_account.return_value = "TEST_ACCOUNT"
        self.session.get_current_role.return_value = "TEST_ROLE"
        bundle = code_bundle.MLCodeBundle("TEST_DB.TEST_SCHEMA.MY_BUNDLE", session=self.session)

        self.assertNotIn("_session", bundle.__getstate__())
        restored = pickle.loads(pickle.dumps(bundle))
        self.assertEqual(restored.name, "TEST_DB.TEST_SCHEMA.MY_BUNDLE")

    def test_register_uploads_pip_requirements(self) -> None:
        resolve_source, upload_payloads, upload_imports, resolve_image, run_query = self._register_patches()
        with resolve_source, upload_payloads, upload_imports, resolve_image, run_query:
            code_bundle.register_code_bundle(
                "./my_project",
                "MY_POOL",
                "@RESULT_STAGE",
                stage_name="payload_stage",
                name="my_bundle",
                pip_requirements=["tabulate==0.9.0", "requests"],
                session=self.session,
            )

        requirements_uploads = [
            call
            for call in self.session.file.put_stream.call_args_list
            if call.kwargs["stage_location"].endswith("/requirements.txt")
        ]
        self.assertEqual(len(requirements_uploads), 1)
        self.assertEqual(
            requirements_uploads[0].kwargs["stage_location"],
            "@TEST_DB.TEST_SCHEMA.PAYLOAD_STAGE/my_bundle/requirements.txt",
        )
        self.assertEqual(requirements_uploads[0].args[0].getvalue().decode(), "tabulate==0.9.0\nrequests")

    def test_register_warns_on_unknown_kwargs(self) -> None:
        resolve_source, upload_payloads, upload_imports, resolve_image, run_query = self._register_patches()
        with (
            resolve_source,
            upload_payloads,
            upload_imports,
            resolve_image,
            run_query,
            self.assertLogs("snowflake.ml.jobs", level="WARNING") as logs,
        ):
            code_bundle.register_code_bundle(
                "./my_project",
                "MY_POOL",
                "@RESULT_STAGE",
                stage_name="payload_stage",
                name="my_bundle",
                bogus_option=1,
                session=self.session,
            )

        self.assertTrue(
            any("Ignoring unknown kwargs" in line and "bogus_option" in line for line in logs.output), logs.output
        )

    def test_register_passes_explicit_runtime_environment_through(self) -> None:
        resolve_source, upload_payloads, upload_imports, resolve_image, run_query = self._register_patches()
        with (
            resolve_source,
            upload_payloads,
            upload_imports,
            resolve_image as mock_resolve_image,
            run_query,
        ):
            code_bundle.register_code_bundle(
                "./my_project",
                "MY_POOL",
                "@RESULT_DB.RESULT_SCHEMA.RESULT_STAGE",
                stage_name="payload_stage",
                name="my_bundle",
                runtime_environment="2.7.2",
                session=self.session,
            )

        mock_resolve_image.assert_called_once_with(self.session, "MY_POOL", "2.7.2")
        self.assertEqual(self._uploaded_manifest()["bundle"]["compute_options"]["runtime_version"], _RESOLVED_IMAGE)

    def test_register_passes_explicit_query_warehouse_through(self) -> None:
        resolve_source, upload_payloads, upload_imports, resolve_image, run_query = self._register_patches()
        with resolve_source, upload_payloads, upload_imports, resolve_image, run_query:
            code_bundle.register_code_bundle(
                "./my_project",
                "MY_POOL",
                "@RESULT_DB.RESULT_SCHEMA.RESULT_STAGE",
                stage_name="payload_stage",
                name="my_bundle",
                query_warehouse="EXPLICIT_WH",
                session=self.session,
            )

        self.assertEqual(self._uploaded_manifest()["bundle"]["compute_options"]["query_warehouse"], "EXPLICIT_WH")

    def test_register_rejects_out_of_range_instance_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_instances must be greater than 0"):
            code_bundle.register_code_bundle(
                "./my_project",
                "MY_POOL",
                "@RESULT_DB.RESULT_SCHEMA.RESULT_STAGE",
                stage_name="payload_stage",
                target_instances=0,
                session=self.session,
            )
        with self.assertRaisesRegex(ValueError, "min_instances must be greater than 0"):
            code_bundle.register_code_bundle(
                "./my_project",
                "MY_POOL",
                "@RESULT_DB.RESULT_SCHEMA.RESULT_STAGE",
                stage_name="payload_stage",
                target_instances=2,
                min_instances=3,
                session=self.session,
            )

    def test_register_auto_generates_name_from_source(self) -> None:
        resolve_source, upload_payloads, upload_imports, resolve_image, run_query = self._register_patches()
        with (
            patch(
                "snowflake.ml.jobs.code_bundle.payload_utils.get_payload_name",
                return_value="my-project_abcd1234",
            ),
            resolve_source,
            upload_payloads,
            upload_imports,
            resolve_image,
            run_query as mock_run_query,
        ):
            result = code_bundle.register_code_bundle(
                "./my-project",
                "MY_POOL",
                "@RESULT_DB.RESULT_SCHEMA.RESULT_STAGE",
                stage_name="payload_stage",
                session=self.session,
            )

        mock_run_query.assert_called_with(
            self.session,
            "CREATE OR REPLACE CODE BUNDLE IDENTIFIER(?) FROM '@TEST_DB.TEST_SCHEMA.PAYLOAD_STAGE/my-project_abcd1234'",
            params=['TEST_DB.TEST_SCHEMA."my-project_abcd1234"'],
        )
        self.assertEqual(result.name, 'TEST_DB.TEST_SCHEMA."my-project_abcd1234"')

    def test_register_explicit_qualified_name_parses_database_and_schema(self) -> None:
        resolve_source, upload_payloads, upload_imports, resolve_image, run_query = self._register_patches()
        with (
            resolve_source,
            upload_payloads,
            upload_imports,
            resolve_image,
            run_query as mock_run_query,
        ):
            result = code_bundle.register_code_bundle(
                "./my_project",
                "MY_POOL",
                "@RESULT_DB.RESULT_SCHEMA.RESULT_STAGE",
                stage_name="payload_stage",
                name="OTHER_DB.OTHER_SCHEMA.my_bundle",
                session=self.session,
            )

        mock_run_query.assert_called_with(
            self.session,
            "CREATE OR REPLACE CODE BUNDLE IDENTIFIER(?) FROM '@OTHER_DB.OTHER_SCHEMA.PAYLOAD_STAGE/my_bundle'",
            params=["OTHER_DB.OTHER_SCHEMA.MY_BUNDLE"],
        )
        self.assertEqual(result.name, "OTHER_DB.OTHER_SCHEMA.MY_BUNDLE")

    def test_register_quoted_name_uses_stage_safe_subpath(self) -> None:
        resolve_source, upload_payloads, upload_imports, resolve_image, run_query = self._register_patches()
        with (
            resolve_source,
            upload_payloads as mock_upload,
            upload_imports,
            resolve_image,
            run_query as mock_run_query,
        ):
            result = code_bundle.register_code_bundle(
                "./my project",
                "MY_POOL",
                "@RESULT_DB.RESULT_SCHEMA.RESULT_STAGE",
                stage_name="payload_stage",
                name='"my bundle"',
                session=self.session,
            )

        uploaded_stage_path = mock_upload.call_args.args[1].as_posix()
        self.assertRegex(uploaded_stage_path, r"^@TEST_DB\.TEST_SCHEMA\.PAYLOAD_STAGE/code_bundle_[0-9a-f]{16}$")
        self.assertEqual(result.name, 'TEST_DB.TEST_SCHEMA."my bundle"')
        self.assertEqual(result._stage_path, uploaded_stage_path)
        mock_run_query.assert_called_with(
            self.session,
            f"CREATE OR REPLACE CODE BUNDLE IDENTIFIER(?) FROM '{uploaded_stage_path}'",
            params=['TEST_DB.TEST_SCHEMA."my bundle"'],
        )

    def test_generated_name_with_spaces_uses_stable_stage_subpath(self) -> None:
        bundle_fqn = 'TEST_DB.TEST_SCHEMA."my project_abcd1234"'
        first_subpath = code_bundle._get_code_bundle_stage_subpath("my project_abcd1234", bundle_fqn)
        second_subpath = code_bundle._get_code_bundle_stage_subpath("my project_abcd1234", bundle_fqn)

        self.assertRegex(first_subpath, r"^code_bundle_[0-9a-f]{16}$")
        self.assertEqual(first_subpath, second_subpath)

    def test_schema_level_identifier_quotes_sql_tokens(self) -> None:
        quoted_name = identifier.get_inferred_name("; DROP TABLE foo; --")
        self.assertEqual(quoted_name, '"; DROP TABLE foo; --"')
        self.assertEqual(
            identifier.get_schema_level_object_identifier("TEST_DB", "TEST_SCHEMA", quoted_name),
            'TEST_DB.TEST_SCHEMA."; DROP TABLE foo; --"',
        )
        quoted_with_embedded_quote = identifier.get_inferred_name('foo"bar')
        self.assertEqual(quoted_with_embedded_quote, '"foo""bar"')
        self.assertEqual(
            identifier.get_schema_level_object_identifier("TEST_DB", "TEST_SCHEMA", quoted_with_embedded_quote),
            'TEST_DB.TEST_SCHEMA."foo""bar"',
        )

    def test_register_binds_quoted_sql_token_name(self) -> None:
        resolve_source, upload_payloads, upload_imports, resolve_image, run_query = self._register_patches()
        with (
            patch(
                "snowflake.ml.jobs.code_bundle.payload_utils.get_payload_name",
                return_value="; DROP TABLE foo; --",
            ),
            patch(
                "snowflake.ml.jobs.code_bundle.stage_utils.try_resolve_stage_path",
                return_value="@payload_stage/safe",
            ),
            resolve_source,
            upload_payloads,
            upload_imports,
            resolve_image,
            run_query as mock_run_query,
        ):
            result = code_bundle.register_code_bundle(
                "./my_project",
                "MY_POOL",
                "@RESULT_DB.RESULT_SCHEMA.RESULT_STAGE",
                stage_name="payload_stage",
                session=self.session,
            )

        mock_run_query.assert_called_with(
            self.session,
            "CREATE OR REPLACE CODE BUNDLE IDENTIFIER(?) FROM '@payload_stage/safe'",
            params=['TEST_DB.TEST_SCHEMA."; DROP TABLE foo; --"'],
        )
        self.assertEqual(result.name, 'TEST_DB.TEST_SCHEMA."; DROP TABLE foo; --"')

    @parameterized.named_parameters(  # type: ignore[misc]
        ("unqualified", "result_stage", "@TEST_DB.TEST_SCHEMA.RESULT_STAGE"),
        ("quoted_unqualified", '"result_stage"', '@TEST_DB.TEST_SCHEMA."result_stage"'),
        ("schema_qualified", "@OTHER_SCHEMA.RESULT_STAGE/out", "@TEST_DB.OTHER_SCHEMA.RESULT_STAGE/out"),
        ("fully_qualified", "@OTHER_DB.OTHER_SCHEMA.RESULT_STAGE", "@OTHER_DB.OTHER_SCHEMA.RESULT_STAGE"),
    )
    def test_register_qualifies_result_stage(self, result_stage: str, expected: str) -> None:
        resolve_source, upload_payloads, upload_imports, resolve_image, run_query = self._register_patches()
        with resolve_source, upload_payloads, upload_imports, resolve_image, run_query:
            code_bundle.register_code_bundle(
                "./my_project",
                "MY_POOL",
                result_stage,
                stage_name="payload_stage",
                name="my_bundle",
                session=self.session,
            )

        self.assertEqual(self._uploaded_manifest()["bundle"]["properties"]["result_stage"], expected)

    @parameterized.named_parameters(  # type: ignore[misc]
        ("unqualified", "payload_stage", "@BUNDLE_DB.BUNDLE_SCHEMA.PAYLOAD_STAGE/my_bundle"),
        ("unqualified_with_subdir", "@payload_stage/sub", "@BUNDLE_DB.BUNDLE_SCHEMA.PAYLOAD_STAGE/sub/my_bundle"),
        ("schema_qualified", "OTHER_SCHEMA.payload_stage", "@BUNDLE_DB.OTHER_SCHEMA.PAYLOAD_STAGE/my_bundle"),
        ("fully_qualified", "@OTHER_DB.OTHER_SCHEMA.payload_stage", "@OTHER_DB.OTHER_SCHEMA.PAYLOAD_STAGE/my_bundle"),
    )
    def test_register_resolves_payload_stage_in_bundle_schema(self, stage_name: str, expected: str) -> None:
        resolve_source, upload_payloads, upload_imports, resolve_image, run_query = self._register_patches()
        with resolve_source, upload_payloads as mock_upload, upload_imports, resolve_image, run_query:
            code_bundle.register_code_bundle(
                "./my_project",
                "MY_POOL",
                "@RESULT_STAGE",
                stage_name=stage_name,
                name="BUNDLE_DB.BUNDLE_SCHEMA.my_bundle",
                session=self.session,
            )

        self.assertEqual(mock_upload.call_args.args[1], PurePosixPath(expected))

    def test_register_clears_bundle_directory_before_upload(self) -> None:
        resolve_source, upload_payloads, upload_imports, resolve_image, run_query = self._register_patches()
        call_order = MagicMock()
        with resolve_source, upload_payloads as mock_upload, upload_imports, resolve_image, run_query as mock_run_query:
            call_order.attach_mock(mock_run_query, "run_query")
            call_order.attach_mock(mock_upload, "upload_payloads")
            code_bundle.register_code_bundle(
                "./my_project",
                "MY_POOL",
                "@RESULT_STAGE",
                stage_name="payload_stage",
                name="my_bundle",
                session=self.session,
            )

        calls = [(name, args) for name, args, _ in call_order.mock_calls]
        remove_index = calls.index(
            ("run_query", (self.session, "REMOVE @TEST_DB.TEST_SCHEMA.PAYLOAD_STAGE/my_bundle/"))
        )
        upload_index = next(index for index, (name, _) in enumerate(calls) if name == "upload_payloads")
        self.assertLess(remove_index, upload_index)
        remove_statements = [args[1] for name, args in calls if name == "run_query" and args[1].startswith("REMOVE")]
        self.assertEqual(remove_statements, ["REMOVE @TEST_DB.TEST_SCHEMA.PAYLOAD_STAGE/my_bundle/"])

    def test_register_qualified_stage_source_without_session_schema(self) -> None:
        self.session.get_current_schema.return_value = None
        _, upload_payloads, upload_imports, resolve_image, run_query = self._register_patches()
        with (
            patch(
                "snowflake.ml.jobs.code_bundle.payload_utils.resolve_source",
                return_value=stage_utils.StagePath("@SRC_DB.SRC_SCHEMA.SRC_STAGE/src/"),
            ),
            upload_payloads as mock_upload,
            upload_imports,
            resolve_image,
            run_query,
        ):
            result = code_bundle.register_code_bundle(
                "@SRC_DB.SRC_SCHEMA.SRC_STAGE/src/",
                "MY_POOL",
                "@RESULT_STAGE",
                stage_name="payload_stage",
                name="BUNDLE_DB.BUNDLE_SCHEMA.my_bundle",
                session=self.session,
            )

        mock_upload.assert_called_once()
        self.assertEqual(result.name, "BUNDLE_DB.BUNDLE_SCHEMA.MY_BUNDLE")

    def test_register_continues_when_remove_is_rejected(self) -> None:
        resolve_source, upload_payloads, upload_imports, resolve_image, run_query = self._register_patches()

        def reject_remove(session: Any, query: str, **kwargs: Any) -> list[Any]:
            if query.startswith("REMOVE"):
                raise sp_exceptions.SnowparkSQLException("Unsupported statement type 'REMOVE_FILES'.")
            return []

        with (
            resolve_source,
            upload_payloads as mock_upload,
            upload_imports,
            resolve_image,
            run_query as mock_run_query,
        ):
            mock_run_query.side_effect = reject_remove
            result = code_bundle.register_code_bundle(
                "./my_project",
                "MY_POOL",
                "@RESULT_STAGE",
                stage_name="payload_stage",
                name="my_bundle",
                session=self.session,
            )

        mock_upload.assert_called_once()
        self.assertTrue(
            any(call.args[1].startswith("CREATE OR REPLACE CODE BUNDLE") for call in mock_run_query.call_args_list)
        )
        self.assertEqual(result.name, "TEST_DB.TEST_SCHEMA.MY_BUNDLE")

    @parameterized.named_parameters(  # type: ignore[misc]
        ("bundle_directory", "@payload_stage/my_bundle/", True),
        ("inside_bundle_directory", "@TEST_DB.TEST_SCHEMA.PAYLOAD_STAGE/my_bundle/src", True),
        ("sibling_with_same_prefix", "@payload_stage/my_bundle2/", False),
        ("other_directory", "@payload_stage/raw_src/", False),
        ("other_schema", "@OTHER_SCHEMA.payload_stage/my_bundle/", False),
    )
    def test_register_rejects_source_inside_bundle_directory(self, source: str, rejected: bool) -> None:
        _, upload_payloads, upload_imports, resolve_image, run_query = self._register_patches()
        with (
            patch(
                "snowflake.ml.jobs.code_bundle.payload_utils.resolve_source",
                return_value=stage_utils.StagePath(source),
            ),
            upload_payloads as mock_upload,
            upload_imports,
            resolve_image,
            run_query as mock_run_query,
        ):
            register = functools.partial(
                code_bundle.register_code_bundle,
                source,
                "MY_POOL",
                "@RESULT_STAGE",
                stage_name="payload_stage",
                name="my_bundle",
                session=self.session,
            )
            if rejected:
                with self.assertRaisesRegex(ValueError, "is inside the code bundle's payload directory"):
                    register()
            else:
                register()

        if rejected:
            mock_upload.assert_not_called()
            self.assertFalse(any(call.args[1].startswith("REMOVE") for call in mock_run_query.call_args_list))
        else:
            mock_upload.assert_called_once()

    def test_register_escapes_quotes_in_source_stage_literal(self) -> None:
        resolve_source, upload_payloads, upload_imports, resolve_image, run_query = self._register_patches()
        with (
            patch(
                "snowflake.ml.jobs.code_bundle.stage_utils.try_resolve_stage_path",
                return_value="@payload_stage/it's\\dir",
            ),
            resolve_source,
            upload_payloads,
            upload_imports,
            resolve_image,
            run_query as mock_run_query,
        ):
            code_bundle.register_code_bundle(
                "./my_project",
                "MY_POOL",
                "@RESULT_DB.RESULT_SCHEMA.RESULT_STAGE",
                stage_name="payload_stage",
                name="my_bundle",
                session=self.session,
            )

        mock_run_query.assert_called_with(
            self.session,
            "CREATE OR REPLACE CODE BUNDLE IDENTIFIER(?) FROM '@payload_stage/it\\'s\\\\dir'",
            params=["TEST_DB.TEST_SCHEMA.MY_BUNDLE"],
        )

    def test_registered_handle_keeps_payload_stage_context_for_delete(self) -> None:
        resolve_source, upload_payloads, upload_imports, resolve_image, run_query = self._register_patches()
        with resolve_source, upload_payloads, upload_imports, resolve_image, run_query:
            bundle = code_bundle.register_code_bundle(
                "./my_project",
                "MY_POOL",
                "@RESULT_DB.RESULT_SCHEMA.RESULT_STAGE",
                stage_name="payload_stage",
                name="my_bundle",
                session=self.session,
            )

        self.session.get_current_database.return_value = "OTHER_DB"
        self.session.get_current_schema.return_value = "OTHER_SCHEMA"
        with patch("snowflake.ml.jobs.code_bundle.query_helper.run_query"):
            bundle.delete()

        self.session.sql.assert_called_once_with("REMOVE @TEST_DB.TEST_SCHEMA.PAYLOAD_STAGE/my_bundle/")

    def test_register_rejects_unquoted_sql_token_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be parsed as a SQL identifier"):
            code_bundle.register_code_bundle(
                "./my_project",
                "MY_POOL",
                "@RESULT_DB.RESULT_SCHEMA.RESULT_STAGE",
                stage_name="payload_stage",
                name="; DROP TABLE foo; --",
                session=self.session,
            )

    def test_register_callable_source_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support callable payloads"):
            code_bundle.register_code_bundle(
                lambda: None,  # type: ignore[arg-type]
                "MY_POOL",
                "@RESULT_DB.RESULT_SCHEMA.RESULT_STAGE",
                stage_name="payload_stage",
                session=self.session,
            )

    def test_register_database_without_schema_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Schema must be specified"):
            code_bundle.register_code_bundle(
                "./my_project",
                "MY_POOL",
                "@RESULT_DB.RESULT_SCHEMA.RESULT_STAGE",
                stage_name="payload_stage",
                database="TEST_DB",
                session=self.session,
            )

    def test_register_no_database_in_context_raises(self) -> None:
        self.session.get_current_database.return_value = None
        with self.assertRaisesRegex(ValueError, "Database must be specified"):
            code_bundle.register_code_bundle(
                "./my_project",
                "MY_POOL",
                "@RESULT_DB.RESULT_SCHEMA.RESULT_STAGE",
                stage_name="payload_stage",
                name="my_bundle",
                session=self.session,
            )


class RegisterJobFromCodeBundleTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.session = MagicMock()
        self.async_job = MagicMock()
        self.async_job.query_id = "01c652b3-0000-0000-0000-000000000000"
        self.async_job.is_failed.return_value = False
        self.session.sql.return_value.collect_nowait.return_value = self.async_job
        self.session.get_current_database.return_value = "SESSION_DB"

    def _mock_history_rows(self, job_id: str | None) -> list[list[str | None]]:
        return [[job_id]] if job_id else []

    def test_trigger_with_bundle_handle_and_args(self) -> None:
        bundle = code_bundle.MLCodeBundle("TEST_DB.TEST_SCHEMA.MY_BUNDLE", session=self.session)
        with patch(
            "snowflake.ml.jobs.code_bundle.query_helper.run_query", return_value=self._mock_history_rows("TRAIN_PY_1")
        ) as mock_run_query:
            result = code_bundle.register_job_from_code_bundle(
                bundle,
                entrypoint="train.py",
                args=["--epochs", "10"],
            )

        self.session.sql.assert_called_once_with(
            "EXECUTE CODE BUNDLE IDENTIFIER(?) ENTRYPOINT = 'train.py' ARGUMENTS = (?, ?)",
            params=["TEST_DB.TEST_SCHEMA.MY_BUNDLE", "--epochs", "10"],
        )
        self.assertEqual(
            mock_run_query.call_args_list,
            [
                call(
                    self.session,
                    "SELECT RUNTIME_NAME FROM TABLE(TEST_DB.INFORMATION_SCHEMA.CODE_BUNDLE_HISTORY(QUERY_ID => ?))",
                    params=[self.async_job.query_id],
                ),
                call(self.session, "DESCRIBE SERVICE IDENTIFIER(?)", params=["TRAIN_PY_1"]),
            ],
        )
        self.assertEqual(result.id, "TRAIN_PY_1")

    def test_trigger_with_bundle_name_string_no_args(self) -> None:
        with patch(
            "snowflake.ml.jobs.code_bundle.query_helper.run_query", return_value=self._mock_history_rows("TRAIN_PY_1")
        ):
            code_bundle.register_job_from_code_bundle(
                "TEST_DB.TEST_SCHEMA.MY_BUNDLE",
                entrypoint="train.py",
                session=self.session,
            )

        self.session.sql.assert_called_once_with(
            "EXECUTE CODE BUNDLE IDENTIFIER(?) ENTRYPOINT = 'train.py'",
            params=["TEST_DB.TEST_SCHEMA.MY_BUNDLE"],
        )

    @parameterized.named_parameters(  # type: ignore[misc]
        ("qualified_bundle", "OTHER_DB.OTHER_SCHEMA.MY_BUNDLE", "SESSION_DB", "OTHER_DB.INFORMATION_SCHEMA"),
        ("unqualified_bundle", "MY_BUNDLE", "SESSION_DB", "SESSION_DB.INFORMATION_SCHEMA"),
        ("no_database_anywhere", "MY_BUNDLE", None, "INFORMATION_SCHEMA"),
    )
    def test_trigger_reads_history_from_bundle_database(
        self, bundle_name: str, session_database: str | None, expected_information_schema: str
    ) -> None:
        self.session.get_current_database.return_value = session_database
        with patch(
            "snowflake.ml.jobs.code_bundle.query_helper.run_query", return_value=self._mock_history_rows("TRAIN_PY_1")
        ) as mock_run_query:
            code_bundle.register_job_from_code_bundle(bundle_name, entrypoint="train.py", session=self.session)

        self.assertEqual(
            mock_run_query.call_args_list[0],
            call(
                self.session,
                f"SELECT RUNTIME_NAME FROM TABLE({expected_information_schema}.CODE_BUNDLE_HISTORY(QUERY_ID => ?))",
                params=[self.async_job.query_id],
            ),
        )

    def test_trigger_escapes_entrypoint_literal(self) -> None:
        with patch(
            "snowflake.ml.jobs.code_bundle.query_helper.run_query", return_value=self._mock_history_rows("TRAIN_PY_1")
        ):
            code_bundle.register_job_from_code_bundle(
                "TEST_DB.TEST_SCHEMA.MY_BUNDLE",
                entrypoint="it's\\dir\nmain.py",
                session=self.session,
            )

        self.session.sql.assert_called_once_with(
            "EXECUTE CODE BUNDLE IDENTIFIER(?) ENTRYPOINT = 'it\\'s\\\\dir\\nmain.py'",
            params=["TEST_DB.TEST_SCHEMA.MY_BUNDLE"],
        )

    def test_trigger_warns_public_preview(self) -> None:
        with (
            patch(
                "snowflake.ml.jobs.code_bundle.query_helper.run_query",
                return_value=self._mock_history_rows("TRAIN_PY_1"),
            ),
            self.assertLogs("snowflake.ml.jobs.code_bundle", level="WARNING") as logs,
        ):
            code_bundle.register_job_from_code_bundle(
                "TEST_DB.TEST_SCHEMA.MY_BUNDLE",
                entrypoint="train.py",
                session=self.session,
            )

        self.assertEqual(
            logs.output,
            [
                "WARNING:snowflake.ml.jobs.code_bundle:Code bundles are in Public Preview since 2.6.0."
                " Do not use them in production."
            ],
        )

    def test_trigger_polls_until_job_id_appears(self) -> None:
        with patch(
            "snowflake.ml.jobs.code_bundle.query_helper.run_query",
            side_effect=[
                self._mock_history_rows(None),
                self._mock_history_rows(None),
                self._mock_history_rows("X"),
                [],  # DESCRIBE SERVICE succeeds
            ],
        ) as mock_run_query, patch("snowflake.ml.jobs.code_bundle.time.sleep") as mock_sleep:
            result = code_bundle.register_job_from_code_bundle(
                "TEST_DB.TEST_SCHEMA.MY_BUNDLE",
                entrypoint="train.py",
                session=self.session,
            )

        self.assertEqual(mock_run_query.call_count, 4)
        self.assertEqual(mock_sleep.call_count, 2)
        self.assertEqual(result.id, "X")

    def test_trigger_waits_until_job_service_exists(self) -> None:
        service_missing = sp_exceptions.SnowparkSQLException("Service does not exist", sql_error_code=2003)
        with patch(
            "snowflake.ml.jobs.code_bundle.query_helper.run_query",
            side_effect=[self._mock_history_rows("X"), service_missing, []],
        ) as mock_run_query, patch("snowflake.ml.jobs.code_bundle.time.sleep") as mock_sleep:
            result = code_bundle.register_job_from_code_bundle(
                "TEST_DB.TEST_SCHEMA.MY_BUNDLE",
                entrypoint="train.py",
                session=self.session,
            )

        queries = [called.args[1] for called in mock_run_query.call_args_list]
        self.assertEqual(
            queries,
            [
                "SELECT RUNTIME_NAME FROM TABLE(TEST_DB.INFORMATION_SCHEMA.CODE_BUNDLE_HISTORY(QUERY_ID => ?))",
                "DESCRIBE SERVICE IDENTIFIER(?)",
                "DESCRIBE SERVICE IDENTIFIER(?)",
            ],
        )
        mock_sleep.assert_called_once_with(5)
        self.assertEqual(result.id, "X")

    def test_trigger_returns_job_after_short_wait_if_service_never_appears(self) -> None:
        service_missing = sp_exceptions.SnowparkSQLException("Service does not exist", sql_error_code=2003)
        clock = iter([0.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
        with patch(
            "snowflake.ml.jobs.code_bundle.query_helper.run_query",
            side_effect=[self._mock_history_rows("X")] + [service_missing] * 10,
        ) as mock_run_query, patch("snowflake.ml.jobs.code_bundle.time.sleep") as mock_sleep, patch(
            "snowflake.ml.jobs.code_bundle.time.monotonic", side_effect=lambda: next(clock)
        ):
            result = code_bundle.register_job_from_code_bundle(
                "TEST_DB.TEST_SCHEMA.MY_BUNDLE",
                entrypoint="train.py",
                session=self.session,
            )

        self.assertEqual(result.id, "X")
        describe_calls = [c for c in mock_run_query.call_args_list if c.args[1] == "DESCRIBE SERVICE IDENTIFIER(?)"]
        # Checks every 5 seconds from 0 to 25 seconds; at 30 seconds the wait is over and the job is returned.
        self.assertEqual(len(describe_calls), 6)
        self.assertEqual([c.args[0] for c in mock_sleep.call_args_list], [5] * 5)

    def test_trigger_returns_job_when_service_check_fails_unexpectedly(self) -> None:
        not_authorized = sp_exceptions.SnowparkSQLException("Insufficient privileges", sql_error_code=3001)
        with patch(
            "snowflake.ml.jobs.code_bundle.query_helper.run_query",
            side_effect=[self._mock_history_rows("X"), not_authorized],
        ) as mock_run_query, patch("snowflake.ml.jobs.code_bundle.time.sleep") as mock_sleep:
            result = code_bundle.register_job_from_code_bundle(
                "TEST_DB.TEST_SCHEMA.MY_BUNDLE",
                entrypoint="train.py",
                session=self.session,
            )

        self.assertEqual(result.id, "X")
        self.assertEqual(mock_run_query.call_count, 2)
        mock_sleep.assert_not_called()

    def test_trigger_times_out_if_job_id_never_appears(self) -> None:
        times = iter([0.0, 0.0, 100.0])
        with patch(
            "snowflake.ml.jobs.code_bundle.query_helper.run_query", return_value=self._mock_history_rows(None)
        ), patch("snowflake.ml.jobs.code_bundle.time.sleep"), patch(
            "snowflake.ml.jobs.code_bundle.time.monotonic", side_effect=lambda: next(times)
        ):
            with self.assertRaisesRegex(TimeoutError, "timed out after 5s waiting for the job"):
                code_bundle.register_job_from_code_bundle(
                    "TEST_DB.TEST_SCHEMA.MY_BUNDLE",
                    entrypoint="train.py",
                    session=self.session,
                    timeout=5,
                )

    def test_trigger_rejects_non_positive_timeout(self) -> None:
        with self.assertRaisesRegex(ValueError, "timeout must be greater than 0"):
            code_bundle.register_job_from_code_bundle(
                "TEST_DB.TEST_SCHEMA.MY_BUNDLE",
                entrypoint="train.py",
                session=self.session,
                timeout=0,
            )

        self.session.sql.assert_not_called()

    @patch("snowflake.ml.jobs.code_bundle.time.sleep", autospec=True)
    @patch("snowflake.ml.jobs.code_bundle.query_helper.run_query", autospec=True, return_value=[])
    def test_trigger_raises_execution_error_without_waiting(
        self, mock_run_query: MagicMock, mock_sleep: MagicMock
    ) -> None:
        self.async_job.is_failed.return_value = True
        self.async_job.result.side_effect = connector_errors.ProgrammingError(
            msg="Entrypoint 'missing.py' not found", errno=2, sfqid=self.async_job.query_id
        )

        with self.assertRaisesRegex(sp_exceptions.SnowparkSQLException, "Entrypoint 'missing.py' not found"):
            code_bundle.register_job_from_code_bundle(
                "TEST_DB.TEST_SCHEMA.MY_BUNDLE",
                entrypoint="missing.py",
                session=self.session,
            )

        mock_run_query.assert_called_once()
        mock_sleep.assert_not_called()


class DeleteCodeBundleTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.session = MagicMock()

    def test_delete_code_bundle_drops_by_name(self) -> None:
        with patch("snowflake.ml.jobs.code_bundle.query_helper.run_query") as mock_run_query:
            code_bundle.delete_code_bundle("TEST_DB.TEST_SCHEMA.MY_BUNDLE", session=self.session)

        mock_run_query.assert_called_once_with(
            self.session,
            "DROP CODE BUNDLE IDENTIFIER(?)",
            params=["TEST_DB.TEST_SCHEMA.MY_BUNDLE"],
        )
        self.session.sql.assert_not_called()

    def test_delete_code_bundle_drops_then_removes_payload_stage(self) -> None:
        bundle = code_bundle.MLCodeBundle(
            "TEST_DB.TEST_SCHEMA.MY_BUNDLE",
            session=self.session,
            stage_path="@payload_stage/my_bundle",
        )
        call_order = MagicMock()
        call_order.attach_mock(self.session.sql, "sql")
        with patch("snowflake.ml.jobs.code_bundle.query_helper.run_query") as mock_run_query:
            call_order.attach_mock(mock_run_query, "run_query")
            bundle.delete()

        self.assertEqual(
            [name for name, _, _ in call_order.mock_calls if name in ("run_query", "sql")],
            ["run_query", "sql"],
        )
        self.session.sql.assert_called_once_with("REMOVE @payload_stage/my_bundle/")
        mock_run_query.assert_called_once_with(
            self.session,
            "DROP CODE BUNDLE IDENTIFIER(?)",
            params=["TEST_DB.TEST_SCHEMA.MY_BUNDLE"],
        )

    def test_delete_code_bundle_keeps_payload_when_drop_fails(self) -> None:
        bundle = code_bundle.MLCodeBundle(
            "TEST_DB.TEST_SCHEMA.MY_BUNDLE",
            session=self.session,
            stage_path="@payload_stage/my_bundle",
        )
        with patch(
            "snowflake.ml.jobs.code_bundle.query_helper.run_query",
            side_effect=sp_exceptions.SnowparkSQLException("Insufficient privileges to operate on code bundle"),
        ):
            with self.assertRaises(sp_exceptions.SnowparkSQLException):
                bundle.delete()

        self.session.sql.assert_not_called()

    def test_delete_code_bundle_binds_quoted_fqn(self) -> None:
        fqn = 'TEST_DB.TEST_SCHEMA."; DROP TABLE foo; --"'
        with patch("snowflake.ml.jobs.code_bundle.query_helper.run_query") as mock_run_query:
            code_bundle.delete_code_bundle(fqn, session=self.session)

        mock_run_query.assert_called_once_with(
            self.session,
            "DROP CODE BUNDLE IDENTIFIER(?)",
            params=[fqn],
        )


if __name__ == "__main__":
    absltest.main()
