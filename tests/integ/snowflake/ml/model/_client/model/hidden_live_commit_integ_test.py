import re
import uuid
from typing import Optional

from absl.testing import absltest

from snowflake.ml.registry import registry
from snowflake.snowpark import exceptions as snowpark_exceptions
from tests.integ.snowflake.ml.test_utils import (
    common_test_base,
    db_manager,
    model_factory,
)

_PENDING_MODEL_NAME_PATTERN = re.compile(r"PENDING_[0-9A-F]{8}_MODEL")
_LIVE_VERSION_NAME_PATTERN = re.compile(r"LIVE_[0-9A-F]{8}_VERSION")


class HiddenLiveCommitIntegTest(common_test_base.CommonTestBase):
    def setUp(self) -> None:
        super().setUp()
        self._run_id = uuid.uuid4().hex
        self._db_manager = db_manager.DBManager(self.session)
        self._test_schema = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "schema"
        ).upper()
        self._db_manager.create_schema(self._test_schema)
        self._db_manager.use_schema(self._test_schema)
        self._registry = registry.Registry(self.session)

    def tearDown(self) -> None:
        self._db_manager.drop_schema(self._test_schema)
        super().tearDown()

    def test_log_model_alter_path_uses_add_live_version_and_commit_rename(self) -> None:
        model, test_features, _ = model_factory.ModelFactory.prepare_sklearn_model()
        model_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "hidden_live_model"
        ).upper()

        self._registry.log_model(
            model=model,
            model_name=model_name,
            version_name="V1",
            sample_input_data=test_features,
        )

        with self.session.query_history() as query_history:
            model_version = self._registry.log_model(
                model=model,
                model_name=model_name,
                version_name="V2",
                sample_input_data=test_features,
            )

        executed_sql = [query.sql_text.upper() for query in query_history.queries]
        add_live_version_sql = [sql for sql in executed_sql if "ADD LIVE VERSION" in sql]
        commit_rename_sql = [sql for sql in executed_sql if "COMMIT VERSION" in sql and "RENAME VERSION TO" in sql]

        if not add_live_version_sql or not commit_rename_sql:
            self.skipTest("Hidden live commit SQL is not supported on this account; log_model fell back to FROM STAGE.")

        self.assertTrue(
            any("RENAME VERSION TO V2" in sql for sql in commit_rename_sql),
            f"Expected COMMIT ... RENAME VERSION TO V2, got: {commit_rename_sql}",
        )
        self.assertEqual(model_version.version_name, "V2")
        logged_model = self._registry.get_model(model_name=model_name)
        self.assertEqual([version.version_name for version in logged_model.versions()], ["V1", "V2"])

    def test_failed_create_keeps_pending_model_and_live_version_hidden(self) -> None:
        """Failed first-version log must not surface PENDING_* models or LIVE_* versions in SHOW SQL."""
        model, test_features, _ = model_factory.ModelFactory.prepare_sklearn_model()
        model_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "hidden_live_fail_create"
        ).upper()

        executed_sql = self._log_model_expecting_failure(
            model=model,
            model_name=model_name,
            version_name="V1",
            sample_input_data=test_features,
        )
        pending_model_name, live_version_name = self._require_hidden_live_create_sql(executed_sql)
        self._assert_commit_version_attempted(executed_sql)

        self._assert_not_listed_in_show_models([model_name, pending_model_name])
        with self.assertRaises(snowpark_exceptions.SnowparkSQLException):
            self._show_version_names(model_name)
        self._assert_show_versions_hides_live_checkout(
            model_name=pending_model_name,
            live_version_name=live_version_name,
            model_must_exist=False,
        )

    def test_failed_add_version_keeps_live_version_hidden(self) -> None:
        """Failed second-version log must not surface LIVE_* versions on the committed model."""
        model, test_features, _ = model_factory.ModelFactory.prepare_sklearn_model()
        model_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "hidden_live_fail_alter"
        ).upper()

        self._registry.log_model(
            model=model,
            model_name=model_name,
            version_name="V1",
            sample_input_data=test_features,
        )

        executed_sql = self._log_model_expecting_failure(
            model=model,
            model_name=model_name,
            version_name="V2",
            sample_input_data=test_features,
        )
        live_version_name = self._require_hidden_live_alter_sql(executed_sql)
        self._assert_commit_version_attempted(executed_sql)

        self.assertIn(model_name, self._show_model_names())
        self._assert_show_versions_hides_live_checkout(
            model_name=model_name,
            live_version_name=live_version_name,
            model_must_exist=True,
            expected_visible_versions=["V1"],
        )

    def _log_model_expecting_failure(
        self,
        *,
        model: object,
        model_name: str,
        version_name: str,
        sample_input_data: object,
    ) -> list[str]:
        # The dependency is a valid client-side spec, so packaging succeeds and the version is
        # rejected only when Snowflake resolves dependencies at commit time. `include_error` is
        # required because the failing statement is otherwise left out of the query history.
        with self.session.query_history(include_error=True) as query_history:
            with self.assertRaises(snowpark_exceptions.SnowparkSQLException):
                self._registry.log_model(
                    model=model,
                    model_name=model_name,
                    version_name=version_name,
                    sample_input_data=sample_input_data,
                    conda_dependencies=["A-NON-EXISTING-PACKAGE==0.1.0"],
                )
        return [query.sql_text.upper() for query in query_history.queries]

    def _assert_commit_version_attempted(self, executed_sql: list[str]) -> None:
        commit_version_sql = [sql for sql in executed_sql if "COMMIT VERSION" in sql]
        self.assertNotEmpty(
            commit_version_sql,
            f"Expected COMMIT VERSION to run and fail, got: {executed_sql}",
        )

    def _require_hidden_live_create_sql(self, executed_sql: list[str]) -> tuple[str, str]:
        create_live_sql = [sql for sql in executed_sql if "WITH LIVE VERSION" in sql]
        if not create_live_sql:
            self.skipTest("Hidden live commit SQL is not supported on this account; log_model fell back to FROM STAGE.")
        pending_model_name = _PENDING_MODEL_NAME_PATTERN.search(create_live_sql[0])
        live_version_name = _LIVE_VERSION_NAME_PATTERN.search(create_live_sql[0])
        if pending_model_name is None or live_version_name is None:
            self.fail(f"Could not parse pending model / live version from: {create_live_sql}")
        return pending_model_name.group(0), live_version_name.group(0)

    def _require_hidden_live_alter_sql(self, executed_sql: list[str]) -> str:
        add_live_sql = [sql for sql in executed_sql if "ADD LIVE VERSION" in sql]
        if not add_live_sql:
            self.skipTest("Hidden live commit SQL is not supported on this account; log_model fell back to FROM STAGE.")
        live_version_name = _LIVE_VERSION_NAME_PATTERN.search(add_live_sql[0])
        if live_version_name is None:
            self.fail(f"Could not parse live version from: {add_live_sql}")
        return live_version_name.group(0)

    def _show_model_names(self) -> list[str]:
        rows = self.session.sql(f"SHOW MODELS IN SCHEMA {self._test_schema}").collect()
        return [str(row["name"]).upper() for row in rows]

    def _show_version_names(self, model_name: str) -> list[str]:
        rows = self.session.sql(f"SHOW VERSIONS IN MODEL {self._test_schema}.{model_name}").collect()
        return [str(row["name"]).upper() for row in rows]

    def _assert_not_listed_in_show_models(self, names: list[str]) -> None:
        shown_names = self._show_model_names()
        for name in names:
            self.assertNotIn(
                name,
                shown_names,
                f"Expected {name} to stay hidden from SHOW MODELS, got: {shown_names}",
            )
        pending_shown = [name for name in shown_names if _PENDING_MODEL_NAME_PATTERN.fullmatch(name)]
        self.assertEmpty(pending_shown, f"Pending models listed by SHOW MODELS: {pending_shown}")

    def _assert_show_versions_hides_live_checkout(
        self,
        *,
        model_name: str,
        live_version_name: str,
        model_must_exist: bool,
        expected_visible_versions: Optional[list[str]] = None,
    ) -> None:
        try:
            shown_versions = self._show_version_names(model_name)
        except snowpark_exceptions.SnowparkSQLException:
            if model_must_exist:
                raise
            return

        self.assertNotIn(
            live_version_name,
            shown_versions,
            (
                f"Expected {live_version_name} to stay hidden from SHOW VERSIONS IN MODEL {model_name}, "
                f"got: {shown_versions}"
            ),
        )
        live_shown = [name for name in shown_versions if _LIVE_VERSION_NAME_PATTERN.fullmatch(name)]
        self.assertEmpty(
            live_shown,
            f"Live versions listed by SHOW VERSIONS IN MODEL {model_name}: {live_shown}",
        )
        if expected_visible_versions is not None:
            self.assertEqual(shown_versions, [version.upper() for version in expected_visible_versions])


if __name__ == "__main__":
    absltest.main()
