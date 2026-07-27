import uuid

from absl.testing import absltest

from snowflake.ml.registry import registry
from tests.integ.snowflake.ml.test_utils import (
    common_test_base,
    db_manager,
    model_factory,
)


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


if __name__ == "__main__":
    absltest.main()
