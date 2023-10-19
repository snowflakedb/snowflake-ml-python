import uuid
from typing import Callable, Tuple

from absl.testing import absltest

from snowflake.ml.registry import model_registry
from snowflake.snowpark import session
from tests.integ.snowflake.ml.test_utils import common_test_base, db_manager


class ModelRegistryCompatTest(common_test_base.CommonTestBase):
    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        super().setUp()
        self.run_id = uuid.uuid4().hex
        self._db_manager = db_manager.DBManager(self.session)
        self.current_db = self.session.get_current_database()
        self.current_schema = self.session.get_current_schema()

    def _prepare_registry_fn_factory(
        self,
    ) -> Tuple[Callable[[session.Session, str], None], Tuple[str]]:
        self.registry_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self.run_id, "registry_db")

        def prepare_registry(session: session.Session, registry_name: str) -> None:
            from snowflake.connector.errors import ProgrammingError
            from snowflake.ml.registry import model_registry

            try:
                model_registry.create_model_registry(session=session, database_name=registry_name)
            except ProgrammingError:
                # Previous versions of library will call use even in the sproc env, which is not allowed.
                # This is to suppress the error
                pass

        return prepare_registry, (self.registry_name,)

    # Starting from 1.0.1 as we had a breaking change at that time.
    # TODO: mypy is giving out error `Cannot infer type argument 1 of "compatibility_test" of "CommonTestBase"  [misc]`
    # Need to figure out the reason and remove ignore
    @common_test_base.CommonTestBase.compatibility_test(
        prepare_fn_factory=_prepare_registry_fn_factory, version_range=">=1.0.1,<=1.0.9"  # type: ignore[misc]
    )
    def test_open_registry_compat_v0(self) -> None:
        try:
            with self.assertRaisesRegex(
                RuntimeError, r"Registry schema version \([0-9]+\) is ahead of deployed schema \(0\)."
            ):
                model_registry.ModelRegistry(
                    session=self.session, database_name=self.registry_name, create_if_not_exists=False
                )
            model_registry.ModelRegistry(
                session=self.session, database_name=self.registry_name, create_if_not_exists=True
            )
        finally:
            self._db_manager.drop_database(self.registry_name, if_exists=True)
            self.session.use_database(self.current_db)
            self.session.use_schema(self.current_schema)


if __name__ == "__main__":
    absltest.main()
