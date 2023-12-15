import inspect
import unittest
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from absl.testing import absltest
from packaging import version

from snowflake.ml.model import type_hints as model_types
from snowflake.ml.registry import registry
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.test_utils import db_manager, test_env_utils


class RegistryModelTestBase(absltest.TestCase):
    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        login_options = connection_params.SnowflakeLoginOptions()

        self._run_id = uuid.uuid4().hex
        self._test_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self._run_id, "db").upper()
        self._test_schema = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "schema"
        ).upper()

        self._session = Session.builder.configs(
            {
                **login_options,
                **{"database": self._test_db, "schema": self._test_schema},
            }
        ).create()

        current_sf_version = test_env_utils.get_current_snowflake_version(self._session)

        if current_sf_version < version.parse("8.0.0"):
            raise unittest.SkipTest("This test requires Snowflake Version 8.0.0 or higher.")

        self._db_manager = db_manager.DBManager(self._session)
        self._db_manager.create_database(self._test_db)
        self._db_manager.create_schema(self._test_schema)
        self._db_manager.cleanup_databases(expire_hours=6)
        self.registry = registry.Registry(self._session)

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._test_db)
        self._session.close()

    def _test_registry_model(
        self,
        model: model_types.SupportedModelType,
        prediction_assert_fns: Dict[str, Tuple[Any, Callable[[Any], Any]]],
        sample_input: Optional[model_types.SupportedDataType] = None,
        additional_dependencies: Optional[List[str]] = None,
        options: Optional[model_types.ModelSaveOption] = None,
    ) -> None:
        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self._session, "snowflake-snowpark-python")
        ]
        if additional_dependencies:
            conda_dependencies.extend(additional_dependencies)

        # Get the name of the caller as the model name
        name = f"model_{inspect.stack()[1].function}"
        version = f"ver_{self._run_id}"
        mv = self.registry.log_model(
            model=model,
            model_name=name,
            version_name=version,
            sample_input_data=sample_input,
            conda_dependencies=conda_dependencies,
            options=options,
        )

        for target_method, (test_input, check_func) in prediction_assert_fns.items():
            res = mv.run(test_input, method_name=target_method)

            check_func(res)

        self.registry.delete_model(model_name=name)

        self.assertNotIn(mv.model_name, [m.name for m in self.registry.list_models()])


if __name__ == "__main__":
    absltest.main()
