import uuid

import numpy as np
from absl.testing import absltest
from sklearn import datasets, linear_model

from snowflake.ml import registry
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.test_utils import db_manager


class RandomVersionNameTest(absltest.TestCase):
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

        self._db_manager = db_manager.DBManager(self._session)
        self._db_manager.create_database(self._test_db)
        self._db_manager.create_schema(self._test_schema)
        self._db_manager.cleanup_databases(expire_hours=6)
        self.registry = registry.Registry(self._session)

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._test_db)
        self._session.close()

    def test_random_version_name(self) -> None:
        iris_X, iris_y = datasets.load_iris(return_X_y=True)
        # LogisticRegression is for classfication task, such as iris
        regr = linear_model.LogisticRegression()
        regr.fit(iris_X, iris_y)
        name = f"model_{self._run_id}"
        mv = self.registry.log_model(regr, model_name=name, sample_input_data=iris_X)
        np.testing.assert_allclose(
            mv.run(iris_X, function_name="predict")["output_feature_0"].values, regr.predict(iris_X)
        )

        self.registry._model_manager._hrid_generator.hrid_to_id(mv.version_name.lower())


if __name__ == "__main__":
    absltest.main()
