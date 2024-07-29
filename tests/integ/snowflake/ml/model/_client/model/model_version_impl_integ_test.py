import glob
import os
import tempfile
import uuid

import numpy as np
from absl.testing import absltest, parameterized
from sklearn import svm

from snowflake.ml.model import ExportMode
from snowflake.ml.registry import registry
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.test_utils import db_manager, model_factory

MODEL_NAME = "TEST_MODEL"
VERSION_NAME = "V1"


class TestModelVersionImplInteg(parameterized.TestCase):
    @classmethod
    def setUpClass(self) -> None:
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

        self.model, self.test_features, _ = model_factory.ModelFactory.prepare_sklearn_model()
        self._mv = self.registry.log_model(
            model=self.model,
            model_name=MODEL_NAME,
            version_name=VERSION_NAME,
            sample_input_data=self.test_features,
        )

    @classmethod
    def tearDownClass(self) -> None:
        self._db_manager.drop_database(self._test_db)
        self._session.close()

    def test_description(self) -> None:
        description = "test description"
        self._mv.description = description
        self.assertEqual(self._mv.description, description)

    def test_metrics(self) -> None:
        self._mv.set_metric("a", 1)
        expected_metrics = {"a": 2, "b": 1.0, "c": True}
        for k, v in expected_metrics.items():
            self._mv.set_metric(k, v)

        self.assertEqual(self._mv.get_metric("a"), expected_metrics["a"])
        self.assertDictEqual(self._mv.show_metrics(), expected_metrics)

        expected_metrics.pop("b")
        self._mv.delete_metric("b")
        self.assertDictEqual(self._mv.show_metrics(), expected_metrics)
        with self.assertRaises(KeyError):
            self._mv.get_metric("b")

    def test_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self._mv.export(tmpdir)
            self.assertLen(list(glob.iglob(os.path.join(tmpdir, "**", "*"), recursive=True)), 14)

        with tempfile.TemporaryDirectory() as tmpdir:
            self._mv.export(tmpdir, export_mode=ExportMode.FULL)
            self.assertLen(list(glob.iglob(os.path.join(tmpdir, "**", "*"), recursive=True)), 26)

    def test_load(self) -> None:
        loaded_model = self._mv.load()
        assert isinstance(loaded_model, svm.SVC)
        np.testing.assert_allclose(loaded_model.predict(self.test_features), self.model.predict(self.test_features))


if __name__ == "__main__":
    absltest.main()
