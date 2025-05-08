import os
import tempfile

import numpy as np
from absl.testing import absltest, parameterized
from packaging import version

from snowflake.ml import jobs
from snowflake.ml._internal import env
from snowflake.ml._internal.utils import identifier
from snowflake.ml.model.model_signature import DataType, FeatureSpec, ModelSignature
from snowflake.ml.registry import Registry
from snowflake.ml.utils import sql_client
from snowflake.snowpark import exceptions as sp_exceptions
from snowflake.snowpark.context import get_active_session
from tests.integ.snowflake.ml.jobs import test_constants
from tests.integ.snowflake.ml.test_utils import db_manager, test_env_utils


@absltest.skipIf(
    (region := test_env_utils.get_current_snowflake_region()) is None
    or region["cloud"] not in test_constants._SUPPORTED_CLOUDS,
    "Test only for SPCS supported clouds",
)
class AccessTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.session = test_env_utils.get_available_session()
        cls.dbm = db_manager.DBManager(cls.session)
        cls.dbm.cleanup_schemas(prefix=test_constants._TEST_SCHEMA, expire_days=1)
        cls.db = cls.session.get_current_database()
        cls.schema = cls.dbm.create_random_schema(prefix=test_constants._TEST_SCHEMA)
        try:
            cls.compute_pool = cls.dbm.create_compute_pool(
                test_constants._TEST_COMPUTE_POOL, sql_client.CreationMode(if_not_exists=True), max_nodes=5
            )
        except sp_exceptions.SnowparkSQLException:
            if not cls.dbm.show_compute_pools(test_constants._TEST_COMPUTE_POOL).count() > 0:
                raise cls.failureException(
                    f"Compute pool {test_constants._TEST_COMPUTE_POOL} not available and could not be created"
                )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.dbm.drop_schema(cls.schema, if_exists=True)
        super().tearDownClass()

    @absltest.skipIf(
        version.Version(env.PYTHON_VERSION) >= version.Version("3.11"),
        "Decorator test only works for Python 3.10 and below due to pickle compatibility",
    )
    def test_save_table(self) -> None:
        table = "TEST_FEATURE"

        @jobs.remote(self.compute_pool, stage_name="payload_stage", session=self.session)
        def save_data() -> None:
            session = get_active_session()
            df = session.create_dataframe([[1, 2], [3, 4]], schema=["feature_1", "feature_2"])
            df.write.mode("overwrite").save_as_table(table)

        job = save_data()
        self.assertEqual(job.wait(), "DONE", job.get_logs())
        result = self.session.table(table).collect()
        self.assertEqual(len(result), 2)
        self.assertTrue(
            all(
                "FEATURE_1" in row
                and "FEATURE_2" in row
                and isinstance(row["FEATURE_1"], int)
                and isinstance(row["FEATURE_2"], int)
                for row in result
            )
        )

    @absltest.skipIf(
        version.Version(env.PYTHON_VERSION) >= version.Version("3.11"),
        "Decorator test only works for Python 3.10 and below due to pickle compatibility",
    )
    def test_save_model_registry(self):
        model_name = "test_model"
        version_name = "v1"

        @jobs.remote(self.compute_pool, stage_name="payload_stage", session=self.session)
        def train_model() -> None:
            import xgboost as xgb

            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
            X_dummy = np.random.rand(2, 2)
            y_dummy = np.array([0, 1])
            model.fit(X_dummy, y_dummy)
            # model register
            session = get_active_session()
            reg = Registry(session=session)
            input_features = [
                FeatureSpec(dtype=DataType.FLOAT, name="feature1"),
                FeatureSpec(dtype=DataType.FLOAT, name="feature2"),
            ]
            output_features = [
                FeatureSpec(dtype=DataType.INT64, name="target"),
            ]

            sig = ModelSignature(inputs=input_features, outputs=output_features)
            reg.log_model(
                model,
                model_name=model_name,
                version_name=version_name,
                conda_dependencies=["scikit-learn"],
                comment="test xgboost model",
                signatures={"predict": sig},
            )

        job = train_model()
        self.assertEqual(job.wait(), "DONE", job.get_logs())
        reg = Registry(session=self.session, database_name=self.db, schema_name=self.schema)
        model = reg.get_model(model_name)
        self.assertIsNotNone(model)
        self.assertEqual(model.name, identifier.resolve_identifier(model_name))
        model_version = model.version(version_name)
        self.assertEqual(model_version.version_name, identifier.resolve_identifier(version_name))

    @absltest.skipIf(
        version.Version(env.PYTHON_VERSION) >= version.Version("3.11"),
        "Decorator test only works for Python 3.10 and below due to pickle compatibility",
    )
    def test_save_stage(self):
        TEST_STAGE = "headless_test_stage"
        TEST_FILE = "test.csv"

        @jobs.remote(self.compute_pool, stage_name="payload_stage", session=self.session)
        def save_stage() -> None:
            session = get_active_session()
            session.sql(f"create or replace stage {TEST_STAGE}").collect()
            df = session.create_dataframe(
                [["John", "Berry"], ["Rick", "Berry"], ["Anthony", "Davis"]], schema=["FIRST_NAME", "LAST_NAME"]
            )
            pdf = df.to_pandas()
            pdf.to_csv(TEST_FILE, index=False)
            session.file.put(TEST_FILE, f"@{TEST_STAGE}")
            if os.path.exists(TEST_FILE):
                os.remove(TEST_FILE)

        job = save_stage()
        self.assertIsNotNone(job)
        self.assertEqual(job.wait(), "DONE", job.get_logs())
        with tempfile.TemporaryDirectory() as tmpdir:
            get_result = self.session.file.get(f"@{TEST_STAGE}", f"{tmpdir}")
            self.assertEqual(len(get_result), 1)
        self.session.sql(f"DROP STAGE IF EXISTS {TEST_STAGE}").collect()


if __name__ == "__main__":
    absltest.main()
