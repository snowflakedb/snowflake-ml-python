import datetime
import uuid

import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model import custom_model, model_signature
from snowflake.ml.registry import registry
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session, exceptions
from tests.integ.snowflake.ml.test_utils import dataframe_utils, db_manager

MODEL_NAME = "TEST_MODEL"
TIMESTAMP_MODEL_NAME = "TEST_TIMESTAMP_MODEL"
VERSION_NAME = "V1"


class DemoModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"]})


class TestInputValidationInteg(parameterized.TestCase):
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

        lm = DemoModel(custom_model.ModelContext())

        self._mv = self.registry.log_model(
            model=lm,
            model_name=MODEL_NAME,
            version_name=VERSION_NAME,
            signatures={
                "predict": model_signature.ModelSignature(
                    inputs=[
                        model_signature.FeatureSpec(name="c1", dtype=model_signature.DataType.INT8),
                        model_signature.FeatureSpec(name="c2", dtype=model_signature.DataType.INT8),
                        model_signature.FeatureSpec(name="c3", dtype=model_signature.DataType.INT8),
                    ],
                    outputs=[
                        model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.INT8),
                    ],
                )
            },
        )

        self._mv_timestamp = self.registry.log_model(
            model=lm,
            model_name=TIMESTAMP_MODEL_NAME,
            version_name=VERSION_NAME,
            signatures={
                "predict": model_signature.ModelSignature(
                    inputs=[
                        model_signature.FeatureSpec(name="c1", dtype=model_signature.DataType.INT8),
                        model_signature.FeatureSpec(name="c2", dtype=model_signature.DataType.INT8),
                        model_signature.FeatureSpec(name="c3", dtype=model_signature.DataType.INT8),
                        model_signature.FeatureSpec(name="c4", dtype=model_signature.DataType.TIMESTAMP_NTZ),
                    ],
                    outputs=[
                        model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.INT8),
                    ],
                )
            },
        )

    @classmethod
    def tearDownClass(self) -> None:
        self._db_manager.drop_database(self._test_db)
        self._session.close()

    def test_default_non_strict(self) -> None:
        pd.testing.assert_frame_equal(
            self._mv.run(pd.DataFrame([[1, 2, 3], [4, 2, 5]])),
            pd.DataFrame([1, 4], columns=["output"]),
            check_dtype=False,
        )

        pd.testing.assert_frame_equal(
            self._mv.run(pd.DataFrame([[1, 2, 3], [257, 2, 5]])),
            pd.DataFrame([1, 1], columns=["output"]),
            check_dtype=False,
        )

        sp_df = self._session.create_dataframe([[1, 2, 3], [4, 2, 5]], schema=['"c1"', '"c2"', '"c3"'])
        y_df_expected = pd.DataFrame([[1, 2, 3, 1], [4, 2, 5, 4]], columns=["c1", "c2", "c3", "output"])
        dataframe_utils.check_sp_df_res(self._mv.run(sp_df), y_df_expected, check_dtype=False)

        sp_df = self._session.create_dataframe([[None, 2, 3], [257, 2, 5]], schema=['"c1"', '"c2"', '"c3"'])
        y_df_expected = pd.DataFrame([[None, 2, 3, None], [257, 2, 5, 1]], columns=["c1", "c2", "c3", "output"])
        dataframe_utils.check_sp_df_res(self._mv.run(sp_df), y_df_expected, check_dtype=False)

        sp_df = self._session.create_dataframe([[1, 2, 3], [257, 2, 5]], schema=['"c1"', '"c2"', '"c3"'])
        with self.assertRaisesRegex(exceptions.SnowparkSQLException, "Python Interpreter Error"):
            self._mv.run(sp_df).collect()

    def test_strict(self) -> None:
        pd.testing.assert_frame_equal(
            self._mv.run(pd.DataFrame([[1, 2, 3], [4, 2, 5]]), strict_input_validation=True),
            pd.DataFrame([1, 4], columns=["output"]),
            check_dtype=False,
        )

        with self.assertRaisesRegex(ValueError, "Data Validation Error"):
            self._mv.run(pd.DataFrame([[1, 2, 4], [257, 2, 5]]), strict_input_validation=True)

        sp_df = self._session.create_dataframe([[1, 2, 3], [4, 2, 5]], schema=['"c1"', '"c2"', '"c3"'])
        y_df_expected = pd.DataFrame([[1, 2, 3, 1], [4, 2, 5, 4]], columns=["c1", "c2", "c3", "output"])
        dataframe_utils.check_sp_df_res(
            self._mv.run(sp_df, strict_input_validation=True), y_df_expected, check_dtype=False
        )

        sp_df = self._session.create_dataframe([[1, 2, 3], [257, 2, 5]], schema=['"c1"', '"c2"', '"c3"'])
        y_df_expected = pd.DataFrame([[1, 2, 3, 1], [257, 2, 5, 257]], columns=["c1", "c2", "c3", "output"])
        with self.assertRaisesRegex(ValueError, "Data Validation Error"):
            self._mv.run(sp_df, strict_input_validation=True)

    def test_timestamps(self) -> None:
        d1 = datetime.datetime(year=2024, month=6, day=21, minute=1, second=1)
        d2 = datetime.datetime(year=2024, month=7, day=11, minute=1, second=1)

        pd.testing.assert_frame_equal(
            self._mv_timestamp.run(pd.DataFrame([[1, 2, 3, d1], [4, 2, 5, d2]]), strict_input_validation=True),
            pd.DataFrame([1, 4], columns=["output"]),
            check_dtype=False,
        )

        with self.assertRaisesRegex(ValueError, "Data Validation Error"):
            self._mv_timestamp.run(pd.DataFrame([[1, 2, 4, d1], [257, 2, 5, d2]]), strict_input_validation=True)

        sp_df = self._session.create_dataframe([[1, 2, 3, d1], [4, 2, 5, d2]], schema=['"c1"', '"c2"', '"c3"', '"c4"'])
        y_df_expected = pd.DataFrame([[1, 2, 3, d1, 1], [4, 2, 5, d2, 4]], columns=["c1", "c2", "c3", "c4", "output"])
        dataframe_utils.check_sp_df_res(
            self._mv_timestamp.run(sp_df, strict_input_validation=True), y_df_expected, check_dtype=False
        )

        sp_df = self._session.create_dataframe(
            [[1, 2, 3, d1], [257, 2, 5, d2]], schema=['"c1"', '"c2"', '"c3"', '"c4"']
        )
        y_df_expected = pd.DataFrame(
            [[1, 2, 3, d1, 1], [257, 2, 5, d2, 257]], columns=["c1", "c2", "c3", "c4", "output"]
        )
        with self.assertRaisesRegex(ValueError, "Data Validation Error"):
            self._mv_timestamp.run(sp_df, strict_input_validation=True)


if __name__ == "__main__":
    absltest.main()
