import numpy as np
import pandas as pd
import xgboost as xgb
from absl.testing import absltest

from snowflake.ml.model._signatures import core, dmatrix_handler
from snowflake.ml.test_utils import exception_utils


class XGBoostDMatrixHandlerTest(absltest.TestCase):
    def test_validate_xgboost_DMatrix(self) -> None:
        data = xgb.DMatrix(np.empty((0, 10)))
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Empty data is found."
        ):
            dmatrix_handler.XGBoostDMatrixHandler.validate(data)

    def test_truncate_xgboost_DMatrix(self) -> None:
        SIG_INFER_ROWS_COUNT_LIMIT = 100
        data = xgb.DMatrix(np.ones((SIG_INFER_ROWS_COUNT_LIMIT + 1, 10)))
        truncated_data = dmatrix_handler.XGBoostDMatrixHandler.truncate(data, 10)
        self.assertEqual(truncated_data.num_row(), 10)

        data = xgb.DMatrix(np.ones((SIG_INFER_ROWS_COUNT_LIMIT - 1, 10)))
        truncated_data = dmatrix_handler.XGBoostDMatrixHandler.truncate(data, 10)
        self.assertEqual(truncated_data.num_row(), 10)

    def test_infer_signature_xgboost_DMatrix(self) -> None:
        data = xgb.DMatrix(np.ones((10, 2)), feature_names=["f1", "f2"], feature_types=["float", "int"])
        self.assertListEqual(
            dmatrix_handler.XGBoostDMatrixHandler.infer_signature(data, role="input"),
            [
                core.FeatureSpec(dtype=core.DataType.DOUBLE, name="f1"),
                core.FeatureSpec(dtype=core.DataType.INT64, name="f2"),
            ],
        )

    def test_convert_to_df_xgboost_DMatrix(self) -> None:
        np_array = np.ones((10, 2))
        data = xgb.DMatrix(np_array, feature_names=["f1", "f2"])
        df = dmatrix_handler.XGBoostDMatrixHandler.convert_to_df(data)
        expected_df = pd.DataFrame(np_array, columns=["f1", "f2"])

        pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)

    def test_convert_from_df_xgboost_DMatrix(self) -> None:
        df = pd.DataFrame(np.ones((10, 2)), columns=["f1", "f2"])
        data = dmatrix_handler.XGBoostDMatrixHandler.convert_from_df(df)
        self.assertEqual(data.num_row(), 10)
        self.assertEqual(data.num_col(), 2)


if __name__ == "__main__":
    absltest.main()
