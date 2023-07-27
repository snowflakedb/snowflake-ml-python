import numpy as np
import pandas as pd
from absl.testing import absltest

from snowflake.ml.model._signatures import core, numpy_handler


class NumpyArrayHandlerTest(absltest.TestCase):
    def test_validate_np_ndarray(self) -> None:
        arr = np.array([])
        with self.assertRaisesRegex(ValueError, "Empty data is found."):
            numpy_handler.NumpyArrayHandler.validate(arr)

        arr = np.array(1)
        with self.assertRaisesRegex(ValueError, "Scalar data is found."):
            numpy_handler.NumpyArrayHandler.validate(arr)

    def test_trunc_np_ndarray(self) -> None:
        arr = np.array([1] * (numpy_handler.NumpyArrayHandler.SIG_INFER_ROWS_COUNT_LIMIT + 1))

        np.testing.assert_equal(
            np.array([1] * (numpy_handler.NumpyArrayHandler.SIG_INFER_ROWS_COUNT_LIMIT)),
            numpy_handler.NumpyArrayHandler.truncate(arr),
        )

        arr = np.array([1] * (numpy_handler.NumpyArrayHandler.SIG_INFER_ROWS_COUNT_LIMIT - 1))

        np.testing.assert_equal(
            arr,
            numpy_handler.NumpyArrayHandler.truncate(arr),
        )

    def test_infer_schema_np_ndarray(self) -> None:
        arr = np.array([1, 2, 3, 4])
        self.assertListEqual(
            numpy_handler.NumpyArrayHandler.infer_signature(arr, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.INT64)],
        )

        arr = np.array([[1, 2], [3, 4]])
        self.assertListEqual(
            numpy_handler.NumpyArrayHandler.infer_signature(arr, role="input"),
            [
                core.FeatureSpec("input_feature_0", core.DataType.INT64),
                core.FeatureSpec("input_feature_1", core.DataType.INT64),
            ],
        )

        arr = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
        self.assertListEqual(
            numpy_handler.NumpyArrayHandler.infer_signature(arr, role="input"),
            [
                core.FeatureSpec("input_feature_0", core.DataType.INT64, shape=(2,)),
                core.FeatureSpec("input_feature_1", core.DataType.INT64, shape=(2,)),
            ],
        )

        arr = np.array([1, 2, 3, 4])
        self.assertListEqual(
            numpy_handler.NumpyArrayHandler.infer_signature(arr, role="output"),
            [core.FeatureSpec("output_feature_0", core.DataType.INT64)],
        )

        arr = np.array([[1, 2], [3, 4]])
        self.assertListEqual(
            numpy_handler.NumpyArrayHandler.infer_signature(arr, role="output"),
            [
                core.FeatureSpec("output_feature_0", core.DataType.INT64),
                core.FeatureSpec("output_feature_1", core.DataType.INT64),
            ],
        )

        arr = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
        self.assertListEqual(
            numpy_handler.NumpyArrayHandler.infer_signature(arr, role="output"),
            [
                core.FeatureSpec("output_feature_0", core.DataType.INT64, shape=(2,)),
                core.FeatureSpec("output_feature_1", core.DataType.INT64, shape=(2,)),
            ],
        )

    def test_convert_to_df_numpy_array(self) -> None:
        arr1 = np.array([1, 2, 3, 4])
        pd.testing.assert_frame_equal(
            numpy_handler.NumpyArrayHandler.convert_to_df(arr1),
            pd.DataFrame([1, 2, 3, 4]),
        )

        arr2 = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        pd.testing.assert_frame_equal(
            numpy_handler.NumpyArrayHandler.convert_to_df(arr2),
            pd.DataFrame([[1, 1], [2, 2], [3, 3], [4, 4]]),
        )

        arr3 = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
        pd.testing.assert_frame_equal(
            numpy_handler.NumpyArrayHandler.convert_to_df(arr3),
            pd.DataFrame(data={0: [np.array([1, 1]), np.array([3, 3])], 1: [np.array([2, 2]), np.array([4, 4])]}),
        )


class SeqOfNumpyArrayHandlerTest(absltest.TestCase):
    def test_validate_list_of_numpy_array(self) -> None:
        lt8 = [pd.DataFrame([1]), pd.DataFrame([2, 3])]
        self.assertFalse(numpy_handler.SeqOfNumpyArrayHandler.can_handle(lt8))

    def test_trunc_np_ndarray(self) -> None:
        arrs = [np.array([1] * (numpy_handler.SeqOfNumpyArrayHandler.SIG_INFER_ROWS_COUNT_LIMIT + 1))] * 2

        for arr in numpy_handler.SeqOfNumpyArrayHandler.truncate(arrs):
            np.testing.assert_equal(
                np.array([1] * (numpy_handler.SeqOfNumpyArrayHandler.SIG_INFER_ROWS_COUNT_LIMIT)), arr
            )

        arrs = [
            np.array([1]),
            np.array([1] * (numpy_handler.SeqOfNumpyArrayHandler.SIG_INFER_ROWS_COUNT_LIMIT - 1)),
        ]

        for arr in numpy_handler.SeqOfNumpyArrayHandler.truncate(arrs):
            np.testing.assert_equal(np.array([1]), arr)

    def test_infer_signature_list_of_numpy_array(self) -> None:
        arr = np.array([1, 2, 3, 4])
        lt = [arr, arr]
        self.assertListEqual(
            numpy_handler.SeqOfNumpyArrayHandler.infer_signature(lt, role="input"),
            [
                core.FeatureSpec("input_feature_0", core.DataType.INT64),
                core.FeatureSpec("input_feature_1", core.DataType.INT64),
            ],
        )

        arr = np.array([[1, 2], [3, 4]])
        lt = [arr, arr]
        self.assertListEqual(
            numpy_handler.SeqOfNumpyArrayHandler.infer_signature(lt, role="input"),
            [
                core.FeatureSpec("input_feature_0", core.DataType.INT64, shape=(2,)),
                core.FeatureSpec("input_feature_1", core.DataType.INT64, shape=(2,)),
            ],
        )

        arr = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
        lt = [arr, arr]
        self.assertListEqual(
            numpy_handler.SeqOfNumpyArrayHandler.infer_signature(lt, role="output"),
            [
                core.FeatureSpec("output_feature_0", core.DataType.INT64, shape=(2, 2)),
                core.FeatureSpec("output_feature_1", core.DataType.INT64, shape=(2, 2)),
            ],
        )

    def test_convert_to_df_list_of_numpy_array(self) -> None:
        arr1 = np.array([1, 2, 3, 4])
        lt = [arr1, arr1]
        pd.testing.assert_frame_equal(
            numpy_handler.SeqOfNumpyArrayHandler.convert_to_df(lt),
            pd.DataFrame([[1, 1], [2, 2], [3, 3], [4, 4]]),
            check_names=False,
        )

        arr2 = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        lt = [arr1, arr2]
        pd.testing.assert_frame_equal(
            numpy_handler.SeqOfNumpyArrayHandler.convert_to_df(lt),
            pd.DataFrame([[1, [1, 1]], [2, [2, 2]], [3, [3, 3]], [4, [4, 4]]]),
        )

        arr = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
        lt = [arr, arr]
        pd.testing.assert_frame_equal(
            numpy_handler.SeqOfNumpyArrayHandler.convert_to_df(lt),
            pd.DataFrame(
                data={
                    0: [[[1, 1], [2, 2]], [[3, 3], [4, 4]]],
                    1: [[[1, 1], [2, 2]], [[3, 3], [4, 4]]],
                }
            ),
        )


if __name__ == "__main__":
    absltest.main()
