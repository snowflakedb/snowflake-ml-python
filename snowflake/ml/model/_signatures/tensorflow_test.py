import numpy as np
import pandas as pd
import tensorflow as tf
from absl.testing import absltest

from snowflake.ml.model._signatures import core, tensorflow_handler, utils
from snowflake.ml.test_utils import exception_utils


class SeqOfTensorflowTensorHandlerTest(absltest.TestCase):
    def test_can_handle_list_tf_tensor(self) -> None:
        lt1 = [tf.constant([1, 2]), tf.constant([1, 2])]
        self.assertTrue(tensorflow_handler.SeqOfTensorflowTensorHandler.can_handle(lt1))

        lt2 = (tf.constant([1, 2]), tf.Variable([1, 2]))
        self.assertTrue(tensorflow_handler.SeqOfTensorflowTensorHandler.can_handle(lt2))

        lt3 = (tf.constant([1, 2]), 3)
        self.assertFalse(tensorflow_handler.SeqOfTensorflowTensorHandler.can_handle(lt3))

        lt4 = ({"a": tf.constant([1, 2])}, 3)
        self.assertFalse(tensorflow_handler.SeqOfTensorflowTensorHandler.can_handle(lt4))

        lt5 = [tf.constant([1, 2]), 3]
        self.assertFalse(tensorflow_handler.SeqOfTensorflowTensorHandler.can_handle(lt5))

        lt6 = [np.array([1, 2, 3, 4]), tf.constant([1, 2])]
        self.assertFalse(tensorflow_handler.SeqOfTensorflowTensorHandler.can_handle(lt6))

    def test_validate_list_of_tf_tensor(self) -> None:
        lt1 = [np.array([1, 4]), np.array([2, 3])]
        self.assertFalse(tensorflow_handler.SeqOfTensorflowTensorHandler.can_handle(lt1))

        lt2 = [np.array([1, 4]), tf.constant([2, 3])]
        self.assertFalse(tensorflow_handler.SeqOfTensorflowTensorHandler.can_handle(lt2))

        lt3 = [tf.constant([1, 4]), tf.constant([2, 3])]
        self.assertTrue(tensorflow_handler.SeqOfTensorflowTensorHandler.can_handle(lt3))

        lt4 = [tf.constant([1, 4]), tf.Variable([2, 3])]
        self.assertTrue(tensorflow_handler.SeqOfTensorflowTensorHandler.can_handle(lt4))

        lt5 = [tf.Variable([1, 4]), tf.Variable([2, 3])]
        self.assertTrue(tensorflow_handler.SeqOfTensorflowTensorHandler.can_handle(lt5))

    def test_validate_tf_tensor(self) -> None:
        t = [tf.constant([])]
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Empty data is found."
        ):
            tensorflow_handler.SeqOfTensorflowTensorHandler.validate(t)

        t = [tf.Variable([1, 2], shape=tf.TensorShape(None))]
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Unknown shape data is found."
        ):
            tensorflow_handler.SeqOfTensorflowTensorHandler.validate(t)

        t = [tf.Variable([[1, 2]], shape=tf.TensorShape([None, 2]))]
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Unknown shape data is found."
        ):
            tensorflow_handler.SeqOfTensorflowTensorHandler.validate(t)

        t = [tf.Variable([[1, 2]], shape=tf.TensorShape([1, None]))]
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Unknown shape data is found."
        ):
            tensorflow_handler.SeqOfTensorflowTensorHandler.validate(t)

        t = [tf.constant(1)]
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Scalar data is found."
        ):
            tensorflow_handler.SeqOfTensorflowTensorHandler.validate(t)

        t = [tf.constant([1])]
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Scalar data is found."
        ):
            tensorflow_handler.SeqOfTensorflowTensorHandler.validate(t)

        t = [tf.Variable(1)]
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Scalar data is found."
        ):
            tensorflow_handler.SeqOfTensorflowTensorHandler.validate(t)

        t = [tf.Variable([1])]
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Scalar data is found."
        ):
            tensorflow_handler.SeqOfTensorflowTensorHandler.validate(t)

        t = [tf.constant([1, 2]), tf.constant(1)]
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Scalar data is found."
        ):
            tensorflow_handler.SeqOfTensorflowTensorHandler.validate(t)

    def test_count_tf_tensor(self) -> None:
        t = [tf.constant([1, 2])]
        self.assertEqual(tensorflow_handler.SeqOfTensorflowTensorHandler.count(t), 2)

        t = [tf.constant([[1, 2]])]
        self.assertEqual(tensorflow_handler.SeqOfTensorflowTensorHandler.count(t), 1)

        t = [tf.Variable([1, 2])]
        self.assertEqual(tensorflow_handler.SeqOfTensorflowTensorHandler.count(t), 2)

        t = [tf.Variable([1, 2], shape=tf.TensorShape(None))]
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Unknown shape data is found."
        ):
            tensorflow_handler.SeqOfTensorflowTensorHandler.validate(t)

        t = [tf.Variable([[1, 2]], shape=tf.TensorShape([None, 2]))]
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Unknown shape data is found."
        ):
            tensorflow_handler.SeqOfTensorflowTensorHandler.validate(t)

        t = [tf.Variable([[1, 2]], shape=tf.TensorShape([1, None]))]
        self.assertEqual(tensorflow_handler.SeqOfTensorflowTensorHandler.count(t), 1)

    def test_trunc_tf_tensor(self) -> None:
        t = [tf.constant([1] * (tensorflow_handler.SeqOfTensorflowTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT + 1))]

        for ts in tensorflow_handler.SeqOfTensorflowTensorHandler.truncate(t):
            tf.assert_equal(
                tf.constant([1] * (tensorflow_handler.SeqOfTensorflowTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT)), ts
            )

        t = [tf.constant([1] * (tensorflow_handler.SeqOfTensorflowTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT - 1))]

        for ts in tensorflow_handler.SeqOfTensorflowTensorHandler.truncate(t):
            tf.assert_equal(
                tf.constant([1] * (tensorflow_handler.SeqOfTensorflowTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT - 1)), ts
            )

        t = [tf.constant([1] * (tensorflow_handler.SeqOfTensorflowTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT + 1))] * 2

        for ts in tensorflow_handler.SeqOfTensorflowTensorHandler.truncate(t):
            tf.assert_equal(
                tf.constant([1] * (tensorflow_handler.SeqOfTensorflowTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT)), ts
            )

        t = [
            tf.constant([1]),
            tf.constant([1] * (tensorflow_handler.SeqOfTensorflowTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT - 1)),
        ]

        for ts in tensorflow_handler.SeqOfTensorflowTensorHandler.truncate(t):
            tf.assert_equal(tf.constant([1]), ts)

    def test_infer_schema_tf_tensor(self) -> None:
        t1 = [tf.constant([1, 2, 3, 4], dtype=tf.int32)]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t1, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.INT32, nullable=False)],
        )

        t2 = [tf.constant([1, 2, 3, 4], dtype=tf.int64)]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t2, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.INT64, nullable=False)],
        )

        t3 = [tf.constant([1, 2, 3, 4], dtype=tf.int16)]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t3, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.INT16, nullable=False)],
        )

        t4 = [tf.constant([1, 2, 3, 4], dtype=tf.int8)]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t4, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.INT8, nullable=False)],
        )

        t5 = [tf.constant([1, 2, 3, 4], dtype=tf.uint32)]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t5, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.UINT32, nullable=False)],
        )

        t6 = [tf.constant([1, 2, 3, 4], dtype=tf.uint64)]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t6, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.UINT64, nullable=False)],
        )

        t7 = [tf.constant([1, 2, 3, 4], dtype=tf.uint16)]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t7, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.UINT16, nullable=False)],
        )

        t8 = [tf.constant([1, 2, 3, 4], dtype=tf.uint8)]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t8, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.UINT8, nullable=False)],
        )

        t9 = [tf.constant([False, True])]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t9, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.BOOL, nullable=False)],
        )

        t10 = [tf.constant([1.2, 3.4], dtype=tf.float32)]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t10, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.FLOAT, nullable=False)],
        )

        t11 = [tf.constant([1.2, 3.4], dtype=tf.float64)]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t11, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.DOUBLE, nullable=False)],
        )

        t12 = [tf.constant([[1, 2], [3, 4]])]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t12, role="input"),
            [
                core.FeatureSpec("input_feature_0", core.DataType.INT32, shape=(2,), nullable=False),
            ],
        )

        t13 = [tf.constant([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t13, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.INT32, shape=(2, 2), nullable=False)],
        )

        t14 = [tf.constant([1, 2, 3, 4])]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t14, role="output"),
            [core.FeatureSpec("output_feature_0", core.DataType.INT32, nullable=False)],
        )

        t15 = [tf.constant([1, 2]), tf.constant([3, 4])]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t15, role="output"),
            [
                core.FeatureSpec("output_feature_0", core.DataType.INT32, nullable=False),
                core.FeatureSpec("output_feature_1", core.DataType.INT32, nullable=False),
            ],
        )

        t16 = [tf.constant([1.2, 2.4]), tf.constant([3, 4])]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t16, role="output"),
            [
                core.FeatureSpec("output_feature_0", core.DataType.FLOAT, nullable=False),
                core.FeatureSpec("output_feature_1", core.DataType.INT32, nullable=False),
            ],
        )

        t17 = [tf.constant([[1, 1], [2, 2]]), tf.constant([[3, 3], [4, 4]])]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t17, role="output"),
            [
                core.FeatureSpec("output_feature_0", core.DataType.INT32, shape=(2,), nullable=False),
                core.FeatureSpec("output_feature_1", core.DataType.INT32, shape=(2,), nullable=False),
            ],
        )

        t18 = [tf.constant([[1, 1], [2, 2]]), tf.constant([[1.5, 6.8], [2.9, 9.2]])]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t18, role="output"),
            [
                core.FeatureSpec("output_feature_0", core.DataType.INT32, shape=(2,), nullable=False),
                core.FeatureSpec("output_feature_1", core.DataType.FLOAT, shape=(2,), nullable=False),
            ],
        )

        t21 = [tf.constant([1, 2, 3, 4], dtype=tf.int32)]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t21, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.INT32, nullable=False)],
        )

        t22 = [tf.constant([1, 2, 3, 4], dtype=tf.int64)]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t22, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.INT64, nullable=False)],
        )

        t23 = [tf.constant([1, 2, 3, 4], dtype=tf.int16)]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t23, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.INT16, nullable=False)],
        )

        t24 = [tf.constant([1, 2, 3, 4], dtype=tf.int8)]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t24, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.INT8, nullable=False)],
        )

        t25 = [tf.constant([1, 2, 3, 4], dtype=tf.uint32)]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t25, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.UINT32, nullable=False)],
        )

        t26 = [tf.constant([1, 2, 3, 4], dtype=tf.uint64)]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t26, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.UINT64, nullable=False)],
        )

        t27 = [tf.constant([1, 2, 3, 4], dtype=tf.uint16)]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t27, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.UINT16, nullable=False)],
        )

        t28 = [tf.constant([1, 2, 3, 4], dtype=tf.uint8)]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t28, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.UINT8, nullable=False)],
        )

        t29 = [tf.constant([False, True])]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t29, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.BOOL, nullable=False)],
        )

        t30 = [tf.constant([1.2, 3.4], dtype=tf.float32)]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t30, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.FLOAT, nullable=False)],
        )

        t31 = [tf.constant([1.2, 3.4], dtype=tf.float64)]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t31, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.DOUBLE, nullable=False)],
        )

        t32 = [tf.constant([[1, 2], [3, 4]])]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t32, role="input"),
            [
                core.FeatureSpec("input_feature_0", core.DataType.INT32, shape=(2,), nullable=False),
            ],
        )

        t33 = [tf.constant([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t33, role="input"),
            [core.FeatureSpec("input_feature_0", core.DataType.INT32, shape=(2, 2), nullable=False)],
        )

        t34 = [tf.constant([1, 2, 3, 4])]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t34, role="output"),
            [core.FeatureSpec("output_feature_0", core.DataType.INT32, nullable=False)],
        )

        t35 = [tf.constant([1, 2]), tf.constant([3, 4])]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t35, role="output"),
            [
                core.FeatureSpec("output_feature_0", core.DataType.INT32, nullable=False),
                core.FeatureSpec("output_feature_1", core.DataType.INT32, nullable=False),
            ],
        )

        t36 = [tf.constant([1.2, 2.4]), tf.constant([3, 4])]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t36, role="output"),
            [
                core.FeatureSpec("output_feature_0", core.DataType.FLOAT, nullable=False),
                core.FeatureSpec("output_feature_1", core.DataType.INT32, nullable=False),
            ],
        )

        t37 = [tf.constant([[1, 1], [2, 2]]), tf.constant([[3, 3], [4, 4]])]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t37, role="output"),
            [
                core.FeatureSpec("output_feature_0", core.DataType.INT32, shape=(2,), nullable=False),
                core.FeatureSpec("output_feature_1", core.DataType.INT32, shape=(2,), nullable=False),
            ],
        )

        t38 = [tf.constant([[1, 1], [2, 2]]), tf.constant([[1.5, 6.8], [2.9, 9.2]])]
        self.assertListEqual(
            tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t38, role="output"),
            [
                core.FeatureSpec("output_feature_0", core.DataType.INT32, shape=(2,), nullable=False),
                core.FeatureSpec("output_feature_1", core.DataType.FLOAT, shape=(2,), nullable=False),
            ],
        )

    def test_convert_to_df_tf_tensor(self) -> None:
        t1 = [tf.constant([1, 2, 3, 4], dtype=tf.int64)]
        pd.testing.assert_frame_equal(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t1),
            pd.DataFrame([1, 2, 3, 4]),
        )

        t2 = [tf.Variable([1, 2, 3, 4], dtype=tf.int64)]
        pd.testing.assert_frame_equal(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t2),
            pd.DataFrame([1, 2, 3, 4]),
        )

        t3 = [tf.constant([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=tf.int64)]
        pd.testing.assert_frame_equal(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t3),
            pd.DataFrame({0: [np.array([1, 1]), np.array([2, 2]), np.array([3, 3]), np.array([4, 4])]}),
        )

        t4 = [tf.constant([[[1, 1], [2, 2]], [[3, 3], [4, 4]]], dtype=tf.int64)]
        pd.testing.assert_frame_equal(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t4),
            pd.DataFrame(data={0: [np.array([[1, 1], [2, 2]]), np.array([[3, 3], [4, 4]])]}),
        )

        t5 = [tf.constant([1, 2], dtype=tf.int64), tf.constant([3, 4], dtype=tf.int64)]
        pd.testing.assert_frame_equal(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t5),
            pd.DataFrame([[1, 3], [2, 4]]),
        )

        t6 = [tf.constant([1.2, 2.4], dtype=tf.float64), tf.constant([3, 4], dtype=tf.int64)]
        pd.testing.assert_frame_equal(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t6),
            pd.DataFrame([[1.2, 3], [2.4, 4]]),
        )

        t7 = [tf.constant([[1, 1], [2, 2]], dtype=tf.int64), tf.constant([[3, 3], [4, 4]], dtype=tf.int64)]
        pd.testing.assert_frame_equal(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t7),
            pd.DataFrame({0: [np.array([1, 1]), np.array([2, 2])], 1: [np.array([3, 3]), np.array([4, 4])]}),
        )

        t8 = [tf.constant([[1, 1], [2, 2]], dtype=tf.int64), tf.constant([[1.5, 6.8], [2.9, 9.2]], dtype=tf.float64)]
        pd.testing.assert_frame_equal(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t8),
            pd.DataFrame({0: [np.array([1, 1]), np.array([2, 2])], 1: [np.array([1.5, 6.8]), np.array([2.9, 9.2])]}),
        )

    def test_convert_from_df_tf_tensor(self) -> None:
        t1 = [tf.constant([1, 2, 3, 4], dtype=tf.int64)]
        for idx, t in enumerate(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t1)
            )
        ):
            tf.assert_equal(t, t1[idx])

        t2 = [tf.Variable([1, 2, 3, 4], dtype=tf.int64)]
        for idx, t in enumerate(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t2)
            )
        ):
            tf.assert_equal(t, t2[idx])

        t3 = [tf.constant([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=tf.int64)]
        for idx, t in enumerate(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t3)
            )
        ):
            tf.assert_equal(t, t3[idx])

        t4 = [tf.constant([[[1, 1], [2, 2]], [[3, 3], [4, 4]]], dtype=tf.int64)]
        for idx, t in enumerate(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t4)
            )
        ):
            tf.assert_equal(t, t4[idx])

        t5 = [tf.constant([1, 2], dtype=tf.int64), tf.constant([3, 4], dtype=tf.int64)]
        for idx, t in enumerate(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t5)
            )
        ):
            tf.assert_equal(t, t5[idx])

        t6 = [tf.constant([1.2, 2.4], dtype=tf.float64), tf.constant([3, 4], dtype=tf.int64)]
        for idx, t in enumerate(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t6)
            )
        ):
            tf.assert_equal(t, t6[idx])

        t7 = [tf.constant([[1, 1], [2, 2]], dtype=tf.int64), tf.constant([[3, 3], [4, 4]], dtype=tf.int64)]
        for idx, t in enumerate(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t7)
            )
        ):
            tf.assert_equal(t, t7[idx])

        t8 = [tf.constant([[1, 1], [2, 2]], dtype=tf.int64), tf.constant([[1.5, 6.8], [2.9, 9.2]], dtype=tf.float64)]
        for idx, t in enumerate(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t8)
            )
        ):
            tf.assert_equal(t, t8[idx])

        t9 = [tf.constant([1, 2, 3, 4])]
        fts = tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t9, role="input")
        for idx, t in enumerate(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                utils.rename_pandas_df(tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t9), fts),
                fts,
            )
        ):
            tf.assert_equal(t, t9[idx])

        t10 = [tf.constant([1.2, 3.4])]
        fts = tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t10, role="input")
        for idx, t in enumerate(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                utils.rename_pandas_df(tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t10), fts),
                fts,
            )
        ):
            tf.assert_equal(t, t10[idx])

        t11 = [tf.constant([[1, 1], [2, 2], [3, 3], [4, 4]])]
        fts = tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t11, role="input")
        for idx, t in enumerate(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                utils.rename_pandas_df(tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t11), fts),
                fts,
            )
        ):
            tf.assert_equal(t, t11[idx])

        t12 = [tf.constant([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])]
        fts = tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t12, role="input")
        for idx, t in enumerate(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                utils.rename_pandas_df(tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t12), fts),
                fts,
            )
        ):
            tf.assert_equal(t, t12[idx])

        t13 = [tf.constant([1, 2]), tf.constant([3, 4])]
        fts = tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t13, role="input")
        for idx, t in enumerate(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                utils.rename_pandas_df(tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t13), fts),
                fts,
            )
        ):
            tf.assert_equal(t, t13[idx])

        t14 = [tf.constant([1.2, 2.4]), tf.constant([3, 4])]
        fts = tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t14, role="input")
        for idx, t in enumerate(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                utils.rename_pandas_df(tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t14), fts),
                fts,
            )
        ):
            tf.assert_equal(t, t14[idx])

        t15 = [tf.constant([[1, 1], [2, 2]]), tf.constant([[3, 3], [4, 4]])]
        fts = tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t15, role="input")
        for idx, t in enumerate(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                utils.rename_pandas_df(tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t15), fts),
                fts,
            )
        ):
            tf.assert_equal(t, t15[idx])

        t16 = [tf.constant([[1, 1], [2, 2]]), tf.constant([[1.5, 6.8], [2.9, 9.2]])]
        fts = tensorflow_handler.SeqOfTensorflowTensorHandler.infer_signature(t16, role="input")
        for idx, t in enumerate(
            tensorflow_handler.SeqOfTensorflowTensorHandler.convert_from_df(
                utils.rename_pandas_df(tensorflow_handler.SeqOfTensorflowTensorHandler.convert_to_df(t16), fts),
                fts,
            )
        ):
            tf.assert_equal(t, t16[idx])


if __name__ == "__main__":
    absltest.main()
