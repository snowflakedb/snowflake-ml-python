import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from absl.testing import absltest

import snowflake.snowpark.types as spt
from snowflake.ml.model import model_signature
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session


class DataTypeTest(absltest.TestCase):
    def test_numpy_type(self) -> None:
        data = np.array([1, 2, 3, 4])
        self.assertEqual(model_signature.DataType.INT64, model_signature.DataType.from_numpy_type(data.dtype))

        data = np.array(["a", "b", "c", "d"])
        self.assertEqual(model_signature.DataType.STRING, model_signature.DataType.from_numpy_type(data.dtype))

    def test_snowpark_type(self) -> None:
        self.assertEqual(model_signature.DataType.INT8, model_signature.DataType.from_snowpark_type(spt.ByteType()))
        self.assertEqual(model_signature.DataType.INT16, model_signature.DataType.from_snowpark_type(spt.ShortType()))
        self.assertEqual(model_signature.DataType.INT32, model_signature.DataType.from_snowpark_type(spt.IntegerType()))
        self.assertEqual(model_signature.DataType.INT64, model_signature.DataType.from_snowpark_type(spt.LongType()))

        self.assertEqual(
            model_signature.DataType.INT64, model_signature.DataType.from_snowpark_type(spt.DecimalType(38, 0))
        )

        self.assertEqual(model_signature.DataType.FLOAT, model_signature.DataType.from_snowpark_type(spt.FloatType()))
        self.assertEqual(model_signature.DataType.DOUBLE, model_signature.DataType.from_snowpark_type(spt.DoubleType()))

        with self.assertRaises(NotImplementedError):
            model_signature.DataType.from_snowpark_type(spt.DecimalType(38, 6))

        self.assertEqual(model_signature.DataType.BOOL, model_signature.DataType.from_snowpark_type(spt.BooleanType()))
        self.assertEqual(model_signature.DataType.STRING, model_signature.DataType.from_snowpark_type(spt.StringType()))
        self.assertEqual(model_signature.DataType.BYTES, model_signature.DataType.from_snowpark_type(spt.BinaryType()))

        self.assertTrue(model_signature.DataType.INT64.is_same_snowpark_type(spt.LongType()))
        self.assertTrue(model_signature.DataType.INT32.is_same_snowpark_type(spt.IntegerType()))
        self.assertTrue(model_signature.DataType.INT16.is_same_snowpark_type(spt.ShortType()))
        self.assertTrue(model_signature.DataType.INT8.is_same_snowpark_type(spt.ByteType()))
        self.assertTrue(model_signature.DataType.UINT64.is_same_snowpark_type(spt.LongType()))
        self.assertTrue(model_signature.DataType.UINT32.is_same_snowpark_type(spt.IntegerType()))
        self.assertTrue(model_signature.DataType.UINT16.is_same_snowpark_type(spt.ShortType()))
        self.assertTrue(model_signature.DataType.UINT8.is_same_snowpark_type(spt.ByteType()))

        self.assertTrue(model_signature.DataType.FLOAT.is_same_snowpark_type(spt.FloatType()))
        self.assertTrue(model_signature.DataType.DOUBLE.is_same_snowpark_type(spt.DoubleType()))

        self.assertTrue(
            model_signature.DataType.INT64.is_same_snowpark_type(incoming_snowpark_type=spt.DecimalType(38, 0))
        )
        self.assertTrue(
            model_signature.DataType.UINT64.is_same_snowpark_type(incoming_snowpark_type=spt.DecimalType(38, 0))
        )


class FeatureSpecTest(absltest.TestCase):
    def test_feature_spec(self) -> None:
        ft = model_signature.FeatureSpec(name="feature", dtype=model_signature.DataType.INT64)
        self.assertEqual(ft, eval(repr(ft), model_signature.__dict__))
        self.assertEqual(ft, model_signature.FeatureSpec.from_dict(ft.to_dict()))
        self.assertEqual(ft.as_snowpark_type(), spt.LongType())

        ft = model_signature.FeatureSpec(name="feature", dtype=model_signature.DataType.INT64, shape=(2,))
        self.assertEqual(ft, eval(repr(ft), model_signature.__dict__))
        self.assertEqual(ft, model_signature.FeatureSpec.from_dict(input_dict=ft.to_dict()))
        self.assertEqual(ft.as_snowpark_type(), spt.ArrayType(spt.LongType()))


class FeatureGroupSpecTest(absltest.TestCase):
    def test_feature_group_spec(self) -> None:
        with self.assertRaisesRegex(ValueError, "No children feature specs."):
            _ = model_signature.FeatureGroupSpec(name="features", specs=[])

        with self.assertRaisesRegex(ValueError, "All children feature specs have to have name."):
            ft1 = model_signature.FeatureSpec(name="feature1", dtype=model_signature.DataType.INT64)
            ft2 = model_signature.FeatureSpec(name="feature2", dtype=model_signature.DataType.INT64)
            ft2._name = None  # type: ignore[assignment]
            _ = model_signature.FeatureGroupSpec(name="features", specs=[ft1, ft2])

        with self.assertRaisesRegex(ValueError, "All children feature specs have to have same type."):
            ft1 = model_signature.FeatureSpec(name="feature1", dtype=model_signature.DataType.INT64)
            ft2 = model_signature.FeatureSpec(name="feature2", dtype=model_signature.DataType.FLOAT)
            _ = model_signature.FeatureGroupSpec(name="features", specs=[ft1, ft2])

        with self.assertRaisesRegex(ValueError, "All children feature specs have to have same shape."):
            ft1 = model_signature.FeatureSpec(name="feature1", dtype=model_signature.DataType.INT64)
            ft2 = model_signature.FeatureSpec(name="feature2", dtype=model_signature.DataType.INT64, shape=(2,))
            fts = model_signature.FeatureGroupSpec(name="features", specs=[ft1, ft2])

        ft1 = model_signature.FeatureSpec(name="feature1", dtype=model_signature.DataType.INT64)
        ft2 = model_signature.FeatureSpec(name="feature2", dtype=model_signature.DataType.INT64)
        fts = model_signature.FeatureGroupSpec(name="features", specs=[ft1, ft2])
        self.assertEqual(fts, eval(repr(fts), model_signature.__dict__))
        self.assertEqual(fts, model_signature.FeatureGroupSpec.from_dict(fts.to_dict()))
        self.assertEqual(fts.as_snowpark_type(), spt.MapType(spt.StringType(), spt.LongType()))

        ft1 = model_signature.FeatureSpec(name="feature1", dtype=model_signature.DataType.INT64, shape=(3,))
        ft2 = model_signature.FeatureSpec(name="feature2", dtype=model_signature.DataType.INT64, shape=(2,))
        fts = model_signature.FeatureGroupSpec(name="features", specs=[ft1, ft2])
        self.assertEqual(fts, eval(repr(fts), model_signature.__dict__))
        self.assertEqual(fts, model_signature.FeatureGroupSpec.from_dict(fts.to_dict()))
        self.assertEqual(fts.as_snowpark_type(), spt.MapType(spt.StringType(), spt.ArrayType(spt.LongType())))


class ModelSignatureTest(absltest.TestCase):
    def test_1(self) -> None:
        s = model_signature.ModelSignature(
            inputs=[
                model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="c1"),
                model_signature.FeatureGroupSpec(
                    name="cg1",
                    specs=[
                        model_signature.FeatureSpec(
                            dtype=model_signature.DataType.FLOAT,
                            name="cc1",
                        ),
                        model_signature.FeatureSpec(
                            dtype=model_signature.DataType.FLOAT,
                            name="cc2",
                        ),
                    ],
                ),
                model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="c2", shape=(-1,)),
            ],
            outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
        )
        target = {
            "inputs": [
                {"type": "FLOAT", "name": "c1"},
                {
                    "feature_group": {
                        "name": "cg1",
                        "specs": [{"type": "FLOAT", "name": "cc1"}, {"type": "FLOAT", "name": "cc2"}],
                    }
                },
                {"type": "FLOAT", "name": "c2", "shape": (-1,)},
            ],
            "outputs": [{"type": "FLOAT", "name": "output"}],
        }
        self.assertDictEqual(s.to_dict(), target)
        self.assertEqual(s, eval(repr(s), model_signature.__dict__))
        self.assertEqual(s, model_signature.ModelSignature.from_dict(s.to_dict()))

    def test_2(self) -> None:
        s = model_signature.ModelSignature(
            inputs=[
                model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="c1"),
                model_signature.FeatureGroupSpec(
                    name="cg1",
                    specs=[
                        model_signature.FeatureSpec(
                            dtype=model_signature.DataType.FLOAT,
                            name="cc1",
                        ),
                        model_signature.FeatureSpec(
                            dtype=model_signature.DataType.FLOAT,
                            name="cc2",
                        ),
                    ],
                ),
                model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="c2", shape=(-1,)),
            ],
            outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
        )
        self.assertEqual(s, eval(repr(s), model_signature.__dict__))
        self.assertEqual(s, model_signature.ModelSignature.from_dict(s.to_dict()))


class PandasDataFrameHandlerTest(absltest.TestCase):
    def test_validate_pd_DataFrame(self) -> None:
        df = pd.DataFrame([])
        with self.assertRaisesRegex(ValueError, "Empty data is found."):
            model_signature._PandasDataFrameHandler.validate(df)

        df = pd.DataFrame([[1, 2], [2, 4]], columns=["a", "a"])
        with self.assertRaisesRegex(ValueError, "Duplicate column index is found"):
            model_signature._PandasDataFrameHandler.validate(df)

        sub_df = pd.DataFrame([2.5, 6.8])
        df = pd.DataFrame([[1, sub_df], [2, sub_df]], columns=["a", "b"])
        with self.assertRaisesRegex(ValueError, "Unsupported type confronted in"):
            model_signature._PandasDataFrameHandler.validate(df)

        df = pd.DataFrame(
            [[1, 2.0, 1, 2.0, 1, 2.0], [2, 4.0, 2, 4.0, 2, 4.0]],
            columns=pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"]),
        )
        with self.assertRaisesRegex(ValueError, "Duplicate column index is found"):
            model_signature._PandasDataFrameHandler.validate(df)

        df = pd.DataFrame([[1, 2], [2, 4]], columns=["a", "a"])
        with self.assertRaisesRegex(ValueError, "Duplicate column index is found"):
            model_signature._PandasDataFrameHandler.validate(df)

        df = pd.DataFrame([[1, "Hello"], [2, [2, 6]]], columns=["a", "b"])
        with self.assertRaisesRegex(ValueError, "Inconsistent type of object"):
            model_signature._PandasDataFrameHandler.validate(df)

        df = pd.DataFrame([[1, 2], [2, [2, 6]]], columns=["a", "b"])
        with self.assertRaisesRegex(ValueError, "Inconsistent type of object"):
            model_signature._PandasDataFrameHandler.validate(df)

        df = pd.DataFrame([[1, [2, [6]]], [2, [2, 6]]], columns=["a", "b"])
        with self.assertRaisesRegex(ValueError, "Ragged nested or Unsupported list-like data"):
            model_signature._PandasDataFrameHandler.validate(df)

        df = pd.DataFrame([[1, [2, 6]], [2, [2, [6]]]], columns=["a", "b"])
        with self.assertRaisesRegex(ValueError, "Ragged nested or Unsupported list-like data"):
            model_signature._PandasDataFrameHandler.validate(df)

        df = pd.DataFrame([[1, [2.5, 6.8]], [2, [2, 6]]], columns=["a", "b"])
        with self.assertRaisesRegex(ValueError, "Inconsistent type of element in object found in column data"):
            model_signature._PandasDataFrameHandler.validate(df)

        df = pd.DataFrame([[1, np.array([2.5, 6.8])], [2, np.array([2, 6])]], columns=["a", "b"])
        with self.assertRaisesRegex(ValueError, "Inconsistent type of element in object found in column data"):
            model_signature._PandasDataFrameHandler.validate(df)

        df = pd.DataFrame([[1, np.array([2.5, 6.8])], [2, 6]], columns=["a", "b"])
        with self.assertRaisesRegex(ValueError, "Inconsistent type of object found in column data"):
            model_signature._PandasDataFrameHandler.validate(df)

    def test_trunc_pd_DataFrame(self) -> None:
        df = pd.DataFrame([1] * (model_signature._PandasDataFrameHandler.SIG_INFER_ROWS_COUNT_LIMIT + 1))

        pd.testing.assert_frame_equal(
            pd.DataFrame([1] * (model_signature._PandasDataFrameHandler.SIG_INFER_ROWS_COUNT_LIMIT)),
            model_signature._PandasDataFrameHandler.truncate(df),
        )

        df = pd.DataFrame([1] * (model_signature._PandasDataFrameHandler.SIG_INFER_ROWS_COUNT_LIMIT - 1))

        pd.testing.assert_frame_equal(
            df,
            model_signature._PandasDataFrameHandler.truncate(df),
        )

    def test_infer_signature_pd_DataFrame(self) -> None:
        df = pd.DataFrame([1, 2, 3, 4])
        self.assertListEqual(
            model_signature._PandasDataFrameHandler.infer_signature(df, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64)],
        )

        df = pd.DataFrame([1, 2, 3, 4], columns=["a"])
        self.assertListEqual(
            model_signature._PandasDataFrameHandler.infer_signature(df, role="input"),
            [model_signature.FeatureSpec("a", model_signature.DataType.INT64)],
        )

        df = pd.DataFrame(["a", "b", "c", "d"], columns=["a"])
        self.assertListEqual(
            model_signature._PandasDataFrameHandler.infer_signature(df, role="input"),
            [model_signature.FeatureSpec("a", model_signature.DataType.STRING)],
        )

        df = pd.DataFrame([ele.encode() for ele in ["a", "b", "c", "d"]], columns=["a"])
        self.assertListEqual(
            model_signature._PandasDataFrameHandler.infer_signature(df, role="input"),
            [model_signature.FeatureSpec("a", model_signature.DataType.BYTES)],
        )

        df = pd.DataFrame([[1, 2.0], [2, 4.0]])
        self.assertListEqual(
            model_signature._PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("input_feature_1", model_signature.DataType.DOUBLE),
            ],
        )

        df = pd.DataFrame([[1, [2.5, 6.8]], [2, [2.5, 6.8]]], columns=["a", "b"])
        self.assertListEqual(
            model_signature._PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                model_signature.FeatureSpec("a", model_signature.DataType.INT64),
                model_signature.FeatureSpec("b", model_signature.DataType.DOUBLE, shape=(2,)),
            ],
        )

        df = pd.DataFrame([[1, [2.5, 6.8]], [2, [2.5]]], columns=["a", "b"])
        self.assertListEqual(
            model_signature._PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                model_signature.FeatureSpec("a", model_signature.DataType.INT64),
                model_signature.FeatureSpec("b", model_signature.DataType.DOUBLE, shape=(-1,)),
            ],
        )

        df = pd.DataFrame([[1, [[2.5], [6.8]]], [2, [[2.5], [6.8]]]], columns=["a", "b"])
        self.assertListEqual(
            model_signature._PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                model_signature.FeatureSpec("a", model_signature.DataType.INT64),
                model_signature.FeatureSpec("b", model_signature.DataType.DOUBLE, shape=(2, 1)),
            ],
        )

        a = np.array([2.5, 6.8])
        df = pd.DataFrame([[1, a], [2, a]], columns=["a", "b"])
        self.assertListEqual(
            model_signature._PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                model_signature.FeatureSpec("a", model_signature.DataType.INT64),
                model_signature.FeatureSpec("b", model_signature.DataType.DOUBLE, shape=(2,)),
            ],
        )

        df = pd.DataFrame([[1, np.array([2.5, 6.8])], [2, np.array([2.5])]], columns=["a", "b"])
        self.assertListEqual(
            model_signature._PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                model_signature.FeatureSpec("a", model_signature.DataType.INT64),
                model_signature.FeatureSpec("b", model_signature.DataType.DOUBLE, shape=(-1,)),
            ],
        )

        a = np.array([[2, 5], [6, 8]])
        df = pd.DataFrame([[1, a], [2, a]], columns=["a", "b"])
        self.assertListEqual(
            model_signature._PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                model_signature.FeatureSpec("a", model_signature.DataType.INT64),
                model_signature.FeatureSpec("b", model_signature.DataType.INT64, shape=(2, 2)),
            ],
        )

        df = pd.DataFrame([[1, 2.0], [2, 4.0]], columns=pd.PeriodIndex(year=[2000, 2002], quarter=[1, 3]))
        self.assertListEqual(
            model_signature._PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                model_signature.FeatureSpec("2000Q1", model_signature.DataType.INT64),
                model_signature.FeatureSpec("2002Q3", model_signature.DataType.DOUBLE),
            ],
        )

        df = pd.DataFrame([[1, 2.0], [2, 4.0]], columns=pd.date_range("2020-01-06", "2020-03-03", freq="MS"))
        self.assertListEqual(
            model_signature._PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                model_signature.FeatureSpec("2020-02-01 00:00:00", model_signature.DataType.INT64),
                model_signature.FeatureSpec("2020-03-01 00:00:00", model_signature.DataType.DOUBLE),
            ],
        )

        df = pd.DataFrame(
            [[1, 2.0], [2, 4.0]], columns=pd.TimedeltaIndex(data=["1 days 02:00:00", "1 days 06:05:01.000030"])
        )
        self.assertListEqual(
            model_signature._PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                model_signature.FeatureSpec("1 days 02:00:00", model_signature.DataType.INT64),
                model_signature.FeatureSpec("1 days 06:05:01.000030", model_signature.DataType.DOUBLE),
            ],
        )

        df = pd.DataFrame([[1, 2.0], [2, 4.0]], columns=pd.interval_range(start=0, end=2))
        self.assertListEqual(
            model_signature._PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                model_signature.FeatureSpec("(0, 1]", model_signature.DataType.INT64),
                model_signature.FeatureSpec("(1, 2]", model_signature.DataType.DOUBLE),
            ],
        )

        arrays = [[1, 2], ["red", "blue"]]
        df = pd.DataFrame([[1, 2.0], [2, 4.0]], columns=pd.MultiIndex.from_arrays(arrays, names=("number", "color")))
        self.assertListEqual(
            model_signature._PandasDataFrameHandler.infer_signature(df, role="input"),
            [
                model_signature.FeatureSpec("(1, 'red')", model_signature.DataType.INT64),
                model_signature.FeatureSpec("(2, 'blue')", model_signature.DataType.DOUBLE),
            ],
        )

        df = pd.DataFrame([1, 2, 3, 4])
        self.assertListEqual(
            model_signature._PandasDataFrameHandler.infer_signature(df, role="output"),
            [model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT64)],
        )

        df = pd.DataFrame([1, 2, 3, 4], columns=["a"])
        self.assertListEqual(
            model_signature._PandasDataFrameHandler.infer_signature(df, role="output"),
            [model_signature.FeatureSpec("a", model_signature.DataType.INT64)],
        )

        df = pd.DataFrame(["a", "b", "c", "d"], columns=["a"])
        self.assertListEqual(
            model_signature._PandasDataFrameHandler.infer_signature(df, role="output"),
            [model_signature.FeatureSpec("a", model_signature.DataType.STRING)],
        )

        df = pd.DataFrame([ele.encode() for ele in ["a", "b", "c", "d"]], columns=["a"])
        self.assertListEqual(
            model_signature._PandasDataFrameHandler.infer_signature(df, role="output"),
            [model_signature.FeatureSpec("a", model_signature.DataType.BYTES)],
        )

        df = pd.DataFrame([[1, 2.0], [2, 4.0]])
        self.assertListEqual(
            model_signature._PandasDataFrameHandler.infer_signature(df, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("output_feature_1", model_signature.DataType.DOUBLE),
            ],
        )

    def test_convert_to_df_pd_DataFrame(self) -> None:
        a = np.array([[2, 5], [6, 8]])
        li = [[2, 5], [6, 8]]
        df1 = pd.DataFrame([[1, a], [2, a]], columns=["a", "b"])
        df2 = pd.DataFrame([[1, li], [2, li]], columns=["a", "b"])
        pd.testing.assert_frame_equal(model_signature._PandasDataFrameHandler.convert_to_df(df1), df2)


class NumpyArrayHandlerTest(absltest.TestCase):
    def test_validate_np_ndarray(self) -> None:
        arr = np.array([])
        with self.assertRaisesRegex(ValueError, "Empty data is found."):
            model_signature._NumpyArrayHandler.validate(arr)

        arr = np.array(1)
        with self.assertRaisesRegex(ValueError, "Scalar data is found."):
            model_signature._NumpyArrayHandler.validate(arr)

    def test_trunc_np_ndarray(self) -> None:
        arr = np.array([1] * (model_signature._NumpyArrayHandler.SIG_INFER_ROWS_COUNT_LIMIT + 1))

        np.testing.assert_equal(
            np.array([1] * (model_signature._NumpyArrayHandler.SIG_INFER_ROWS_COUNT_LIMIT)),
            model_signature._NumpyArrayHandler.truncate(arr),
        )

        arr = np.array([1] * (model_signature._NumpyArrayHandler.SIG_INFER_ROWS_COUNT_LIMIT - 1))

        np.testing.assert_equal(
            arr,
            model_signature._NumpyArrayHandler.truncate(arr),
        )

    def test_infer_schema_np_ndarray(self) -> None:
        arr = np.array([1, 2, 3, 4])
        self.assertListEqual(
            model_signature._NumpyArrayHandler.infer_signature(arr, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64)],
        )

        arr = np.array([[1, 2], [3, 4]])
        self.assertListEqual(
            model_signature._NumpyArrayHandler.infer_signature(arr, role="input"),
            [
                model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("input_feature_1", model_signature.DataType.INT64),
            ],
        )

        arr = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
        self.assertListEqual(
            model_signature._NumpyArrayHandler.infer_signature(arr, role="input"),
            [
                model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64, shape=(2,)),
                model_signature.FeatureSpec("input_feature_1", model_signature.DataType.INT64, shape=(2,)),
            ],
        )

        arr = np.array([1, 2, 3, 4])
        self.assertListEqual(
            model_signature._NumpyArrayHandler.infer_signature(arr, role="output"),
            [model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT64)],
        )

        arr = np.array([[1, 2], [3, 4]])
        self.assertListEqual(
            model_signature._NumpyArrayHandler.infer_signature(arr, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("output_feature_1", model_signature.DataType.INT64),
            ],
        )

        arr = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
        self.assertListEqual(
            model_signature._NumpyArrayHandler.infer_signature(arr, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT64, shape=(2,)),
                model_signature.FeatureSpec("output_feature_1", model_signature.DataType.INT64, shape=(2,)),
            ],
        )

    def test_convert_to_df_numpy_array(self) -> None:
        arr1 = np.array([1, 2, 3, 4])
        pd.testing.assert_frame_equal(
            model_signature._NumpyArrayHandler.convert_to_df(arr1),
            pd.DataFrame([1, 2, 3, 4]),
        )

        arr2 = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        pd.testing.assert_frame_equal(
            model_signature._NumpyArrayHandler.convert_to_df(arr2),
            pd.DataFrame([[1, 1], [2, 2], [3, 3], [4, 4]]),
        )

        arr3 = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
        pd.testing.assert_frame_equal(
            model_signature._NumpyArrayHandler.convert_to_df(arr3),
            pd.DataFrame(data={0: [np.array([1, 1]), np.array([3, 3])], 1: [np.array([2, 2]), np.array([4, 4])]}),
        )


class SeqOfNumpyArrayHandlerTest(absltest.TestCase):
    def test_validate_list_of_numpy_array(self) -> None:
        lt8 = [pd.DataFrame([1]), pd.DataFrame([2, 3])]
        self.assertFalse(model_signature._SeqOfNumpyArrayHandler.can_handle(lt8))

    def test_trunc_np_ndarray(self) -> None:
        arrs = [np.array([1] * (model_signature._SeqOfNumpyArrayHandler.SIG_INFER_ROWS_COUNT_LIMIT + 1))] * 2

        for arr in model_signature._SeqOfNumpyArrayHandler.truncate(arrs):
            np.testing.assert_equal(
                np.array([1] * (model_signature._SeqOfNumpyArrayHandler.SIG_INFER_ROWS_COUNT_LIMIT)), arr
            )

        arrs = [
            np.array([1]),
            np.array([1] * (model_signature._SeqOfNumpyArrayHandler.SIG_INFER_ROWS_COUNT_LIMIT - 1)),
        ]

        for arr in model_signature._SeqOfNumpyArrayHandler.truncate(arrs):
            np.testing.assert_equal(np.array([1]), arr)

    def test_infer_signature_list_of_numpy_array(self) -> None:
        arr = np.array([1, 2, 3, 4])
        lt = [arr, arr]
        self.assertListEqual(
            model_signature._SeqOfNumpyArrayHandler.infer_signature(lt, role="input"),
            [
                model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("input_feature_1", model_signature.DataType.INT64),
            ],
        )

        arr = np.array([[1, 2], [3, 4]])
        lt = [arr, arr]
        self.assertListEqual(
            model_signature._SeqOfNumpyArrayHandler.infer_signature(lt, role="input"),
            [
                model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64, shape=(2,)),
                model_signature.FeatureSpec("input_feature_1", model_signature.DataType.INT64, shape=(2,)),
            ],
        )

        arr = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
        lt = [arr, arr]
        self.assertListEqual(
            model_signature._SeqOfNumpyArrayHandler.infer_signature(lt, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT64, shape=(2, 2)),
                model_signature.FeatureSpec("output_feature_1", model_signature.DataType.INT64, shape=(2, 2)),
            ],
        )

    def test_convert_to_df_list_of_numpy_array(self) -> None:
        arr1 = np.array([1, 2, 3, 4])
        lt = [arr1, arr1]
        pd.testing.assert_frame_equal(
            model_signature._SeqOfNumpyArrayHandler.convert_to_df(lt),
            pd.DataFrame([[1, 1], [2, 2], [3, 3], [4, 4]]),
            check_names=False,
        )

        arr2 = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        lt = [arr1, arr2]
        pd.testing.assert_frame_equal(
            model_signature._SeqOfNumpyArrayHandler.convert_to_df(lt),
            pd.DataFrame([[1, [1, 1]], [2, [2, 2]], [3, [3, 3]], [4, [4, 4]]]),
        )

        arr = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
        lt = [arr, arr]
        pd.testing.assert_frame_equal(
            model_signature._SeqOfNumpyArrayHandler.convert_to_df(lt),
            pd.DataFrame(
                data={
                    0: [[[1, 1], [2, 2]], [[3, 3], [4, 4]]],
                    1: [[[1, 1], [2, 2]], [[3, 3], [4, 4]]],
                }
            ),
        )


class ListOfBuiltinsHandlerTest(absltest.TestCase):
    def test_validate_list_builtins(self) -> None:
        lt6 = ["Hello", [2, 3]]
        with self.assertRaisesRegex(ValueError, "Inconsistent type of object found in data"):
            model_signature._ListOfBuiltinHandler.validate(lt6)  # type:ignore[arg-type]

        lt7 = [[1], [2, 3]]
        with self.assertRaisesRegex(ValueError, "Ill-shaped list data"):
            model_signature._ListOfBuiltinHandler.validate(lt7)

        lt8 = [pd.DataFrame([1]), pd.DataFrame([2, 3])]
        self.assertFalse(model_signature._ListOfBuiltinHandler.can_handle(lt8))

    def test_infer_signature_list_builtins(self) -> None:
        lt1 = [1, 2, 3, 4]
        self.assertListEqual(
            model_signature._ListOfBuiltinHandler.infer_signature(lt1, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64)],
        )

        lt2 = ["a", "b", "c", "d"]
        self.assertListEqual(
            model_signature._ListOfBuiltinHandler.infer_signature(lt2, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.STRING)],
        )

        lt3 = [ele.encode() for ele in lt2]
        self.assertListEqual(
            model_signature._ListOfBuiltinHandler.infer_signature(lt3, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.BYTES)],
        )

        lt4 = [[1, 2], [3, 4]]
        self.assertListEqual(
            model_signature._ListOfBuiltinHandler.infer_signature(lt4, role="input"),
            [
                model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("input_feature_1", model_signature.DataType.INT64),
            ],
        )

        lt5 = [[1, 2.0], [3, 4]]  # This is not encouraged and will have type error, but we support it.
        self.assertListEqual(
            model_signature._ListOfBuiltinHandler.infer_signature(lt5, role="input"),  # type:ignore[arg-type]
            [
                model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("input_feature_1", model_signature.DataType.DOUBLE),
            ],
        )

        lt6 = [[[1, 1], [2, 2]], [[3, 3], [4, 4]]]
        self.assertListEqual(
            model_signature._ListOfBuiltinHandler.infer_signature(lt6, role="input"),
            [
                model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64, shape=(2,)),
                model_signature.FeatureSpec("input_feature_1", model_signature.DataType.INT64, shape=(2,)),
            ],
        )


class SeqOfPyTorchTensorHandlerTest(absltest.TestCase):
    def test_validate_list_of_pytorch_tensor(self) -> None:
        lt1 = [np.array([1, 4]), np.array([2, 3])]
        self.assertFalse(model_signature._SeqOfPyTorchTensorHandler.can_handle(lt1))

        lt2 = [np.array([1, 4]), torch.Tensor([2, 3])]
        self.assertFalse(model_signature._SeqOfPyTorchTensorHandler.can_handle(lt2))

        lt3 = [torch.Tensor([1, 4]), torch.Tensor([2, 3])]
        self.assertTrue(model_signature._SeqOfPyTorchTensorHandler.can_handle(lt3))

    def test_validate_torch_tensor(self) -> None:
        t = [torch.Tensor([])]
        with self.assertRaisesRegex(ValueError, "Empty data is found."):
            model_signature._SeqOfPyTorchTensorHandler.validate(t)

        t = [torch.Tensor(1)]
        with self.assertRaisesRegex(ValueError, "Scalar data is found."):
            model_signature._SeqOfPyTorchTensorHandler.validate(t)

        t = [torch.Tensor([1, 2]), torch.Tensor(1)]
        with self.assertRaisesRegex(ValueError, "Scalar data is found."):
            model_signature._SeqOfPyTorchTensorHandler.validate(t)

    def test_trunc_torch_tensor(self) -> None:
        t = [torch.Tensor([1] * (model_signature._SeqOfPyTorchTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT + 1))]

        for ts in model_signature._SeqOfPyTorchTensorHandler.truncate(t):
            torch.testing.assert_close(  # type:ignore[attr-defined]
                torch.Tensor([1] * (model_signature._SeqOfPyTorchTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT)), ts
            )

        t = [torch.Tensor([1] * (model_signature._SeqOfPyTorchTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT - 1))]

        for ts in model_signature._SeqOfPyTorchTensorHandler.truncate(t):
            torch.testing.assert_close(  # type:ignore[attr-defined]
                torch.Tensor([1] * (model_signature._SeqOfPyTorchTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT - 1)), ts
            )

        t = [torch.Tensor([1] * (model_signature._SeqOfPyTorchTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT + 1))] * 2

        for ts in model_signature._SeqOfPyTorchTensorHandler.truncate(t):
            torch.testing.assert_close(  # type:ignore[attr-defined]
                torch.Tensor([1] * (model_signature._SeqOfPyTorchTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT)), ts
            )

        t = [
            torch.Tensor([1]),
            torch.Tensor([1] * (model_signature._SeqOfPyTorchTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT - 1)),
        ]

        for ts in model_signature._SeqOfPyTorchTensorHandler.truncate(t):
            torch.testing.assert_close(  # type:ignore[attr-defined]
                torch.Tensor([1]), ts
            )

    def test_infer_schema_torch_tensor(self) -> None:
        t1 = [torch.IntTensor([1, 2, 3, 4])]
        self.assertListEqual(
            model_signature._SeqOfPyTorchTensorHandler.infer_signature(t1, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT32)],
        )

        t2 = [torch.LongTensor([1, 2, 3, 4])]
        self.assertListEqual(
            model_signature._SeqOfPyTorchTensorHandler.infer_signature(t2, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64)],
        )

        t3 = [torch.ShortTensor([1, 2, 3, 4])]
        self.assertListEqual(
            model_signature._SeqOfPyTorchTensorHandler.infer_signature(t3, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT16)],
        )

        t4 = [torch.CharTensor([1, 2, 3, 4])]
        self.assertListEqual(
            model_signature._SeqOfPyTorchTensorHandler.infer_signature(t4, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT8)],
        )

        t5 = [torch.ByteTensor([1, 2, 3, 4])]
        self.assertListEqual(
            model_signature._SeqOfPyTorchTensorHandler.infer_signature(t5, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.UINT8)],
        )

        t6 = [torch.BoolTensor([False, True])]
        self.assertListEqual(
            model_signature._SeqOfPyTorchTensorHandler.infer_signature(t6, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.BOOL)],
        )

        t7 = [torch.FloatTensor([1.2, 3.4])]
        self.assertListEqual(
            model_signature._SeqOfPyTorchTensorHandler.infer_signature(t7, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.FLOAT)],
        )

        t8 = [torch.DoubleTensor([1.2, 3.4])]
        self.assertListEqual(
            model_signature._SeqOfPyTorchTensorHandler.infer_signature(t8, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.DOUBLE)],
        )

        t9 = [torch.LongTensor([[1, 2], [3, 4]])]
        self.assertListEqual(
            model_signature._SeqOfPyTorchTensorHandler.infer_signature(t9, role="input"),
            [
                model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64, shape=(2,)),
            ],
        )

        t10 = [torch.LongTensor([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])]
        self.assertListEqual(
            model_signature._SeqOfPyTorchTensorHandler.infer_signature(t10, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64, shape=(2, 2))],
        )

        t11 = [torch.LongTensor([1, 2, 3, 4])]
        self.assertListEqual(
            model_signature._SeqOfPyTorchTensorHandler.infer_signature(t11, role="output"),
            [model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT64)],
        )

        t12 = [torch.LongTensor([1, 2]), torch.LongTensor([3, 4])]
        self.assertListEqual(
            model_signature._SeqOfPyTorchTensorHandler.infer_signature(t12, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("output_feature_1", model_signature.DataType.INT64),
            ],
        )

        t13 = [torch.FloatTensor([1.2, 2.4]), torch.LongTensor([3, 4])]
        self.assertListEqual(
            model_signature._SeqOfPyTorchTensorHandler.infer_signature(t13, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.FLOAT),
                model_signature.FeatureSpec("output_feature_1", model_signature.DataType.INT64),
            ],
        )

        t14 = [torch.LongTensor([[1, 1], [2, 2]]), torch.LongTensor([[3, 3], [4, 4]])]
        self.assertListEqual(
            model_signature._SeqOfPyTorchTensorHandler.infer_signature(t14, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT64, shape=(2,)),
                model_signature.FeatureSpec("output_feature_1", model_signature.DataType.INT64, shape=(2,)),
            ],
        )

        t15 = [torch.LongTensor([[1, 1], [2, 2]]), torch.DoubleTensor([[1.5, 6.8], [2.9, 9.2]])]
        self.assertListEqual(
            model_signature._SeqOfPyTorchTensorHandler.infer_signature(t15, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT64, shape=(2,)),
                model_signature.FeatureSpec("output_feature_1", model_signature.DataType.DOUBLE, shape=(2,)),
            ],
        )

    def test_convert_to_df_torch_tensor(self) -> None:
        t1 = [torch.LongTensor([1, 2, 3, 4])]
        pd.testing.assert_frame_equal(
            model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t1),
            pd.DataFrame([1, 2, 3, 4]),
        )

        t2 = [torch.DoubleTensor([1, 2, 3, 4])]
        t2[0].requires_grad = True
        pd.testing.assert_frame_equal(
            model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t2),
            pd.DataFrame([1, 2, 3, 4], dtype=np.double),
        )

        t3 = [torch.LongTensor([[1, 1], [2, 2], [3, 3], [4, 4]])]
        pd.testing.assert_frame_equal(
            model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t3),
            pd.DataFrame({0: [np.array([1, 1]), np.array([2, 2]), np.array([3, 3]), np.array([4, 4])]}),
        )

        t4 = [torch.LongTensor([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])]
        pd.testing.assert_frame_equal(
            model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t4),
            pd.DataFrame(data={0: [np.array([[1, 1], [2, 2]]), np.array([[3, 3], [4, 4]])]}),
        )

        t5 = [torch.LongTensor([1, 2]), torch.LongTensor([3, 4])]
        pd.testing.assert_frame_equal(
            model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t5),
            pd.DataFrame([[1, 3], [2, 4]]),
        )

        t6 = [torch.DoubleTensor([1.2, 2.4]), torch.LongTensor([3, 4])]
        pd.testing.assert_frame_equal(
            model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t6),
            pd.DataFrame([[1.2, 3], [2.4, 4]]),
        )

        t7 = [torch.LongTensor([[1, 1], [2, 2]]), torch.LongTensor([[3, 3], [4, 4]])]
        pd.testing.assert_frame_equal(
            model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t7),
            pd.DataFrame({0: [np.array([1, 1]), np.array([2, 2])], 1: [np.array([3, 3]), np.array([4, 4])]}),
        )

        t8 = [torch.LongTensor([[1, 1], [2, 2]]), torch.DoubleTensor([[1.5, 6.8], [2.9, 9.2]])]
        pd.testing.assert_frame_equal(
            model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t8),
            pd.DataFrame({0: [np.array([1, 1]), np.array([2, 2])], 1: [np.array([1.5, 6.8]), np.array([2.9, 9.2])]}),
        )

    def test_convert_from_df_torch_tensor(self) -> None:
        t1 = [torch.LongTensor([1, 2, 3, 4])]
        for idx, t in enumerate(
            model_signature._SeqOfPyTorchTensorHandler.convert_from_df(
                model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t1)
            )
        ):
            torch.testing.assert_close(t, t1[idx])  # type:ignore[attr-defined]

        t2 = [torch.DoubleTensor([1, 2, 3, 4])]
        for idx, t in enumerate(
            model_signature._SeqOfPyTorchTensorHandler.convert_from_df(
                model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t2)
            )
        ):
            torch.testing.assert_close(t, t2[idx])  # type:ignore[attr-defined]

        t3 = [torch.LongTensor([[1, 1], [2, 2], [3, 3], [4, 4]])]
        for idx, t in enumerate(
            model_signature._SeqOfPyTorchTensorHandler.convert_from_df(
                model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t3)
            )
        ):
            torch.testing.assert_close(t, t3[idx])  # type:ignore[attr-defined]

        t4 = [torch.LongTensor([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])]
        for idx, t in enumerate(
            model_signature._SeqOfPyTorchTensorHandler.convert_from_df(
                model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t4)
            )
        ):
            torch.testing.assert_close(t, t4[idx])  # type:ignore[attr-defined]

        t5 = [torch.LongTensor([1, 2]), torch.LongTensor([3, 4])]
        for idx, t in enumerate(
            model_signature._SeqOfPyTorchTensorHandler.convert_from_df(
                model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t5)
            )
        ):
            torch.testing.assert_close(t, t5[idx])  # type:ignore[attr-defined]

        t6 = [torch.DoubleTensor([1.2, 2.4]), torch.LongTensor([3, 4])]
        for idx, t in enumerate(
            model_signature._SeqOfPyTorchTensorHandler.convert_from_df(
                model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t6)
            )
        ):
            torch.testing.assert_close(t, t6[idx])  # type:ignore[attr-defined]

        t7 = [torch.LongTensor([[1, 1], [2, 2]]), torch.LongTensor([[3, 3], [4, 4]])]
        for idx, t in enumerate(
            model_signature._SeqOfPyTorchTensorHandler.convert_from_df(
                model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t7)
            )
        ):
            torch.testing.assert_close(t, t7[idx])  # type:ignore[attr-defined]

        t8 = [torch.LongTensor([[1, 1], [2, 2]]), torch.DoubleTensor([[1.5, 6.8], [2.9, 9.2]])]
        for idx, t in enumerate(
            model_signature._SeqOfPyTorchTensorHandler.convert_from_df(
                model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t8)
            )
        ):
            torch.testing.assert_close(t, t8[idx])  # type:ignore[attr-defined]

        t9 = [torch.IntTensor([1, 2, 3, 4])]
        fts = model_signature._SeqOfPyTorchTensorHandler.infer_signature(t9, role="input")
        for idx, t in enumerate(
            model_signature._SeqOfPyTorchTensorHandler.convert_from_df(
                model_signature._rename_pandas_df(model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t9), fts),
                fts,
            )
        ):
            torch.testing.assert_close(t, t9[idx])  # type:ignore[attr-defined]

        t10 = [torch.tensor([1.2, 3.4])]
        fts = model_signature._SeqOfPyTorchTensorHandler.infer_signature(t10, role="input")
        for idx, t in enumerate(
            model_signature._SeqOfPyTorchTensorHandler.convert_from_df(
                model_signature._rename_pandas_df(model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t10), fts),
                fts,
            )
        ):
            torch.testing.assert_close(t, t10[idx])  # type:ignore[attr-defined]

        t11 = [torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]])]
        fts = model_signature._SeqOfPyTorchTensorHandler.infer_signature(t11, role="input")
        for idx, t in enumerate(
            model_signature._SeqOfPyTorchTensorHandler.convert_from_df(
                model_signature._rename_pandas_df(model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t11), fts),
                fts,
            )
        ):
            torch.testing.assert_close(t, t11[idx])  # type:ignore[attr-defined]

        t12 = [torch.tensor([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])]
        fts = model_signature._SeqOfPyTorchTensorHandler.infer_signature(t12, role="input")
        for idx, t in enumerate(
            model_signature._SeqOfPyTorchTensorHandler.convert_from_df(
                model_signature._rename_pandas_df(model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t12), fts),
                fts,
            )
        ):
            torch.testing.assert_close(t, t12[idx])  # type:ignore[attr-defined]

        t13 = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        fts = model_signature._SeqOfPyTorchTensorHandler.infer_signature(t13, role="input")
        for idx, t in enumerate(
            model_signature._SeqOfPyTorchTensorHandler.convert_from_df(
                model_signature._rename_pandas_df(model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t13), fts),
                fts,
            )
        ):
            torch.testing.assert_close(t, t13[idx])  # type:ignore[attr-defined]

        t14 = [torch.tensor([1.2, 2.4]), torch.tensor([3, 4])]
        fts = model_signature._SeqOfPyTorchTensorHandler.infer_signature(t14, role="input")
        for idx, t in enumerate(
            model_signature._SeqOfPyTorchTensorHandler.convert_from_df(
                model_signature._rename_pandas_df(model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t14), fts),
                fts,
            )
        ):
            torch.testing.assert_close(t, t14[idx])  # type:ignore[attr-defined]

        t15 = [torch.tensor([[1, 1], [2, 2]]), torch.tensor([[3, 3], [4, 4]])]
        fts = model_signature._SeqOfPyTorchTensorHandler.infer_signature(t15, role="input")
        for idx, t in enumerate(
            model_signature._SeqOfPyTorchTensorHandler.convert_from_df(
                model_signature._rename_pandas_df(model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t15), fts),
                fts,
            )
        ):
            torch.testing.assert_close(t, t15[idx])  # type:ignore[attr-defined]

        t16 = [torch.tensor([[1, 1], [2, 2]]), torch.tensor([[1.5, 6.8], [2.9, 9.2]])]
        fts = model_signature._SeqOfPyTorchTensorHandler.infer_signature(t16, role="input")
        for idx, t in enumerate(
            model_signature._SeqOfPyTorchTensorHandler.convert_from_df(
                model_signature._rename_pandas_df(model_signature._SeqOfPyTorchTensorHandler.convert_to_df(t16), fts),
                fts,
            )
        ):
            torch.testing.assert_close(t, t16[idx])  # type:ignore[attr-defined]


class SeqOfTensorflowTensorHandlerTest(absltest.TestCase):
    def test_validate_list_of_tf_tensor(self) -> None:
        lt1 = [np.array([1, 4]), np.array([2, 3])]
        self.assertFalse(model_signature._SeqOfTensorflowTensorHandler.can_handle(lt1))

        lt2 = [np.array([1, 4]), tf.constant([2, 3])]
        self.assertFalse(model_signature._SeqOfTensorflowTensorHandler.can_handle(lt2))

        lt3 = [tf.constant([1, 4]), tf.constant([2, 3])]
        self.assertTrue(model_signature._SeqOfTensorflowTensorHandler.can_handle(lt3))

        lt4 = [tf.constant([1, 4]), tf.Variable([2, 3])]
        self.assertTrue(model_signature._SeqOfTensorflowTensorHandler.can_handle(lt4))

        lt5 = [tf.Variable([1, 4]), tf.Variable([2, 3])]
        self.assertTrue(model_signature._SeqOfTensorflowTensorHandler.can_handle(lt5))

    def test_validate_tf_tensor(self) -> None:
        t = [tf.constant([])]
        with self.assertRaisesRegex(ValueError, "Empty data is found."):
            model_signature._SeqOfTensorflowTensorHandler.validate(t)

        t = [tf.Variable([1, 2], shape=tf.TensorShape(None))]
        with self.assertRaisesRegex(ValueError, "Unknown shape data is found."):
            model_signature._SeqOfTensorflowTensorHandler.validate(t)

        t = [tf.Variable([[1, 2]], shape=tf.TensorShape([None, 2]))]
        with self.assertRaisesRegex(ValueError, "Unknown shape data is found."):
            model_signature._SeqOfTensorflowTensorHandler.validate(t)

        t = [tf.Variable([[1, 2]], shape=tf.TensorShape([1, None]))]
        with self.assertRaisesRegex(ValueError, "Unknown shape data is found."):
            model_signature._SeqOfTensorflowTensorHandler.validate(t)

        t = [tf.constant(1)]
        with self.assertRaisesRegex(ValueError, "Scalar data is found."):
            model_signature._SeqOfTensorflowTensorHandler.validate(t)

        t = [tf.constant([1])]
        with self.assertRaisesRegex(ValueError, "Scalar data is found."):
            model_signature._SeqOfTensorflowTensorHandler.validate(t)

        t = [tf.Variable(1)]
        with self.assertRaisesRegex(ValueError, "Scalar data is found."):
            model_signature._SeqOfTensorflowTensorHandler.validate(t)

        t = [tf.Variable([1])]
        with self.assertRaisesRegex(ValueError, "Scalar data is found."):
            model_signature._SeqOfTensorflowTensorHandler.validate(t)

        t = [tf.constant([1, 2]), tf.constant(1)]
        with self.assertRaisesRegex(ValueError, "Scalar data is found."):
            model_signature._SeqOfTensorflowTensorHandler.validate(t)

    def test_count_tf_tensor(self) -> None:
        t = [tf.constant([1, 2])]
        self.assertEqual(model_signature._SeqOfTensorflowTensorHandler.count(t), 2)

        t = [tf.constant([[1, 2]])]
        self.assertEqual(model_signature._SeqOfTensorflowTensorHandler.count(t), 1)

        t = [tf.Variable([1, 2])]
        self.assertEqual(model_signature._SeqOfTensorflowTensorHandler.count(t), 2)

        t = [tf.Variable([1, 2], shape=tf.TensorShape(None))]
        with self.assertRaisesRegex(ValueError, "Unknown shape data is found."):
            model_signature._SeqOfTensorflowTensorHandler.validate(t)

        t = [tf.Variable([[1, 2]], shape=tf.TensorShape([None, 2]))]
        with self.assertRaisesRegex(ValueError, "Unknown shape data is found."):
            model_signature._SeqOfTensorflowTensorHandler.validate(t)

        t = [tf.Variable([[1, 2]], shape=tf.TensorShape([1, None]))]
        self.assertEqual(model_signature._SeqOfTensorflowTensorHandler.count(t), 1)

    def test_trunc_tf_tensor(self) -> None:
        t = [tf.constant([1] * (model_signature._SeqOfTensorflowTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT + 1))]

        for ts in model_signature._SeqOfTensorflowTensorHandler.truncate(t):
            tf.assert_equal(
                tf.constant([1] * (model_signature._SeqOfTensorflowTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT)), ts
            )

        t = [tf.constant([1] * (model_signature._SeqOfTensorflowTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT - 1))]

        for ts in model_signature._SeqOfTensorflowTensorHandler.truncate(t):
            tf.assert_equal(
                tf.constant([1] * (model_signature._SeqOfTensorflowTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT - 1)), ts
            )

        t = [tf.constant([1] * (model_signature._SeqOfTensorflowTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT + 1))] * 2

        for ts in model_signature._SeqOfTensorflowTensorHandler.truncate(t):
            tf.assert_equal(
                tf.constant([1] * (model_signature._SeqOfTensorflowTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT)), ts
            )

        t = [
            tf.constant([1]),
            tf.constant([1] * (model_signature._SeqOfTensorflowTensorHandler.SIG_INFER_ROWS_COUNT_LIMIT - 1)),
        ]

        for ts in model_signature._SeqOfTensorflowTensorHandler.truncate(t):
            tf.assert_equal(tf.constant([1]), ts)

    def test_infer_schema_tf_tensor(self) -> None:
        t1 = [tf.constant([1, 2, 3, 4], dtype=tf.int32)]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t1, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT32)],
        )

        t2 = [tf.constant([1, 2, 3, 4], dtype=tf.int64)]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t2, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64)],
        )

        t3 = [tf.constant([1, 2, 3, 4], dtype=tf.int16)]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t3, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT16)],
        )

        t4 = [tf.constant([1, 2, 3, 4], dtype=tf.int8)]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t4, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT8)],
        )

        t5 = [tf.constant([1, 2, 3, 4], dtype=tf.uint32)]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t5, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.UINT32)],
        )

        t6 = [tf.constant([1, 2, 3, 4], dtype=tf.uint64)]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t6, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.UINT64)],
        )

        t7 = [tf.constant([1, 2, 3, 4], dtype=tf.uint16)]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t7, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.UINT16)],
        )

        t8 = [tf.constant([1, 2, 3, 4], dtype=tf.uint8)]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t8, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.UINT8)],
        )

        t9 = [tf.constant([False, True])]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t9, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.BOOL)],
        )

        t10 = [tf.constant([1.2, 3.4], dtype=tf.float32)]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t10, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.FLOAT)],
        )

        t11 = [tf.constant([1.2, 3.4], dtype=tf.float64)]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t11, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.DOUBLE)],
        )

        t12 = [tf.constant([[1, 2], [3, 4]])]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t12, role="input"),
            [
                model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT32, shape=(2,)),
            ],
        )

        t13 = [tf.constant([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t13, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT32, shape=(2, 2))],
        )

        t14 = [tf.constant([1, 2, 3, 4])]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t14, role="output"),
            [model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT32)],
        )

        t15 = [tf.constant([1, 2]), tf.constant([3, 4])]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t15, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT32),
                model_signature.FeatureSpec("output_feature_1", model_signature.DataType.INT32),
            ],
        )

        t16 = [tf.constant([1.2, 2.4]), tf.constant([3, 4])]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t16, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.FLOAT),
                model_signature.FeatureSpec("output_feature_1", model_signature.DataType.INT32),
            ],
        )

        t17 = [tf.constant([[1, 1], [2, 2]]), tf.constant([[3, 3], [4, 4]])]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t17, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT32, shape=(2,)),
                model_signature.FeatureSpec("output_feature_1", model_signature.DataType.INT32, shape=(2,)),
            ],
        )

        t18 = [tf.constant([[1, 1], [2, 2]]), tf.constant([[1.5, 6.8], [2.9, 9.2]])]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t18, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT32, shape=(2,)),
                model_signature.FeatureSpec("output_feature_1", model_signature.DataType.FLOAT, shape=(2,)),
            ],
        )

        t21 = [tf.constant([1, 2, 3, 4], dtype=tf.int32)]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t21, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT32)],
        )

        t22 = [tf.constant([1, 2, 3, 4], dtype=tf.int64)]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t22, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64)],
        )

        t23 = [tf.constant([1, 2, 3, 4], dtype=tf.int16)]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t23, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT16)],
        )

        t24 = [tf.constant([1, 2, 3, 4], dtype=tf.int8)]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t24, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT8)],
        )

        t25 = [tf.constant([1, 2, 3, 4], dtype=tf.uint32)]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t25, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.UINT32)],
        )

        t26 = [tf.constant([1, 2, 3, 4], dtype=tf.uint64)]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t26, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.UINT64)],
        )

        t27 = [tf.constant([1, 2, 3, 4], dtype=tf.uint16)]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t27, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.UINT16)],
        )

        t28 = [tf.constant([1, 2, 3, 4], dtype=tf.uint8)]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t28, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.UINT8)],
        )

        t29 = [tf.constant([False, True])]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t29, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.BOOL)],
        )

        t30 = [tf.constant([1.2, 3.4], dtype=tf.float32)]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t30, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.FLOAT)],
        )

        t31 = [tf.constant([1.2, 3.4], dtype=tf.float64)]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t31, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.DOUBLE)],
        )

        t32 = [tf.constant([[1, 2], [3, 4]])]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t32, role="input"),
            [
                model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT32, shape=(2,)),
            ],
        )

        t33 = [tf.constant([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t33, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT32, shape=(2, 2))],
        )

        t34 = [tf.constant([1, 2, 3, 4])]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t34, role="output"),
            [model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT32)],
        )

        t35 = [tf.constant([1, 2]), tf.constant([3, 4])]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t35, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT32),
                model_signature.FeatureSpec("output_feature_1", model_signature.DataType.INT32),
            ],
        )

        t36 = [tf.constant([1.2, 2.4]), tf.constant([3, 4])]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t36, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.FLOAT),
                model_signature.FeatureSpec("output_feature_1", model_signature.DataType.INT32),
            ],
        )

        t37 = [tf.constant([[1, 1], [2, 2]]), tf.constant([[3, 3], [4, 4]])]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t37, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT32, shape=(2,)),
                model_signature.FeatureSpec("output_feature_1", model_signature.DataType.INT32, shape=(2,)),
            ],
        )

        t38 = [tf.constant([[1, 1], [2, 2]]), tf.constant([[1.5, 6.8], [2.9, 9.2]])]
        self.assertListEqual(
            model_signature._SeqOfTensorflowTensorHandler.infer_signature(t38, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT32, shape=(2,)),
                model_signature.FeatureSpec("output_feature_1", model_signature.DataType.FLOAT, shape=(2,)),
            ],
        )

    def test_convert_to_df_tf_tensor(self) -> None:
        t1 = [tf.constant([1, 2, 3, 4], dtype=tf.int64)]
        pd.testing.assert_frame_equal(
            model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t1),
            pd.DataFrame([1, 2, 3, 4]),
        )

        t2 = [tf.Variable([1, 2, 3, 4], dtype=tf.int64)]
        pd.testing.assert_frame_equal(
            model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t2),
            pd.DataFrame([1, 2, 3, 4]),
        )

        t3 = [tf.constant([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=tf.int64)]
        pd.testing.assert_frame_equal(
            model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t3),
            pd.DataFrame({0: [np.array([1, 1]), np.array([2, 2]), np.array([3, 3]), np.array([4, 4])]}),
        )

        t4 = [tf.constant([[[1, 1], [2, 2]], [[3, 3], [4, 4]]], dtype=tf.int64)]
        pd.testing.assert_frame_equal(
            model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t4),
            pd.DataFrame(data={0: [np.array([[1, 1], [2, 2]]), np.array([[3, 3], [4, 4]])]}),
        )

        t5 = [tf.constant([1, 2], dtype=tf.int64), tf.constant([3, 4], dtype=tf.int64)]
        pd.testing.assert_frame_equal(
            model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t5),
            pd.DataFrame([[1, 3], [2, 4]]),
        )

        t6 = [tf.constant([1.2, 2.4], dtype=tf.float64), tf.constant([3, 4], dtype=tf.int64)]
        pd.testing.assert_frame_equal(
            model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t6),
            pd.DataFrame([[1.2, 3], [2.4, 4]]),
        )

        t7 = [tf.constant([[1, 1], [2, 2]], dtype=tf.int64), tf.constant([[3, 3], [4, 4]], dtype=tf.int64)]
        pd.testing.assert_frame_equal(
            model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t7),
            pd.DataFrame({0: [np.array([1, 1]), np.array([2, 2])], 1: [np.array([3, 3]), np.array([4, 4])]}),
        )

        t8 = [tf.constant([[1, 1], [2, 2]], dtype=tf.int64), tf.constant([[1.5, 6.8], [2.9, 9.2]], dtype=tf.float64)]
        pd.testing.assert_frame_equal(
            model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t8),
            pd.DataFrame({0: [np.array([1, 1]), np.array([2, 2])], 1: [np.array([1.5, 6.8]), np.array([2.9, 9.2])]}),
        )

    def test_convert_from_df_tf_tensor(self) -> None:
        t1 = [tf.constant([1, 2, 3, 4], dtype=tf.int64)]
        for idx, t in enumerate(
            model_signature._SeqOfTensorflowTensorHandler.convert_from_df(
                model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t1)
            )
        ):
            tf.assert_equal(t, t1[idx])

        t2 = [tf.Variable([1, 2, 3, 4], dtype=tf.int64)]
        for idx, t in enumerate(
            model_signature._SeqOfTensorflowTensorHandler.convert_from_df(
                model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t2)
            )
        ):
            tf.assert_equal(t, t2[idx])

        t3 = [tf.constant([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=tf.int64)]
        for idx, t in enumerate(
            model_signature._SeqOfTensorflowTensorHandler.convert_from_df(
                model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t3)
            )
        ):
            tf.assert_equal(t, t3[idx])

        t4 = [tf.constant([[[1, 1], [2, 2]], [[3, 3], [4, 4]]], dtype=tf.int64)]
        for idx, t in enumerate(
            model_signature._SeqOfTensorflowTensorHandler.convert_from_df(
                model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t4)
            )
        ):
            tf.assert_equal(t, t4[idx])

        t5 = [tf.constant([1, 2], dtype=tf.int64), tf.constant([3, 4], dtype=tf.int64)]
        for idx, t in enumerate(
            model_signature._SeqOfTensorflowTensorHandler.convert_from_df(
                model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t5)
            )
        ):
            tf.assert_equal(t, t5[idx])

        t6 = [tf.constant([1.2, 2.4], dtype=tf.float64), tf.constant([3, 4], dtype=tf.int64)]
        for idx, t in enumerate(
            model_signature._SeqOfTensorflowTensorHandler.convert_from_df(
                model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t6)
            )
        ):
            tf.assert_equal(t, t6[idx])

        t7 = [tf.constant([[1, 1], [2, 2]], dtype=tf.int64), tf.constant([[3, 3], [4, 4]], dtype=tf.int64)]
        for idx, t in enumerate(
            model_signature._SeqOfTensorflowTensorHandler.convert_from_df(
                model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t7)
            )
        ):
            tf.assert_equal(t, t7[idx])

        t8 = [tf.constant([[1, 1], [2, 2]], dtype=tf.int64), tf.constant([[1.5, 6.8], [2.9, 9.2]], dtype=tf.float64)]
        for idx, t in enumerate(
            model_signature._SeqOfTensorflowTensorHandler.convert_from_df(
                model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t8)
            )
        ):
            tf.assert_equal(t, t8[idx])

        t9 = [tf.constant([1, 2, 3, 4])]
        fts = model_signature._SeqOfTensorflowTensorHandler.infer_signature(t9, role="input")
        for idx, t in enumerate(
            model_signature._SeqOfTensorflowTensorHandler.convert_from_df(
                model_signature._rename_pandas_df(model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t9), fts),
                fts,
            )
        ):
            tf.assert_equal(t, t9[idx])

        t10 = [tf.constant([1.2, 3.4])]
        fts = model_signature._SeqOfTensorflowTensorHandler.infer_signature(t10, role="input")
        for idx, t in enumerate(
            model_signature._SeqOfTensorflowTensorHandler.convert_from_df(
                model_signature._rename_pandas_df(
                    model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t10), fts
                ),
                fts,
            )
        ):
            tf.assert_equal(t, t10[idx])

        t11 = [tf.constant([[1, 1], [2, 2], [3, 3], [4, 4]])]
        fts = model_signature._SeqOfTensorflowTensorHandler.infer_signature(t11, role="input")
        for idx, t in enumerate(
            model_signature._SeqOfTensorflowTensorHandler.convert_from_df(
                model_signature._rename_pandas_df(
                    model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t11), fts
                ),
                fts,
            )
        ):
            tf.assert_equal(t, t11[idx])

        t12 = [tf.constant([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])]
        fts = model_signature._SeqOfTensorflowTensorHandler.infer_signature(t12, role="input")
        for idx, t in enumerate(
            model_signature._SeqOfTensorflowTensorHandler.convert_from_df(
                model_signature._rename_pandas_df(
                    model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t12), fts
                ),
                fts,
            )
        ):
            tf.assert_equal(t, t12[idx])

        t13 = [tf.constant([1, 2]), tf.constant([3, 4])]
        fts = model_signature._SeqOfTensorflowTensorHandler.infer_signature(t13, role="input")
        for idx, t in enumerate(
            model_signature._SeqOfTensorflowTensorHandler.convert_from_df(
                model_signature._rename_pandas_df(
                    model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t13), fts
                ),
                fts,
            )
        ):
            tf.assert_equal(t, t13[idx])

        t14 = [tf.constant([1.2, 2.4]), tf.constant([3, 4])]
        fts = model_signature._SeqOfTensorflowTensorHandler.infer_signature(t14, role="input")
        for idx, t in enumerate(
            model_signature._SeqOfTensorflowTensorHandler.convert_from_df(
                model_signature._rename_pandas_df(
                    model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t14), fts
                ),
                fts,
            )
        ):
            tf.assert_equal(t, t14[idx])

        t15 = [tf.constant([[1, 1], [2, 2]]), tf.constant([[3, 3], [4, 4]])]
        fts = model_signature._SeqOfTensorflowTensorHandler.infer_signature(t15, role="input")
        for idx, t in enumerate(
            model_signature._SeqOfTensorflowTensorHandler.convert_from_df(
                model_signature._rename_pandas_df(
                    model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t15), fts
                ),
                fts,
            )
        ):
            tf.assert_equal(t, t15[idx])

        t16 = [tf.constant([[1, 1], [2, 2]]), tf.constant([[1.5, 6.8], [2.9, 9.2]])]
        fts = model_signature._SeqOfTensorflowTensorHandler.infer_signature(t16, role="input")
        for idx, t in enumerate(
            model_signature._SeqOfTensorflowTensorHandler.convert_from_df(
                model_signature._rename_pandas_df(
                    model_signature._SeqOfTensorflowTensorHandler.convert_to_df(t16), fts
                ),
                fts,
            )
        ):
            tf.assert_equal(t, t16[idx])


class SnowParkDataFrameHandlerTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._session.close()

    def test_validate_snowpark_df(self) -> None:
        schema = spt.StructType([spt.StructField('"a"', spt.VariantType()), spt.StructField('"b"', spt.StringType())])
        df = self._session.create_dataframe([[1, "snow"], [3, "flake"]], schema)
        with self.assertRaisesRegex(ValueError, "Unsupported data type"):
            model_signature._SnowparkDataFrameHandler.validate(df)

    def test_infer_schema_snowpark_df(self) -> None:
        schema = spt.StructType([spt.StructField('"a"', spt.LongType()), spt.StructField('"b"', spt.StringType())])
        df = self._session.create_dataframe([[1, "snow"], [3, "flake"]], schema)
        self.assertListEqual(
            model_signature._SnowparkDataFrameHandler.infer_signature(df, role="input"),
            [
                model_signature.FeatureSpec("a", model_signature.DataType.INT64),
                model_signature.FeatureSpec("b", model_signature.DataType.STRING),
            ],
        )

        schema = spt.StructType([spt.StructField('"""a"""', spt.LongType()), spt.StructField('"b"', spt.StringType())])
        df = self._session.create_dataframe([[1, "snow"], [3, "flake"]], schema)
        self.assertListEqual(
            model_signature._SnowparkDataFrameHandler.infer_signature(df, role="input"),
            [
                model_signature.FeatureSpec('"a"', model_signature.DataType.INT64),
                model_signature.FeatureSpec("b", model_signature.DataType.STRING),
            ],
        )

        schema = spt.StructType([spt.StructField('"""a"""', spt.ArrayType(spt.LongType()))])
        df = self._session.create_dataframe([[[1, 3]]], schema)
        with self.assertRaises(NotImplementedError):
            model_signature._SnowparkDataFrameHandler.infer_signature(df, role="input"),

    def test_validate_data_with_features(self) -> None:
        fts = [
            model_signature.FeatureSpec("a", model_signature.DataType.INT64),
            model_signature.FeatureSpec("b", model_signature.DataType.INT64),
        ]
        df = self._session.create_dataframe([{'"a"': 1}, {'"b"': 2}])
        with self.assertWarnsRegex(RuntimeWarning, "Nullable column [^\\s]* provided"):
            model_signature._validate_snowpark_data(df, fts)

        fts = [
            model_signature.FeatureSpec("a", model_signature.DataType.INT64),
            model_signature.FeatureSpec("b", model_signature.DataType.STRING),
        ]
        schema = spt.StructType([spt.StructField('"a"', spt.LongType()), spt.StructField('"b"', spt.StringType())])
        df = self._session.create_dataframe([[1, "snow"], [3, "flake"]], schema)
        model_signature._validate_snowpark_data(df, fts)

        schema = spt.StructType([spt.StructField('"a"', spt.LongType()), spt.StructField('"b"', spt.IntegerType())])
        df = self._session.create_dataframe([[1, 3], [3, 9]], schema)
        with self.assertRaisesRegex(ValueError, "Feature type [^\\s]* is not met by column"):
            model_signature._validate_snowpark_data(df, fts)

        schema = spt.StructType([spt.StructField('"a1"', spt.LongType()), spt.StructField('"b"', spt.StringType())])
        df = self._session.create_dataframe([[1, "snow"], [3, "flake"]], schema)
        with self.assertRaisesRegex(ValueError, "feature [^\\s]* does not exist in data."):
            model_signature._validate_snowpark_data(df, fts)

        df = self._session.create_dataframe([{'"a"': 1}, {'"b"': 2}])
        with self.assertRaisesRegex(ValueError, "Feature type [^\\s]* is not met by column"):
            model_signature._validate_snowpark_data(df, fts)

        fts = [
            model_signature.FeatureSpec("a", model_signature.DataType.INT64, shape=(-1,)),
        ]
        schema = spt.StructType([spt.StructField('"a"', spt.ArrayType(spt.LongType()))])
        df = self._session.create_dataframe([[[1, 3]]], schema)
        with self.assertWarns(RuntimeWarning):
            model_signature._validate_snowpark_data(df, fts)

    def test_convert_to_and_from_df(self) -> None:
        pd_df = pd.DataFrame([1, 2, 3, 4], columns=["col_0"])
        sp_df = model_signature._SnowparkDataFrameHandler.convert_from_df(self._session, pd_df, keep_order=False)
        pd.testing.assert_frame_equal(
            pd_df, model_signature._SnowparkDataFrameHandler.convert_to_df(sp_df), check_dtype=False
        )

        pd_df = pd.DataFrame([[1, 2], [3, 4]], columns=["col_0", "col_1"])
        sp_df = model_signature._SnowparkDataFrameHandler.convert_from_df(self._session, pd_df, keep_order=False)
        pd.testing.assert_frame_equal(
            pd_df, model_signature._SnowparkDataFrameHandler.convert_to_df(sp_df), check_dtype=False
        )

        pd_df = pd.DataFrame([[1.2, 2.4], [3, 4]], columns=["col_0", "col_1"])
        sp_df = model_signature._SnowparkDataFrameHandler.convert_from_df(self._session, pd_df, keep_order=False)
        pd.testing.assert_frame_equal(
            pd_df, model_signature._SnowparkDataFrameHandler.convert_to_df(sp_df), check_dtype=False
        )

        pd_df = pd.DataFrame([[1, [[2.5], [6.8]]], [2, [[2.5], [6.8]]]], columns=["a", "b"])
        sp_df = model_signature._SnowparkDataFrameHandler.convert_from_df(self._session, pd_df, keep_order=False)
        pd.testing.assert_frame_equal(
            pd_df, model_signature._SnowparkDataFrameHandler.convert_to_df(sp_df), check_dtype=False
        )

        a = np.array([2.5, 6.8])
        pd_df = pd.DataFrame([[1, a], [2, a]], columns=["a", "b"])
        sp_df = model_signature._SnowparkDataFrameHandler.convert_from_df(self._session, pd_df, keep_order=False)
        pd.testing.assert_frame_equal(
            pd_df, model_signature._SnowparkDataFrameHandler.convert_to_df(sp_df), check_dtype=False
        )


class ModelSignatureMiscTest(absltest.TestCase):
    def test_rename_features(self) -> None:
        model_signature._rename_features([])

        fts = [model_signature.FeatureSpec("a", model_signature.DataType.INT64)]
        self.assertListEqual(
            model_signature._rename_features(fts, ["b"]),
            [model_signature.FeatureSpec("b", model_signature.DataType.INT64)],
        )

        fts = [model_signature.FeatureSpec("a", model_signature.DataType.INT64, shape=(2,))]
        self.assertListEqual(
            model_signature._rename_features(fts, ["b"]),
            [model_signature.FeatureSpec("b", model_signature.DataType.INT64, shape=(2,))],
        )

        fts = [model_signature.FeatureSpec("a", model_signature.DataType.INT64, shape=(2,))]
        model_signature._rename_features(fts)

        with self.assertRaises(ValueError):
            fts = [model_signature.FeatureSpec("a", model_signature.DataType.INT64, shape=(2,))]
            model_signature._rename_features(fts, ["b", "c"])

    def test_infer_signature(self) -> None:
        df = pd.DataFrame([1, 2, 3, 4])
        self.assertListEqual(
            model_signature._infer_signature(df, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64)],
        )

        arr = np.array([1, 2, 3, 4])
        self.assertListEqual(
            model_signature._infer_signature(arr, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64)],
        )

        lt1 = [1, 2, 3, 4]
        self.assertListEqual(
            model_signature._infer_signature(lt1, role="input"),
            [model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64)],
        )

        lt2 = [[1, 2], [3, 4]]
        self.assertListEqual(
            model_signature._infer_signature(lt2, role="input"),
            [
                model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("input_feature_1", model_signature.DataType.INT64),
            ],
        )

        lt3 = [arr, arr]
        self.assertListEqual(
            model_signature._infer_signature(lt3, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("output_feature_1", model_signature.DataType.INT64),
            ],
        )

        self.assertListEqual(
            model_signature._infer_signature(lt3, role="input"),
            [
                model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("input_feature_1", model_signature.DataType.INT64),
            ],
        )

        torch_tensor = torch.LongTensor([1, 2, 3, 4])
        lt4 = [torch_tensor, torch_tensor]
        self.assertListEqual(
            model_signature._infer_signature(lt4, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("output_feature_1", model_signature.DataType.INT64),
            ],
        )

        self.assertListEqual(
            model_signature._infer_signature(lt4, role="input"),
            [
                model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("input_feature_1", model_signature.DataType.INT64),
            ],
        )

        tf_tensor = tf.constant([1, 2, 3, 4], dtype=tf.int64)
        lt5 = [tf_tensor, tf_tensor]
        self.assertListEqual(
            model_signature._infer_signature(lt5, role="output"),
            [
                model_signature.FeatureSpec("output_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("output_feature_1", model_signature.DataType.INT64),
            ],
        )

        self.assertListEqual(
            model_signature._infer_signature(lt5, role="input"),
            [
                model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("input_feature_1", model_signature.DataType.INT64),
            ],
        )

        df = pd.DataFrame([1, 2, 3, 4])
        lt = [df, arr]
        with self.assertRaises(NotImplementedError):
            model_signature._infer_signature(lt, role="input")

        with self.assertRaises(ValueError):
            model_signature._infer_signature([True, 1], role="input")

        with self.assertRaises(NotImplementedError):
            model_signature._infer_signature(1, role="input")

        with self.assertRaises(NotImplementedError):
            model_signature._infer_signature([], role="input")

    def test_validate_pandas_df(self) -> None:
        fts = [
            model_signature.FeatureSpec("a", model_signature.DataType.INT64),
            model_signature.FeatureSpec("b", model_signature.DataType.INT64),
        ]

        model_signature._validate_pandas_df(pd.DataFrame([[2, 5], [6, 8]], columns=["a", "b"]), fts)

        with self.assertRaisesRegex(ValueError, "Feature type [^\\s]* is not met by all elements"):
            model_signature._validate_pandas_df(pd.DataFrame([[2.5, 5], [6.8, 8]], columns=["a", "b"]), fts)

        with self.assertRaisesRegex(ValueError, "feature [^\\s]* does not exist in data."):
            model_signature._validate_pandas_df(pd.DataFrame([5, 6], columns=["a"]), fts)

        model_signature._validate_pandas_df(pd.DataFrame([5, 6], columns=["a"]), fts[:1])

        with self.assertRaisesRegex(ValueError, "feature [^\\s]* does not exist in data."):
            model_signature._validate_pandas_df(pd.DataFrame([[2, 5], [6, 8]], columns=["c", "d"]), fts)

        with self.assertRaisesRegex(ValueError, "Feature is a scalar feature while list data is provided."):
            model_signature._validate_pandas_df(
                pd.DataFrame([[1, [2.5, 6.8]], [2, [2.5, 6.8]]], columns=["a", "b"]), fts
            )

        fts = [
            model_signature.FeatureSpec("a", model_signature.DataType.INT64),
            model_signature.FeatureSpec("b", model_signature.DataType.DOUBLE, shape=(2,)),
        ]

        model_signature._validate_pandas_df(pd.DataFrame([[1, [2.5, 6.8]], [2, [2.5, 6.8]]], columns=["a", "b"]), fts)

        with self.assertRaisesRegex(ValueError, "Feature is a array type feature while scalar data is provided."):
            model_signature._validate_pandas_df(pd.DataFrame([[2, 2.5], [6, 6.8]], columns=["a", "b"]), fts)

        with self.assertRaisesRegex(ValueError, "Feature shape [\\(\\)0-9,\\s-]* is not met by all elements"):
            model_signature._validate_pandas_df(
                pd.DataFrame([[1, [2.5, 6.8, 6.8]], [2, [2.5, 6.8, 6.8]]], columns=["a", "b"]), fts
            )

        with self.assertRaisesRegex(ValueError, "Feature shape [\\(\\)0-9,\\s-]* is not met by all elements"):
            model_signature._validate_pandas_df(
                pd.DataFrame([[1, [2.5, 6.8]], [2, [2.5, 6.8, 6.8]]], columns=["a", "b"]), fts
            )

        with self.assertRaisesRegex(ValueError, "Feature type [^\\s]* is not met by all elements"):
            model_signature._validate_pandas_df(pd.DataFrame([[1, [2, 5]], [2, [6, 8]]], columns=["a", "b"]), fts)

        model_signature._validate_pandas_df(
            pd.DataFrame([[1, np.array([2.5, 6.8])], [2, np.array([2.5, 6.8])]], columns=["a", "b"]), fts
        )

        with self.assertRaisesRegex(ValueError, "Feature shape [\\(\\)0-9,\\s-]* is not met by all elements"):
            model_signature._validate_pandas_df(
                pd.DataFrame([[1, np.array([2.5, 6.8, 6.8])], [2, np.array([2.5, 6.8, 6.8])]], columns=["a", "b"]), fts
            )

        with self.assertRaisesRegex(ValueError, "Feature shape [\\(\\)0-9,\\s-]* is not met by all elements"):
            model_signature._validate_pandas_df(
                pd.DataFrame([[1, np.array([2.5, 6.8])], [2, np.array([2.5, 6.8, 6.8])]], columns=["a", "b"]), fts
            )

        with self.assertRaisesRegex(ValueError, "Feature type [^\\s]* is not met by all elements"):
            model_signature._validate_pandas_df(
                pd.DataFrame([[1, np.array([2, 5])], [2, np.array([6, 8])]], columns=["a", "b"]), fts
            )

        with self.assertRaisesRegex(ValueError, "Feature is a array type feature while scalar data is provided."):
            model_signature._validate_pandas_df(
                pd.DataFrame([ele.encode() for ele in ["a", "b", "c", "d"]], columns=["b"]), fts[-1:]
            )

        with self.assertRaisesRegex(ValueError, "Feature is a array type feature while scalar data is provided."):
            model_signature._validate_pandas_df(pd.DataFrame(["a", "b", "c", "d"], columns=["b"]), fts[-1:])

        fts = [
            model_signature.FeatureSpec("a", model_signature.DataType.INT64),
            model_signature.FeatureSpec("b", model_signature.DataType.DOUBLE, shape=(-1,)),
        ]

        model_signature._validate_pandas_df(pd.DataFrame([[1, [2.5, 6.8]], [2, [2.5, 6.8]]], columns=["a", "b"]), fts)

        model_signature._validate_pandas_df(
            pd.DataFrame([[1, [2.5, 6.8, 6.8]], [2, [2.5, 6.8, 6.8]]], columns=["a", "b"]), fts
        )

        model_signature._validate_pandas_df(
            pd.DataFrame([[1, [2.5, 6.8]], [2, [2.5, 6.8, 6.8]]], columns=["a", "b"]), fts
        )

        with self.assertRaisesRegex(ValueError, "Feature type [^\\s]* is not met by all elements"):
            model_signature._validate_pandas_df(pd.DataFrame([[1, [2, 5]], [2, [6, 8]]], columns=["a", "b"]), fts)

        model_signature._validate_pandas_df(
            pd.DataFrame([[1, np.array([2.5, 6.8])], [2, np.array([2.5, 6.8])]], columns=["a", "b"]), fts
        )

        model_signature._validate_pandas_df(
            pd.DataFrame([[1, np.array([2.5, 6.8, 6.8])], [2, np.array([2.5, 6.8, 6.8])]], columns=["a", "b"]), fts
        )

        model_signature._validate_pandas_df(
            pd.DataFrame([[1, np.array([2.5, 6.8])], [2, np.array([2.5, 6.8, 6.8])]], columns=["a", "b"]), fts
        )

        with self.assertRaisesRegex(ValueError, "Feature type [^\\s]* is not met by all elements"):
            model_signature._validate_pandas_df(
                pd.DataFrame([[1, np.array([2, 5])], [2, np.array([6, 8])]], columns=["a", "b"]), fts
            )

        fts = [
            model_signature.FeatureSpec("a", model_signature.DataType.INT64),
            model_signature.FeatureSpec("b", model_signature.DataType.DOUBLE, shape=(2, 1)),
        ]

        model_signature._validate_pandas_df(
            pd.DataFrame([[1, [[2.5], [6.8]]], [2, [[2.5], [6.8]]]], columns=["a", "b"]), fts
        )

        with self.assertRaisesRegex(ValueError, "Feature shape [\\(\\)0-9,\\s-]* is not met by all elements"):
            model_signature._validate_pandas_df(
                pd.DataFrame([[1, [[2.5], [6.8]]], [2, [[2.5], [6.8], [6.8]]]], columns=["a", "b"]), fts
            )

        with self.assertRaisesRegex(ValueError, "Feature shape [\\(\\)0-9,\\s-]* is not met by all elements"):
            model_signature._validate_pandas_df(
                pd.DataFrame([[1, [2.5, 6.8]], [2, [2.5, 6.8]]], columns=["a", "b"]), fts
            )

        with self.assertRaisesRegex(ValueError, "Feature type [^\\s]* is not met by all elements"):
            model_signature._validate_pandas_df(
                pd.DataFrame([[1, [[2], [5]]], [2, [[6], [8]]]], columns=["a", "b"]), fts
            )

        model_signature._validate_pandas_df(
            pd.DataFrame([[1, np.array([[2.5], [6.8]])], [2, np.array([[2.5], [6.8]])]], columns=["a", "b"]), fts
        )

        with self.assertRaisesRegex(ValueError, "Feature shape [\\(\\)0-9,\\s-]* is not met by all elements"):
            model_signature._validate_pandas_df(
                pd.DataFrame([[1, np.array([[2.5], [6.8]])], [2, np.array([[2.5], [6.8], [6.8]])]], columns=["a", "b"]),
                fts,
            )

        with self.assertRaisesRegex(ValueError, "Feature shape [\\(\\)0-9,\\s-]* is not met by all elements"):
            model_signature._validate_pandas_df(
                pd.DataFrame([[1, np.array([2.5, 6.8])], [2, np.array([2.5, 6.8])]], columns=["a", "b"]), fts
            )

        with self.assertRaisesRegex(ValueError, "Feature type [^\\s]* is not met by all elements"):
            model_signature._validate_pandas_df(
                pd.DataFrame([[1, np.array([[2], [5]])], [2, np.array([[6], [8]])]], columns=["a", "b"]), fts
            )

        fts = [model_signature.FeatureSpec("a", model_signature.DataType.STRING)]
        model_signature._validate_pandas_df(pd.DataFrame(["a", "b", "c", "d"], columns=["a"]), fts)

        with self.assertRaisesRegex(ValueError, "Feature type [^\\s]* is not met by all elements"):
            model_signature._validate_pandas_df(
                pd.DataFrame([ele.encode() for ele in ["a", "b", "c", "d"]], columns=["a"]), fts
            )

        with self.assertRaisesRegex(ValueError, "Feature is a scalar feature while list data is provided."):
            model_signature._validate_pandas_df(pd.DataFrame(data={"a": [[1, 2]]}), fts)

        with self.assertRaisesRegex(ValueError, "Feature is a scalar feature while array data is provided."):
            model_signature._validate_pandas_df(pd.DataFrame(data={"a": [np.array([1, 2])]}), fts)

        fts = [model_signature.FeatureSpec("a", model_signature.DataType.BYTES)]
        model_signature._validate_pandas_df(
            pd.DataFrame([ele.encode() for ele in ["a", "b", "c", "d"]], columns=["a"]), fts
        )

        with self.assertRaisesRegex(ValueError, "Feature type [^\\s]* is not met by all elements"):
            model_signature._validate_pandas_df(pd.DataFrame(["a", "b", "c", "d"], columns=["a"]), fts)

        with self.assertRaisesRegex(ValueError, "Feature is a scalar feature while list data is provided."):
            model_signature._validate_pandas_df(pd.DataFrame(data={"a": [[1, 2]]}), fts)

        with self.assertRaisesRegex(ValueError, "Feature is a scalar feature while array data is provided."):
            model_signature._validate_pandas_df(pd.DataFrame(data={"a": [np.array([1, 2])]}), fts)

    def test_rename_pandas_df(self) -> None:
        fts = [
            model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64),
            model_signature.FeatureSpec("input_feature_1", model_signature.DataType.INT64),
        ]

        df = pd.DataFrame([[2, 5], [6, 8]], columns=["a", "b"])

        pd.testing.assert_frame_equal(df, model_signature._rename_pandas_df(df, fts))

        df = pd.DataFrame([[2, 5], [6, 8]])

        pd.testing.assert_frame_equal(df, model_signature._rename_pandas_df(df, fts), check_names=False)
        pd.testing.assert_index_equal(
            pd.Index(["input_feature_0", "input_feature_1"]), model_signature._rename_pandas_df(df, fts).columns
        )

    def test_validate_data_with_features(self) -> None:
        fts = [
            model_signature.FeatureSpec("input_feature_0", model_signature.DataType.INT64),
            model_signature.FeatureSpec("input_feature_1", model_signature.DataType.INT64),
        ]

        with self.assertRaisesRegex(ValueError, "Empty data is found."):
            model_signature._convert_and_validate_local_data(np.array([]), fts)

        with self.assertRaisesRegex(ValueError, "Scalar data is found."):
            model_signature._convert_and_validate_local_data(np.array(5), fts)

        with self.assertRaisesRegex(ValueError, "Feature type [^\\s]* is not met by all elements"):
            model_signature._convert_and_validate_local_data(np.array([[2.5, 5], [6.8, 8]]), fts)

        with self.assertRaisesRegex(ValueError, "Un-supported type <class 'list'> provided."):
            model_signature._convert_and_validate_local_data([], fts)

        with self.assertRaisesRegex(ValueError, "Inconsistent type of object found in data"):
            model_signature._convert_and_validate_local_data([1, [1, 1]], fts)

        with self.assertRaisesRegex(ValueError, "Ill-shaped list data"):
            model_signature._convert_and_validate_local_data([[1], [1, 1]], fts)

        with self.assertRaisesRegex(ValueError, "Feature type [^\\s]* is not met by all elements"):
            model_signature._convert_and_validate_local_data([[2.1, 5.0], [6.8, 8.0]], fts)

        with self.assertRaisesRegex(ValueError, "Feature type [^\\s]* is not met by all elements"):
            model_signature._convert_and_validate_local_data(pd.DataFrame([[2.5, 5], [6.8, 8]]), fts)

        with self.assertRaisesRegex(ValueError, "Data does not have the same number of features as signature"):
            model_signature._convert_and_validate_local_data(pd.DataFrame([5, 6]), fts)

        with self.assertRaisesRegex(ValueError, "Data does not have the same number of features as signature."):
            model_signature._convert_and_validate_local_data(np.array([5, 6]), fts)

        with self.assertRaisesRegex(ValueError, "feature [^\\s]* does not exist in data."):
            model_signature._convert_and_validate_local_data(pd.DataFrame([[2, 5], [6, 8]], columns=["a", "b"]), fts)

        df = model_signature._convert_and_validate_local_data(np.array([5, 6]), fts[:1])
        self.assertListEqual(df.columns.to_list(), ["input_feature_0"])

        df = model_signature._convert_and_validate_local_data(pd.DataFrame([5, 6]), fts[:1])
        self.assertListEqual(df.columns.to_list(), ["input_feature_0"])

        df = model_signature._convert_and_validate_local_data([5, 6], fts[:1])
        self.assertListEqual(df.columns.to_list(), ["input_feature_0"])

        df = model_signature._convert_and_validate_local_data(np.array([[2, 5], [6, 8]]), fts)
        self.assertListEqual(df.columns.to_list(), ["input_feature_0", "input_feature_1"])

        df = model_signature._convert_and_validate_local_data(pd.DataFrame([[2, 5], [6, 8]]), fts)
        self.assertListEqual(df.columns.to_list(), ["input_feature_0", "input_feature_1"])

        df = model_signature._convert_and_validate_local_data(
            pd.DataFrame([[2, 5], [6, 8]], columns=["a", "b"]),
            [
                model_signature.FeatureSpec("a", model_signature.DataType.INT64),
                model_signature.FeatureSpec("b", model_signature.DataType.INT64),
            ],
        )
        self.assertListEqual(df.columns.to_list(), ["a", "b"])

        df = model_signature._convert_and_validate_local_data([[2, 5], [6, 8]], fts)
        self.assertListEqual(df.columns.to_list(), ["input_feature_0", "input_feature_1"])


if __name__ == "__main__":
    absltest.main()
