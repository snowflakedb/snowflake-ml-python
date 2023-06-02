import numpy as np
import pandas as pd
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


class FeatureSpecTest(absltest.TestCase):
    def test_feature_spec(self) -> None:
        ft = model_signature.FeatureSpec(name="feature", dtype=model_signature.DataType.INT64)
        self.assertEqual(ft, eval(repr(ft), model_signature.__dict__))
        self.assertEqual(ft, model_signature.FeatureSpec.from_dict(ft.to_dict()))
        self.assertEqual(ft.as_snowpark_type(), spt.IntegerType())

        ft = model_signature.FeatureSpec(name="feature", dtype=model_signature.DataType.INT64, shape=(2,))
        self.assertEqual(ft, eval(repr(ft), model_signature.__dict__))
        self.assertEqual(ft, model_signature.FeatureSpec.from_dict(ft.to_dict()))
        self.assertEqual(ft.as_snowpark_type(), spt.ArrayType(spt.IntegerType()))


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
        self.assertEqual(fts.as_snowpark_type(), spt.MapType(spt.StringType(), spt.IntegerType()))

        ft1 = model_signature.FeatureSpec(name="feature1", dtype=model_signature.DataType.INT64, shape=(3,))
        ft2 = model_signature.FeatureSpec(name="feature2", dtype=model_signature.DataType.INT64, shape=(2,))
        fts = model_signature.FeatureGroupSpec(name="features", specs=[ft1, ft2])
        self.assertEqual(fts, eval(repr(fts), model_signature.__dict__))
        self.assertEqual(fts, model_signature.FeatureGroupSpec.from_dict(fts.to_dict()))
        self.assertEqual(fts.as_snowpark_type(), spt.MapType(spt.StringType(), spt.ArrayType(spt.IntegerType())))


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


class ListOfNumpyArrayHandlerTest(absltest.TestCase):
    def test_validate_list_of_numpy_array(self) -> None:
        lt8 = [pd.DataFrame([1]), pd.DataFrame([2, 3])]
        self.assertFalse(model_signature._ListOfNumpyArrayHandler.can_handle(lt8))

    def test_trunc_np_ndarray(self) -> None:
        arrs = [np.array([1] * (model_signature._ListOfNumpyArrayHandler.SIG_INFER_ROWS_COUNT_LIMIT + 1))] * 2

        for arr in model_signature._ListOfNumpyArrayHandler.truncate(arrs):
            np.testing.assert_equal(
                np.array([1] * (model_signature._ListOfNumpyArrayHandler.SIG_INFER_ROWS_COUNT_LIMIT)), arr
            )

        arrs = [
            np.array([1]),
            np.array([1] * (model_signature._ListOfNumpyArrayHandler.SIG_INFER_ROWS_COUNT_LIMIT - 1)),
        ]

        for arr in model_signature._ListOfNumpyArrayHandler.truncate(arrs):
            np.testing.assert_equal(np.array([1]), arr)

    def test_infer_signature_list_of_numpy_array(self) -> None:
        arr = np.array([1, 2, 3, 4])
        lt = [arr, arr]
        self.assertListEqual(
            model_signature._ListOfNumpyArrayHandler.infer_signature(lt, role="input"),
            [
                model_signature.FeatureSpec("input_0_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("input_1_feature_0", model_signature.DataType.INT64),
            ],
        )

        arr = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
        lt = [arr, arr]
        self.assertListEqual(
            model_signature._ListOfNumpyArrayHandler.infer_signature(lt, role="output"),
            [
                model_signature.FeatureSpec("output_0_feature_0", model_signature.DataType.INT64, shape=(2,)),
                model_signature.FeatureSpec("output_0_feature_1", model_signature.DataType.INT64, shape=(2,)),
                model_signature.FeatureSpec("output_1_feature_0", model_signature.DataType.INT64, shape=(2,)),
                model_signature.FeatureSpec("output_1_feature_1", model_signature.DataType.INT64, shape=(2,)),
            ],
        )

    def test_convert_to_df_list_of_numpy_array(self) -> None:
        arr1 = np.array([1, 2, 3, 4])
        lt = [arr1, arr1]
        pd.testing.assert_frame_equal(
            model_signature._ListOfNumpyArrayHandler.convert_to_df(lt),
            pd.DataFrame([[1, 1], [2, 2], [3, 3], [4, 4]]),
            check_names=False,
        )

        arr2 = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        lt = [arr1, arr2]
        pd.testing.assert_frame_equal(
            model_signature._ListOfNumpyArrayHandler.convert_to_df(lt),
            pd.DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]),
        )

        arr = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
        lt = [arr, arr]
        pd.testing.assert_frame_equal(
            model_signature._ListOfNumpyArrayHandler.convert_to_df(lt),
            pd.DataFrame(
                data={
                    0: [np.array([1, 1]), np.array([3, 3])],
                    1: [np.array([2, 2]), np.array([4, 4])],
                    2: [np.array([1, 1]), np.array([3, 3])],
                    3: [np.array([2, 2]), np.array([4, 4])],
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

        lt = [arr, arr]
        self.assertListEqual(
            model_signature._infer_signature(lt, role="output"),
            [
                model_signature.FeatureSpec("output_0_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("output_1_feature_0", model_signature.DataType.INT64),
            ],
        )

        self.assertListEqual(
            model_signature._infer_signature(lt, role="input"),
            [
                model_signature.FeatureSpec("input_0_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("input_1_feature_0", model_signature.DataType.INT64),
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
