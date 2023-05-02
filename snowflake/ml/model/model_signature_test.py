import numpy as np
import pandas as pd
from absl.testing import absltest

import snowflake.snowpark.types as spt
from snowflake.ml.model import model_signature


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
        with self.assertRaises(ValueError):
            _ = model_signature.FeatureGroupSpec(name="features", specs=[])

        with self.assertRaises(ValueError):
            ft1 = model_signature.FeatureSpec(name="feature1", dtype=model_signature.DataType.INT64)
            ft2 = model_signature.FeatureSpec(name="feature2", dtype=model_signature.DataType.INT64)
            ft2._name = None  # type: ignore[assignment]
            _ = model_signature.FeatureGroupSpec(name="features", specs=[ft1, ft2])

        with self.assertRaises(ValueError):
            ft1 = model_signature.FeatureSpec(name="feature1", dtype=model_signature.DataType.INT64)
            ft2 = model_signature.FeatureSpec(name="feature2", dtype=model_signature.DataType.FLOAT)
            _ = model_signature.FeatureGroupSpec(name="features", specs=[ft1, ft2])

        with self.assertRaises(ValueError):
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

    def test_infer_signature_pd_DataFrame(self) -> None:
        df = pd.DataFrame([])
        with self.assertRaises(ValueError):
            self.assertEmpty(model_signature._infer_signature_pd_DataFrame(df))

        df = pd.DataFrame([1, 2, 3, 4])
        self.assertListEqual(
            model_signature._infer_signature_pd_DataFrame(df),
            [model_signature.FeatureSpec("feature_0", model_signature.DataType.INT64)],
        )

        df = pd.DataFrame([1, 2, 3, 4], columns=["a"])
        self.assertListEqual(
            model_signature._infer_signature_pd_DataFrame(df),
            [model_signature.FeatureSpec("a", model_signature.DataType.INT64)],
        )

        df = pd.DataFrame(["a", "b", "c", "d"], columns=["a"])
        self.assertListEqual(
            model_signature._infer_signature_pd_DataFrame(df),
            [model_signature.FeatureSpec("a", model_signature.DataType.STRING)],
        )

        df = pd.DataFrame([ele.encode() for ele in ["a", "b", "c", "d"]], columns=["a"])
        self.assertListEqual(
            model_signature._infer_signature_pd_DataFrame(df),
            [model_signature.FeatureSpec("a", model_signature.DataType.BYTES)],
        )

        df = pd.DataFrame([[1, 2.0], [2, 4.0]])
        self.assertListEqual(
            model_signature._infer_signature_pd_DataFrame(df),
            [
                model_signature.FeatureSpec("feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("feature_1", model_signature.DataType.DOUBLE),
            ],
        )

        df = pd.DataFrame([[1, [2.5, 6.8]], [2, [2.5, 6.8]]], columns=["a", "b"])
        self.assertListEqual(
            model_signature._infer_signature_pd_DataFrame(df),
            [
                model_signature.FeatureSpec("a", model_signature.DataType.INT64),
                model_signature.FeatureSpec("b", model_signature.DataType.DOUBLE, shape=(2,)),
            ],
        )

        df = pd.DataFrame([[1, [2.5, 6.8]], [2, [2, 6]]], columns=["a", "b"])
        with self.assertRaises(ValueError):
            model_signature._infer_signature_pd_DataFrame(df)

        df = pd.DataFrame([[1, [2.5, 6.8]], [2, [2.5]]], columns=["a", "b"])
        with self.assertRaises(NotImplementedError):
            model_signature._infer_signature_pd_DataFrame(df)

        df = pd.DataFrame([[1, [[2.5], [6.8]]], [2, [[2.5], [6.8]]]], columns=["a", "b"])
        self.assertListEqual(
            model_signature._infer_signature_pd_DataFrame(df),
            [
                model_signature.FeatureSpec("a", model_signature.DataType.INT64),
                model_signature.FeatureSpec("b", model_signature.DataType.DOUBLE, shape=(2, 1)),
            ],
        )

        a = np.array([2.5, 6.8])
        df = pd.DataFrame([[1, a], [2, a]], columns=["a", "b"])
        self.assertListEqual(
            model_signature._infer_signature_pd_DataFrame(df),
            [
                model_signature.FeatureSpec("a", model_signature.DataType.INT64),
                model_signature.FeatureSpec("b", model_signature.DataType.DOUBLE, shape=(2,)),
            ],
        )

        df = pd.DataFrame([[1, np.array([2.5, 6.8])], [2, np.array([2, 6])]], columns=["a", "b"])
        with self.assertRaises(ValueError):
            model_signature._infer_signature_pd_DataFrame(df)

        df = pd.DataFrame([[1, np.array([2.5, 6.8])], [2, np.array([2.5])]], columns=["a", "b"])
        with self.assertRaises(NotImplementedError):
            model_signature._infer_signature_pd_DataFrame(df)

        a = np.array([[2, 5], [6, 8]])
        df = pd.DataFrame([[1, a], [2, a]], columns=["a", "b"])
        self.assertListEqual(
            model_signature._infer_signature_pd_DataFrame(df),
            [
                model_signature.FeatureSpec("a", model_signature.DataType.INT64),
                model_signature.FeatureSpec("b", model_signature.DataType.INT64, shape=(2, 2)),
            ],
        )

        a = pd.DataFrame([2.5, 6.8])
        df = pd.DataFrame([[1, a], [2, a]], columns=["a", "b"])
        with self.assertRaises(NotImplementedError):
            model_signature._infer_signature_pd_DataFrame(df)

        df = pd.DataFrame([[1, 2.0], [2, 4.0]], columns=pd.PeriodIndex(year=[2000, 2002], quarter=[1, 3]))
        self.assertListEqual(
            model_signature._infer_signature_pd_DataFrame(df),
            [
                model_signature.FeatureSpec("2000Q1", model_signature.DataType.INT64),
                model_signature.FeatureSpec("2002Q3", model_signature.DataType.DOUBLE),
            ],
        )

        df = pd.DataFrame([[1, 2.0], [2, 4.0]], columns=pd.date_range("2020-01-06", "2020-03-03", freq="MS"))
        self.assertListEqual(
            model_signature._infer_signature_pd_DataFrame(df),
            [
                model_signature.FeatureSpec("2020-02-01 00:00:00", model_signature.DataType.INT64),
                model_signature.FeatureSpec("2020-03-01 00:00:00", model_signature.DataType.DOUBLE),
            ],
        )

        df = pd.DataFrame(
            [[1, 2.0], [2, 4.0]], columns=pd.TimedeltaIndex(data=["1 days 02:00:00", "1 days 06:05:01.000030"])
        )
        self.assertListEqual(
            model_signature._infer_signature_pd_DataFrame(df),
            [
                model_signature.FeatureSpec("1 days 02:00:00", model_signature.DataType.INT64),
                model_signature.FeatureSpec("1 days 06:05:01.000030", model_signature.DataType.DOUBLE),
            ],
        )

        df = pd.DataFrame([[1, 2.0], [2, 4.0]], columns=pd.interval_range(start=0, end=2))
        self.assertListEqual(
            model_signature._infer_signature_pd_DataFrame(df),
            [
                model_signature.FeatureSpec("(0, 1]", model_signature.DataType.INT64),
                model_signature.FeatureSpec("(1, 2]", model_signature.DataType.DOUBLE),
            ],
        )

        df = pd.DataFrame(
            [[1, 2.0, 1, 2.0, 1, 2.0], [2, 4.0, 2, 4.0, 2, 4.0]],
            columns=pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"]),
        )
        with self.assertRaises(NotImplementedError):
            model_signature._infer_signature_pd_DataFrame(df)

        arrays = [[1, 2], ["red", "blue"]]
        df = pd.DataFrame([[1, 2.0], [2, 4.0]], columns=pd.MultiIndex.from_arrays(arrays, names=("number", "color")))
        print(model_signature._infer_signature_pd_DataFrame(df))
        self.assertListEqual(
            model_signature._infer_signature_pd_DataFrame(df),
            [
                model_signature.FeatureSpec("(1, 'red')", model_signature.DataType.INT64),
                model_signature.FeatureSpec("(2, 'blue')", model_signature.DataType.DOUBLE),
            ],
        )

    def test_infer_signature_np_ndarray(self) -> None:
        arr = np.array([])
        with self.assertRaises(ValueError):
            self.assertEmpty(model_signature._infer_signature_np_ndarray(arr))

        arr = np.array([1, 2, 3, 4])
        self.assertListEqual(
            model_signature._infer_signature_np_ndarray(arr),
            [model_signature.FeatureSpec("feature_0", model_signature.DataType.INT64)],
        )

        arr = np.array([[1, 2], [3, 4]])
        self.assertListEqual(
            model_signature._infer_signature_np_ndarray(arr),
            [
                model_signature.FeatureSpec("feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("feature_1", model_signature.DataType.INT64),
            ],
        )

        arr = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
        self.assertListEqual(
            model_signature._infer_signature_np_ndarray(arr),
            [
                model_signature.FeatureSpec("feature_0", model_signature.DataType.INT64, shape=(2,)),
                model_signature.FeatureSpec("feature_1", model_signature.DataType.INT64, shape=(2,)),
            ],
        )

    def test_infer_signature_list_of(self) -> None:
        arr = np.array([1, 2, 3, 4])
        lt = [arr, arr]
        self.assertListEqual(
            model_signature._infer_signature_list_multioutput(lt),
            [
                model_signature.FeatureSpec("output_0_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("output_1_feature_0", model_signature.DataType.INT64),
            ],
        )

        arr = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
        lt = [arr, arr]
        self.assertListEqual(
            model_signature._infer_signature_list_multioutput(lt),
            [
                model_signature.FeatureSpec("output_0_feature_0", model_signature.DataType.INT64, shape=(2,)),
                model_signature.FeatureSpec("output_0_feature_1", model_signature.DataType.INT64, shape=(2,)),
                model_signature.FeatureSpec("output_1_feature_0", model_signature.DataType.INT64, shape=(2,)),
                model_signature.FeatureSpec("output_1_feature_1", model_signature.DataType.INT64, shape=(2,)),
            ],
        )

    def test_infer_signature_list_builtins(self) -> None:
        lt1 = [1, 2, 3, 4]
        self.assertListEqual(
            model_signature._infer_signature_list_builtins(lt1),
            [model_signature.FeatureSpec("feature_0", model_signature.DataType.INT64)],
        )

        lt2 = ["a", "b", "c", "d"]
        self.assertListEqual(
            model_signature._infer_signature_list_builtins(lt2),
            [model_signature.FeatureSpec("feature_0", model_signature.DataType.STRING)],
        )

        lt3 = [ele.encode() for ele in lt2]
        self.assertListEqual(
            model_signature._infer_signature_list_builtins(lt3),
            [model_signature.FeatureSpec("feature_0", model_signature.DataType.BYTES)],
        )

        lt4 = [[1, 2], [3, 4]]
        self.assertListEqual(
            model_signature._infer_signature_list_builtins(lt4),
            [
                model_signature.FeatureSpec("feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("feature_1", model_signature.DataType.INT64),
            ],
        )

        lt5 = [[1, 2.0], [3, 4]]
        self.assertListEqual(
            model_signature._infer_signature_list_builtins(lt5),
            [
                model_signature.FeatureSpec("feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("feature_1", model_signature.DataType.DOUBLE),
            ],
        )

        lt6 = [[[1, 1], [2, 2]], [[3, 3], [4, 4]]]
        self.assertListEqual(
            model_signature._infer_signature_list_builtins(lt6),
            [
                model_signature.FeatureSpec("feature_0", model_signature.DataType.INT64, shape=(2,)),
                model_signature.FeatureSpec("feature_1", model_signature.DataType.INT64, shape=(2,)),
            ],
        )

        lt7 = [[1], [2, 3]]
        with self.assertRaises(ValueError):
            model_signature._infer_signature_list_builtins(lt7)

        lt8 = [pd.DataFrame([1]), pd.DataFrame([2, 3])]
        with self.assertRaises(NotImplementedError):
            model_signature._infer_signature_list_builtins(lt8)

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
            model_signature._infer_signature(df),
            [model_signature.FeatureSpec("feature_0", model_signature.DataType.INT64)],
        )

        arr = np.array([1, 2, 3, 4])
        self.assertListEqual(
            model_signature._infer_signature(arr),
            [model_signature.FeatureSpec("feature_0", model_signature.DataType.INT64)],
        )

        lt1 = [1, 2, 3, 4]
        self.assertListEqual(
            model_signature._infer_signature(lt1),
            [model_signature.FeatureSpec("feature_0", model_signature.DataType.INT64)],
        )

        lt2 = [[1, 2], [3, 4]]
        self.assertListEqual(
            model_signature._infer_signature(lt2),
            [
                model_signature.FeatureSpec("feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("feature_1", model_signature.DataType.INT64),
            ],
        )

        lt = [arr, arr]
        self.assertListEqual(
            model_signature._infer_signature(lt, is_output=True),
            [
                model_signature.FeatureSpec("output_0_feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("output_1_feature_0", model_signature.DataType.INT64),
            ],
        )

        self.assertListEqual(
            model_signature._infer_signature(lt),
            [
                model_signature.FeatureSpec("feature_0", model_signature.DataType.INT64),
                model_signature.FeatureSpec("feature_1", model_signature.DataType.INT64),
                model_signature.FeatureSpec("feature_2", model_signature.DataType.INT64),
                model_signature.FeatureSpec("feature_3", model_signature.DataType.INT64),
            ],
        )

        df = pd.DataFrame([1, 2, 3, 4])
        lt = [df, arr]
        with self.assertRaises(ValueError):
            model_signature._infer_signature(lt)

        with self.assertRaises(ValueError):
            model_signature._infer_signature([True, 1])

        with self.assertRaises(NotImplementedError):
            model_signature._infer_signature(1)

        with self.assertRaises(NotImplementedError):
            model_signature._infer_signature([])

    def test_validate_data_with_features(self) -> None:
        fts = [
            model_signature.FeatureSpec("feature_0", model_signature.DataType.INT64),
            model_signature.FeatureSpec("feature_1", model_signature.DataType.INT64),
        ]

        with self.assertRaises(ValueError):
            model_signature._validate_data_with_features_and_convert_to_df(fts, np.array([]))

        with self.assertRaises(ValueError):
            model_signature._validate_data_with_features_and_convert_to_df(fts, np.array(5))

        with self.assertRaises(NotImplementedError):
            model_signature._validate_data_with_features_and_convert_to_df(fts, [])

        with self.assertRaises(ValueError):
            model_signature._validate_data_with_features_and_convert_to_df(fts, [1, [1, 1]])

        with self.assertRaises(ValueError):
            model_signature._validate_data_with_features_and_convert_to_df(fts, [[1], [1, 1]])

        with self.assertRaises(ValueError):
            model_signature._validate_data_with_features_and_convert_to_df(fts, pd.DataFrame([5, 6]))

        with self.assertRaises(ValueError):
            model_signature._validate_data_with_features_and_convert_to_df(fts, np.array([5, 6]))

        df = model_signature._validate_data_with_features_and_convert_to_df(fts[:1], np.array([5, 6]))
        self.assertListEqual(df.columns.to_list(), ["feature_0"])

        df = model_signature._validate_data_with_features_and_convert_to_df(fts[:1], pd.DataFrame([5, 6]))
        self.assertListEqual(df.columns.to_list(), ["feature_0"])

        df = model_signature._validate_data_with_features_and_convert_to_df(fts[:1], [5, 6])
        self.assertListEqual(df.columns.to_list(), ["feature_0"])

        df = model_signature._validate_data_with_features_and_convert_to_df(fts, np.array([[2, 5], [6, 8]]))
        self.assertListEqual(df.columns.to_list(), ["feature_0", "feature_1"])

        df = model_signature._validate_data_with_features_and_convert_to_df(fts, pd.DataFrame([[2, 5], [6, 8]]))
        self.assertListEqual(df.columns.to_list(), ["feature_0", "feature_1"])

        df = model_signature._validate_data_with_features_and_convert_to_df(
            fts, pd.DataFrame([[2, 5], [6, 8]], columns=["a", "b"])
        )
        self.assertListEqual(df.columns.to_list(), ["a", "b"])

        df = model_signature._validate_data_with_features_and_convert_to_df(fts, [[2, 5], [6, 8]])
        self.assertListEqual(df.columns.to_list(), ["feature_0", "feature_1"])


if __name__ == "__main__":
    absltest.main()
