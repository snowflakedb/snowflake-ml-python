import numpy as np
from absl.testing import absltest

import snowflake.snowpark.types as spt
from snowflake.ml.model._signatures import core
from snowflake.ml.test_utils import exception_utils


class DataTypeTest(absltest.TestCase):
    def test_numpy_type(self) -> None:
        data = np.array([1, 2, 3, 4])
        self.assertEqual(core.DataType.INT64, core.DataType.from_numpy_type(data.dtype))

        data = np.array(["a", "b", "c", "d"])
        self.assertEqual(core.DataType.STRING, core.DataType.from_numpy_type(data.dtype))

    def test_snowpark_type(self) -> None:
        self.assertEqual(core.DataType.INT8, core.DataType.from_snowpark_type(spt.ByteType()))
        self.assertEqual(core.DataType.INT16, core.DataType.from_snowpark_type(spt.ShortType()))
        self.assertEqual(core.DataType.INT32, core.DataType.from_snowpark_type(spt.IntegerType()))
        self.assertEqual(core.DataType.INT64, core.DataType.from_snowpark_type(spt.LongType()))

        self.assertEqual(core.DataType.INT64, core.DataType.from_snowpark_type(spt.DecimalType(38, 0)))

        self.assertEqual(core.DataType.FLOAT, core.DataType.from_snowpark_type(spt.FloatType()))
        self.assertEqual(core.DataType.DOUBLE, core.DataType.from_snowpark_type(spt.DoubleType()))

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=NotImplementedError,
            expected_regex="Type .+ is not supported as a DataType.",
        ):
            core.DataType.from_snowpark_type(spt.DecimalType(38, 6))

        self.assertEqual(core.DataType.BOOL, core.DataType.from_snowpark_type(spt.BooleanType()))
        self.assertEqual(core.DataType.STRING, core.DataType.from_snowpark_type(spt.StringType()))
        self.assertEqual(core.DataType.BYTES, core.DataType.from_snowpark_type(spt.BinaryType()))

        self.assertTrue(core.DataType.INT64.is_same_snowpark_type(spt.LongType()))
        self.assertTrue(core.DataType.INT32.is_same_snowpark_type(spt.IntegerType()))
        self.assertTrue(core.DataType.INT16.is_same_snowpark_type(spt.ShortType()))
        self.assertTrue(core.DataType.INT8.is_same_snowpark_type(spt.ByteType()))
        self.assertTrue(core.DataType.UINT64.is_same_snowpark_type(spt.LongType()))
        self.assertTrue(core.DataType.UINT32.is_same_snowpark_type(spt.IntegerType()))
        self.assertTrue(core.DataType.UINT16.is_same_snowpark_type(spt.ShortType()))
        self.assertTrue(core.DataType.UINT8.is_same_snowpark_type(spt.ByteType()))

        self.assertTrue(core.DataType.FLOAT.is_same_snowpark_type(spt.FloatType()))
        self.assertTrue(core.DataType.DOUBLE.is_same_snowpark_type(spt.DoubleType()))

        self.assertTrue(core.DataType.INT64.is_same_snowpark_type(incoming_snowpark_type=spt.DecimalType(38, 0)))
        self.assertTrue(core.DataType.UINT64.is_same_snowpark_type(incoming_snowpark_type=spt.DecimalType(38, 0)))


class FeatureSpecTest(absltest.TestCase):
    def test_feature_spec(self) -> None:
        ft = core.FeatureSpec(name="feature", dtype=core.DataType.INT64)
        self.assertEqual(ft, eval(repr(ft), core.__dict__))
        self.assertEqual(ft, core.FeatureSpec.from_dict(ft.to_dict()))
        self.assertEqual(ft.as_snowpark_type(), spt.LongType())

        ft = core.FeatureSpec(name="feature", dtype=core.DataType.INT64, shape=(2,))
        self.assertEqual(ft, eval(repr(ft), core.__dict__))
        self.assertEqual(ft, core.FeatureSpec.from_dict(input_dict=ft.to_dict()))
        self.assertEqual(ft.as_snowpark_type(), spt.ArrayType(spt.LongType()))


class FeatureGroupSpecTest(absltest.TestCase):
    def test_feature_group_spec(self) -> None:
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="No children feature specs."
        ):
            _ = core.FeatureGroupSpec(name="features", specs=[])

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="All children feature specs have to have name.",
        ):
            ft1 = core.FeatureSpec(name="feature1", dtype=core.DataType.INT64)
            ft2 = core.FeatureSpec(name="feature2", dtype=core.DataType.INT64)
            ft2._name = None  # type: ignore[assignment]
            _ = core.FeatureGroupSpec(name="features", specs=[ft1, ft2])

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="All children feature specs have to have same type.",
        ):
            ft1 = core.FeatureSpec(name="feature1", dtype=core.DataType.INT64)
            ft2 = core.FeatureSpec(name="feature2", dtype=core.DataType.FLOAT)
            _ = core.FeatureGroupSpec(name="features", specs=[ft1, ft2])

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="All children feature specs have to have same shape.",
        ):
            ft1 = core.FeatureSpec(name="feature1", dtype=core.DataType.INT64)
            ft2 = core.FeatureSpec(name="feature2", dtype=core.DataType.INT64, shape=(2,))
            fts = core.FeatureGroupSpec(name="features", specs=[ft1, ft2])

        ft1 = core.FeatureSpec(name="feature1", dtype=core.DataType.INT64)
        ft2 = core.FeatureSpec(name="feature2", dtype=core.DataType.INT64)
        fts = core.FeatureGroupSpec(name="features", specs=[ft1, ft2])
        self.assertEqual(fts, eval(repr(fts), core.__dict__))
        self.assertEqual(fts, core.FeatureGroupSpec.from_dict(fts.to_dict()))
        self.assertEqual(fts.as_snowpark_type(), spt.MapType(spt.StringType(), spt.LongType()))

        ft1 = core.FeatureSpec(name="feature1", dtype=core.DataType.INT64, shape=(3,))
        ft2 = core.FeatureSpec(name="feature2", dtype=core.DataType.INT64, shape=(2,))
        fts = core.FeatureGroupSpec(name="features", specs=[ft1, ft2])
        self.assertEqual(fts, eval(repr(fts), core.__dict__))
        self.assertEqual(fts, core.FeatureGroupSpec.from_dict(fts.to_dict()))
        self.assertEqual(fts.as_snowpark_type(), spt.MapType(spt.StringType(), spt.ArrayType(spt.LongType())))


class ModelSignatureTest(absltest.TestCase):
    def test_1(self) -> None:
        s = core.ModelSignature(
            inputs=[
                core.FeatureSpec(dtype=core.DataType.FLOAT, name="c1"),
                core.FeatureGroupSpec(
                    name="cg1",
                    specs=[
                        core.FeatureSpec(
                            dtype=core.DataType.FLOAT,
                            name="cc1",
                        ),
                        core.FeatureSpec(
                            dtype=core.DataType.FLOAT,
                            name="cc2",
                        ),
                    ],
                ),
                core.FeatureSpec(dtype=core.DataType.FLOAT, name="c2", shape=(-1,)),
            ],
            outputs=[core.FeatureSpec(name="output", dtype=core.DataType.FLOAT)],
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
        self.assertEqual(s, eval(repr(s), core.__dict__))
        self.assertEqual(s, core.ModelSignature.from_dict(s.to_dict()))

    def test_2(self) -> None:
        s = core.ModelSignature(
            inputs=[
                core.FeatureSpec(dtype=core.DataType.FLOAT, name="c1"),
                core.FeatureGroupSpec(
                    name="cg1",
                    specs=[
                        core.FeatureSpec(
                            dtype=core.DataType.FLOAT,
                            name="cc1",
                        ),
                        core.FeatureSpec(
                            dtype=core.DataType.FLOAT,
                            name="cc2",
                        ),
                    ],
                ),
                core.FeatureSpec(dtype=core.DataType.FLOAT, name="c2", shape=(-1,)),
            ],
            outputs=[core.FeatureSpec(name="output", dtype=core.DataType.FLOAT)],
        )
        self.assertEqual(s, eval(repr(s), core.__dict__))
        self.assertEqual(s, core.ModelSignature.from_dict(s.to_dict()))


if __name__ == "__main__":
    absltest.main()
