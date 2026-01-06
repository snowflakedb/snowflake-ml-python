import numpy as np
import pandas as pd
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

        data = pd.Series([1, 2, 3, 4]).convert_dtypes()
        self.assertEqual(core.DataType.INT64, core.DataType.from_numpy_type(data.dtype))

        data = pd.Series(["a", "b", "c", "d"]).convert_dtypes()
        self.assertEqual(core.DataType.STRING, core.DataType.from_numpy_type(data.dtype))

    def test_snowpark_type(self) -> None:
        self.assertEqual(core.DataType.INT8, core.DataType.from_snowpark_type(spt.ByteType()))
        self.assertEqual(core.DataType.INT16, core.DataType.from_snowpark_type(spt.ShortType()))
        self.assertEqual(core.DataType.INT32, core.DataType.from_snowpark_type(spt.IntegerType()))
        self.assertEqual(core.DataType.INT64, core.DataType.from_snowpark_type(spt.LongType()))

        self.assertEqual(core.DataType.INT64, core.DataType.from_snowpark_type(spt.DecimalType(38, 0)))

        self.assertEqual(core.DataType.FLOAT, core.DataType.from_snowpark_type(spt.FloatType()))
        self.assertEqual(core.DataType.DOUBLE, core.DataType.from_snowpark_type(spt.DoubleType()))

        self.assertEqual(core.DataType.DOUBLE, core.DataType.from_snowpark_type(spt.DecimalType(38, 6)))
        self.assertEqual(core.DataType.BOOL, core.DataType.from_snowpark_type(spt.BooleanType()))
        self.assertEqual(core.DataType.STRING, core.DataType.from_snowpark_type(spt.StringType()))
        self.assertEqual(core.DataType.BYTES, core.DataType.from_snowpark_type(spt.BinaryType()))

    def test_python_type(self) -> None:
        """Test conversion from Python built-in types to DataType."""
        self.assertEqual(core.DataType.INT64, core.DataType.from_python_type(int))
        self.assertEqual(core.DataType.DOUBLE, core.DataType.from_python_type(float))
        self.assertEqual(core.DataType.STRING, core.DataType.from_python_type(str))
        self.assertEqual(core.DataType.BOOL, core.DataType.from_python_type(bool))

    def test_python_type_unsupported(self) -> None:
        """Test that unsupported Python types raise NotImplementedError."""
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=NotImplementedError, expected_regex="not supported as a DataType"
        ):
            core.DataType.from_python_type(list)

        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=NotImplementedError, expected_regex="not supported as a DataType"
        ):
            core.DataType.from_python_type(dict)


class FeatureSpecTest(absltest.TestCase):
    def test_feature_spec(self) -> None:
        ft = core.FeatureSpec(name="feature", dtype=core.DataType.INT64)
        self.assertEqual(ft, eval(repr(ft), core.__dict__))
        self.assertEqual(ft, core.FeatureSpec.from_dict(ft.to_dict()))
        self.assertEqual(ft.as_snowpark_type(), spt.LongType())
        self.assertEqual(ft.as_dtype(), pd.Int64Dtype())
        self.assertEqual(ft.as_dtype(force_numpy_dtype=True), np.int64)

        ft = core.FeatureSpec(name="feature", dtype=core.DataType.INT64, nullable=False)
        self.assertEqual(ft, eval(repr(ft), core.__dict__))
        self.assertEqual(ft, core.FeatureSpec.from_dict(ft.to_dict()))
        self.assertEqual(ft.as_snowpark_type(), spt.LongType())
        self.assertEqual(ft.as_dtype(), np.int64)
        self.assertEqual(ft.as_dtype(force_numpy_dtype=True), np.int64)

        ft = core.FeatureSpec(name="feature", dtype=core.DataType.INT64, shape=(2,))
        self.assertEqual(ft, eval(repr(ft), core.__dict__))
        self.assertEqual(ft, core.FeatureSpec.from_dict(input_dict=ft.to_dict()))
        self.assertEqual(ft.as_snowpark_type(), spt.ArrayType(spt.LongType()))
        self.assertEqual(ft.as_dtype(), np.object_)
        self.assertEqual(ft.as_dtype(force_numpy_dtype=True), np.object_)


class FeatureGroupSpecTest(absltest.TestCase):
    def test_feature_group_spec(self) -> None:
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="No children feature specs."
        ):
            _ = core.FeatureGroupSpec(name="features", specs=[])

        ft1 = core.FeatureSpec(name="feature1", dtype=core.DataType.INT64, nullable=True)
        ft2 = core.FeatureSpec(name="feature2", dtype=core.DataType.DOUBLE, nullable=False)
        fts = core.FeatureGroupSpec(name="features", specs=[ft1, ft2])
        self.assertEqual(fts, eval(repr(fts), core.__dict__))
        self.assertEqual(fts, core.FeatureGroupSpec.from_dict(fts.to_dict()))
        self.assertEqual(
            fts.as_snowpark_type(),
            spt.StructType(
                [
                    spt.StructField("feature1", spt.LongType(), True),
                    spt.StructField("feature2", spt.DoubleType(), False),
                ]
            ),
        )
        self.assertEqual(np.object_, fts.as_dtype())

        ft1 = core.FeatureSpec(name="feature1", dtype=core.DataType.INT64, shape=(3,), nullable=True)
        ft2 = core.FeatureSpec(name="feature2", dtype=core.DataType.INT64, shape=(2,), nullable=False)
        ft3 = core.FeatureGroupSpec(name="features", specs=[ft1, ft2], shape=(-1,))
        fts = core.FeatureGroupSpec(name="features", specs=[ft1, ft3], shape=(-1,))
        self.assertEqual(fts, eval(repr(fts), core.__dict__))
        self.assertEqual(fts, core.FeatureGroupSpec.from_dict(fts.to_dict()))
        self.assertEqual(
            fts.as_snowpark_type(),
            spt.ArrayType(
                spt.StructType(
                    [
                        spt.StructField("feature1", spt.ArrayType(spt.LongType()), True),
                        spt.StructField(
                            "features",
                            spt.ArrayType(
                                spt.StructType(
                                    [
                                        spt.StructField("feature1", spt.ArrayType(spt.LongType()), True),
                                        spt.StructField("feature2", spt.ArrayType(spt.LongType()), False),
                                    ]
                                )
                            ),
                        ),
                    ]
                ),
            ),
        )
        self.assertEqual(np.object_, fts.as_dtype())


class ParamSpecTest(absltest.TestCase):
    def test_param_spec_scalar_and_serialization(self) -> None:
        """Test scalar ParamSpec creation, properties, and round-trip serialization."""
        # Scalar (shape=None)
        param = core.ParamSpec(name="threshold", dtype=core.DataType.FLOAT, default_value=0.5)
        self.assertEqual(param.name, "threshold")
        self.assertEqual(param.dtype, core.DataType.FLOAT)
        self.assertEqual(param.default_value, 0.5)
        self.assertIsNone(param.shape)

        # Serialization round-trip
        self.assertEqual(param, core.ParamSpec.from_dict(param.to_dict()))
        self.assertEqual(param, eval(repr(param), core.__dict__))

    def test_param_spec_with_shape(self) -> None:
        """Test ParamSpec with shape for array parameters."""
        # Fixed shape array
        param = core.ParamSpec(name="weights", dtype=core.DataType.FLOAT, default_value=[1.0, 2.0, 3.0], shape=(3,))
        self.assertEqual(param.shape, (3,))
        self.assertEqual(param.default_value, [1.0, 2.0, 3.0])

        # Serialization includes shape
        param_dict = param.to_dict()
        self.assertEqual(param_dict["shape"], (3,))
        self.assertEqual(param, core.ParamSpec.from_dict(param_dict))

        # Variable length shape (-1)
        param_var = core.ParamSpec(name="ids", dtype=core.DataType.INT64, default_value=[1, 2], shape=(-1,))
        self.assertEqual(param_var.shape, (-1,))
        self.assertEqual(param_var, core.ParamSpec.from_dict(param_var.to_dict()))

    def test_param_spec_validation(self) -> None:
        """Test validation for dtype, shape, and default_value compatibility."""
        # Invalid shape type
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=TypeError, expected_regex="Shape should be a tuple"
        ):
            core.ParamSpec(
                name="bad", dtype=core.DataType.FLOAT, default_value=0.5, shape=[3]  # type: ignore[arg-type]
            )

        # Shape mismatch: expected scalar, got array
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Expected scalar value"
        ):
            core.ParamSpec(name="bad", dtype=core.DataType.FLOAT, default_value=[1.0, 2.0])

        # Shape mismatch: expected array, got scalar
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Expected 1-dimensional"
        ):
            core.ParamSpec(name="bad", dtype=core.DataType.FLOAT, default_value=1.0, shape=(3,))

        # Dimension size mismatch
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Dimension 0: expected 3, got 2"
        ):
            core.ParamSpec(name="bad", dtype=core.DataType.FLOAT, default_value=[1.0, 2.0], shape=(3,))

        # None default value is allowed (means no default)
        param_with_none = core.ParamSpec(name="optional", dtype=core.DataType.INT64, default_value=None)
        self.assertIsNone(param_with_none.default_value)


class ParamGroupSpecTest(absltest.TestCase):
    def test_param_group_spec_with_shape_and_nesting(self) -> None:
        """Test ParamGroupSpec creation, nesting, shape, and serialization."""
        # Create nested structure with shapes
        p1 = core.ParamSpec(name="lr", dtype=core.DataType.FLOAT, default_value=0.01)
        p2 = core.ParamSpec(name="momentum", dtype=core.DataType.FLOAT, default_value=0.9)
        pg_optimizer = core.ParamGroupSpec(name="optimizer", specs=[p1, p2])

        p3 = core.ParamSpec(name="epochs", dtype=core.DataType.INT64, default_value=10)
        pg_training = core.ParamGroupSpec(name="training", specs=[p3, pg_optimizer], shape=(2,))

        # Verify structure
        self.assertEqual(pg_training.name, "training")
        self.assertEqual(pg_training.shape, (2,))
        self.assertEqual(len(pg_training.specs), 2)
        self.assertIsInstance(pg_training.specs[1], core.ParamGroupSpec)

        # Serialization round-trip
        self.assertEqual(pg_training, core.ParamGroupSpec.from_dict(pg_training.to_dict()))
        self.assertEqual(pg_training, eval(repr(pg_training), core.__dict__))

        # Equality
        pg_same = core.ParamGroupSpec(name="training", specs=[p3, pg_optimizer], shape=(2,))
        pg_diff_shape = core.ParamGroupSpec(name="training", specs=[p3, pg_optimizer], shape=(3,))
        self.assertEqual(pg_training, pg_same)
        self.assertNotEqual(pg_training, pg_diff_shape)
        self.assertNotEqual(pg_training, "not a param group spec")

    def test_param_group_spec_empty_specs_error(self) -> None:
        """Test that empty specs raises error."""
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="No children param specs."
        ):
            core.ParamGroupSpec(name="empty", specs=[])


class ModelSignatureTest(absltest.TestCase):
    def test_1(self) -> None:
        s = core.ModelSignature(
            inputs=[
                core.FeatureSpec(dtype=core.DataType.FLOAT, name="c1"),
                core.FeatureGroupSpec(
                    name="cg1",
                    specs=[
                        core.FeatureSpec(dtype=core.DataType.FLOAT, name="cc1", nullable=True),
                        core.FeatureSpec(
                            dtype=core.DataType.FLOAT,
                            name="cc2",
                            nullable=False,
                        ),
                    ],
                ),
                core.FeatureGroupSpec(
                    name="cg2",
                    specs=[
                        core.FeatureSpec(dtype=core.DataType.FLOAT, name="cc1", shape=(-1,), nullable=True),
                        core.FeatureSpec(
                            dtype=core.DataType.FLOAT,
                            name="cc2",
                            shape=(2,),
                            nullable=False,
                        ),
                    ],
                    shape=(3,),
                ),
                core.FeatureSpec(dtype=core.DataType.FLOAT, name="c2", shape=(-1,)),
            ],
            outputs=[core.FeatureSpec(name="output", dtype=core.DataType.FLOAT)],
        )
        target = {
            "inputs": [
                {"type": "FLOAT", "name": "c1", "nullable": True},
                {
                    "name": "cg1",
                    "specs": [
                        {"type": "FLOAT", "name": "cc1", "nullable": True},
                        {"type": "FLOAT", "name": "cc2", "nullable": False},
                    ],
                },
                {
                    "name": "cg2",
                    "specs": [
                        {"type": "FLOAT", "name": "cc1", "shape": (-1,), "nullable": True},
                        {"type": "FLOAT", "name": "cc2", "shape": (2,), "nullable": False},
                    ],
                    "shape": (3,),
                },
                {"type": "FLOAT", "name": "c2", "shape": (-1,), "nullable": True},
            ],
            "outputs": [{"type": "FLOAT", "name": "output", "nullable": True}],
            "params": [],
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
                        core.FeatureSpec(dtype=core.DataType.FLOAT, name="cc1", nullable=True),
                        core.FeatureSpec(
                            dtype=core.DataType.FLOAT,
                            name="cc2",
                            nullable=False,
                        ),
                    ],
                ),
                core.FeatureGroupSpec(
                    name="cg2",
                    specs=[
                        core.FeatureSpec(dtype=core.DataType.FLOAT, name="cc1", shape=(-1,), nullable=True),
                        core.FeatureSpec(
                            dtype=core.DataType.FLOAT,
                            name="cc2",
                            shape=(2,),
                            nullable=False,
                        ),
                    ],
                    shape=(3,),
                ),
                core.FeatureSpec(dtype=core.DataType.FLOAT, name="c2", shape=(-1,)),
            ],
            outputs=[core.FeatureSpec(name="output", dtype=core.DataType.FLOAT)],
        )
        self.assertEqual(s, eval(repr(s), core.__dict__))
        self.assertEqual(s, core.ModelSignature.from_dict(s.to_dict()))

    def test_with_param_group_spec(self) -> None:
        """Test ModelSignature with ParamGroupSpec."""
        # Create nested param group structure
        optimizer_params = core.ParamGroupSpec(
            name="optimizer",
            specs=[
                core.ParamSpec(name="learning_rate", dtype=core.DataType.FLOAT, default_value=0.01),
                core.ParamSpec(name="momentum", dtype=core.DataType.FLOAT, default_value=0.9),
            ],
        )
        training_params = core.ParamGroupSpec(
            name="training",
            specs=[
                core.ParamSpec(name="epochs", dtype=core.DataType.INT64, default_value=10),
                optimizer_params,
            ],
        )

        s = core.ModelSignature(
            inputs=[core.FeatureSpec(dtype=core.DataType.FLOAT, name="input")],
            outputs=[core.FeatureSpec(name="output", dtype=core.DataType.FLOAT)],
            params=[
                core.ParamSpec(name="threshold", dtype=core.DataType.FLOAT, default_value=0.5),
                training_params,
            ],
        )

        # Test to_dict structure
        s_dict = s.to_dict()
        self.assertEqual(len(s_dict["params"]), 2)
        self.assertEqual(s_dict["params"][0]["name"], "threshold")
        self.assertIn("specs", s_dict["params"][1])  # training_params is a group

        # Test round-trip serialization
        self.assertEqual(s, core.ModelSignature.from_dict(s.to_dict()))

        # Test repr/eval round-trip
        self.assertEqual(s, eval(repr(s), core.__dict__))

    def test_repr_html_happy_path(self) -> None:
        """Test _repr_html_ method for ModelSignature with various feature types."""
        # Create a comprehensive ModelSignature with different feature types
        signature = core.ModelSignature(
            inputs=[
                # Simple feature with nullable=True (default)
                core.FeatureSpec(dtype=core.DataType.FLOAT, name="temperature"),
                # Non-nullable feature
                core.FeatureSpec(dtype=core.DataType.INT64, name="user_id", nullable=False),
                # Feature with shape
                core.FeatureSpec(dtype=core.DataType.DOUBLE, name="embeddings", shape=(128,)),
                # Feature group with nested features
                core.FeatureGroupSpec(
                    name="user_profile",
                    specs=[
                        core.FeatureSpec(dtype=core.DataType.STRING, name="username", nullable=False),
                        core.FeatureSpec(dtype=core.DataType.INT32, name="age", nullable=True),
                        core.FeatureSpec(dtype=core.DataType.BOOL, name="is_premium"),
                    ],
                ),
                # Feature group with shape and nested arrays
                core.FeatureGroupSpec(
                    name="interaction_history",
                    specs=[
                        core.FeatureSpec(dtype=core.DataType.INT64, name="item_id", shape=(-1,), nullable=True),
                        core.FeatureSpec(dtype=core.DataType.FLOAT, name="ratings", shape=(5,), nullable=False),
                    ],
                    shape=(10,),
                ),
            ],
            outputs=[
                # Output feature with simple type
                core.FeatureSpec(name="prediction", dtype=core.DataType.FLOAT),
                # Output feature with shape
                core.FeatureSpec(name="confidence_scores", dtype=core.DataType.DOUBLE, shape=(3,)),
            ],
        )

        # Generate HTML representation
        html = signature._repr_html_()

        # Verify HTML structure and content
        self.assertIn("Model Signature", html)

        # Check that it's properly formatted HTML
        self.assertIn("<div style=", html)
        self.assertIn("font-family: Helvetica, Arial, sans-serif", html)

        # Verify inputs section
        self.assertIn("Inputs", html)
        self.assertIn("<details", html)
        self.assertIn("<summary", html)

        # Check input features are present
        self.assertIn("temperature", html)
        self.assertIn("FLOAT", html)
        self.assertIn("user_id", html)
        self.assertIn("INT64", html)
        self.assertIn("not nullable", html)  # For non-nullable user_id
        self.assertIn("embeddings", html)
        self.assertIn("shape=(128,)", html)

        # Check feature groups
        self.assertIn("user_profile", html)
        self.assertIn("(group)", html)
        self.assertIn("username", html)
        self.assertIn("STRING", html)
        self.assertIn("age", html)
        self.assertIn("INT32", html)
        self.assertIn("is_premium", html)
        self.assertIn("BOOL", html)

        # Check nested feature group with shape
        self.assertIn("interaction_history", html)
        self.assertIn("item_id", html)
        self.assertIn("shape=(-1,)", html)
        self.assertIn("ratings", html)
        self.assertIn("shape=(5,)", html)
        # Note: Group shape is not displayed in HTML output

        # Verify outputs section
        self.assertIn("Outputs", html)
        self.assertIn("prediction", html)
        self.assertIn("confidence_scores", html)
        self.assertIn("shape=(3,)", html)

        # Check for proper indentation/nesting structure
        self.assertIn("border-left: 2px solid #e0e0e0", html)  # Nested group styling
        self.assertIn("margin-left:", html)  # Indentation

        # Verify collapsible structure
        self.assertIn("open", html)  # Details should be open by default

    def test_name_validation(self) -> None:
        """Test name validation for ModelSignature."""
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Found duplicate parameter named resolved as param.",
        ):
            core.ModelSignature(
                inputs=[core.FeatureSpec(dtype=core.DataType.FLOAT, name="input")],
                outputs=[core.FeatureSpec(name="output", dtype=core.DataType.FLOAT)],
                params=[
                    core.ParamSpec(name="param", dtype=core.DataType.FLOAT, default_value=0.1),
                    core.ParamSpec(name="param", dtype=core.DataType.FLOAT, default_value=0.5),
                ],
            )

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Found duplicate parameter named resolved as param.",
        ):
            core.ModelSignature(
                inputs=[core.FeatureSpec(dtype=core.DataType.FLOAT, name="input")],
                outputs=[core.FeatureSpec(name="output", dtype=core.DataType.FLOAT)],
                params=[
                    core.ParamGroupSpec(
                        name="param_group",
                        specs=[core.ParamSpec(name="param", dtype=core.DataType.FLOAT, default_value=0.5)],
                    ),
                    core.ParamSpec(name="param", dtype=core.DataType.FLOAT, default_value=0.5),
                ],
            )

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="Found duplicate parameter named resolved as param.",
        ):
            core.ModelSignature(
                inputs=[core.FeatureSpec(dtype=core.DataType.FLOAT, name="input")],
                outputs=[core.FeatureSpec(name="output", dtype=core.DataType.FLOAT)],
                params=[
                    core.ParamGroupSpec(
                        name="param_group",
                        specs=[
                            core.ParamSpec(name="param", dtype=core.DataType.FLOAT, default_value=0.1),
                            core.ParamSpec(name="param", dtype=core.DataType.FLOAT, default_value=0.5),
                        ],
                    ),
                ],
            )

        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex=r"Found parameter\(s\) with the same name as input feature\(s\): Param",
        ):
            core.ModelSignature(
                inputs=[core.FeatureSpec(dtype=core.DataType.FLOAT, name="param")],
                outputs=[core.FeatureSpec(name="output", dtype=core.DataType.FLOAT)],
                params=[core.ParamSpec(name="Param", dtype=core.DataType.FLOAT, default_value=0.5)],
            )

        # Test ParamGroupSpec param name conflicting with input feature
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex=r"Found parameter\(s\) with the same name as input feature\(s\): Param",
        ):
            core.ModelSignature(
                inputs=[core.FeatureSpec(dtype=core.DataType.FLOAT, name="param")],
                outputs=[core.FeatureSpec(name="output", dtype=core.DataType.FLOAT)],
                params=[
                    core.ParamGroupSpec(
                        name="param_group",
                        specs=[core.ParamSpec(name="Param", dtype=core.DataType.FLOAT, default_value=0.5)],
                    ),
                ],
            )


if __name__ == "__main__":
    absltest.main()
