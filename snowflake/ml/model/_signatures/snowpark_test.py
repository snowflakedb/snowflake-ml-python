import datetime
import decimal
from typing import cast
from unittest import mock

import numpy as np
import pandas as pd
from absl.testing import absltest

import snowflake.snowpark.types as spt
from snowflake.ml.model import model_signature
from snowflake.ml.model._signatures import core, snowpark_handler
from snowflake.ml.test_utils import exception_utils, mock_session
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session


class SnowParkDataFrameHandlerTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._session.close()

    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.c_session = cast(Session, self.m_session)

    def test_validate_snowpark_df(self) -> None:
        schema = spt.StructType([spt.StructField('"a"', spt.VariantType()), spt.StructField('"b"', spt.StringType())])
        df = self._session.create_dataframe([[1, "snow"], [3, "flake"]], schema)
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Unsupported data type"
        ):
            snowpark_handler.SnowparkDataFrameHandler.validate(df)

    def test_infer_schema_snowpark_df(self) -> None:
        schema = spt.StructType([spt.StructField('"a"', spt.LongType()), spt.StructField('"b"', spt.StringType())])
        df = self._session.create_dataframe([[1, "snow"], [3, "flake"]], schema)
        self.assertListEqual(
            snowpark_handler.SnowparkDataFrameHandler.infer_signature(df, role="input"),
            [
                core.FeatureSpec("a", core.DataType.INT8),
                core.FeatureSpec("b", core.DataType.STRING),
            ],
        )

        schema = spt.StructType([spt.StructField('"""a"""', spt.LongType()), spt.StructField('"b"', spt.StringType())])
        df = self._session.create_dataframe([[1, "snow"], [3, "flake"]], schema)
        self.assertListEqual(
            snowpark_handler.SnowparkDataFrameHandler.infer_signature(df, role="input"),
            [
                core.FeatureSpec('"a"', core.DataType.INT8),
                core.FeatureSpec("b", core.DataType.STRING),
            ],
        )

        schema = spt.StructType([spt.StructField('"""a"""', spt.ArrayType(spt.LongType()))])
        df = self._session.create_dataframe([[[1, 3]]], schema)
        self.assertListEqual(
            snowpark_handler.SnowparkDataFrameHandler.infer_signature(df, role="input"),
            [
                core.FeatureSpec('"a"', core.DataType.INT64, shape=(2,)),
            ],
        )

    def test_validate_data_with_features(self) -> None:
        fts = [
            core.FeatureSpec("a", core.DataType.UINT8),
            core.FeatureSpec("b", core.DataType.INT64),
        ]
        df = self._session.create_dataframe([{'"a"': 1}, {'"b"': 2}])
        self.assertEqual(
            model_signature._validate_snowpark_data(df, fts), model_signature.SnowparkIdentifierRule.INFERRED
        )

        fts = [
            core.FeatureSpec("a", core.DataType.UINT8),
            core.FeatureSpec("b", core.DataType.INT64),
        ]
        df = self._session.create_dataframe([{"a": 1}, {"b": 2}])
        self.assertEqual(
            model_signature._validate_snowpark_data(df, fts), model_signature.SnowparkIdentifierRule.NORMALIZED
        )

        fts = [
            core.FeatureSpec("a", core.DataType.UINT8),
            core.FeatureSpec("b", core.DataType.INT64),
        ]
        df = self._session.create_dataframe([{"A": 1}, {"B": 2}])
        self.assertEqual(
            model_signature._validate_snowpark_data(df, fts), model_signature.SnowparkIdentifierRule.NORMALIZED
        )

        fts = [
            core.FeatureSpec('"a"', core.DataType.UINT8),
            core.FeatureSpec('"b"', core.DataType.INT64),
        ]
        df = self._session.create_dataframe([{'"a"': 1}, {'"b"': 2}])
        self.assertEqual(
            model_signature._validate_snowpark_data(df, fts), model_signature.SnowparkIdentifierRule.NORMALIZED
        )

        fts = [
            core.FeatureSpec('"a"', core.DataType.UINT8),
            core.FeatureSpec('"b"', core.DataType.INT64),
        ]
        df = self._session.create_dataframe([{'"""a"""': 1}, {'"""b"""': 2}])
        self.assertEqual(
            model_signature._validate_snowpark_data(df, fts), model_signature.SnowparkIdentifierRule.INFERRED
        )

        fts = [
            core.FeatureSpec('"a"', core.DataType.UINT8),
            core.FeatureSpec('"b"', core.DataType.INT64),
        ]
        df = self._session.create_dataframe([{'"A"': 1}, {'"b"': 2}])
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="feature [^\\s]* does not exist in data."
        ):
            model_signature._validate_snowpark_data(df, fts)

        fts = [
            core.FeatureSpec('"a"', core.DataType.UINT8),
            core.FeatureSpec('"b"', core.DataType.INT64),
        ]
        df = self._session.create_dataframe([{"A": 1}, {'"b"': 2}])
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="feature [^\\s]* does not exist in data."
        ):
            model_signature._validate_snowpark_data(df, fts)

        fts = [
            core.FeatureSpec("a", core.DataType.INT16),
            core.FeatureSpec("b", core.DataType.UINT32),
        ]
        df = self._session.create_dataframe([{'"a"': 1}, {'"b"': -2}])
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Feature type [^\\s]* is not met by column"
        ):
            model_signature._validate_snowpark_data(df, fts, strict=True)

        model_signature._validate_snowpark_data(df, fts)

        fts = [
            core.FeatureSpec("a", core.DataType.INT8),
            core.FeatureSpec("b", core.DataType.INT32),
        ]
        df = self._session.create_dataframe([{'"a"': 129}, {'"b"': -2}])
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Feature type [^\\s]* is not met by column"
        ):
            model_signature._validate_snowpark_data(df, fts, strict=True)

        model_signature._validate_snowpark_data(df, fts)

        fts = [
            core.FeatureSpec("a", core.DataType.INT16),
            core.FeatureSpec("b", core.DataType.UINT32),
        ]
        schema = spt.StructType(
            [spt.StructField('"a"', spt.DecimalType(6, 0)), spt.StructField('"b"', spt.DecimalType(12, 0))]
        )
        df = self._session.create_dataframe(
            [[decimal.Decimal(1), decimal.Decimal(1)], [decimal.Decimal(1), decimal.Decimal(1)]], schema
        )
        model_signature._validate_snowpark_data(df, fts)

        fts = [
            core.FeatureSpec("a", core.DataType.INT16),
            core.FeatureSpec("b", core.DataType.UINT32),
        ]
        schema = spt.StructType(
            [spt.StructField('"a"', spt.DecimalType(6, 2)), spt.StructField('"b"', spt.DecimalType(12, 0))]
        )
        df = self._session.create_dataframe(
            [[decimal.Decimal(1), decimal.Decimal(1)], [decimal.Decimal(1), decimal.Decimal(1)]], schema
        )
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Feature type [^\\s]* is not met by column"
        ):
            model_signature._validate_snowpark_data(df, fts)

        fts = [
            core.FeatureSpec("a", core.DataType.INT16),
            core.FeatureSpec("b", core.DataType.UINT32),
        ]
        schema = spt.StructType(
            [spt.StructField('"a"', spt.DecimalType(6, 0)), spt.StructField('"b"', spt.DecimalType(12, 0))]
        )
        df = self._session.create_dataframe(
            [[decimal.Decimal(1), decimal.Decimal(-1)], [decimal.Decimal(1), decimal.Decimal(1)]], schema
        )
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Feature type [^\\s]* is not met by column"
        ):
            model_signature._validate_snowpark_data(df, fts, strict=True)

        model_signature._validate_snowpark_data(df, fts)

        fts = [
            core.FeatureSpec("a", core.DataType.UINT8),
            core.FeatureSpec("b", core.DataType.FLOAT),
        ]
        df = self._session.create_dataframe([{'"a"': 1}, {'"b"': 2}])
        model_signature._validate_snowpark_data(df, fts)

        fts = [
            core.FeatureSpec("a", core.DataType.UINT8),
            core.FeatureSpec("b", core.DataType.FLOAT),
        ]
        df = self._session.create_dataframe([{'"a"': 1}, {'"b"': 2.0}])
        model_signature._validate_snowpark_data(df, fts)

        fts = [
            core.FeatureSpec("a", core.DataType.UINT8),
            core.FeatureSpec("b", core.DataType.FLOAT),
        ]
        df = self._session.create_dataframe([{'"a"': 1}, {'"b"': 98765432109876543210987654321098765432}])
        model_signature._validate_snowpark_data(df, fts)

        fts = [
            core.FeatureSpec("a", core.DataType.INT16),
            core.FeatureSpec("b", core.DataType.FLOAT),
        ]
        schema = spt.StructType(
            [spt.StructField('"a"', spt.DecimalType(6, 0)), spt.StructField('"b"', spt.DecimalType(38, 0))]
        )
        df = self._session.create_dataframe(
            [
                [decimal.Decimal(1), decimal.Decimal(1)],
                [decimal.Decimal(1), decimal.Decimal(98765432109876543210987654321098765432)],
            ],
            schema,
        )
        model_signature._validate_snowpark_data(df, fts)

        fts = [
            core.FeatureSpec("a", core.DataType.INT16),
            core.FeatureSpec("b", core.DataType.FLOAT),
        ]
        schema = spt.StructType(
            [spt.StructField('"a"', spt.DecimalType(6, 0)), spt.StructField('"b"', spt.DecimalType(6, 2))]
        )
        df = self._session.create_dataframe(
            [
                [decimal.Decimal(1), decimal.Decimal(2.5)],
                [decimal.Decimal(1), decimal.Decimal(6.8)],
            ],
            schema,
        )
        with self.assertWarnsRegex(UserWarning, "is being automatically converted to DOUBLE in the Snowpark DataFrame"):
            model_signature._validate_snowpark_data(df, fts)

        fts = [
            core.FeatureSpec("a", core.DataType.INT16),
            core.FeatureSpec("b", core.DataType.FLOAT),
        ]
        schema = spt.StructType(
            [spt.StructField('"a"', spt.DecimalType(6, 0)), spt.StructField('"b"', spt.DoubleType())]
        )
        df = self._session.create_dataframe([[decimal.Decimal(1), -2.0], [decimal.Decimal(1), 1e58]], schema)
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Feature type [^\\s]* is not met by column"
        ):
            model_signature._validate_snowpark_data(df, fts, strict=True)

        model_signature._validate_snowpark_data(df, fts)

        fts = [
            core.FeatureSpec("a", core.DataType.INT64),
            core.FeatureSpec("b", core.DataType.STRING),
        ]
        schema = spt.StructType([spt.StructField('"a"', spt.LongType()), spt.StructField('"b"', spt.StringType())])
        df = self._session.create_dataframe([[1, "snow"], [3, "flake"]], schema)
        model_signature._validate_snowpark_data(df, fts)

        schema = spt.StructType([spt.StructField('"a"', spt.LongType()), spt.StructField('"b"', spt.IntegerType())])
        df = self._session.create_dataframe([[1, 3], [3, 9]], schema)
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Feature type [^\\s]* is not met by column"
        ):
            model_signature._validate_snowpark_data(df, fts)

        schema = spt.StructType([spt.StructField('"a1"', spt.LongType()), spt.StructField('"b"', spt.StringType())])
        df = self._session.create_dataframe([[1, "snow"], [3, "flake"]], schema)
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="feature [^\\s]* does not exist in data."
        ):
            model_signature._validate_snowpark_data(df, fts)

        df = self._session.create_dataframe([{'"a"': 1}, {'"b"': 2}])
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="Feature type [^\\s]* is not met by column"
        ):
            model_signature._validate_snowpark_data(df, fts)

        fts = [
            core.FeatureSpec("a", core.DataType.INT64, shape=(-1,)),
        ]
        schema = spt.StructType([spt.StructField('"a"', spt.ArrayType(spt.LongType()))])
        df = self._session.create_dataframe([[[1, 3]]], schema)
        with self.assertWarns(RuntimeWarning):
            model_signature._validate_snowpark_data(df, fts)

    def test_convert_to_and_from_df(self) -> None:
        pd_df = pd.DataFrame([1, 2, 3, 4], columns=["col_0"])
        sp_df = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(self._session, pd_df, keep_order=False)
        pd.testing.assert_frame_equal(
            pd_df, snowpark_handler.SnowparkDataFrameHandler.convert_to_df(sp_df), check_dtype=False
        )

        pd_df = pd.DataFrame([[1, 2], [3, 4]], columns=["col_0", "col_1"])
        sp_df = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(self._session, pd_df, keep_order=False)
        pd.testing.assert_frame_equal(
            pd_df, snowpark_handler.SnowparkDataFrameHandler.convert_to_df(sp_df), check_dtype=False
        )

        pd_df = pd.DataFrame([[1.2, 2.4], [3, 4]], columns=["col_0", "col_1"])
        sp_df = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(self._session, pd_df, keep_order=False)
        pd.testing.assert_frame_equal(
            pd_df, snowpark_handler.SnowparkDataFrameHandler.convert_to_df(sp_df), check_dtype=False
        )

        pd_df = pd.DataFrame([[1, [[2.5], [6.8]]], [2, [[2.5], [6.8]]]], columns=["a", "b"])
        sp_df = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(self._session, pd_df, keep_order=False)
        pd.testing.assert_frame_equal(
            pd_df, snowpark_handler.SnowparkDataFrameHandler.convert_to_df(sp_df), check_dtype=False
        )

        a = np.array([2.5, 6.8])
        pd_df = pd.DataFrame([[1, a], [2, a]], columns=["a", "b"])
        sp_df = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(self._session, pd_df, keep_order=False)
        pd.testing.assert_frame_equal(
            pd_df, snowpark_handler.SnowparkDataFrameHandler.convert_to_df(sp_df), check_dtype=False
        )

        pd_df = pd.DataFrame([1, 2, 3, 4], columns=["col_0"])
        sp_df = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(
            self._session,
            pd_df,
            keep_order=False,
            features=[model_signature.FeatureSpec(name="col_0", dtype=model_signature.DataType.INT64)],
        )
        pd.testing.assert_frame_equal(
            pd_df, snowpark_handler.SnowparkDataFrameHandler.convert_to_df(sp_df), check_dtype=False
        )

        d = datetime.datetime(year=2024, month=6, day=21, hour=1, minute=1, second=1)
        pd_df = pd.DataFrame([[1, d], [2, d]], columns=["col_0", "col_1"])
        sp_df = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(
            self._session,
            pd_df,
            keep_order=False,
            features=[
                model_signature.FeatureSpec(name="col_0", dtype=model_signature.DataType.INT64),
                model_signature.FeatureSpec(name="col_1", dtype=model_signature.DataType.TIMESTAMP_NTZ),
            ],
        )
        pd.testing.assert_frame_equal(
            pd_df, snowpark_handler.SnowparkDataFrameHandler.convert_to_df(sp_df), check_dtype=False
        )

        pd_df = pd.DataFrame({"a": [1, 2], "b": [[2.5], None]}, columns=["a", "b"])
        sp_df = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(self._session, pd_df, keep_order=False)
        pd.testing.assert_frame_equal(
            pd_df, snowpark_handler.SnowparkDataFrameHandler.convert_to_df(sp_df), check_dtype=False
        )

    def test_is_quoted_identifiers_ignore_case_enabled(self) -> None:
        def create_mock_row(value: str) -> mock.Mock:
            mock_row = mock.Mock()
            mock_row.value = value
            return mock_row

        with mock.patch.object(self.c_session, "sql") as mock_sql:
            mock_sql.return_value.collect.return_value = [create_mock_row("FALSE")]

            statement_params = {"QUOTED_IDENTIFIERS_IGNORE_CASE": "TRUE"}
            result = snowpark_handler.SnowparkDataFrameHandler._is_quoted_identifiers_ignore_case_enabled(
                self.c_session, statement_params
            )
            self.assertTrue(result, "statement_params should take precedence over session parameter")

        with mock.patch.object(self.c_session, "sql") as mock_sql:
            mock_sql.return_value.collect.return_value = [create_mock_row("FALSE")]

            result = snowpark_handler.SnowparkDataFrameHandler._is_quoted_identifiers_ignore_case_enabled(
                self.c_session, None
            )
            self.assertFalse(result, "Should use session parameter when statement_params is None")

        with mock.patch.object(self.c_session, "sql") as mock_sql:
            mock_sql.return_value.collect.return_value = [create_mock_row("TRUE")]

            statement_params = {}
            result = snowpark_handler.SnowparkDataFrameHandler._is_quoted_identifiers_ignore_case_enabled(
                self.c_session, statement_params
            )
            self.assertTrue(result, "Should fallback to session parameter when statement_params is empty")

        with mock.patch.object(self.c_session, "sql") as mock_sql:
            mock_sql.return_value.collect.return_value = [create_mock_row("FALSE")]

            for param_value in ["TRUE", "True", "true"]:
                statement_params = {"QUOTED_IDENTIFIERS_IGNORE_CASE": param_value}
                result = snowpark_handler.SnowparkDataFrameHandler._is_quoted_identifiers_ignore_case_enabled(
                    self.c_session, statement_params
                )
                self.assertTrue(result, f"Failed for parameter value: {param_value}")

        with mock.patch.object(self.c_session, "sql") as mock_sql:
            mock_sql.return_value.collect.return_value = [create_mock_row("TRUE")]

            for param_value in ["FALSE", "False", "false"]:
                statement_params = {"QUOTED_IDENTIFIERS_IGNORE_CASE": param_value}
                result = snowpark_handler.SnowparkDataFrameHandler._is_quoted_identifiers_ignore_case_enabled(
                    self.c_session, statement_params
                )
                self.assertFalse(result, f"Failed for parameter value: {param_value}")

        with mock.patch.object(self.c_session, "sql") as mock_sql:
            mock_sql.return_value.collect.return_value = [create_mock_row("FALSE")]

            statement_params = {"quoted_identifiers_ignore_case": "true"}
            result = snowpark_handler.SnowparkDataFrameHandler._is_quoted_identifiers_ignore_case_enabled(
                self.c_session, statement_params
            )
            self.assertTrue(result, "Parameter name should be case insensitive")


if __name__ == "__main__":
    absltest.main()
