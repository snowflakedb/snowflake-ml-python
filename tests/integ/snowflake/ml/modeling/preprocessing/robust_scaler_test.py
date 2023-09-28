#!/usr/bin/env python3
import importlib
import os
import pickle
import sys
import tempfile
from typing import List, Tuple

import cloudpickle
import joblib
import numpy as np
import pandas as pd
from absl.testing import parameterized
from absl.testing.absltest import main
from sklearn.preprocessing import RobustScaler as SklearnRobustScaler

from snowflake.ml.modeling.preprocessing import (  # type: ignore[attr-defined]
    RobustScaler,
)
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.modeling.framework import utils as framework_utils
from tests.integ.snowflake.ml.modeling.framework.utils import (
    DATA,
    ID_COL,
    NUMERIC_COLS,
    OUTPUT_COLS,
    SCHEMA,
    equal_optional_of,
)


class RobustScalerTest(parameterized.TestCase):
    """Test RobustScaler."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        self._to_be_deleted_files: List[str] = []

    def tearDown(self) -> None:
        self._session.close()
        for filepath in self._to_be_deleted_files:
            if os.path.exists(filepath):
                os.remove(filepath)

    @parameterized.parameters(  # type: ignore[misc]
        {
            "params": {
                "with_scaling": True,
                "with_centering": True,
                "quantile_range": (25.0, 75.0),
                "unit_variance": False,
            }
        },
        {
            "params": {
                "with_scaling": True,
                "with_centering": True,
                "quantile_range": (25.0, 75.0),
                "unit_variance": True,
            }
        },
        {
            "params": {
                "with_scaling": False,
                "with_centering": True,
                "quantile_range": (25.0, 75.0),
                "unit_variance": False,
            }
        },
        {
            "params": {
                "with_scaling": True,
                "with_centering": False,
                "quantile_range": (25.0, 75.0),
                "unit_variance": False,
            }
        },
        {
            "params": {
                "with_scaling": False,
                "with_centering": False,
                "quantile_range": (25.0, 75.0),
                "unit_variance": False,
            }
        },
        {
            "params": {
                "with_scaling": True,
                "with_centering": True,
                "quantile_range": (0.0, 100.0),
                "unit_variance": False,
            }
        },
        {
            "params": {
                "with_scaling": True,
                "with_centering": True,
                "quantile_range": (50.0, 50.0),
                "unit_variance": False,
            }
        },
    )
    def test_fit(self, params) -> None:
        """
        Verify fitted states.

        Args:
            params: Dict with a single `params` key, which maps to the desired RobustScaler constructor arguments.
        """
        input_cols = NUMERIC_COLS
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)
        equality_func = equal_optional_of(np.allclose)

        for _df in [df_pandas, df]:
            scaler = RobustScaler(**params).set_input_cols(input_cols)
            scaler.fit(_df)

            actual_center = (
                scaler._convert_attribute_dict_to_ndarray(scaler.center_) if scaler.center_ is not None else None
            )
            actual_scale = (
                scaler._convert_attribute_dict_to_ndarray(scaler.scale_) if scaler.scale_ is not None else None
            )

            # sklearn
            scaler_sklearn = SklearnRobustScaler(**params)
            scaler_sklearn.fit(df_pandas[input_cols])

            self.assertTrue(equality_func(actual_center, scaler_sklearn.center_))
            self.assertTrue(equality_func(actual_scale, scaler_sklearn.scale_))

    def _run_and_compare(
        self,
        with_centering: bool,
        with_scaling: bool,
        quantile_range: Tuple[float, float],
        unit_variance: bool,
        expect_exception: bool = False,
    ) -> None:
        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL
        input_cols_extended = input_cols.copy()
        input_cols_extended.append(id_col)
        input_df_pandas, input_df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)

        # Fit with snowpark DataFrame, transform with snowpark DataFrame
        scaler = RobustScaler(
            with_scaling=with_scaling,
            with_centering=with_centering,
            quantile_range=quantile_range,
            unit_variance=unit_variance,
            input_cols=input_cols,
            output_cols=output_cols,
        )
        if expect_exception:
            with self.assertRaises(ValueError):
                scaler.fit(input_df)
            return
        scaler.fit(input_df)
        transformed_df = scaler.transform(input_df[input_cols_extended])
        actual_arr = transformed_df.sort(id_col)[output_cols].to_pandas().to_numpy()

        # Fit with pandas DataFrame, transform with snowpark DataFrame
        scaler2 = RobustScaler(
            with_scaling=with_scaling,
            with_centering=with_centering,
            quantile_range=quantile_range,
            unit_variance=unit_variance,
            input_cols=input_cols,
            output_cols=output_cols,
        )
        if expect_exception:
            with self.assertRaises(ValueError):
                scaler2.fit(input_df)
            return
        scaler2.fit(input_df_pandas)
        transformed_df = scaler2.transform(input_df[input_cols_extended])
        actual_arr2 = transformed_df.sort(id_col)[output_cols].to_pandas().to_numpy()

        # sklearn
        scaler_sklearn = SklearnRobustScaler(
            with_scaling=with_scaling,
            with_centering=with_centering,
            quantile_range=quantile_range,
            unit_variance=unit_variance,
        )
        scaler_sklearn.fit(input_df_pandas[input_cols])
        expected_arr = scaler_sklearn.transform(input_df_pandas.sort_values(by=[id_col])[input_cols])

        np.testing.assert_allclose(actual_arr, expected_arr)
        np.testing.assert_allclose(actual_arr2, expected_arr)

    def test_transform_default(self) -> None:
        self._run_and_compare(with_scaling=True, with_centering=True, quantile_range=(25.0, 75.0), unit_variance=False)

    def test_transform_with_unit_variance(self) -> None:
        self._run_and_compare(with_scaling=True, with_centering=True, quantile_range=(25.0, 75.0), unit_variance=True)

    def test_transform_with_no_scaling(self) -> None:
        self._run_and_compare(with_scaling=False, with_centering=True, quantile_range=(25.0, 75.0), unit_variance=False)

    def test_transform_with_no_centering(self) -> None:
        self._run_and_compare(with_scaling=True, with_centering=False, quantile_range=(25.0, 75.0), unit_variance=False)

    def test_transform_with_no_scaling_and_centering(self) -> None:
        self._run_and_compare(
            with_scaling=False, with_centering=False, quantile_range=(25.0, 75.0), unit_variance=False
        )

    def test_transform_with_quantile_range_0_100(self) -> None:
        self._run_and_compare(with_scaling=True, with_centering=True, quantile_range=(0.0, 100.0), unit_variance=False)

    def test_transform_with_quantile_range_50_50(self) -> None:
        self._run_and_compare(with_scaling=True, with_centering=True, quantile_range=(50.0, 50.0), unit_variance=False)

    def test_transform_with_quantile_range_75_25(self) -> None:
        self._run_and_compare(
            with_scaling=True,
            with_centering=True,
            quantile_range=(75.0, 25.0),
            unit_variance=False,
            expect_exception=True,
        )

    def test_transform_pandas_input(self) -> None:
        data, schema = DATA, SCHEMA
        input_cols, output_cols = NUMERIC_COLS, OUTPUT_COLS
        input_df_pandas = pd.DataFrame(data, columns=schema)
        input_df = self._session.create_dataframe(input_df_pandas)

        scaler = RobustScaler(
            input_cols=input_cols,
            output_cols=output_cols,
        )
        scaler.fit(input_df)
        transformed_df = scaler.transform(input_df_pandas[input_cols])
        actual_arr = transformed_df[output_cols].to_numpy()

        scaler_sklearn = SklearnRobustScaler()
        scaler_sklearn.fit(input_df_pandas[input_cols])
        transformed_df_sklearn = scaler_sklearn.transform(input_df_pandas[input_cols])
        expected_arr = transformed_df_sklearn

        np.testing.assert_allclose(actual_arr, expected_arr)

    def test_serde(self) -> None:
        """
        Test serialization and deserialization via cloudpickle, pickle, and joblib.
        """
        data, schema = DATA, SCHEMA
        input_cols, output_cols, id_col = NUMERIC_COLS, OUTPUT_COLS, ID_COL

        # fit in session 1
        df_pandas, df1 = framework_utils.get_df(self._session, data, schema, np.nan)

        scaler = RobustScaler().set_input_cols(input_cols).set_output_cols(output_cols)
        scaler.fit(df1)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as file:
            self._to_be_deleted_files.append(file.name)
            scaler_dump_cloudpickle = cloudpickle.dumps(scaler)
            scaler_dump_pickle = pickle.dumps(scaler)
            joblib.dump(scaler, file.name)

            self._session.close()

            # transform in session 2
            self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
            _, df2 = framework_utils.get_df(self._session, data, schema, np.nan)
            input_cols_extended = input_cols.copy()
            input_cols_extended.append(id_col)

            importlib.reload(sys.modules["snowflake.ml.modeling.preprocessing.robust_scaler"])

            # cloudpickle
            scaler_load_cloudpickle = cloudpickle.loads(scaler_dump_cloudpickle)
            transformed_df_cloudpickle = scaler_load_cloudpickle.transform(df2[input_cols_extended])
            actual_arr_cloudpickle = transformed_df_cloudpickle.sort(id_col)[output_cols].to_pandas().to_numpy()

            # pickle
            scaler_load_pickle = pickle.loads(scaler_dump_pickle)
            transformed_df_pickle = scaler_load_pickle.transform(df2[input_cols_extended])
            actual_arr_pickle = transformed_df_pickle.sort(id_col)[output_cols].to_pandas().to_numpy()

            # joblib
            scaler_load_joblib = joblib.load(file.name)
            transformed_df_joblib = scaler_load_joblib.transform(df2[input_cols_extended])
            actual_arr_joblib = transformed_df_joblib.sort(id_col)[output_cols].to_pandas().to_numpy()

            # sklearn
            scaler_sklearn = SklearnRobustScaler()
            scaler_sklearn.fit(df_pandas[input_cols])
            sklearn_arr = scaler_sklearn.transform(df_pandas[input_cols])

            np.testing.assert_allclose(actual_arr_cloudpickle, sklearn_arr)
            np.testing.assert_allclose(actual_arr_pickle, sklearn_arr)
            np.testing.assert_allclose(actual_arr_joblib, sklearn_arr)


if __name__ == "__main__":
    main()
