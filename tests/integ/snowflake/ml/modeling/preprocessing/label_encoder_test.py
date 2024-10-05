import importlib
import os
import sys
import tempfile
from typing import List
from unittest import TestCase

import cloudpickle
import numpy as np
from absl.testing.absltest import main
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder

from snowflake.ml.modeling.preprocessing import (  # type: ignore[attr-defined]
    LabelEncoder,
)
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.modeling.framework import utils as framework_utils
from tests.integ.snowflake.ml.modeling.framework.utils import (
    DATA,
    DATA_BOOLEAN,
    DATA_NONE_NAN,
    ID_COL,
    SCHEMA,
    SCHEMA_BOOLEAN,
)

INPUT_COL = SCHEMA[1]
OUTPUT_COL = "_TEST"


class LabelEncoderTest(TestCase):
    """Test LabelEncoder."""

    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        self._to_be_deleted_files: List[str] = []

    def tearDown(self) -> None:
        self._session.close()
        for filepath in self._to_be_deleted_files:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_fit(self) -> None:
        """
        Verify fitted categories.

        Raises
        ------
        AssertionError
            If the fitted categories do not match those of the sklearn encoder.
        """
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA)

        encoder = LabelEncoder(input_cols=INPUT_COL, output_cols=OUTPUT_COL)
        encoder.fit(df)

        encoder_sklearn = SklearnLabelEncoder()
        encoder_sklearn.fit(df_pandas[INPUT_COL])

        np.testing.assert_equal(encoder.classes_, encoder_sklearn.classes_)

    def test_fit_nones(self) -> None:
        """
        Verify fitted categories with `None`s in the label column.

        Raises
        ------
        AssertionError
            If the fitted categories do not match those of the sklearn encoder.
        """
        df_pandas, df = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA)

        encoder = LabelEncoder(input_cols=INPUT_COL, output_cols=OUTPUT_COL)
        encoder.fit(df)

        encoder_sklearn = SklearnLabelEncoder()
        encoder_sklearn.fit(df_pandas[INPUT_COL])

        np.testing.assert_equal(encoder.classes_, encoder_sklearn.classes_)

    def test_transform_snowpark(self) -> None:
        """
        Verify transformed dataset.

        Raises
        ------
        AssertionError
            If the transformed dataset does not match the output of the sklearn encoder.
        """
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA)
        encoder = LabelEncoder(input_cols=INPUT_COL, output_cols=OUTPUT_COL)
        encoder.fit(df)

        transformed_dataset = encoder.transform(df)
        transformed_dataset_output_col = transformed_dataset.to_pandas()[[OUTPUT_COL]].to_numpy().flatten()

        sklearn_input_ndarray = df_pandas[INPUT_COL]
        sklearn_encoder = SklearnLabelEncoder()
        sklearn_encoder.fit(sklearn_input_ndarray)
        sklearn_transformed_dataset = sklearn_encoder.transform(sklearn_input_ndarray)

        np.testing.assert_equal(sklearn_transformed_dataset, transformed_dataset_output_col)

    def test_transform_snowpark_none(self) -> None:
        """
        Verify transformed dataset containing `None`s.

        Raises
        ------
        AssertionError
            If `None` is not included in classes.
            If snowpark label encoder transformation does not match the equivalent sklearn transformation.
        """
        df_none_nan_pandas, df_none_nan = framework_utils.get_df(self._session, DATA_NONE_NAN, SCHEMA)
        encoder = LabelEncoder(input_cols=INPUT_COL, output_cols=OUTPUT_COL)
        encoder.fit(df_none_nan)

        # `None` is included.
        self.assertTrue(None in encoder.classes_)

        transformed_dataset = encoder.transform(df_none_nan)
        transformed_dataset_output_col = transformed_dataset.to_pandas()[[OUTPUT_COL]].to_numpy().flatten()

        sklearn_input_ndarray = df_none_nan_pandas[INPUT_COL]
        sklearn_encoder = SklearnLabelEncoder()
        sklearn_encoder.fit(sklearn_input_ndarray)
        sklearn_transformed_dataset = sklearn_encoder.transform(sklearn_input_ndarray)

        np.testing.assert_equal(sklearn_transformed_dataset, transformed_dataset_output_col)

        # Attempt to transform the dataset without `None`s should raise ValueError.
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA)
        with self.assertRaises(ValueError):
            encoder.transform(df)

    def test_transform_snowpark_boolean(self) -> None:
        """
        Verify transformed dataset containing booleans.

        Raises
        ------
        AssertionError
            If snowpark label encoder transformation does not match the equivalent sklearn transformation.
        """
        input_col = SCHEMA_BOOLEAN[3]
        df_pandas, df = framework_utils.get_df(self._session, DATA_BOOLEAN, SCHEMA_BOOLEAN, np.nan)
        encoder = LabelEncoder(input_cols=input_col, output_cols=OUTPUT_COL)
        encoder.fit(df)

        transformed_dataset = encoder.transform(df)
        transformed_dataset_output_col = transformed_dataset.to_pandas()[[OUTPUT_COL]].to_numpy().flatten()

        sklearn_input_ndarray = df_pandas[input_col]
        sklearn_encoder = SklearnLabelEncoder()
        sklearn_encoder.fit(sklearn_input_ndarray)
        sklearn_transformed_dataset = sklearn_encoder.transform(sklearn_input_ndarray)

        np.testing.assert_equal(sklearn_transformed_dataset, transformed_dataset_output_col)

    def test_transform_sklearn(self) -> None:
        """ """
        df_pandas, df = framework_utils.get_df(self._session, DATA, SCHEMA, np.nan)
        encoder = LabelEncoder(input_cols=INPUT_COL, output_cols=OUTPUT_COL)
        encoder.fit(df)

        transformed_dataset = encoder.transform(df_pandas)

        sklearn_encoder = SklearnLabelEncoder()
        df_pandas_y = df_pandas[[INPUT_COL]]
        sklearn_encoder.fit(df_pandas_y)
        sklearn_transformed_dataset = sklearn_encoder.transform(df_pandas_y)

        np.testing.assert_equal(sklearn_transformed_dataset, transformed_dataset[[OUTPUT_COL]].to_numpy().flatten())

    def test_serde(self) -> None:
        """
        Test serialization and deserialization via cloudpickle, pickle, and joblib.

        Raises
        ------
        AssertionError
            If the deserialized results do not match the sklearn results.
        """
        data, schema, id_col = DATA, SCHEMA, ID_COL
        input_cols, output_cols = [SCHEMA[1]], ["_TEST"]

        # fit in session 1
        df_pandas, df1 = framework_utils.get_df(self._session, data, schema, np.nan)

        label_encoder = LabelEncoder().set_input_cols(input_cols).set_output_cols(output_cols)
        label_encoder.fit(df1)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as file:
            self._to_be_deleted_files.append(file.name)
            label_encoder_dump_cloudpickle = cloudpickle.dumps(label_encoder)
            # disabling pickle and joblib serde due to the below error
            # _pickle.PicklingError: Can't pickle <class 'snowflake.ml.modeling.preprocessing.label_encoder.LabelEncoder'>: it's not the same object as snowflake.ml.modeling.preprocessing.label_encoder.LabelEncoder # noqa: E501
            # label_encoder_dump_pickle = pickle.dumps(label_encoder)
            # joblib.dump(label_encoder, file.name)

            self._session.close()

            # transform in session 2
            self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
            _, df2 = framework_utils.get_df(self._session, data, schema, np.nan)
            input_cols_extended = input_cols.copy()
            input_cols_extended.append(id_col)

            importlib.reload(sys.modules["snowflake.ml.modeling.preprocessing.label_encoder"])

            # cloudpickle
            label_encoder_load_cloudpickle = cloudpickle.loads(label_encoder_dump_cloudpickle)
            transformed_df_cloudpickle = label_encoder_load_cloudpickle.transform(df2)
            actual_arr_cloudpickle = transformed_df_cloudpickle[output_cols].to_pandas().to_numpy().flatten()

            # pickle
            # label_encoder_load_pickle = pickle.loads(label_encoder_dump_pickle)
            # transformed_df_pickle = label_encoder_load_pickle.transform(df2)
            # actual_arr_pickle = transformed_df_pickle[output_cols].to_pandas().to_numpy().flatten()

            # joblib
            # label_encoder_load_joblib = joblib.load(file.name)
            # transformed_df_joblib = label_encoder_load_joblib.transform(df2)
            # actual_arr_joblib = transformed_df_joblib[output_cols].to_pandas().to_numpy().flatten()

            # sklearn
            label_encoder_sklearn = SklearnLabelEncoder()
            label_encoder_sklearn.fit(df_pandas[input_cols])
            sklearn_arr = label_encoder_sklearn.transform(df_pandas[input_cols])

            np.testing.assert_allclose(actual_arr_cloudpickle, sklearn_arr)
            # np.testing.assert_allclose(actual_arr_pickle, sklearn_arr)
            # np.testing.assert_allclose(actual_arr_joblib, sklearn_arr)


if __name__ == "__main__":
    main()
