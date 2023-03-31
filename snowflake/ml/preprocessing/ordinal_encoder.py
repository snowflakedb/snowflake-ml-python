#!/usr/bin/env python3
#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
import inspect
import numbers
import uuid
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn import utils as sklearn_utils
from sklearn.preprocessing import OrdinalEncoder as SklearnOrdinalEncoder

import snowflake.snowpark.functions as F
import snowflake.snowpark.types as T
from snowflake.ml.framework.base import BaseEstimator, BaseTransformer
from snowflake.ml.utils import telemetry
from snowflake.snowpark import DataFrame, DataFrameWriter, Window
from snowflake.snowpark._internal import utils as snowpark_internal_utils

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "Preprocessing"
_COLUMN_NAME = "_COLUMN_NAME"
# random suffix as this column is dropped during transform
_CATEGORY = f"'_CATEGORY_{snowpark_internal_utils.generate_random_alphanumeric()}'"
_INDEX = "_INDEX"

# constants used to validate the compatibility of the kwargs passed to the sklearn
# transformer with the sklearn version
_SKLEARN_INITIAL_KEYWORDS = "categories"  # initial keywords in sklearn
_SKLEARN_UNUSED_KEYWORDS = "dtype"  # sklearn keywords that are unused in snowml
_SNOWML_ONLY_KEYWORDS = ["input_cols", "output_cols"]  # snowml only keywords not present in sklearn

# Added keywords mapped to the sklearn versions in which they were added. Update mappings in new
# sklearn versions to support parameter validation.
_SKLEARN_ADDED_KEYWORD_TO_VERSION_DICT = {
    "handle_unknown": "0.24",
    "unknown_value": "0.24",
    "encoded_missing_value": "1.1",
}


class OrdinalEncoder(BaseEstimator, BaseTransformer):
    """
    Encode categorical features as an integer column.

    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are converted to ordinal integers. This results in
    a single column of integers (0 to n_categories - 1) per feature.

    Args:
        categories: 'auto' or dict {column_name: ndarray([category])}, default='auto'
            Categories (unique values) per feature:
            - 'auto': Determine categories automatically from the training data.
            - dict: ``categories[column_name]`` holds the categories expected in
              the column provided. The passed categories should not mix strings
              and numeric values within a single feature, and should be sorted in
              case of numeric values.
            The used categories can be found in the ``categories_`` attribute.
        handle_unknown: {'error', 'use_encoded_value'}, default='error'
            When set to 'error' an error will be raised in case an unknown
            categorical feature is present during transform. When set to
            'use_encoded_value', the encoded value of unknown categories will be
            set to the value given for the parameter `unknown_value`.
        unknown_value: int or np.nan, default=None
            When the parameter handle_unknown is set to 'use_encoded_value', this
            parameter is required and will set the encoded value of unknown
            categories. It has to be distinct from the values used to encode any of
            the categories in `fit`.
        encoded_missing_value: Encoded value of missing categories.
        input_cols: Single or multiple input columns.
        output_cols: Single or multiple output columns.
        drop_input_cols: Remove input columns from output if set True. False by default.

    Attributes:
        categories_: The categories of each feature determined during fitting.
    """

    def __init__(
        self,
        *,
        categories: Union[str, Dict[str, np.ndarray]] = "auto",
        handle_unknown: str = "error",
        unknown_value: Optional[Union[int, float]] = None,
        encoded_missing_value: Union[int, float] = np.nan,
        input_cols: Optional[Union[str, Iterable[str]]] = None,
        output_cols: Optional[Union[str, Iterable[str]]] = None,
        drop_input_cols: Optional[bool] = False,
    ) -> None:
        """See class-level docstring."""
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value

        self.categories_: Dict[str, np.ndarray] = {}
        self._categories_list: List[np.ndarray] = []
        self._missing_indices: Dict[int, int] = {}
        self._vocab_table_name = "snowml_preprocessing_ordinal_encoder_temp_table_" + uuid.uuid4().hex

        BaseEstimator.__init__(self)
        BaseTransformer.__init__(self, drop_input_cols=drop_input_cols)

        self.set_input_cols(input_cols)
        self.set_output_cols(output_cols)

    def _reset(self) -> None:
        """
        Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        super()._reset()
        self.categories_ = {}
        self._categories_list = []
        self._missing_indices = {}

        if hasattr(self, "_state_pandas"):
            del self._state_pandas

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def fit(self, dataset: Union[DataFrame, pd.DataFrame]) -> "OrdinalEncoder":
        """
        Fit the OrdinalEncoder to dataset.

        Args:
            dataset: Input dataset.

        Returns:
            Fitted encoder.

        Raises:
            TypeError: If the input dataset is neither a pandas or Snowpark DataFrame.
        """
        self._reset()
        self._validate_keywords()
        super()._check_input_cols()

        if isinstance(dataset, pd.DataFrame):
            self._fit_sklearn(dataset)
        elif isinstance(dataset, DataFrame):
            self._fit_snowpark(dataset)
        else:
            raise TypeError(
                f"Unexpected dataset type: {type(dataset)}."
                "Supported dataset types: snowpark.DataFrame, pandas.DataFrame."
            )

        self._is_fitted = True
        return self

    def _fit_sklearn(self, dataset: pd.DataFrame) -> None:
        dataset = self._use_input_cols_only(dataset)
        sklearn_encoder = self._create_unfitted_sklearn_object()
        sklearn_encoder.fit(dataset[self.input_cols])

        self._categories_list = sklearn_encoder.categories_

        # Set `categories_` and `_state_pandas`
        if len(self.input_cols) != len(self._categories_list):
            raise ValueError("The derived categories mismatch the supplied input columns.")

        _state_pandas_ordinals: List[pd.DataFrame] = []
        for (i, input_col) in enumerate(sorted(self.input_cols)):
            self.categories_[input_col] = self._categories_list[i]
            # A column with values [a, b, b, None, a] will get mapped into a `_column_ordinals`
            # DataFrame with values: [[a, 0.0], [b, 1.0], [None, NaN]]
            _column_ordinals = (
                dataset[input_col]
                .drop_duplicates()
                .sort_values()
                .reset_index(drop=True)
                .rename_axis(_INDEX)
                .to_frame(_CATEGORY)
                .reset_index()
                .reindex(columns=[_CATEGORY, _INDEX])
                .astype({_INDEX: "float64"})
            )
            _column_ordinals.loc[pd.isnull(_column_ordinals[_CATEGORY]), [_CATEGORY, _INDEX]] = (
                None,
                self.encoded_missing_value,
            )
            _column_ordinals.insert(0, _COLUMN_NAME, input_col)
            _state_pandas_ordinals.append(_column_ordinals)

        self._state_pandas = pd.concat(_state_pandas_ordinals, ignore_index=True)
        self._validate_unknown_value()
        self._check_missing_categories()

    def _fit_snowpark(self, dataset: DataFrame) -> None:
        self._fit_category_state(dataset)
        self._validate_unknown_value()
        self._check_missing_categories()

    def _fit_category_state(self, dataset: DataFrame) -> None:
        """Get and index the categories of dataset. Fitted states are saved as a temp
        table `self._state_table`. Fitted categories are assigned to the object.

        Args:
            dataset: Input dataset.
        """
        # columns: COLUMN_NAME, CATEGORY, INDEX
        state_df = self._get_category_index_state_df(dataset)
        # save the dataframe on server side so that transform doesn't need to upload
        statement_params_save_as_table = telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=_SUBPROJECT,
            function_name=telemetry.get_statement_params_full_func_name(
                inspect.currentframe(), self.__class__.__name__
            ),
            api_calls=[DataFrameWriter.save_as_table],
        )
        state_df.write.save_as_table(
            self._vocab_table_name,
            mode="overwrite",
            table_type="temporary",
            statement_params=statement_params_save_as_table,
        )
        statement_params_to_pandas = telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=_SUBPROJECT,
            function_name=telemetry.get_statement_params_full_func_name(
                inspect.currentframe(), self.__class__.__name__
            ),
            api_calls=[DataFrame.to_pandas],
        )
        self._state_pandas = state_df.to_pandas(statement_params=statement_params_to_pandas)

        self._assign_categories()

    def _get_category_index_state_df(self, dataset: DataFrame) -> DataFrame:
        """
        Get and index the categories of each input column in dataset.

        Args:
            dataset: Input dataset.

        Returns:
            State dataframe with columns [COLUMN_NAME, CATEGORY, INDEX].
        """
        state_df: Optional[DataFrame] = None
        for input_col in self.input_cols:
            distinct_dataset = dataset[[input_col]].distinct()

            # encode non-missing categories
            encoded_value_columns = [
                F.lit(input_col).alias(_COLUMN_NAME),
                F.col(input_col).alias(_CATEGORY),
                (F.dense_rank().over(Window.order_by(input_col)) - 1)
                .cast(T.FloatType())
                .alias(_INDEX),  # index categories
            ]
            encoded_value_df = (
                distinct_dataset.filter(F.col(input_col).is_not_null())
                .sort(F.col(input_col).asc())
                .select(encoded_value_columns)
            )

            # encode missing categories
            encoded_missing_value_columns = [
                F.lit(input_col).alias(_COLUMN_NAME),
                F.col(input_col).alias(_CATEGORY),
                F.lit(self.encoded_missing_value).alias(_INDEX),  # index missing categories
            ]
            encoded_missing_value_df = distinct_dataset.filter(F.col(input_col).is_null()).select(
                encoded_missing_value_columns
            )

            all_encoded_value_df = encoded_value_df.union(encoded_missing_value_df)
            state_df = state_df.union(all_encoded_value_df) if state_df is not None else all_encoded_value_df

        return state_df

    def _assign_categories(self) -> None:
        """
        Assign the categories to the object.

        Raises:
            ValueError: If:
                - `self.categories` is provided and the number of categories mismatches
                  the number of input columns.
                - `self.categories` is provided and the column names mismatch the input
                  column names.
        """
        partial_state_arr = self._state_pandas[[_COLUMN_NAME, _CATEGORY]].to_numpy()
        column_names_arr = partial_state_arr[:, 0]
        categories_arr = partial_state_arr[:, 1]

        grouped_categories = {
            col_name: categories_arr[column_names_arr == col_name] for col_name in np.unique(column_names_arr)
        }

        # sort categories with None at the end
        # {column_name: ndarray([category])}
        categories = {
            col_name: np.concatenate((np.sort(cats[~pd.isnull(cats)]), cats[pd.isnull(cats)]))
            for col_name, cats in grouped_categories.items()
        }

        if isinstance(self.categories, str):
            if self.categories == "auto":
                self.categories_ = categories
            else:
                raise ValueError(f"Unsupported value {self.categories} for parameter `categories`.")

        else:
            if self.handle_unknown == "error":
                for input_col in self.input_cols:
                    given_cats = set(self.categories[input_col].tolist())
                    found_cats = set(categories[input_col].tolist())
                    if not found_cats.issubset(given_cats):
                        msg = f"Found unknown categories {found_cats - given_cats} in column {input_col} during fit"
                        raise ValueError(msg)
            self.categories_ = self.categories

        # list of ndarray same as `sklearn.preprocessing.OrdinalEncoder.categories_`
        self._categories_list = []
        for input_col in self.input_cols:
            self._categories_list.append(self.categories_[input_col])

    def _validate_unknown_value(self) -> None:
        """
        When `self.handle_unknown="use_encoded_value"`, validate that
        `self.unknown_value` is not used to encode any known category.

        Raises:
            ValueError: If unknown categories exist in the fitted dataset.
        """
        if self.handle_unknown == "use_encoded_value":
            for feature_cats in self._categories_list:
                if isinstance(self.unknown_value, numbers.Integral) and 0 <= self.unknown_value < len(feature_cats):
                    raise ValueError(
                        "The used value for unknown_value "
                        f"{self.unknown_value} is one of the "
                        "values already used for encoding the "
                        "seen categories."
                    )

    def _check_missing_categories(self) -> None:
        """
        Add missing categories to `self._missing_indices`.
        Validate `self.encoded_missing_value`.
        """
        # stores the missing indices per category
        for cat_idx, categories_for_idx in enumerate(self._categories_list):
            for i, cat in enumerate(categories_for_idx.tolist()):
                if cat is None or cat is np.nan:
                    self._missing_indices[cat_idx] = i
                    break

        self._validate_encoded_missing_value()

    def _validate_encoded_missing_value(self) -> None:
        """
        When missing categories exist, validate that `self.encoded_missing_value`
        is not used to encode any known category.

        Raises:
            ValueError: If missing categories exist and `self.encoded_missing_value` is already
                used to encode a known category.
        """
        if self._missing_indices:
            if not sklearn_utils.is_scalar_nan(self.encoded_missing_value):
                # Features are invalid when they contain a missing category
                # and encoded_missing_value was already used to encode a
                # known category
                invalid_features = [
                    cat_idx
                    for cat_idx, categories_for_idx in enumerate(self._categories_list)
                    if cat_idx in self._missing_indices and 0 <= self.encoded_missing_value < len(categories_for_idx)
                ]

                if invalid_features:
                    # Use feature names if they are available
                    if hasattr(self, "feature_names_in_"):
                        invalid_features = self.feature_names_in_[invalid_features]
                    raise ValueError(
                        f"encoded_missing_value ({self.encoded_missing_value}) "
                        "is already used to encode a known category in features: "
                        f"{invalid_features}"
                    )

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def transform(self, dataset: Union[DataFrame, pd.DataFrame]) -> Union[DataFrame, pd.DataFrame]:
        """
        Transform dataset to ordinal codes.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.

        Raises:
            RuntimeError: If transformer is not fitted first.
            TypeError: If the input dataset is neither a pandas or Snowpark DataFrame.
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer not fitted before calling transform().")
        super()._check_input_cols()
        super()._check_output_cols()

        if isinstance(dataset, DataFrame):
            output_df = self._transform_snowpark(dataset)
        elif isinstance(dataset, pd.DataFrame):
            output_df = self._transform_sklearn(dataset)
        else:
            raise TypeError(
                f"Unexpected dataset type: {type(dataset)}."
                "Supported dataset types: snowpark.DataFrame, pandas.DataFrame."
            )

        return self._drop_input_columns(output_df) if self._drop_input_cols is True else output_df

    def _transform_snowpark(self, dataset: DataFrame) -> DataFrame:
        """
        Transform Snowpark dataframe to ordinal codes.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.
        """
        state_df = (
            dataset._session.table(self._vocab_table_name)
            if dataset._session._table_exists(self._vocab_table_name)
            else dataset._session.create_dataframe(self._state_pandas)
        )

        # replace NULL with nan
        null_category_state_df = state_df.filter(F.col(_CATEGORY).is_null()).with_column(
            _INDEX, F.lit(self.encoded_missing_value)
        )
        state_df = state_df.filter(F.col(_CATEGORY).is_not_null()).union_by_name(null_category_state_df)

        transformed_dataset = dataset
        for idx, input_col in enumerate(self.input_cols):
            output_col = self.output_cols[idx]
            input_col_state_df = state_df.filter(F.col(_COLUMN_NAME) == input_col)[
                [_CATEGORY, _INDEX]
            ].with_column_renamed(_INDEX, output_col)

            # index values through a join operation over dataset and its states
            transformed_dataset = transformed_dataset.join(
                input_col_state_df,
                on=transformed_dataset[input_col].equal_null(input_col_state_df[_CATEGORY]),
                how="left",
            ).drop(_CATEGORY)

        transformed_dataset = self._handle_unknown_in_transform(transformed_dataset)

        return transformed_dataset

    def _create_unfitted_sklearn_object(self) -> SklearnOrdinalEncoder:
        sklearn_args = self.get_sklearn_args(
            default_sklearn_obj=SklearnOrdinalEncoder(),
            sklearn_initial_keywords=_SKLEARN_INITIAL_KEYWORDS,
            sklearn_unused_keywords=_SKLEARN_UNUSED_KEYWORDS,
            snowml_only_keywords=_SNOWML_ONLY_KEYWORDS,
            sklearn_added_keyword_to_version_dict=_SKLEARN_ADDED_KEYWORD_TO_VERSION_DICT,
        )
        return SklearnOrdinalEncoder(**sklearn_args)

    def _create_sklearn_object(self) -> SklearnOrdinalEncoder:
        """
        Get an equivalent sklearn OrdinalEncoder.

        Returns:
            Sklearn OrdinalEncoder.
        """
        encoder = self._create_unfitted_sklearn_object()
        encoder.categories_ = self._categories_list
        encoder._missing_indices = self._missing_indices
        return encoder

    def _validate_keywords(self) -> None:
        if isinstance(self.categories, str) and self.categories != "auto":
            raise ValueError(f"Unsupported value {self.categories} for parameter `categories`.")
        elif isinstance(self.categories, dict):
            if len(self.categories) != len(self.input_cols):
                raise ValueError("The number of categories mismatches the number of input columns.")
            elif set(self.categories.keys()) != set(self.input_cols):
                raise ValueError("The column names of categories mismatch the column names of input columns.")

        if self.handle_unknown not in {"error", "use_encoded_value"}:
            msg = "handle_unknown should be one of 'error', 'use_encoded_value' " f"got {self.handle_unknown}."
            raise ValueError(msg)

        if self.handle_unknown == "use_encoded_value":
            if not (
                sklearn_utils.is_scalar_nan(self.unknown_value) or isinstance(self.unknown_value, numbers.Integral)
            ):
                raise TypeError(
                    "unknown_value should be an integer or "
                    "np.nan when "
                    "handle_unknown is 'use_encoded_value', "
                    f"got {self.unknown_value}."
                )
        elif self.unknown_value is not None:
            raise TypeError(
                "unknown_value should only be set when "
                "handle_unknown is 'use_encoded_value', "
                f"got {self.unknown_value}."
            )

    def _handle_unknown_in_transform(self, transformed_dataset: DataFrame) -> DataFrame:
        """
        Handle unknown values in the transformed dataset.

        Args:
            transformed_dataset: Transformed dataset without unknown values handled.

        Returns:
            Transformed dataset with unknown values handled.

        Raises:
            ValueError: If `self.handle_unknown="error"` and unknown values exist in the
                transformed dataset.
        """
        if self.handle_unknown == "error":
            # dataframe with unknown values
            # columns: COLUMN_NAME, UNKNOWN_VALUE
            unknown_df: Optional[DataFrame] = None
            for idx, input_col in enumerate(self.input_cols):
                output_col = self.output_cols[idx]
                unknown_columns = [
                    F.lit(input_col),
                    F.col(input_col),
                ]
                temp_df = (
                    transformed_dataset[[input_col, output_col]]
                    .distinct()
                    .filter(F.col(output_col).is_null())
                    .select(unknown_columns)
                    .to_df(["COLUMN_NAME", "UNKNOWN_VALUE"])
                )
                unknown_df = unknown_df.union_by_name(temp_df) if unknown_df is not None else temp_df

            if unknown_df is None:
                raise ValueError("snowml internal error caused by handle_unknown='error': empty input columns")

            statement_params = telemetry.get_function_usage_statement_params(
                project=_PROJECT,
                subproject=_SUBPROJECT,
                function_name=telemetry.get_statement_params_full_func_name(
                    inspect.currentframe(), self.__class__.__name__
                ),
                api_calls=[DataFrame.to_pandas],
            )
            unknown_pandas = unknown_df.to_pandas(statement_params=statement_params)
            if unknown_pandas.shape[0] > 0:
                msg = f"Found unknown categories during transform:\n" f"{unknown_pandas.to_string()}"
                raise ValueError(msg)

        if self.handle_unknown == "use_encoded_value":
            # left outer join has already filled unknown values with null
            if not (self.unknown_value is None or sklearn_utils.is_scalar_nan(self.unknown_value)):
                transformed_dataset = transformed_dataset.fillna(self.unknown_value, self.output_cols)

        return transformed_dataset
