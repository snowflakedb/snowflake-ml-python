#!/usr/bin/env python3
import copy
from typing import Any, Dict, Iterable, Optional, Type, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import impute

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml.modeling.framework import _utils, base
from snowflake.snowpark import functions as F, types as T
from snowflake.snowpark._internal import utils as snowpark_utils

_SUBPROJECT = "Impute"

STRATEGY_TO_STATE_DICT = {
    "constant": None,
    "mean": _utils.NumericStatistics.MEAN,
    "median": _utils.NumericStatistics.MEDIAN,
    "most_frequent": _utils.BasicStatistics.MODE,
}

SNOWFLAKE_DATATYPE_TO_NUMPY_DTYPE_MAP: Dict[Type[T.DataType], npt.DTypeLike] = {
    T.ByteType: np.dtype("int8"),
    T.ShortType: np.dtype("int16"),
    T.IntegerType: np.dtype("int32"),
    T.LongType: np.dtype("int64"),
    T.FloatType: np.dtype("float64"),
    T.DoubleType: np.dtype("float64"),
    T.DecimalType: np.dtype("object"),
    T.StringType: np.dtype("str"),
}

# Constants used to validate the compatibility of the kwargs passed to the sklearn
# transformer with the sklearn version.
_SKLEARN_INITIAL_KEYWORDS = [
    "missing_values",
    "strategy",
    "fill_value",
]  # initial keywords in sklearn
_SKLEARN_UNUSED_KEYWORDS = [
    "verbose",
    "copy",
    "add_indicator",
]  # sklearn keywords that are unused in snowml
_SNOWML_ONLY_KEYWORDS = [
    "input_cols",
    "output_cols",
]  # snowml only keywords not present in sklearn

# Added keywords mapped to the sklearn versions in which they were added. Update mappings in new
# sklearn versions to support parameter validation.
_SKLEARN_ADDED_KEYWORD_TO_VERSION_DICT = {
    "keep_empty_features": "1.2",
}

# Added keyword argument values mapped to the sklearn versions in which they were added. Update
# mappings in new sklearn versions to support parameter validation.
_SKLEARN_ADDED_KWARG_VALUE_TO_VERSION_DICT = {
    "strategy": {"constant": "0.20"},
}

_NUMERIC_TYPES = [
    T.ByteType(),
    T.ShortType(),
    T.IntegerType(),
    T.LongType(),
    T.FloatType(),
    T.DoubleType(),
]


class SimpleImputer(base.BaseTransformer):
    def __init__(
        self,
        *,
        missing_values: Optional[Union[int, float, str, np.float64]] = np.nan,
        strategy: Optional[str] = "mean",
        fill_value: Optional[Union[str, float]] = None,
        input_cols: Optional[Union[str, Iterable[str]]] = None,
        output_cols: Optional[Union[str, Iterable[str]]] = None,
        drop_input_cols: Optional[bool] = False,
    ) -> None:
        """
        Univariate imputer for completing missing values with simple strategies.
        Note that the `add_indicator` param/functionality is not implemented.

        Args:
            missing_values: The values to treat as missing and impute during transform.
            strategy: The imputation strategy.
                * If "mean", replace missing values using the mean along each column.
                  Can only be used with numeric data.
                * If "median", replace missing values using the median along each column.
                  Can only be used with numeric data.
                * If "most_frequent", replace missing using the most frequent value along each column.
                  Can be used with strings or numeric data.
                  If there is more than one such value, only the smallest is returned.
                * If "constant", replace the missing values with `fill_value`. Can be used with strings or numeric data.
            fill_value:
                When `strategy == "constant"`, `fill_value` is used to replace all occurrences of `missing_values`.
                For string or object data types, `fill_value` must be a string. If `None`, `fill_value` will be 0 when
                imputing numerical data and `missing_value` for strings and object data types.
            input_cols:
                Columns to use as inputs during fit or transform.
            output_cols:
                New column labels for the columns that will contain the output of a transform.
            drop_input_cols: Remove input columns from output if set True. False by default.

        Attributes:
        statistics_: dict {input_col: stats_value}
            Dict containing the imputation fill value for each feature. Computing statistics can result in `np.nan`
            values. During `transform`, features corresponding to `np.nan` statistics will be discarded.
        n_features_in_: int
            Number of features seen during `fit`.
        feature_names_in_: ndarray of shape (n_features_in,)
            Names of features seen during `fit`.

        TODO(thoyt): Implement logic for `add_indicator` parameter and `indicator_` attribute. Requires
            `snowflake.ml.impute.MissingIndicator` to be implemented.

        Raises:
            SnowflakeMLException: If strategy is invalid, or if fill value is specified for strategy that isn't
                "constant".
        """
        super().__init__(drop_input_cols=drop_input_cols)
        if strategy in STRATEGY_TO_STATE_DICT:
            self.strategy = strategy
        else:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    f"Invalid strategy {strategy}. Strategy must be one of {STRATEGY_TO_STATE_DICT.keys()}"
                ),
            )

        # Check that the strategy is "constant" if `fill_value` is specified.
        if fill_value is not None and strategy != "constant":
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError("fill_value may only be specified if the strategy is 'constant'."),
            )

        self.fill_value = fill_value
        self.missing_values = missing_values
        # TODO(hayu): [SNOW-752265] Support SimpleImputer keep_empty_features.
        #  Add back when `keep_empty_features` is supported.
        # self.keep_empty_features = keep_empty_features

        self.set_input_cols(input_cols)
        self.set_output_cols(output_cols)

    def _reset(self) -> None:
        """
        Reset internal data-dependent state of the imputer, if necessary.
        __init__ parameters are not touched.
        """
        super()._reset()
        # Checking one attribute is enough, because they are all set during fit. Attributes should be deleted
        # since they are undefined after `__init__`.
        if hasattr(self, "statistics_"):
            del self.statistics_
            del self.n_features_in_
            del self.feature_names_in_
            del self._sklearn_fit_dtype

    def _get_dataset_input_col_datatypes(self, dataset: snowpark.DataFrame) -> Dict[str, T.DataType]:
        """
        Checks that the input columns are all the same datatype category(except for most_frequent strategy) and
        returns the datatype.

        Args:
            dataset: The input dataframe.

        Returns:
            The datatype of the input columns.

        Raises:
            SnowflakeMLException: If the input columns are not all the same datatype category or if the datatype is not
                supported.
        """

        def check_type_consistency(col_types: Dict[str, T.DataType]) -> None:
            is_numeric_type = None
            for col_name, col_type in col_types.items():
                if is_numeric_type is None:
                    is_numeric_type = True if col_type in _NUMERIC_TYPES else False
                if (col_type in _NUMERIC_TYPES) ^ is_numeric_type:
                    raise exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_DATA_TYPE,
                        original_exception=TypeError(
                            f"Inconsistent input column types. Column {col_name} type {col_type} does not match"
                            " previous type category."
                        ),
                    )

        input_col_datatypes = {}
        for field in dataset.schema.fields:
            if field.name in self.input_cols:
                if not any(
                    isinstance(field.datatype, potential_type)
                    for potential_type in SNOWFLAKE_DATATYPE_TO_NUMPY_DTYPE_MAP.keys()
                ):
                    raise exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_DATA_TYPE,
                        original_exception=TypeError(
                            f"Input column type {field.datatype} is not supported by the SimpleImputer."
                        ),
                    )
                input_col_datatypes[field.name] = field.datatype
        if self.strategy != "most_frequent":
            check_type_consistency(input_col_datatypes)

        return input_col_datatypes

    @telemetry.send_api_usage_telemetry(project=base.PROJECT, subproject=_SUBPROJECT)
    def fit(self, dataset: snowpark.DataFrame) -> "SimpleImputer":
        """
        Compute values to impute for the dataset according to the strategy.

        Args:
            dataset: Input dataset.

        Returns:
            Fitted simple imputer.
        """
        super()._check_input_cols()

        # In order to fit, the input columns should have the same type.
        input_col_datatypes = self._get_dataset_input_col_datatypes(dataset)

        self.statistics_: Dict[str, Any] = {}
        statement_params = telemetry.get_statement_params(base.PROJECT, _SUBPROJECT, self.__class__.__name__)

        if self.strategy == "constant":
            if self.fill_value is None:
                # Select a fill_value according to the datatypes of the input columns. If the input columns are all
                # numeric, the fill_value is 0, and if there are any string columns, we will use "missing_data".
                # We could potentially improve this by using a different fill value per column according to its type,
                # but use this algorithm for sklearn compatibility.
                self.fill_value = 0
                for input_col_datatype in list(input_col_datatypes):
                    if isinstance(input_col_datatype, T.StringType):
                        self.fill_value = "missing_data"
                        break

            for input_col in self.input_cols:
                # Check whether input column is empty if necessary.
                if (
                    # TODO(hayu): [SNOW-752265] Support SimpleImputer keep_empty_features.
                    #  Add back when `keep_empty_features` is supported.
                    # not self.keep_empty_features
                    # and dataset.filter(F.col(input_col).is_not_null()).count(statement_params=statement_params) == 0
                    dataset.filter(F.col(input_col).is_not_null()).count(statement_params=statement_params)
                    == 0
                ):
                    self.statistics_[input_col] = np.nan
                else:
                    self.statistics_[input_col] = self.fill_value
        else:
            state = STRATEGY_TO_STATE_DICT[self.strategy]
            assert state is not None
            dataset_copy = copy.copy(dataset)
            if not pd.isna(self.missing_values):
                # Replace `self.missing_values` with null to avoid including it when computing states.
                dataset_copy = dataset_copy.na.replace(self.missing_values, None)
            _computed_states = self._compute(dataset_copy, self.input_cols, states=[state])
            for input_col in self.input_cols:
                statistic = _computed_states[input_col][state]
                self.statistics_[input_col] = np.nan if statistic is None else statistic
                if self.strategy == "mean":
                    self.statistics_[input_col] = float(self.statistics_[input_col])
                elif self.strategy == "most_frequent":
                    # Check if there is only one occurrence of the value. If so, the statistic should be the minimum
                    # value in the dataset.
                    if dataset.filter(F.col(input_col) == statistic).count(statement_params=statement_params) == 1:
                        statistic_min = self._compute(dataset, [input_col], states=[_utils.NumericStatistics.MIN])
                        self.statistics_[input_col] = statistic_min[input_col][_utils.NumericStatistics.MIN]

        self.n_features_in_ = len(self.input_cols)
        self.feature_names_in_ = self.input_cols

        # This attribute is set during `fit` by sklearn objects. In order to avoid fitting
        # the sklearn object directly when creating the sklearn simple imputer, we have to
        # set this property.
        self._sklearn_fit_dtype = max(  # type:ignore[type-var]
            SNOWFLAKE_DATATYPE_TO_NUMPY_DTYPE_MAP[type(input_col_datatypes[input_col])] for input_col in self.input_cols
        )

        self._is_fitted = True
        return self

    @telemetry.send_api_usage_telemetry(project=base.PROJECT, subproject=_SUBPROJECT)
    @telemetry.add_stmt_params_to_df(project=base.PROJECT, subproject=_SUBPROJECT)
    def transform(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> Union[snowpark.DataFrame, pd.DataFrame]:
        """
        Transform the input dataset by imputing the computed statistics in the input columns.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.
        """
        self._enforce_fit()
        super()._check_input_cols()
        super()._check_output_cols()
        super()._check_dataset_type(dataset)

        if isinstance(dataset, snowpark.DataFrame):
            output_df = self._transform_snowpark(dataset)
        else:
            output_df = self._transform_sklearn(dataset)

        return self._drop_input_columns(output_df) if self._drop_input_cols is True else output_df

    def _transform_snowpark(self, dataset: snowpark.DataFrame) -> snowpark.DataFrame:
        """
        Perform imputation in snowpark dataframe.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.
        """
        passthrough_columns = [c for c in dataset.columns if c not in self.output_cols]
        output_columns = [F.col(input_col) for input_col in self.input_cols]
        transformed_dataset: snowpark.DataFrame = dataset.with_columns(self.output_cols, output_columns)

        # Get the type of input columns.
        input_col_datatypes = self._get_dataset_input_col_datatypes(dataset)

        # The output columns that are copies of input columns can't be imputed unless the input columns are temporarily
        # renamed.
        temp_input_cols = []
        for input_col, output_col in zip(self.input_cols, self.output_cols):
            if input_col != output_col:
                temp_input_col = f"{input_col}_{snowpark_utils.generate_random_alphanumeric()}"
                transformed_dataset = transformed_dataset.with_column_renamed(input_col, temp_input_col)
                temp_input_cols.append(temp_input_col)

        # In each column, replace the instances of `_missing_values` with the statistic. If the statistic
        # is nan and keep_missing_values is True, replace the values with `0` unless `fill_value` is specified.
        # Otherwise, drop the output column.
        for input_col, output_col in zip(self.input_cols, self.output_cols):
            statistic = self.statistics_[input_col]
            fill_value = statistic
            if pd.isna(statistic):
                # TODO(hayu): [SNOW-752265] Support SimpleImputer keep_empty_features.
                #  Add back when `keep_empty_features` is supported.
                # if not self.keep_empty_features:
                #     # Drop the column and continue.
                #     transformed_dataset.drop(output_col)
                #     continue
                # Drop the column.
                transformed_dataset.drop(output_col)

            if self.missing_values is not None and pd.isna(self.missing_values):
                # Use `fillna` for replacing nans. Check if the column has a string data type, or coerce a float.
                if not isinstance(input_col_datatypes[input_col], T.StringType):
                    statistic = float(statistic)
                transformed_dataset = transformed_dataset.na.fill({output_col: statistic})
            else:
                transformed_dataset = transformed_dataset.na.replace(
                    self.missing_values,
                    fill_value,
                    subset=[output_col],
                )

        # Rename the input_cols as needed.
        for input_col, output_col in zip(self.input_cols, self.output_cols):
            if input_col != output_col:
                temp_input_col = temp_input_cols.pop(0)
                transformed_dataset = transformed_dataset.with_column_renamed(temp_input_col, input_col)

        # Reorder columns. Passthrough columns are added at the right to the output of the transformers.
        transformed_dataset = transformed_dataset[self.output_cols + passthrough_columns]
        return transformed_dataset

    def _create_sklearn_object(self) -> impute.SimpleImputer:
        """
        Get an equivalent sklearn SimpleImputer.

        Returns:
            Sklearn SimpleImputer.
        """
        sklearn_args = self.get_sklearn_args(
            default_sklearn_obj=impute.SimpleImputer(),
            sklearn_initial_keywords=_SKLEARN_INITIAL_KEYWORDS,
            sklearn_unused_keywords=_SKLEARN_UNUSED_KEYWORDS,
            snowml_only_keywords=_SNOWML_ONLY_KEYWORDS,
            sklearn_added_keyword_to_version_dict=_SKLEARN_ADDED_KEYWORD_TO_VERSION_DICT,
        )

        simple_imputer = impute.SimpleImputer(**sklearn_args)
        if self._is_fitted:
            simple_imputer.statistics_ = np.array(list(self.statistics_.values()))
            simple_imputer.n_features_in_ = self.n_features_in_
            simple_imputer.feature_names_in_ = self.feature_names_in_
            simple_imputer._fit_dtype = self._sklearn_fit_dtype

        return simple_imputer
