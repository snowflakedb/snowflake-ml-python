#!/usr/bin/env python3
import numbers
import uuid
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn import preprocessing, utils as sklearn_utils

from snowflake import snowpark
from snowflake.ml._internal import telemetry, type_utils
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml._internal.utils import identifier
from snowflake.ml.modeling.framework import _utils, base
from snowflake.snowpark import functions as F, types as T

_COLUMN_NAME = "_COLUMN_NAME"
_CATEGORY = "_CATEGORY"
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


class OrdinalEncoder(base.BaseTransformer):
    r"""Encodes categorical features as an integer array.

    In other words, each category (i.e., distinct numeric or string value) is assigned an integer value, starting
    with zero.

    For more details on what this transformer does, see [sklearn.preprocessing.OrdinalEncoder]
    (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html).

    Args:
        categories: The string 'auto' (the default) causes the categories to be extracted from the input columns.
            To specify the categories yourself, pass a dictionary mapping the column name to an ndarray containing the
            categories.
        handle_unknown: Specifies how unknown categories are handled during transformation. Applicable only if\
            categories is not 'auto'.
            Valid values are:
                - 'error': Raise an error if an unknown category is present during transform (default).
                - 'use_encoded_value': When an unknown category is encountered during transform, the specified
                    encoded_missing_value (below) is used.
        encoded_missing_value: The value to be used to encode unknown categories.
        input_cols: The name(s) of one or more columns in a DataFrame containing a feature to be encoded.
        output_cols: The name(s) of one or more columns in a DataFrame in which results will be stored. The number of
            columns specified must match the number of input columns.
        drop_input_cols: Remove input columns from output if set True. False by default.

    Attributes:
        categories_ (dict of ndarray): The categories of each feature determined during fitting. Maps input column
            names to an array of the detected categories.
            Attributes are valid only after fit() has been called.
    """

    def __init__(
        self,
        *,
        categories: Union[str, Dict[str, type_utils.LiteralNDArrayType]] = "auto",
        handle_unknown: str = "error",
        unknown_value: Optional[Union[int, float]] = None,
        encoded_missing_value: Union[int, float] = np.nan,
        input_cols: Optional[Union[str, Iterable[str]]] = None,
        output_cols: Optional[Union[str, Iterable[str]]] = None,
        drop_input_cols: Optional[bool] = False,
    ) -> None:
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
        super().__init__(drop_input_cols=drop_input_cols)
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value

        self.categories_: Dict[str, type_utils.LiteralNDArrayType] = {}
        self._categories_list: List[type_utils.LiteralNDArrayType] = []
        self._missing_indices: Dict[int, int] = {}
        self._infrequent_enabled = False
        self._vocab_table_name = "snowml_preprocessing_ordinal_encoder_temp_table_" + uuid.uuid4().hex

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
        project=base.PROJECT,
        subproject=base.SUBPROJECT,
    )
    def fit(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> "OrdinalEncoder":
        """
        Fit the OrdinalEncoder to dataset.

        Validates the transformer arguments and derives the list of categories (distinct values) from the data, making
        this list available as an attribute of the transformer instance (see Attributes).

        Returns the transformer instance.

        Args:
            dataset: Input dataset.

        Returns:
            Fitted encoder.
        """
        self._reset()
        self._validate_keywords()
        super()._check_input_cols()
        super()._check_dataset_type(dataset)

        if isinstance(dataset, pd.DataFrame):
            self._fit_sklearn(dataset)
        else:
            self._fit_snowpark(dataset)
        self._validate_unknown_value()
        self._check_missing_categories()

        self._is_fitted = True
        return self

    def _fit_sklearn(self, dataset: pd.DataFrame) -> None:
        dataset = self._use_input_cols_only(dataset)
        sklearn_encoder = self._create_unfitted_sklearn_object()
        sklearn_encoder.fit(dataset[self.input_cols])

        self._categories_list = sklearn_encoder.categories_

        _state_pandas_ordinals: List[pd.DataFrame] = []
        for idx, input_col in enumerate(sorted(self.input_cols)):
            self.categories_[input_col] = self._categories_list[idx]
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

    def _fit_snowpark(self, dataset: snowpark.DataFrame) -> None:
        self._fit_category_state(dataset)

    def _fit_category_state(self, dataset: snowpark.DataFrame) -> None:
        """
        Get and index the categories of dataset. Fitted states are saved as a temp
        table `self._state_table`. Fitted categories are assigned to the object.

        Args:
            dataset: Input dataset.
        """
        # columns: COLUMN_NAME, CATEGORY, INDEX
        state_df = self._get_category_index_state_df(dataset)
        # save the dataframe on server side so that transform doesn't need to upload
        state_df.write.save_as_table(
            self._vocab_table_name,
            mode="overwrite",
            table_type="temporary",
            statement_params=telemetry.get_statement_params(base.PROJECT, base.SUBPROJECT, self.__class__.__name__),
        )
        self._state_pandas = state_df.to_pandas(
            statement_params=telemetry.get_statement_params(base.PROJECT, base.SUBPROJECT, self.__class__.__name__)
        )

        self._assign_categories()

    def _get_category_index_state_df(self, dataset: snowpark.DataFrame) -> snowpark.DataFrame:
        """
        Get and index the categories of each input column in dataset.
        If `categories` is provided, use the given categories with orders preserved;
        plus if `self.handle_unknown="error"`, check if the given categories
        contain all categories in dataset.

        Args:
            dataset: Input dataset.

        Returns:
            State dataframe with columns [COLUMN_NAME, CATEGORY, INDEX].

        Raises:
            SnowflakeMLException: If `self.categories` is provided, `self.handle_unknown="error"`,
                and unknown categories exist in dataset.
        """
        # states of categories found in dataset
        found_state_df: Optional[snowpark.DataFrame] = None
        for input_col in self.input_cols:
            distinct_dataset = dataset[[input_col]].distinct()

            # encode non-missing categories
            encoded_value_columns = [
                F.lit(input_col).alias(_COLUMN_NAME),
                F.col(input_col).cast(T.StringType()).alias(_CATEGORY),
                (F.dense_rank().over(snowpark.Window.order_by(input_col)) - 1)
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
                F.col(input_col).cast(T.StringType()).alias(_CATEGORY),
                # index missing categories
                F.lit(self.encoded_missing_value).alias(_INDEX),
            ]
            encoded_missing_value_df = distinct_dataset.filter(F.col(input_col).is_null()).select(
                encoded_missing_value_columns
            )

            all_encoded_value_df = encoded_value_df.union(encoded_missing_value_df)
            found_state_df = (
                found_state_df.union(all_encoded_value_df) if found_state_df is not None else all_encoded_value_df
            )

        assert found_state_df is not None
        if self.categories != "auto":
            state_data = []
            assert isinstance(self.categories, dict)
            for input_col, cats in self.categories.items():
                for idx, cat in enumerate(cats.tolist()):
                    state_data.append([input_col, cat, idx])
            # states of given categories
            assert dataset._session is not None
            given_state_df = dataset._session.create_dataframe(
                data=state_data, schema=[_COLUMN_NAME, _CATEGORY, _INDEX]
            )

            # check given categories
            if self.handle_unknown == "error":
                unknown_df = (
                    found_state_df[[_COLUMN_NAME, _CATEGORY]]
                    .subtract(given_state_df[[_COLUMN_NAME, _CATEGORY]])
                    .to_df(["COLUMN_NAME", "UNKNOWN_VALUE"])
                )
                unknown_pandas = unknown_df.to_pandas(
                    statement_params=telemetry.get_statement_params(
                        base.PROJECT, base.SUBPROJECT, self.__class__.__name__
                    )
                )
                if not unknown_pandas.empty:
                    msg = f"Found unknown categories during fit:\n{unknown_pandas.to_string()}"
                    raise exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_DATA,
                        original_exception=ValueError(msg),
                    )

            return given_state_df

        return found_state_df

    def _assign_categories(self) -> None:
        """Assign the categories to the object."""
        if isinstance(self.categories, str):
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
            self.categories_ = categories
        else:
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
            SnowflakeMLException: If unknown categories exist in the fitted dataset.
        """
        if self.handle_unknown == "use_encoded_value":
            for feature_cats in self._categories_list:
                if isinstance(self.unknown_value, numbers.Integral) and 0 <= self.unknown_value < len(feature_cats):
                    raise exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_ATTRIBUTE,
                        original_exception=ValueError(
                            "The used value for unknown_value "
                            f"{self.unknown_value} is one of the "
                            "values already used for encoding the "
                            "seen categories."
                        ),
                    )

    def _check_missing_categories(self) -> None:
        """
        Add missing categories to `self._missing_indices`.
        Validate `self.encoded_missing_value`.
        """
        # stores the missing indices per category
        for cat_idx, categories_for_idx in enumerate(self._categories_list):
            for idx, cat in enumerate(categories_for_idx.tolist()):
                if cat is None or cat is np.nan:
                    self._missing_indices[cat_idx] = idx
                    break

        self._validate_encoded_missing_value()

    def _validate_encoded_missing_value(self) -> None:
        """
        When missing categories exist, validate that `self.encoded_missing_value`
        is not used to encode any known category.

        Raises:
            SnowflakeMLException: If missing categories exist and `self.encoded_missing_value` is already
                used to encode a known category.
        """
        if self._missing_indices:
            if not sklearn_utils.is_scalar_nan(self.encoded_missing_value):
                # Features are invalid when they contain a missing category
                # and encoded_missing_value was already used to encode a
                # known category
                invalid_features = [
                    self.input_cols[cat_idx]
                    for cat_idx, categories_for_idx in enumerate(self._categories_list)
                    if cat_idx in self._missing_indices and 0 <= self.encoded_missing_value < len(categories_for_idx)
                ]

                if invalid_features:
                    raise exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_ATTRIBUTE,
                        original_exception=ValueError(
                            f"encoded_missing_value ({self.encoded_missing_value}) is already used to encode a known "
                            f"category in features: {invalid_features}."
                        ),
                    )

    @telemetry.send_api_usage_telemetry(
        project=base.PROJECT,
        subproject=base.SUBPROJECT,
    )
    @telemetry.add_stmt_params_to_df(
        project=base.PROJECT,
        subproject=base.SUBPROJECT,
    )
    def transform(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> Union[snowpark.DataFrame, pd.DataFrame]:
        """
        Transform dataset to ordinal codes.

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
        Transform Snowpark dataframe to ordinal codes.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.
        """
        passthrough_columns = [c for c in dataset.columns if c not in self.output_cols]
        assert dataset._session is not None
        state_df = (
            dataset._session.table(self._vocab_table_name)
            if _utils.table_exists(
                dataset._session,
                self._vocab_table_name,
                telemetry.get_statement_params(base.PROJECT, base.SUBPROJECT, self.__class__.__name__),
            )
            else dataset._session.create_dataframe(self._state_pandas)
        )

        # replace NULL with nan
        null_category_state_df = state_df.filter(F.col(_CATEGORY).is_null()).with_column(
            _INDEX, F.lit(self.encoded_missing_value)
        )
        state_df = state_df.filter(F.col(_CATEGORY).is_not_null()).union_by_name(null_category_state_df)

        suffix = "_" + uuid.uuid4().hex.upper()
        transformed_dataset = dataset

        for idx, input_col in enumerate(self.input_cols):
            output_col = self.output_cols[idx]
            input_col_state_df = state_df.filter(F.col(_COLUMN_NAME) == input_col)[
                [_CATEGORY, _INDEX]
            ].with_column_renamed(_INDEX, output_col)

            # index values through a join operation over dataset and its states
            # In case of inplace transform, origin column name adds suffix (lsuffix=suffix)
            transformed_dataset = (
                transformed_dataset.join(
                    input_col_state_df,
                    on=transformed_dataset[input_col].cast(T.StringType()).equal_null(input_col_state_df[_CATEGORY]),
                    how="left",
                    lsuffix=suffix,
                )
                .drop(_CATEGORY)
                .drop(identifier.concat_names([input_col, suffix]))
            )

            # in case of duplicate column, filter them
            output_cols = transformed_dataset.columns
            if output_col not in output_cols:
                output_cols.append(output_col)
            transformed_dataset = transformed_dataset[output_cols]

        if _CATEGORY + suffix in transformed_dataset.columns:
            transformed_dataset = transformed_dataset.with_column_renamed(F.col(_CATEGORY + suffix), _CATEGORY)

        transformed_dataset = self._handle_unknown_in_transform(transformed_dataset)
        # Reorder columns. Passthrough columns are added at the right to the output of the transformers.
        transformed_dataset = transformed_dataset[self.output_cols + passthrough_columns]
        return transformed_dataset

    def _create_unfitted_sklearn_object(self) -> preprocessing.OrdinalEncoder:
        sklearn_args = self.get_sklearn_args(
            default_sklearn_obj=preprocessing.OrdinalEncoder(),
            sklearn_initial_keywords=_SKLEARN_INITIAL_KEYWORDS,
            sklearn_unused_keywords=_SKLEARN_UNUSED_KEYWORDS,
            snowml_only_keywords=_SNOWML_ONLY_KEYWORDS,
            sklearn_added_keyword_to_version_dict=_SKLEARN_ADDED_KEYWORD_TO_VERSION_DICT,
        )
        return preprocessing.OrdinalEncoder(**sklearn_args)

    def _create_sklearn_object(self) -> preprocessing.OrdinalEncoder:
        """
        Get an equivalent sklearn OrdinalEncoder.

        Returns:
            Sklearn OrdinalEncoder.
        """
        encoder = self._create_unfitted_sklearn_object()
        if self._is_fitted:
            encoder.categories_ = self._categories_list
            encoder._missing_indices = self._missing_indices
            encoder._infrequent_enabled = self._infrequent_enabled
        return encoder

    def _validate_keywords(self) -> None:
        if isinstance(self.categories, str) and self.categories != "auto":
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=ValueError(f"Unsupported `categories` value: {self.categories}."),
            )
        elif isinstance(self.categories, dict):
            if len(self.categories) != len(self.input_cols):
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ATTRIBUTE,
                    original_exception=ValueError(
                        f"The number of categories ({len(self.categories)}) mismatches the number of input columns "
                        f"({len(self.input_cols)})."
                    ),
                )
            elif set(self.categories.keys()) != set(self.input_cols):
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ATTRIBUTE,
                    original_exception=ValueError(
                        "The column names of categories mismatch the column names of input columns."
                    ),
                )

        if self.handle_unknown not in {"error", "use_encoded_value"}:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=ValueError(
                    f"`handle_unknown` must be one of 'error', 'use_encoded_value', got {self.handle_unknown}."
                ),
            )

        if self.handle_unknown == "use_encoded_value":
            if not (
                sklearn_utils.is_scalar_nan(self.unknown_value) or isinstance(self.unknown_value, numbers.Integral)
            ):
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ATTRIBUTE,
                    original_exception=TypeError(
                        "`unknown_value` must be an integer or np.nan when `handle_unknown` is 'use_encoded_value', "
                        f"got {self.unknown_value}."
                    ),
                )
        elif self.unknown_value is not None:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=TypeError(
                    "`unknown_value` must only be set when `handle_unknown` is 'use_encoded_value', "
                    f"got {self.unknown_value}."
                ),
            )

    def _handle_unknown_in_transform(self, transformed_dataset: snowpark.DataFrame) -> snowpark.DataFrame:
        """
        Handle unknown values in the transformed dataset.

        Args:
            transformed_dataset: Transformed dataset without unknown values handled.

        Returns:
            Transformed dataset with unknown values handled.

        Raises:
            SnowflakeMLException: If `self.handle_unknown="error"` and unknown values exist in the
                transformed dataset.
        """
        if self.handle_unknown == "error":
            # dataframe with unknown values
            # columns: COLUMN_NAME, UNKNOWN_VALUE
            unknown_df: Optional[snowpark.DataFrame] = None
            for idx, input_col in enumerate(self.input_cols):
                output_col = self.output_cols[idx]
                unknown_columns = [
                    F.lit(input_col),
                    F.col(input_col),
                ]
                temp_df = (
                    transformed_dataset[list({input_col, output_col})]
                    .distinct()
                    .filter(F.col(output_col).is_null())
                    .select(unknown_columns)
                    .to_df(["COLUMN_NAME", "UNKNOWN_VALUE"])
                )
                unknown_df = unknown_df.union_by_name(temp_df) if unknown_df is not None else temp_df

            if unknown_df is None:
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_PYTHON_ERROR,
                    original_exception=ValueError(
                        "Internal error caused by handle_unknown='error': empty input columns."
                    ),
                )

            unknown_pandas = unknown_df.to_pandas(
                statement_params=telemetry.get_statement_params(base.PROJECT, base.SUBPROJECT, self.__class__.__name__)
            )
            if not unknown_pandas.empty:
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_DATA,
                    original_exception=ValueError(
                        f"Found unknown categories during transform:\n{unknown_pandas.to_string()}"
                    ),
                )

        if self.handle_unknown == "use_encoded_value":
            # left outer join has already filled unknown values with null
            if not (self.unknown_value is None or sklearn_utils.is_scalar_nan(self.unknown_value)):
                transformed_dataset = transformed_dataset.na.fill(self.unknown_value, self.output_cols)

        return transformed_dataset
