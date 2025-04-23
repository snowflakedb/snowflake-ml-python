#!/usr/bin/env python3
import numbers
import uuid
from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd
from sklearn import preprocessing

from snowflake import snowpark
from snowflake.ml._internal import telemetry, type_utils
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml._internal.utils import identifier, import_utils
from snowflake.ml.modeling.framework import _utils, base
from snowflake.snowpark import functions as F, types as T
from snowflake.snowpark._internal import utils as snowpark_utils

is_scalar_nan = import_utils.import_with_fallbacks(
    "sklearn.utils.is_scalar_nan", "sklearn.utils._missing.is_scalar_nan"
)

_COLUMN_NAME = "_COLUMN_NAME"
_CATEGORY = "_CATEGORY"
_INDEX = "_INDEX"
_COLUMN_BATCH_SIZE = 20

# constants used to validate the compatibility of the kwargs passed to the sklearn
# transformer with the sklearn version
_SKLEARN_INITIAL_KEYWORDS = "categories"  # initial keywords in sklearn
_SKLEARN_UNUSED_KEYWORDS = "dtype"  # sklearn keywords that are unused in snowml
_SNOWML_ONLY_KEYWORDS = ["input_cols", "output_cols", "passthrough_cols"]  # snowml only keywords not present in sklearn

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
        categories: Union[str, List[type_utils.LiteralNDArrayType], Dict[str, type_utils.LiteralNDArrayType]],
        default="auto"
            The string 'auto' (the default) causes the categories to be extracted from the input columns.
            To specify the categories yourself, pass either (1) a list of ndarrays containing the categories or
            (2) a dictionary mapping the column name to an ndarray containing the
            categories.

        handle_unknown: str, default="error"
            Specifies how unknown categories are handled during transformation. Applicable only if
            categories is not 'auto'.
            Valid values are:
                - 'error': Raise an error if an unknown category is present during transform (default).
                - 'use_encoded_value': When an unknown category is encountered during transform, the specified
                    encoded_missing_value (below) is used.

        unknown_value: Optional[Union[int, float]], default=None
            When the parameter handle_unknown is set to 'use_encoded_value', this parameter is required and
            will set the encoded value of unknown categories. It has to be distinct from the values used to encode any
            of the categories in `fit`.

        encoded_missing_value: Union[int, float], default=np.nan
            The value to be used to encode unknown categories.

        input_cols: Optional[Union[str, List[str]]], default=None
            The name(s) of one or more columns in the input DataFrame containing feature(s) to be encoded. Input
            columns must be specified before fit with this argument or after initialization with the
            `set_input_cols` method. This argument is optional for API consistency.

        output_cols: Optional[Union[str, List[str]]], default=None
            The prefix to be used for encoded output for each input column. The number of
            output column prefixes specified must equal the number of input columns. Output column prefixes must be
            specified before transform with this argument or after initialization with the `set_output_cols` method.

        passthrough_cols: Optional[Union[str, List[str]]], default=None
            A string or a list of strings indicating column names to be excluded from any
            operations (such as train, transform, or inference). These specified column(s)
            will remain untouched throughout the process. This option is helpful in scenarios
            requiring automatic input_cols inference, but need to avoid using specific
            columns, like index columns, during training or inference.

        drop_input_cols: Optional[bool], default=False
            Remove input columns from output if set True. False by default.

    Attributes:
        categories_ (dict of ndarray): List[type_utils.LiteralNDArrayType]
            The categories of each feature determined during fitting. Maps input column
            names to an array of the detected categories.
            Attributes are valid only after fit() has been called.
    """

    def __init__(
        self,
        *,
        categories: Union[str, list[type_utils.LiteralNDArrayType], dict[str, type_utils.LiteralNDArrayType]] = "auto",
        handle_unknown: str = "error",
        unknown_value: Optional[Union[int, float]] = None,
        encoded_missing_value: Union[int, float] = np.nan,
        input_cols: Optional[Union[str, Iterable[str]]] = None,
        output_cols: Optional[Union[str, Iterable[str]]] = None,
        passthrough_cols: Optional[Union[str, Iterable[str]]] = None,
        drop_input_cols: Optional[bool] = False,
    ) -> None:
        """
        Encode categorical features as an integer column.

        The input to this transformer should be an array-like of integers or
        strings, denoting the values taken on by categorical (discrete) features.
        The features are converted to ordinal integers. This results in
        a single column of integers (0 to n_categories - 1) per feature.

        Args:
            categories: 'auto', list of array-like, or dict {column_name: ndarray([category])}, default='auto'
                Categories (unique values) per feature:
                - 'auto': Determine categories automatically from the training data.
                - list: ``categories[i]`` holds the categories expected in the ith
                  column. The passed categories should not mix strings and numeric
                  values within a single feature, and should be sorted in case of
                  numeric values.
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
            passthrough_cols: A string or a list of strings indicating column names to be excluded from any
                operations (such as train, transform, or inference). These specified column(s)
                will remain untouched throughout the process. This option is helful in scenarios
                requiring automatic input_cols inference, but need to avoid using specific
                columns, like index columns, during in training or inference.
            drop_input_cols: Remove input columns from output if set True. False by default.

        Attributes:
            categories_: The categories of each feature determined during fitting.
        """
        super().__init__(drop_input_cols=drop_input_cols)
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value

        self.categories_: dict[str, type_utils.LiteralNDArrayType] = {}
        self._categories_list: list[type_utils.LiteralNDArrayType] = []
        self._missing_indices: dict[int, int] = {}
        self._infrequent_enabled = False
        self._vocab_table_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.TABLE)

        self.set_input_cols(input_cols)
        self.set_output_cols(output_cols)
        self.set_passthrough_cols(passthrough_cols)

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

    def _fit(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> "OrdinalEncoder":
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

        _state_pandas_ordinals: list[pd.DataFrame] = []
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
            if isinstance(self.categories, list):
                categories_map = {col_name: cats for col_name, cats in zip(self.input_cols, self.categories)}
            elif isinstance(self.categories, dict):
                categories_map = self.categories
            else:
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=ValueError(
                        f"Invalid type {type(self.categories)} provided for argument `categories`"
                    ),
                )

            for input_col, cats in categories_map.items():
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
        elif isinstance(self.categories, list):
            self.categories_ = {col_name: cats for col_name, cats in zip(self.input_cols, self.categories)}
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
            if not is_scalar_nan(self.encoded_missing_value):
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

        for batch_start in range(0, len(self.input_cols), _COLUMN_BATCH_SIZE):
            batch_end = min(batch_start + _COLUMN_BATCH_SIZE, len(self.input_cols))
            batch_input_cols = self.input_cols[batch_start:batch_end]
            batch_output_cols = self.output_cols[batch_start:batch_end]

            for input_col, output_col in zip(batch_input_cols, batch_output_cols):
                input_col_state_df = state_df.filter(F.col(_COLUMN_NAME) == input_col)[
                    [_CATEGORY, _INDEX]
                ].with_column_renamed(_INDEX, output_col)

                # index values through a join operation over dataset and its states
                # In case of inplace transform, origin column name adds suffix (lsuffix=suffix)
                transformed_dataset = (
                    transformed_dataset.join(
                        input_col_state_df,
                        on=transformed_dataset[input_col]
                        .cast(T.StringType())
                        .equal_null(input_col_state_df[_CATEGORY]),
                        how="left",
                        lsuffix=suffix,
                    )
                    .drop(_CATEGORY)
                    .drop(identifier.concat_names([input_col, suffix]))
                )

            batch_table_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.TABLE)
            transformed_dataset.write.save_as_table(
                batch_table_name,
                mode="overwrite",
                table_type="temporary",
                statement_params=telemetry.get_statement_params(base.PROJECT, base.SUBPROJECT, self.__class__.__name__),
            )
            assert transformed_dataset._session is not None
            transformed_dataset = transformed_dataset._session.table(batch_table_name)

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
        if "categories" in sklearn_args and isinstance(sklearn_args["categories"], dict):
            # sklearn requires a list of array-like to satisfy the `categories` arg
            try:
                sklearn_args["categories"] = [sklearn_args["categories"][input_col] for input_col in self.input_cols]
            except KeyError as e:
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=e,
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
        elif isinstance(self.categories, (dict, list)):
            if len(self.categories) != len(self.input_cols):
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ATTRIBUTE,
                    original_exception=ValueError(
                        f"The number of categories ({len(self.categories)}) mismatches the number of input columns "
                        f"({len(self.input_cols)})."
                    ),
                )
            elif isinstance(self.categories, dict) and set(self.categories.keys()) != set(self.input_cols):
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
            if not (is_scalar_nan(self.unknown_value) or isinstance(self.unknown_value, numbers.Integral)):
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
        """
        if self.handle_unknown == "error":
            # batch columns to avoid query compilation OOM
            self._check_unknown(
                transformed_dataset,
                batch=len(self.input_cols) > _COLUMN_BATCH_SIZE,
                statement_params=telemetry.get_statement_params(base.PROJECT, base.SUBPROJECT, self.__class__.__name__),
            )

        if self.handle_unknown == "use_encoded_value":
            # left outer join has already filled unknown values with null
            if not (self.unknown_value is None or is_scalar_nan(self.unknown_value)):
                transformed_dataset = transformed_dataset.na.fill(self.unknown_value, self.output_cols)

        return transformed_dataset

    def _check_unknown(
        self,
        dataset: snowpark.DataFrame,
        statement_params: dict[str, Any],
        batch: bool = False,
    ) -> None:
        """
        Check if there are unknown values in the output of the given dataset.

        Args:
            dataset: Dataset to check.
            statement_params: Statement parameters for telemetry tracking.
            batch: Whether to batch the dataset.

        Raises:
            SnowflakeMLException: If unknown values exist in the output of the given dataset.
        """

        def create_unknown_df(
            dataset: snowpark.DataFrame,
            input_cols: list[str],
            output_cols: list[str],
        ) -> snowpark.DataFrame:
            # dataframe with unknown values
            # columns: COLUMN_NAME, UNKNOWN_VALUE
            unknown_df: Optional[snowpark.DataFrame] = None
            for input_col, output_col in zip(input_cols, output_cols):
                unknown_columns = [
                    F.lit(input_col),
                    F.col(input_col),
                ]
                temp_df = (
                    dataset[list({input_col, output_col})]
                    .distinct()
                    .filter(F.col(output_col).is_null())
                    .select(unknown_columns)
                    .to_df(["COLUMN_NAME", "UNKNOWN_VALUE"])
                )
                unknown_df = unknown_df.union_by_name(temp_df) if unknown_df is not None else temp_df
            assert unknown_df is not None, "Internal error by handle_unknown='error': Empty input columns."
            return unknown_df

        unknown_pandas_list = []
        if batch:
            batch_writes = []
            for batch_start in range(0, len(self.input_cols), _COLUMN_BATCH_SIZE):
                batch_end = min(batch_start + _COLUMN_BATCH_SIZE, len(self.input_cols))
                batch_input_cols = self.input_cols[batch_start:batch_end]
                batch_output_cols = self.output_cols[batch_start:batch_end]
                batch_dataset = dataset[list(set(batch_input_cols + batch_output_cols))]
                batch_table_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.TABLE)
                job = batch_dataset.write.save_as_table(
                    batch_table_name,
                    mode="overwrite",
                    table_type="temporary",
                    block=False,
                    statement_params=statement_params,
                )
                batch_writes.append((job, batch_table_name, batch_input_cols, batch_output_cols))

            to_pandas_async_jobs = []
            for job, batch_table_name, batch_input_cols, batch_output_cols in batch_writes:
                job.result(result_type="no_result")
                assert dataset._session is not None
                unknown_df = create_unknown_df(
                    dataset._session.table(batch_table_name), batch_input_cols, batch_output_cols
                )
                job = unknown_df.to_pandas(block=False, statement_params=statement_params)
                to_pandas_async_jobs.append(job)

            for job in to_pandas_async_jobs:
                unknown_pandas = job.result(result_type="pandas")
                if not unknown_pandas.empty:
                    unknown_pandas_list.append(unknown_pandas)
        else:
            unknown_df = create_unknown_df(dataset, self.input_cols, self.output_cols)
            unknown_pandas = unknown_df.to_pandas(statement_params=statement_params)
            if not unknown_pandas.empty:
                unknown_pandas_list.append(unknown_pandas)

        if unknown_pandas_list:
            concat_unknown_pandas = pd.concat(unknown_pandas_list, ignore_index=True)
            msg = f"Found unknown categories during transform:\n{concat_unknown_pandas.to_string()}"
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_DATA,
                original_exception=ValueError(msg),
            )
