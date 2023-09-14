#!/usr/bin/env python3
import numbers
import uuid
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn
from packaging import version
from scipy import sparse
from sklearn import preprocessing, utils as sklearn_utils

from snowflake import snowpark
from snowflake.ml._internal import telemetry, type_utils
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml._internal.utils import identifier
from snowflake.ml.model import model_signature
from snowflake.ml.modeling.framework import _utils, base
from snowflake.snowpark import functions as F, types as T
from snowflake.snowpark._internal.utils import (
    TempObjectType,
    generate_random_alphanumeric,
    random_name_for_temp_object,
)

_INFREQUENT_CATEGORY = "_INFREQUENT"
_COLUMN_NAME = "_COLUMN_NAME"
_CATEGORY = "_CATEGORY"
_COUNT = "_COUNT"
_STATE = "_STATE"
_FITTED_CATEGORY = "_FITTED_CATEGORY"
_ENCODING = "_ENCODING"
_ENCODED_VALUE = "_ENCODED_VALUE"
_N_FEATURES_OUT = "_N_FEATURES_OUT"

# constants used to validate the compatibility of the kwargs passed to the sklearn
# transformer with the sklearn version
_SKLEARN_INITIAL_KEYWORDS = ("sparse", "handle_unknown")  # initial keywords in sklearn
_SKLEARN_UNUSED_KEYWORDS = "dtype"  # sklearn keywords that are unused in snowml
_SNOWML_ONLY_KEYWORDS = ["input_cols", "output_cols"]  # snowml only keywords not present in sklearn

# Added keywords mapped to the sklearn versions in which they were added. Update mappings in new
# sklearn versions to support parameter validation.
_SKLEARN_ADDED_KEYWORD_TO_VERSION_DICT = {
    "categories": "0.20",
    "drop": "0.20",
    "min_frequency": "1.1",
    "max_categories": "1.1",
    "sparse_output": "1.2",
}

# Added keyword argument values mapped to the sklearn versions in which they were added. Update
# mappings in new sklearn versions to support parameter validation.
_SKLEARN_ADDED_KWARG_VALUE_TO_VERSION_DICT = {
    "drop": {"if_binary": "0.23"},
    "handle_unknown": {"infrequent_if_exist": "1.1"},
}

# Deprecated keywords mapped to the sklearn versions in which they were deprecated. Update mappings
# in new sklearn versions to support parameter validation.
_SKLEARN_DEPRECATED_KEYWORD_TO_VERSION_DICT = {
    "sparse": "1.2",
}

# Removed keywords mapped to the sklearn versions in which they were removed. Update mappings in
# new sklearn versions to support parameter validation.
_SKLEARN_REMOVED_KEYWORD_TO_VERSION_DICT = {
    "sparse": "1.4",
}


class OneHotEncoder(base.BaseTransformer):
    r"""Encode categorical features as a one-hot numeric array.

    The feature is converted to a matrix containing a column for each category. For each row, a column is 0 if the
    category is absent, or 1 if it exists. The categories can be detected from the data, or you can provide them.
    If you provide the categories, you can handle unknown categories in one of several different ways
    (see handle_unknown parameter below).

    Categories that do not appear frequently in a feature may be consolidated into a pseudo-category
    called “infrequent.” The threshold below which a category is considered “infrequent” is configurable using
    the min_frequency parameter.

    It is useful to drop one category from features in situations where perfectly collinear features cause problems,
    such as when feeding the resulting data into an unregularized linear regression model. However, dropping
    a category breaks the symmetry of the original representation and can therefore induce a bias in downstream models,
    for instance for penalized linear classification or regression models. You can choose from a handful of strategies
    for specifying the category to be dropped. See drop parameter below.

    The results of one-hot encoding can be represented in two ways.
        * Dense representation creates a binary column for each category. For each row, exactly one column will
            contain a 1.
        * Sparse representation creates a compressed sparse row (CSR) matrix that indicates which columns contain a
            nonzero value in each row. As all columns but one contain zeroes, this is an efficient way to represent
            the results.

    The order of input columns are preserved as the order of features.

    For more details on what this transformer does, see [sklearn.preprocessing.OneHotEncoder]
    (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).

    Args:
        categories: 'auto' or dict {column_name: ndarray([category])}, default='auto'
            Categories (unique values) per feature:
            - 'auto': Determine categories automatically from the training data.
            - dict: ``categories[column_name]`` holds the categories expected in
              the column provided. The passed categories should not mix strings
              and numeric values within a single feature, and should be sorted in
              case of numeric values.
            The used categories can be found in the ``categories_`` attribute.
        drop: {‘first’, ‘if_binary’} or an array-like of shape (n_features,), default=None
            Specifies a methodology to use to drop one of the categories per
            feature. This is useful in situations where perfectly collinear
            features cause problems, such as when feeding the resulting data
            into an unregularized linear regression model.
            However, dropping one category breaks the symmetry of the original
            representation and can therefore induce a bias in downstream models,
            for instance for penalized linear classification or regression models.
            - None: retain all features (the default).
            - 'first': drop the first category in each feature. If only one
              category is present, the feature will be dropped entirely.
            - 'if_binary': drop the first category in each feature with two
              categories. Features with 1 or more than 2 categories are
              left intact.
            - array: ``drop[i]`` is the category in feature ``input_cols[i]`` that
              should be dropped.
            When `max_categories` or `min_frequency` is configured to group
            infrequent categories, the dropping behavior is handled after the
            grouping.
        sparse: bool, default=False
            Will return a column with sparse representation if set True else will return
            a separate column for each category.
        handle_unknown: {'error', 'ignore'}, default='error'
            Specifies the way unknown categories are handled during :meth:`transform`.
            - 'error': Raise an error if an unknown category is present during transform.
            - 'ignore': When an unknown category is encountered during
              transform, the resulting one-hot encoded columns for this feature
              will be all zeros.
        min_frequency: int or float, default=None
            Specifies the minimum frequency below which a category will be
            considered infrequent.
            - If `int`, categories with a smaller cardinality will be considered
              infrequent.
            - If `float`, categories with a smaller cardinality than
              `min_frequency * n_samples`  will be considered infrequent.
        max_categories: int, default=None
            Specifies an upper limit to the number of output features for each input
            feature when considering infrequent categories. If there are infrequent
            categories, `max_categories` includes the category representing the
            infrequent categories along with the frequent categories. If `None`,
            there is no limit to the number of output features.
        input_cols: str or Iterable [column_name], default=None
            Single or multiple input columns.
        output_cols: str or Iterable [column_name], default=None
            Single or multiple output columns.
        drop_input_cols: Remove input columns from output if set True. False by default.

    Attributes:
        categories_: dict {column_name: ndarray([category])}
            The categories of each feature determined during fitting.
        drop_idx_: ndarray([index]) of shape (n_features,)
            - ``drop_idx_[i]`` is the index in ``_categories_list[i]`` of the category
              to be dropped for each feature.
            - ``drop_idx_[i] = None`` if no category is to be dropped from the
              feature with index ``i``, e.g. when `drop='if_binary'` and the
              feature isn't binary.
            - ``drop_idx_ = None`` if all the transformed features will be
              retained.
            If infrequent categories are enabled by setting `min_frequency` or
            `max_categories` to a non-default value and `drop_idx[i]` corresponds
            to a infrequent category, then the entire infrequent category is
            dropped.
        infrequent_categories_: list [ndarray([category])]
            Defined only if infrequent categories are enabled by setting
            `min_frequency` or `max_categories` to a non-default value.
            `infrequent_categories_[i]` are the infrequent categories for feature
            ``input_cols[i]``. If the feature ``input_cols[i]`` has no infrequent
            categories `infrequent_categories_[i]` is None.
    """

    def __init__(
        self,
        *,
        categories: Union[str, Dict[str, type_utils.LiteralNDArrayType]] = "auto",
        drop: Optional[Union[str, npt.ArrayLike]] = None,
        sparse: bool = False,
        handle_unknown: str = "error",
        min_frequency: Optional[Union[int, float]] = None,
        max_categories: Optional[int] = None,
        input_cols: Optional[Union[str, Iterable[str]]] = None,
        output_cols: Optional[Union[str, Iterable[str]]] = None,
        drop_input_cols: Optional[bool] = False,
    ) -> None:
        """See class-level docstring."""
        super().__init__(drop_input_cols=drop_input_cols)
        self.categories = categories
        self.drop = drop
        self.sparse = sparse
        self.handle_unknown = handle_unknown
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self._infrequent_enabled = (
            self.max_categories is not None and self.max_categories >= 1
        ) or self.min_frequency is not None

        # Fit state
        self.categories_: Dict[str, type_utils.LiteralNDArrayType] = {}
        self._categories_list: List[type_utils.LiteralNDArrayType] = []
        self.drop_idx_: Optional[npt.NDArray[np.int_]] = None
        self._drop_idx_after_grouping: Optional[npt.NDArray[np.int_]] = None
        self._n_features_outs: List[int] = []
        self._snowpark_cols: Dict[str, List[str]] = dict()

        # Fit state if output columns are set before fitting
        self._dense_output_cols_mappings: Dict[str, List[str]] = {}
        self._inferred_output_cols: List[str] = []

        self.set_input_cols(input_cols)
        self.set_output_cols(output_cols)

    @property
    def infrequent_categories_(self) -> List[Optional[type_utils.LiteralNDArrayType]]:
        """Infrequent categories for each feature."""
        # raises an AttributeError if `_infrequent_indices` is not defined
        infrequent_indices = self._infrequent_indices
        return [
            None if indices is None else category[indices]
            for category, indices in zip(
                self._categories_list,
                infrequent_indices,
            )
        ]

    def _reset(self) -> None:
        """Reset internal data-dependent state. Constructor parameters are not touched."""
        super()._reset()
        self.categories_ = {}
        self._categories_list = []
        self.drop_idx_ = None
        self._drop_idx_after_grouping = None
        self._n_features_outs = []
        self._dense_output_cols_mappings = {}

        if hasattr(self, "_infrequent_indices"):
            del self._infrequent_indices
        if hasattr(self, "_default_to_infrequent_mappings"):
            del self._default_to_infrequent_mappings
        if hasattr(self, "_state_pandas"):
            del self._state_pandas

    @telemetry.send_api_usage_telemetry(
        project=base.PROJECT,
        subproject=base.SUBPROJECT,
    )
    def fit(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> "OneHotEncoder":
        """
        Fit OneHotEncoder to dataset.

        Validates the transformer arguments and derives the list of categories (distinct values) from the data,
        making this list available as an attribute of the transformer instance (see Attributes).

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
        self._is_fitted = True

        if not self.sparse and self.output_cols:
            self._handle_dense_output_cols()
        if self.output_cols:
            self._handle_inferred_output_cols(dataset)

        return self

    def _fit_sklearn(self, dataset: pd.DataFrame) -> None:
        dataset = self._use_input_cols_only(dataset)
        sklearn_encoder = self._create_unfitted_sklearn_object()
        sklearn_encoder.fit(dataset)

        self._categories_list = sklearn_encoder.categories_
        self.drop_idx_ = sklearn_encoder.drop_idx_
        if version.parse(sklearn.__version__) >= version.parse("1.2.2"):
            self._drop_idx_after_grouping = sklearn_encoder._drop_idx_after_grouping
        else:
            self._drop_idx_after_grouping = sklearn_encoder.drop_idx_
        self._n_features_outs = sklearn_encoder._n_features_outs

        _state_pandas_counts: List[pd.DataFrame] = []
        for idx, input_col in enumerate(self.input_cols):
            self.categories_[input_col] = self._categories_list[idx]
            _column_counts = (
                dataset.value_counts(subset=[input_col], dropna=False)
                .rename_axis(_CATEGORY)
                .to_frame(_COUNT)
                .reset_index()
            )
            _column_counts.insert(0, _COLUMN_NAME, input_col)
            _state_pandas_counts.append(_column_counts)

        # Set infrequent mappings
        if self._infrequent_enabled:
            self._infrequent_indices = sklearn_encoder._infrequent_indices
            self._default_to_infrequent_mappings = sklearn_encoder._default_to_infrequent_mappings

        # Set `_state_pandas`
        self._state_pandas = pd.concat(_state_pandas_counts, ignore_index=True)
        self._update_categories_state()

    def _fit_snowpark(self, dataset: snowpark.DataFrame) -> None:
        # StructType[[StructField(COLUMN, TYPE, nullable=True), ...]
        self._dataset_schema = dataset.schema
        self._snowpark_cols["input_cols"] = dataset.select(self.input_cols).columns
        self._snowpark_cols["sorted_input_cols"] = dataset.select(sorted(self.input_cols)).columns
        fit_results = self._fit_category_state(dataset, return_counts=self._infrequent_enabled)
        if self._infrequent_enabled:
            self._fit_infrequent_category_mapping(fit_results["n_samples"], fit_results["category_counts"])
        self._set_drop_idx()
        self._n_features_outs = self._compute_n_features_outs()
        self._update_categories_state()

    def _fit_category_state(self, dataset: snowpark.DataFrame, return_counts: bool) -> Dict[str, Any]:
        """
        Get the number of samples, categories and (optional) category counts of dataset.
        Fitted categories are assigned to the object.

        Args:
            dataset: Input dataset.
            return_counts: Whether to return category counts.

        Returns:
            Dict with `n_samples` and (optionally) `category_counts` of the dataset.

        Raises:
            SnowflakeMLException: Empty data.
        """
        # columns: COLUMN_NAME, CATEGORY, COUNT
        state_df = self._get_category_count_state_df(dataset)
        self._state_pandas = state_df.to_pandas(
            statement_params=telemetry.get_statement_params(base.PROJECT, base.SUBPROJECT, self.__class__.__name__)
        )
        if self._state_pandas.empty:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_DATA,
                original_exception=ValueError("Empty data while a minimum of 1 sample is required."),
            )

        # columns: COLUMN_NAME, STATE
        # state object: {category: count}
        state_object_pandas = self._get_state_object_pandas_df()

        # {column_name: {category: count}}
        state = state_object_pandas.set_index(_COLUMN_NAME).to_dict()[_STATE]

        self._assign_categories(state_object_pandas)

        n_samples = sum(state[self.input_cols[0]].values()) if self.input_cols else 0
        output = {"n_samples": n_samples}
        if return_counts:
            output["category_counts"] = state
        return output

    def _get_category_count_state_df(self, dataset: snowpark.DataFrame) -> snowpark.DataFrame:
        """
        Get the number of samples, categories and (optional) category counts of each
        input column in dataset.
        If `categories` is provided, use the given categories;
        plus if `self.handle_unknown="error"`, check if the given categories
        contain all categories in dataset.

        Args:
            dataset: Input dataset.

        Returns:
            State dataframe with columns [COLUMN_NAME, CATEGORY, COUNT].

        Raises:
            SnowflakeMLException: If `self.categories` is provided, `self.handle_unknown="error"`,
                and unknown categories exist in dataset.
        """
        # states of categories found in dataset
        found_state_df: Optional[snowpark.DataFrame] = None
        for input_col in self.input_cols:
            state_columns = [
                F.lit(input_col).alias(_COLUMN_NAME),
                F.col(input_col).cast(T.StringType()).alias(_CATEGORY),
                F.iff(
                    # null or nan values
                    F.col(input_col).is_null() | (F.col(input_col).cast(T.StringType()).equal_nan()),
                    # count null and nan values
                    F.sum(
                        F.iff(
                            F.col(input_col).is_null() | (F.col(input_col).cast(T.StringType()).equal_nan()),
                            1,
                            0,
                        )
                    ).over(snowpark.Window.partition_by(input_col)),
                    # count values that are not null or nan
                    F.count(input_col).over(snowpark.Window.partition_by(input_col)),
                ).alias(_COUNT),
            ]
            temp_df = dataset.select(state_columns).distinct()
            found_state_df = found_state_df.union_by_name(temp_df) if found_state_df is not None else temp_df

        assert found_state_df is not None
        if self.categories != "auto":
            state_data = []
            assert isinstance(self.categories, dict)
            for input_col, cats in self.categories.items():
                for cat in cats.tolist():
                    state_data.append([input_col, cat])
            # states of given categories
            assert dataset._session is not None
            given_state_df = dataset._session.create_dataframe(data=state_data, schema=[_COLUMN_NAME, _CATEGORY])
            given_state_df = (
                given_state_df.join(
                    found_state_df,
                    (given_state_df[_COLUMN_NAME] == found_state_df[_COLUMN_NAME])
                    & (given_state_df[_CATEGORY] == found_state_df[_CATEGORY]),
                    "left",
                )
                .select(
                    given_state_df[_COLUMN_NAME].alias(_COLUMN_NAME),
                    given_state_df[_CATEGORY].alias(_CATEGORY),
                    found_state_df[_COUNT],
                )
                .fillna({_COUNT: 0})
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
                    raise exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_DATA,
                        original_exception=ValueError(
                            f"Found unknown categories during fit:\n{unknown_pandas.to_string()}"
                        ),
                    )

            return given_state_df

        return found_state_df

    def _get_state_object_pandas_df(self) -> pd.DataFrame:
        """
        Convert `self._state_pandas` to a state object pandas dataframe where states
        are in the form of objects.

        Returns:
            Pandas dataframe with columns [COLUMN_NAME, STATE], where STATE contains
                state objects: {category: count}.
        """
        state_object_pandas = (
            self._state_pandas.groupby(_COLUMN_NAME)
            .apply(lambda x: dict(zip(x[_CATEGORY], x[_COUNT])))
            .reset_index(name=_STATE)
        )
        return state_object_pandas

    def _assign_categories(self, state_object_pandas: pd.DataFrame) -> None:
        """
        Assign the categories to the object.

        Args:
            state_object_pandas: Pandas dataframe with columns [COLUMN_NAME, STATE],
                where STATE contains state objects: {category: count}.

        Raises:
            SnowflakeMLException: If `self.categories` is an unsupported value.
        """
        if isinstance(self.categories, str):
            if self.categories != "auto":
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ATTRIBUTE,
                    original_exception=ValueError(f"Unsupported value {self.categories} for parameter `categories`."),
                )

            categories_col = "CATEGORIES"

            # dataframe with the category array of each input column in dataset
            # columns: COLUMN_NAME, CATEGORIES
            categories_pandas = state_object_pandas
            categories_pandas[_STATE] = (
                categories_pandas[_STATE]
                .map(lambda x: sorted(list(x.keys()), key=lambda v: (v is None, v)))
                .map(lambda x: np.array(x))
            )
            categories_pandas = categories_pandas.rename(columns={_STATE: categories_col})

            # {column_name: ndarray([category])}
            categories: Dict[str, type_utils.LiteralNDArrayType] = categories_pandas.set_index(_COLUMN_NAME).to_dict()[
                categories_col
            ]
            # Giving the original type back to categories.
            for idx, k in enumerate(categories.keys()):
                v = categories[k]
                # Schema column names are case insensitive. Using snowpark's dataset to maintain schema's
                # column name consistency. Because the key of categories_pandas is sorted, we need sorted
                # input cols from snowpark as well.
                _dataset_schema_key = self._snowpark_cols["sorted_input_cols"][idx]
                snowml_type = model_signature.DataType.from_snowpark_type(
                    self._dataset_schema[_dataset_schema_key].datatype
                )
                # Don't convert the boolean type, numpy is unable to switch from string to boolean.
                # Boolean types would be treated as string
                if snowml_type not in [model_signature.DataType.BOOL]:
                    # If the category contains None values - None stays None; other values are converted. Type unchanged
                    if pd.isnull(v).any():
                        categories[k] = np.where(pd.isnull(v), v, v.astype(snowml_type._numpy_type))
                    # Otherwise, convert the whole array, changing the array type.
                    else:
                        categories[k] = v.astype(snowml_type._numpy_type)
                else:
                    # Custom function to convert string to bool
                    # Vectorize the function to work with arrays
                    vectorized_func = _utils._handle_str_bool_type()
                    if pd.isnull(v).any():
                        categories[k] = np.where(pd.isnull(v), v, vectorized_func(v))
                    # Otherwise, convert the whole array, changing the array type.
                    else:
                        categories[k] = vectorized_func(v)
            self.categories_ = categories
        else:
            self.categories_ = self.categories

        # list of ndarray same as `sklearn.preprocessing.OneHotEncoder.categories_`
        for input_col in self.input_cols:
            self._categories_list.append(self.categories_[input_col])

    def _update_categories_state(self) -> None:
        self._add_n_features_out_state()
        self._add_fitted_category_state()
        self._add_encoding_state()
        self._remove_dropped_categories_state()

    def _remove_dropped_categories_state(self) -> None:
        """Remove dropped categories from `self._state_pandas` if `self.drop` is set."""
        if self.drop is None:
            return

        for idx, input_col in enumerate(self.input_cols):
            if self.drop_idx_ is not None and self.drop_idx_[idx] is not None:
                drop_cat = self.categories_[input_col][self.drop_idx_[idx]]
                self._state_pandas.drop(
                    self._state_pandas[
                        (self._state_pandas[_COLUMN_NAME] == input_col) & (self._state_pandas[_CATEGORY] == drop_cat)
                    ].index,
                    inplace=True,
                )

    def _add_n_features_out_state(self) -> None:
        """Add the n features out column to `self._state_pandas`."""

        def map_n_features_out(row: pd.Series) -> int:
            col_idx = self.input_cols.index(row[_COLUMN_NAME])
            n_features_out: int = self._n_features_outs[col_idx]
            return n_features_out

        self._state_pandas[_N_FEATURES_OUT] = self._state_pandas.apply(lambda x: map_n_features_out(x), axis=1)

    def _add_fitted_category_state(self) -> None:
        """
        Add the fitted category column to `self._state_pandas` where infrequent
        categories are categorized.
        """
        if not self._infrequent_enabled:
            self._state_pandas[_FITTED_CATEGORY] = self._state_pandas[_CATEGORY]
            return

        def map_fitted_category(row: pd.Series) -> str:
            cat: str = row[_CATEGORY]
            col_idx = self.input_cols.index(row[_COLUMN_NAME])
            infrequent_cats = self.infrequent_categories_[col_idx]
            if infrequent_cats is not None and cat in infrequent_cats:
                return _INFREQUENT_CATEGORY
            return cat

        self._state_pandas[_FITTED_CATEGORY] = self._state_pandas.apply(lambda x: map_fitted_category(x), axis=1)

    def _add_encoding_state(self) -> None:
        """Add the encoding column to `self._state_pandas`."""

        def map_encoding(row: pd.Series) -> int:
            input_col = row[_COLUMN_NAME]
            col_idx = self.input_cols.index(input_col)

            # whether there are infrequent categories in the input column
            has_infrequent_categories = self._infrequent_enabled and self.infrequent_categories_[col_idx] is not None

            cat = row[_CATEGORY]
            if hasattr(self, "_dataset_schema") and not pd.isnull(cat):  # Do not convert when it is null
                row_element = np.array([row[_CATEGORY]])
                _dataset_schema_key = self._snowpark_cols["input_cols"][col_idx]
                snowml_type = model_signature.DataType.from_snowpark_type(
                    self._dataset_schema[_dataset_schema_key].datatype
                )
                # Don't convert the boolean type, it would be treated as string
                if snowml_type not in [model_signature.DataType.BOOL]:
                    cat = row_element.astype(snowml_type._numpy_type)[0]
                else:
                    if not pd.isnull(cat) and isinstance(cat, str):
                        cat = _utils.str_to_bool(cat)
            # np.isnan cannot be applied to object or string dtypes, use pd.isnull instead
            cat_idx = (
                np.where(pd.isnull(self.categories_[input_col]))[0][0]
                if isinstance(cat, float) and np.isnan(cat)
                else np.where(self.categories_[input_col] == cat)[0][0]
            )
            if has_infrequent_categories:
                if self._default_to_infrequent_mappings[col_idx] is None:
                    raise exceptions.SnowflakeMLException(
                        error_code=error_codes.INTERNAL_PYTHON_ERROR,
                        original_exception=RuntimeError(
                            f"`self._default_to_infrequent_mappings[{col_idx}]` is None while infrequent categories "
                            f"exist in '{input_col}'."
                        ),
                    )
                encoding: int = self._default_to_infrequent_mappings[col_idx][cat_idx]
            else:
                encoding = cat_idx

            # decrement the encoding if it occurs after that of the dropped category
            if (
                self._drop_idx_after_grouping is not None
                and self._drop_idx_after_grouping[col_idx] is not None
                and encoding > self._drop_idx_after_grouping[col_idx]
            ):
                encoding -= 1

            return encoding

        self._state_pandas[_ENCODING] = self._state_pandas.apply(lambda x: map_encoding(x), axis=1)

    @telemetry.send_api_usage_telemetry(
        project=base.PROJECT,
        subproject=base.SUBPROJECT,
    )
    @telemetry.add_stmt_params_to_df(
        project=base.PROJECT,
        subproject=base.SUBPROJECT,
    )
    def transform(
        self, dataset: Union[snowpark.DataFrame, pd.DataFrame]
    ) -> Union[snowpark.DataFrame, pd.DataFrame, sparse.csr_matrix]:
        """
        Transform dataset using one-hot encoding.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset. The output type depends on the input dataset type:
                - If input is DataFrame, returns DataFrame
                - If input is a pd.DataFrame and `self.sparse=True`, returns `csr_matrix`
                - If input is a pd.DataFrame and `self.sparse=False`, returns `pd.DataFrame`
        """
        self._enforce_fit()
        super()._check_input_cols()
        super()._check_output_cols()
        super()._check_dataset_type(dataset)

        # output columns are unset before fitting
        if not self.sparse and not self._dense_output_cols_mappings:
            self._handle_dense_output_cols()

        if isinstance(dataset, snowpark.DataFrame):
            output_df = self._transform_snowpark(dataset)
        else:
            output_df = self._transform_sklearn(dataset)

        return self._drop_input_columns(output_df) if self._drop_input_cols is True else output_df

    def _transform_snowpark(self, dataset: snowpark.DataFrame) -> snowpark.DataFrame:
        """
        Transform Snowpark dataframe using one-hot encoding.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.
        """
        if self.sparse:
            return self._transform_snowpark_sparse(dataset)
        else:
            return self._transform_snowpark_dense(dataset)

    def _transform_snowpark_sparse(self, dataset: snowpark.DataFrame) -> snowpark.DataFrame:
        """
        Transform Snowpark dataframe using one-hot encoding when
        `self.sparse=True`. Return the sparse representation where
        the transformed output is
        {column_index: 1.0, "array_length": length} for each value
        representing the corresponding 1 in the matrix.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset in the sparse representation.
        """
        # TODO(hayu): [SNOW-752263] Support OneHotEncoder handle_unknown="infrequent_if_exist".
        #  Add back when `handle_unknown="infrequent_if_exist"` is supported.
        # TODO: [SNOW-720743] Support replacing values in a variant column
        # if self.handle_unknown == "infrequent_if_exist":
        #     return self._transform_snowpark_sparse_udf(dataset)

        state_pandas = self._state_pandas

        def map_encoded_value(row: pd.Series) -> Dict[str, Any]:
            n_features_out = row[_N_FEATURES_OUT]
            encoding = row[_ENCODING]
            encoded_value = {str(encoding): 1, "array_length": n_features_out}
            return encoded_value

        # TODO: [SNOW-730357] Support NUMBER as the key of Snowflake OBJECT for OneHotEncoder sparse output
        state_pandas[_ENCODED_VALUE] = state_pandas.apply(lambda x: map_encoded_value(x), axis=1)

        # columns: COLUMN_NAME, CATEGORY, COUNT, FITTED_CATEGORY, ENCODING, N_FEATURES_OUT, ENCODED_VALUE
        assert dataset._session is not None
        state_df = dataset._session.create_dataframe(state_pandas)

        suffix = "_" + uuid.uuid4().hex.upper()
        transformed_dataset = dataset
        original_dataset_cols = transformed_dataset.columns[:]
        all_output_cols = []
        suffixed_input_cols = []
        joined_input_cols = []
        for idx, input_col in enumerate(self.input_cols):
            output_col = self.output_cols[idx]
            all_output_cols += [output_col]
            input_col_state_df = state_df.filter(F.col(_COLUMN_NAME) == input_col)[
                [_CATEGORY, _ENCODED_VALUE]
            ].with_column_renamed(_ENCODED_VALUE, output_col)

            # index values through a left join over the dataset and its states
            transformed_dataset = transformed_dataset.join(
                input_col_state_df,
                on=transformed_dataset[input_col].equal_null(input_col_state_df[_CATEGORY]),
                how="left",
                lsuffix=suffix,
            ).drop(_CATEGORY)

            # handle identical input & output cols
            if input_col == output_col:
                col = identifier.concat_names([input_col, suffix])
                suffixed_input_cols.append(col)
                joined_input_cols.append(col)
            else:
                joined_input_cols.append(input_col)

        if not self._inferred_output_cols:
            self._inferred_output_cols = transformed_dataset[all_output_cols].columns

        transformed_dataset = self._handle_unknown_in_transform(transformed_dataset, joined_input_cols)
        # Reorder columns. Passthrough columns are added at the right to the output of the transformers.
        passthrough_cols = list(set(original_dataset_cols) - set(all_output_cols))
        transformed_dataset = transformed_dataset.drop(suffixed_input_cols)[all_output_cols + passthrough_cols]
        return transformed_dataset

    def _transform_snowpark_dense(self, dataset: snowpark.DataFrame) -> snowpark.DataFrame:
        """
        Transform Snowpark dataframe using one-hot encoding when
        `self.sparse=False`. Return the dense representation. For
        `self.input_cols[i]`, its output columns are named as
        "'OUTPUT_category'", where "OUTPUT" is `self.output_cols[i]`,
        and "category" is `self.categories_[input_col][j]`.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset in the dense representation.
        """
        state_pandas = self._state_pandas

        def map_encoded_value(row: pd.Series) -> List[int]:
            n_features_out = row[_N_FEATURES_OUT]
            encoding = row[_ENCODING]
            encoded_value = [0] * n_features_out
            encoded_value[encoding] = 1
            return encoded_value

        state_pandas[_ENCODED_VALUE] = state_pandas.apply(lambda x: map_encoded_value(x), axis=1)

        for input_col in self.input_cols:
            # split encoded values to columns
            input_col_state_pandas = state_pandas.loc[state_pandas[_COLUMN_NAME] == input_col][
                [_COLUMN_NAME, _CATEGORY, _ENCODED_VALUE]
            ]
            split_pandas = pd.DataFrame(
                input_col_state_pandas[_ENCODED_VALUE].tolist(), columns=self._dense_output_cols_mappings[input_col]
            )
            split_pandas[_COLUMN_NAME] = input_col_state_pandas[_COLUMN_NAME].values
            split_pandas[_CATEGORY] = input_col_state_pandas[_CATEGORY].values

            # merge split encoding columns to the state pandas
            state_pandas = state_pandas.merge(split_pandas, on=[_COLUMN_NAME, _CATEGORY], how="left")

        # columns: COLUMN_NAME, CATEGORY, COUNT, FITTED_CATEGORY, ENCODING, N_FEATURES_OUT, ENCODED_VALUE, OUTPUT_CATs
        assert dataset._session is not None
        state_df = dataset._session.create_dataframe(state_pandas)

        transformed_dataset = dataset
        original_dataset_columns = transformed_dataset.columns[:]
        all_output_cols = []
        for input_col in self.input_cols:
            output_cols = [
                identifier.quote_name_without_upper_casing(col) for col in self._dense_output_cols_mappings[input_col]
            ]
            all_output_cols += output_cols
            input_col_state_df = state_df.filter(F.col(_COLUMN_NAME) == input_col)[output_cols + [_CATEGORY]]

            # index values through a left join over the dataset and its states
            transformed_dataset = transformed_dataset.join(
                input_col_state_df,
                on=transformed_dataset[input_col].equal_null(input_col_state_df[_CATEGORY]),
                how="left",
            )[transformed_dataset.columns + output_cols]

        if not self._inferred_output_cols:
            self._inferred_output_cols = transformed_dataset[all_output_cols].columns

        transformed_dataset = self._handle_unknown_in_transform(transformed_dataset)
        # Reorder columns. Passthrough columns are added at the right to the output of the transformers.
        transformed_dataset = transformed_dataset[all_output_cols + original_dataset_columns]
        return transformed_dataset

    def _transform_snowpark_sparse_udf(self, dataset: snowpark.DataFrame) -> snowpark.DataFrame:
        """
        Transform Snowpark dataframe using one-hot encoding when
        `self.sparse=True`. Return the sparse representation where
        the transformed output is {column_index: 1.0} for each value
        representing the corresponding 1 in the matrix.

        Temp pandas UDF `one_hot_encoder_sparse_transform` is registered.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset in the sparse representation.
        """
        encoder_sklearn = self.to_sklearn()
        udf_name = random_name_for_temp_object(TempObjectType.FUNCTION)

        @F.pandas_udf(  # type: ignore
            is_permanent=False,
            name=udf_name,
            replace=True,
            return_type=T.PandasSeriesType(T.ArrayType(T.MapType(T.FloatType(), T.FloatType()))),
            input_types=[T.PandasDataFrameType([T.StringType() for _ in range(len(self.input_cols))])],
            packages=["numpy", "scikit-learn"],
            statement_params=telemetry.get_statement_params(base.PROJECT, base.SUBPROJECT, self.__class__.__name__),
        )
        def one_hot_encoder_sparse_transform(data: pd.DataFrame) -> List[List[Optional[Dict[Any, Any]]]]:
            data = data.replace({np.nan: None})  # fill NA with None as represented in `categories_`
            transformed_csr = encoder_sklearn.transform(data)
            transformed_coo = transformed_csr.tocoo()

            coo_idx = 0
            transformed_vals = []
            for _, row in data.iterrows():
                base_encoding = 0
                row_transformed_vals: List[Optional[Dict[Any, Any]]] = []
                for col_idx, val in row.items():
                    if val in encoder_sklearn.categories_[col_idx] or encoder_sklearn.handle_unknown != "ignore":
                        if col_idx > 0:
                            base_encoding += encoder_sklearn._n_features_outs[col_idx - 1]
                        coo_col_val = transformed_coo.col[coo_idx].item() - base_encoding  # column independent encoding
                        coo_data_val = transformed_coo.data[coo_idx].item()
                        row_transformed_vals.append({coo_col_val: coo_data_val})
                        coo_idx += 1
                    else:
                        row_transformed_vals.append(None)
                transformed_vals.append(row_transformed_vals)
            return transformed_vals

        # encoded column returned by `one_hot_encoder_sparse_transform`
        encoded_output_col = f"'ENCODED_OUTPUT_{generate_random_alphanumeric()}'"
        encoded_column = one_hot_encoder_sparse_transform(self.input_cols)  # type: ignore
        encoded_dataset = dataset.with_column(encoded_output_col, encoded_column)

        # output columns of value {column_index: 1.0} or null
        output_columns = []
        for idx, _ in enumerate(self.input_cols):
            output_columns.append(F.col(encoded_output_col)[idx])

        transformed_dataset: snowpark.DataFrame = encoded_dataset.with_columns(self.output_cols, output_columns).drop(
            encoded_output_col
        )
        return transformed_dataset

    def _transform_sklearn(self, dataset: pd.DataFrame) -> Union[pd.DataFrame, sparse.csr_matrix]:
        """
        Transform pandas dataframe using sklearn one-hot encoding.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.
        """
        encoder_sklearn = self.to_sklearn()
        transformed_dataset = encoder_sklearn.transform(dataset[self.input_cols])

        if self.sparse:
            return transformed_dataset

        if not self._inferred_output_cols:
            self._inferred_output_cols = self._get_inferred_output_cols()

        dataset = dataset.copy()
        dataset[self.get_output_cols()] = transformed_dataset
        return dataset

    def _create_unfitted_sklearn_object(self) -> preprocessing.OneHotEncoder:
        sklearn_args = self.get_sklearn_args(
            default_sklearn_obj=preprocessing.OneHotEncoder(),
            sklearn_initial_keywords=_SKLEARN_INITIAL_KEYWORDS,
            sklearn_unused_keywords=_SKLEARN_UNUSED_KEYWORDS,
            snowml_only_keywords=_SNOWML_ONLY_KEYWORDS,
            sklearn_added_keyword_to_version_dict=_SKLEARN_ADDED_KEYWORD_TO_VERSION_DICT,
            sklearn_added_kwarg_value_to_version_dict=_SKLEARN_ADDED_KWARG_VALUE_TO_VERSION_DICT,
            sklearn_deprecated_keyword_to_version_dict=_SKLEARN_DEPRECATED_KEYWORD_TO_VERSION_DICT,
            sklearn_removed_keyword_to_version_dict=_SKLEARN_REMOVED_KEYWORD_TO_VERSION_DICT,
        )
        return preprocessing.OneHotEncoder(**sklearn_args)

    def _create_sklearn_object(self) -> preprocessing.OneHotEncoder:
        """
        Get an equivalent sklearn OneHotEncoder. This can only be called after the transformer is fit.

        Returns:
            Sklearn OneHotEncoder.
        """
        encoder = self._create_unfitted_sklearn_object()

        if self._is_fitted:
            encoder.categories_ = self._categories_list
            encoder.drop_idx_ = self.drop_idx_
            encoder._drop_idx_after_grouping = self._drop_idx_after_grouping
            encoder._infrequent_enabled = self._infrequent_enabled
            encoder._n_features_outs = self._n_features_outs

            if self._infrequent_enabled:
                encoder._infrequent_indices = self._infrequent_indices
                encoder._default_to_infrequent_mappings = self._default_to_infrequent_mappings

        return encoder

    def _validate_keywords(self) -> None:
        # categories
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

        # drop: array-like object is validated in `_compute_drop_idx`
        if isinstance(self.drop, str) and self.drop not in {"first", "if_binary"}:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=ValueError(
                    "`drop` must be one of 'first', 'if_binary', an array-like of shape (n_features,), or None, "
                    f"got {self.drop}."
                ),
            )

        # handle_unknown
        # TODO(hayu): [SNOW-752263] Support OneHotEncoder handle_unknown="infrequent_if_exist".
        #  Add back when `handle_unknown="infrequent_if_exist"` is supported.
        # if self.handle_unknown not in {"error", "ignore", "infrequent_if_exist"}:
        #     msg = "`handle_unknown` must be one of 'error', 'ignore', 'infrequent_if_exist', got {}."
        #     raise ValueError(msg.format(self.handle_unknown))
        if self.handle_unknown not in {"error", "ignore"}:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=ValueError(
                    f"`handle_unknown` must be one of 'error', 'ignore', got {self.handle_unknown}."
                ),
            )

        # min_frequency
        if isinstance(self.min_frequency, numbers.Integral):
            if not int(self.min_frequency) >= 1:
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ATTRIBUTE,
                    original_exception=ValueError(
                        "`min_frequency` must be an integer at least 1, a float in (0.0, 1.0), or None, "
                        f"got integer {self.min_frequency}."
                    ),
                )
        elif isinstance(self.min_frequency, numbers.Real):
            if not (0.0 < float(self.min_frequency) < 1.0):
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ATTRIBUTE,
                    original_exception=ValueError(
                        "`min_frequency` must be an integer at least 1, a float in (0.0, 1.0), or None, "
                        f"got float {self.min_frequency}."
                    ),
                )

    def _handle_unknown_in_transform(
        self,
        transformed_dataset: snowpark.DataFrame,
        input_cols: Optional[List[str]] = None,
    ) -> snowpark.DataFrame:
        """
        Handle unknown values in the transformed dataset.

        Args:
            transformed_dataset: Transformed dataset without unknown values handled.
            input_cols: Input columns (may be suffixed).

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
            cols = input_cols or self.input_cols
            for idx, input_col in enumerate(cols):
                output_col = self.output_cols[idx]
                check_col = output_col
                if not self.sparse:
                    output_cat_cols = [
                        identifier.quote_name_without_upper_casing(col)
                        for col in self._dense_output_cols_mappings[input_col]
                    ]
                    if not output_cat_cols:
                        continue
                    check_col = output_cat_cols[0]

                unknown_columns = [
                    F.lit(self.input_cols[idx]),
                    F.col(input_col),
                ]
                temp_df = (
                    transformed_dataset[[input_col, check_col]]
                    .distinct()
                    .filter(F.col(check_col).is_null())
                    .select(unknown_columns)
                    .to_df(["COLUMN_NAME", "UNKNOWN_VALUE"])
                )
                unknown_df = unknown_df.union_by_name(temp_df) if unknown_df is not None else temp_df

            if unknown_df:
                unknown_pandas = unknown_df.to_pandas(
                    statement_params=telemetry.get_statement_params(
                        base.PROJECT, base.SUBPROJECT, self.__class__.__name__
                    )
                )
                if not unknown_pandas.empty:
                    raise exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_DATA,
                        original_exception=ValueError(
                            f"Found unknown categories during transform:\n{unknown_pandas.to_string()}"
                        ),
                    )
        if self.handle_unknown == "ignore" and not self.sparse:
            transformed_dataset = transformed_dataset.na.fill(0, self._inferred_output_cols)

        # TODO(hayu): [SNOW-752263] Support OneHotEncoder handle_unknown="infrequent_if_exist".
        #  Add back when `handle_unknown="infrequent_if_exist"` is supported.
        # if self.handle_unknown == "infrequent_if_exist" and not self.sparse:
        #     all_output_freq_cols = []
        #     all_output_infreq_cols = []
        #     for idx, _ in enumerate(self.input_cols):
        #         output_col = self.output_cols[idx]
        #         output_freq_cols = [
        #             x for x in transformed_dataset.columns if f"{output_col}_" in x and _INFREQUENT_CATEGORY not in x
        #         ]
        #         output_infreq_cols = [
        #             x for x in transformed_dataset.columns if f"{output_col}_" in x and _INFREQUENT_CATEGORY in x
        #         ]
        #         all_output_freq_cols.extend(output_freq_cols)
        #         all_output_infreq_cols.extend(output_infreq_cols)
        #     transformed_dataset = transformed_dataset.fillna(0, all_output_freq_cols)
        #     transformed_dataset = transformed_dataset.fillna(1, all_output_infreq_cols)

        return transformed_dataset

    def _map_drop_idx_to_infrequent(self, feature_idx: int, drop_idx: int) -> int:
        """
        Convert `drop_idx` into the index for infrequent categories.
        If there are no infrequent categories, then `drop_idx` is
        returned. This method is called in `_set_drop_idx` when the `drop`
        parameter is an array-like.

        Same as :meth:`sklearn.preprocessing.OneHotEncoder._map_drop_idx_to_infrequent`
        with `self.categories_` replaced by `self._categories_list`.

        Args:
            feature_idx: Feature (input column) index in self.input_cols.
            drop_idx: Index of the category to drop in categories of the given feature.

        Returns:
            Converted drop index with infrequent encoding considered.

        Raises:
            SnowflakeMLException: If the category to drop is infrequent.
        """
        if not self._infrequent_enabled:
            return drop_idx

        default_to_infrequent: Optional[List[int]] = self._default_to_infrequent_mappings[feature_idx]
        if default_to_infrequent is None:
            return drop_idx

        # Raise error when explicitly dropping a category that is infrequent
        infrequent_indices = self._infrequent_indices[feature_idx]
        if infrequent_indices is not None and drop_idx in infrequent_indices:
            categories = self._categories_list[feature_idx]
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_DATA,
                original_exception=ValueError(
                    f"Unable to drop category {categories[drop_idx]!r} from feature "
                    f"{feature_idx} because it is infrequent."
                ),
            )
        return default_to_infrequent[drop_idx]

    def _set_drop_idx(self) -> None:
        """
        Compute the drop indices associated with `self.categories_`.

        Same as :meth:`sklearn.preprocessing.OneHotEncoder._set_drop_idx`
        with `self.categories_` replaced by `self._categories_list`.

        If `self.drop` is:
        - `None`, No categories have been dropped.
        - `'first'`, All zeros to drop the first category.
        - `'if_binary'`, All zeros if the category is binary and `None`
          otherwise.
        - array-like, The indices of the categories that match the
          categories in `self.drop`. If the dropped category is an infrequent
          category, then the index for the infrequent category is used. This
          means that the entire infrequent category is dropped.

        This method defines a public `drop_idx_` and a private
        `_drop_idx_after_grouping`.

        - `drop_idx_`: Public facing API that references the drop category in
          `self.categories_`.
        - `_drop_idx_after_grouping`: Used internally to drop categories *after* the
          infrequent categories are grouped together.

        If there are no infrequent categories or drop is `None`, then
        `drop_idx_=_drop_idx_after_grouping`.

        Raises:
            SnowflakeMLException: If `self.drop` is array-like:
                - `self.drop` cannot be converted to a ndarray.
                - The length of `self.drop` is not equal to the number of input columns.
                - The categories to drop are not found.
        """
        if self.drop is None:
            drop_idx_after_grouping = None
        elif isinstance(self.drop, str):
            if self.drop == "first":
                drop_idx_after_grouping = np.zeros(len(self._categories_list), dtype=object)
            else:  # if_binary
                n_features_out_no_drop = [len(cat) for cat in self._categories_list]
                if self._infrequent_enabled:
                    for i, infreq_idx in enumerate(self._infrequent_indices):
                        if infreq_idx is None:
                            continue
                        n_features_out_no_drop[i] -= infreq_idx.size - 1

                drop_idx_after_grouping = np.array(
                    [0 if n_features_out == 2 else None for n_features_out in n_features_out_no_drop],
                    dtype=object,
                )
        else:
            try:
                drop_array = np.asarray(self.drop, dtype=object)
                droplen = len(drop_array)
            except (ValueError, TypeError):
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ATTRIBUTE,
                    original_exception=ValueError(
                        "`drop` must be one of 'first', 'if_binary', an array-like of "  # type: ignore[str-bytes-safe]
                        f"shape (n_features,), or None, got {self.drop}."
                    ),
                )
            if droplen != len(self._categories_list):
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ATTRIBUTE,
                    original_exception=ValueError(
                        f"`drop` must have length equal to the number of features ({len(self._categories_list)}), "
                        f"got {droplen}."
                    ),
                )
            missing_drops = []
            drop_indices = []
            for feature_idx, (drop_val, cat_list) in enumerate(zip(drop_array, self._categories_list)):
                if not sklearn_utils.is_scalar_nan(drop_val):
                    drop_idx = np.where(cat_list == drop_val)[0]
                    if drop_idx.size:  # found drop idx
                        drop_indices.append(self._map_drop_idx_to_infrequent(feature_idx, drop_idx[0]))
                    else:
                        missing_drops.append((feature_idx, drop_val))
                    continue

                # drop_val is nan, find nan in categories manually
                for cat_idx, cat in enumerate(cat_list):
                    if sklearn_utils.is_scalar_nan(cat):
                        drop_indices.append(self._map_drop_idx_to_infrequent(feature_idx, cat_idx))
                        break
                else:  # loop did not break thus drop is missing
                    missing_drops.append((feature_idx, drop_val))

            if any(missing_drops):
                msg = (
                    "The following categories were supposed to be "
                    "dropped, but were not found in the training "
                    "data.\n{}".format("\n".join([f"Category: {c}, Feature: {v}" for c, v in missing_drops]))
                )
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_DATA,
                    original_exception=ValueError(msg),
                )
            drop_idx_after_grouping = np.array(drop_indices, dtype=object)

        # `_drop_idx_after_grouping` are the categories to drop *after* the infrequent
        # categories are grouped together. If needed, we remap `drop_idx` back
        # to the categories seen in `self.categories_`.
        self._drop_idx_after_grouping = drop_idx_after_grouping

        if not self._infrequent_enabled or drop_idx_after_grouping is None:
            self.drop_idx_ = self._drop_idx_after_grouping
        else:
            drop_idx_ = []
            for feature_idx, drop_idx in enumerate(drop_idx_after_grouping):
                default_to_infrequent = self._default_to_infrequent_mappings[feature_idx]
                if drop_idx is None or default_to_infrequent is None:
                    orig_drop_idx = drop_idx
                else:
                    orig_drop_idx = np.flatnonzero(default_to_infrequent == drop_idx)[0]

                drop_idx_.append(orig_drop_idx)

            self.drop_idx_ = np.asarray(drop_idx_, dtype=object)

    def _fit_infrequent_category_mapping(
        self, n_samples: int, category_counts: Dict[str, Dict[str, Dict[str, int]]]
    ) -> None:
        """
        Fit infrequent categories.

        Defines the private attribute: `_default_to_infrequent_mappings`. For
        feature `i`, `_default_to_infrequent_mappings[i]` defines the mapping
        from the integer encoding returned by `super().transform()` into
        infrequent categories. If `_default_to_infrequent_mappings[i]` is None,
        there were no infrequent categories in the training set.

        For example if categories 0, 2 and 4 were frequent, while categories
        1, 3, 5 were infrequent for feature 7, then these categories are mapped
        to a single output:
        `_default_to_infrequent_mappings[7] = ndarray([0, 3, 1, 3, 2, 3])`

        Defines private attribute: `_infrequent_indices`. `_infrequent_indices[i]`
        is a ndarray of indices such that
        `categories_[i][_infrequent_indices[i]]` are all the infrequent category
        labels. If the feature `i` has no infrequent categories
        `_infrequent_indices[i]` is None.

        Same as :meth:`sklearn.preprocessing.OneHotEncoder._fit_infrequent_category_mapping`
        with the type of `category_counts` changed from `list of ndarray` to `dict`.

        Args:
            n_samples: Number of samples in training set.
            category_counts: A dict of {column_name: {category: count}}
        """
        category_counts_list = []  # list of ndarray
        for idx, input_col in enumerate(self.input_cols):
            counts = np.vectorize(lambda x: category_counts[input_col][x])(self._categories_list[idx])
            category_counts_list.append(np.array(counts))
        self._infrequent_indices = [
            self._identify_infrequent(category_count, n_samples) for category_count in category_counts_list
        ]

        # compute mapping from default mapping to infrequent mapping
        self._default_to_infrequent_mappings = []

        for cats, infreq_idx in zip(self._categories_list, self._infrequent_indices):
            # no infrequent categories
            if infreq_idx is None:
                self._default_to_infrequent_mappings.append(None)
                continue

            n_cats = len(cats)
            # infrequent indices exist
            mapping = np.empty(n_cats, dtype=np.int64)
            n_infrequent_cats = infreq_idx.size

            # infrequent categories are mapped to the last element.
            n_frequent_cats = n_cats - n_infrequent_cats
            mapping[infreq_idx] = n_frequent_cats

            frequent_indices = np.setdiff1d(np.arange(n_cats), infreq_idx)
            mapping[frequent_indices] = np.arange(n_frequent_cats)

            self._default_to_infrequent_mappings.append(mapping)

    def _identify_infrequent(
        self, category_count: npt.NDArray[np.int_], n_samples: int
    ) -> Optional[npt.NDArray[np.int_]]:
        """
        Compute the infrequent indices.

        Same as :meth:`sklearn.preprocessing.OneHotEncoder._identify_infrequent`
        with `col_idx` removed.

        Args:
            category_count: ndarray of shape (n_cardinality,) of category counts.
            n_samples: Number of samples.

        Returns:
            Indices of infrequent categories. If there are no infrequent categories,
            None is returned.
        """
        if isinstance(self.min_frequency, numbers.Integral):
            infrequent_mask = category_count < self.min_frequency
        elif isinstance(self.min_frequency, numbers.Real):
            min_frequency_abs = n_samples * self.min_frequency
            infrequent_mask = category_count < min_frequency_abs
        else:
            infrequent_mask = np.zeros(category_count.shape[0], dtype=bool)

        n_current_features = category_count.size - infrequent_mask.sum() + 1
        if self.max_categories is not None and self.max_categories < n_current_features:
            # stable sort to preserve original count order
            smallest_levels = np.argsort(category_count, kind="mergesort")[: -self.max_categories + 1]
            infrequent_mask[smallest_levels] = True

        output = np.flatnonzero(infrequent_mask)
        return output if output.size > 0 else None

    def _compute_n_features_outs(self) -> List[int]:
        """Compute the n_features_out for each input feature."""
        output = [len(cats) for cats in self._categories_list]

        if self.drop_idx_ is not None:
            for idx, drop_idx in enumerate(self.drop_idx_):
                if drop_idx is not None:
                    output[idx] -= 1

        if not self._infrequent_enabled:
            return output

        # infrequent is enabled, the number of features out are reduced
        # because the infrequent categories are grouped together
        for idx, infreq_idx in enumerate(self._infrequent_indices):
            if infreq_idx is None:
                continue
            output[idx] -= infreq_idx.size - 1

        return output

    def get_output_cols(self) -> List[str]:
        """
        Output columns getter.

        Returns:
            Output columns.
        """
        return self._inferred_output_cols

    def _get_inferred_output_cols(self) -> List[str]:
        """
        Get output column names meeting Snowflake requirements.
        Only useful when fitting a pandas dataframe.

        Returns:
            Inferred output columns.
        """
        cols = (
            self.output_cols
            if self.sparse
            else [col for input_col in self.input_cols for col in self._dense_output_cols_mappings[input_col]]
        )
        return [identifier.get_inferred_name(c) for c in cols]

    def _handle_dense_output_cols(self) -> None:
        """Assign input column to dense output columns mappings to `self._dense_output_cols_mappings`."""
        for idx, input_col in enumerate(self.input_cols):
            output_col = self.output_cols[idx]
            n_features_out = self._n_features_outs[idx]
            self._dense_output_cols_mappings[input_col] = []

            # whether there are infrequent categories in the input column
            has_infrequent_categories = self._infrequent_enabled and self.infrequent_categories_[idx] is not None

            # integer encoding of infrequent categories
            # in `_default_to_infrequent_mappings[idx]`
            infrequent_encoding = None
            if has_infrequent_categories:
                infrequent_idx = self._infrequent_indices[idx][0]
                infrequent_encoding = self._default_to_infrequent_mappings[idx][infrequent_idx]
                if (
                    self._drop_idx_after_grouping is not None
                    and self._drop_idx_after_grouping[idx] is not None
                    and infrequent_encoding > self._drop_idx_after_grouping[idx]
                ):
                    infrequent_encoding -= 1

            # get output column names
            for encoding in range(n_features_out):
                # increment the encoding if it occurs after that of the dropped category
                orig_encoding = encoding
                if (
                    (not has_infrequent_categories or encoding != infrequent_encoding)
                    and self._drop_idx_after_grouping is not None
                    and self._drop_idx_after_grouping[idx] is not None
                    and encoding >= self._drop_idx_after_grouping[idx]
                ):
                    orig_encoding += 1

                if has_infrequent_categories:
                    if encoding == infrequent_encoding:
                        cat = _INFREQUENT_CATEGORY
                    else:
                        cat_idx = np.where(self._default_to_infrequent_mappings[idx] == orig_encoding)[0][0]
                        cat = self.categories_[input_col][cat_idx]
                else:
                    cat = self.categories_[input_col][orig_encoding]
                if cat and isinstance(cat, str):
                    cat = cat.replace('"', "'")
                self._dense_output_cols_mappings[input_col].append(f"{output_col}_{cat}")

    def _handle_inferred_output_cols(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> None:
        """
        Assign output column names used to transform pandas dataframes to `self._inferred_output_cols`.
        This ensures consistent (double quoted) column names in Snowpark and pandas transformed dataframes.

        Args:
            dataset: Input dataset.
        """
        if isinstance(dataset, snowpark.DataFrame):
            temp = self.handle_unknown
            self.handle_unknown = "ignore"
            self.transform(dataset[self.input_cols].limit(0))
            self.handle_unknown = temp
        else:
            self._inferred_output_cols = self._get_inferred_output_cols()

    def get_sklearn_args(
        self,
        default_sklearn_obj: Optional[object] = None,
        sklearn_initial_keywords: Optional[Union[str, Iterable[str]]] = None,
        sklearn_unused_keywords: Optional[Union[str, Iterable[str]]] = None,
        snowml_only_keywords: Optional[Union[str, Iterable[str]]] = None,
        sklearn_added_keyword_to_version_dict: Optional[Dict[str, str]] = None,
        sklearn_added_kwarg_value_to_version_dict: Optional[Dict[str, Dict[str, str]]] = None,
        sklearn_deprecated_keyword_to_version_dict: Optional[Dict[str, str]] = None,
        sklearn_removed_keyword_to_version_dict: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Modified snowflake.ml.framework.base.Base.get_sklearn_args with `sparse` and `sparse_output` handling."""
        default_sklearn_args = _utils.get_default_args(default_sklearn_obj.__class__.__init__)
        given_args = self.get_params()

        # replace 'sparse' with 'sparse_output' when scikit-learn>=1.2
        sklearn_version = sklearn.__version__
        if version.parse(sklearn_version) >= version.parse(_SKLEARN_DEPRECATED_KEYWORD_TO_VERSION_DICT["sparse"]):
            given_args["sparse_output"] = given_args.pop("sparse")

        sklearn_args: Dict[str, Any] = _utils.get_filtered_valid_sklearn_args(
            args=given_args,
            default_sklearn_args=default_sklearn_args,
            sklearn_initial_keywords=sklearn_initial_keywords,
            sklearn_unused_keywords=sklearn_unused_keywords,
            snowml_only_keywords=snowml_only_keywords,
            sklearn_added_keyword_to_version_dict=sklearn_added_keyword_to_version_dict,
            sklearn_added_kwarg_value_to_version_dict=sklearn_added_kwarg_value_to_version_dict,
            sklearn_deprecated_keyword_to_version_dict=sklearn_deprecated_keyword_to_version_dict,
            sklearn_removed_keyword_to_version_dict=sklearn_removed_keyword_to_version_dict,
        )
        return sklearn_args
