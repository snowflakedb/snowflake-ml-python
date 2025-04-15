#!/usr/bin/env python3
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import _data as sklearn_preprocessing_data

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml.modeling.framework import _utils, base
from snowflake.snowpark import functions as F, types as T


class MinMaxScaler(base.BaseTransformer):
    r"""Transforms features by scaling each feature to a given range, by default between zero and one.

    Values must be of float type. Each feature is scaled and translated independently.

    For more details on what this transformer does, see [sklearn.preprocessing.MinMaxScaler]
    (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).

    Args:
        feature_range: Tuple[float, float], default=(0, 1)
            Desired range of transformed data (default is 0 to 1).

        clip: bool, default=False
            Whether to clip transformed values of held-out data to the specified feature range (default is True).

        input_cols: Optional[Union[str, List[str]]], default=None
            The name(s) of one or more columns in the input DataFrame containing feature(s) to be scaled. Input
            columns must be specified before fit with this argument or after initialization with the
            `set_input_cols` method. This argument is optional for API consistency.

        output_cols: Optional[Union[str, List[str]]], default=None
            The name(s) to assign output columns in the output DataFrame. The number of
            columns specified must equal the number of input columns. Output columns must be specified before transform
            with this argument or after initialization with the `set_output_cols` method. This argument is optional for
            API consistency.

        passthrough_cols: Optional[Union[str, List[str]]], default=None
            A string or a list of strings indicating column names to be excluded from any
            operations (such as train, transform, or inference). These specified column(s)
            will remain untouched throughout the process. This option is helpful in scenarios
            requiring automatic input_cols inference, but need to avoid using specific
            columns, like index columns, during training or inference.

        drop_input_cols: Optional[bool], default=False
            Remove input columns from output if set True. False by default.

    Attributes:
        min_: Dict[str, float]
            dict {column_name: value} or None. Per-feature adjustment for minimum.

        scale_: Dict[str, float]
            dict {column_name: value} or None. Per-feature relative scaling factor.

        data_min_: Dict[str, float]
            dict {column_name: value} or None. Per-feature minimum seen in the data.

        data_max_: Dict[str, float]
            dict {column_name: value} or None. Per-feature maximum seen in the data.

        data_range_: Dict[str, float]
            dict {column_name: value} or None. Per-feature range seen in the data as a (min, max) tuple.
    """

    def __init__(
        self,
        *,
        feature_range: tuple[float, float] = (0, 1),
        clip: bool = False,
        input_cols: Optional[Union[str, Iterable[str]]] = None,
        output_cols: Optional[Union[str, Iterable[str]]] = None,
        passthrough_cols: Optional[Union[str, Iterable[str]]] = None,
        drop_input_cols: Optional[bool] = False,
    ) -> None:
        """
        Transform features by scaling each feature to a given range.

        Args:
            feature_range: Desired range of transformed data.
            clip: Set to True to clip transformed values of held-out data to provided `feature range`.
            input_cols: Single or multiple input columns.
            output_cols: Single or multiple output columns.
            passthrough_cols: A string or a list of strings indicating column names to be excluded from any
                operations (such as train, transform, or inference). These specified column(s)
                will remain untouched throughout the process. This option is helful in scenarios
                requiring automatic input_cols inference, but need to avoid using specific
                columns, like index columns, during in training or inference.
            drop_input_cols: Remove input columns from output if set True. False by default.

        Attributes:
            min_: Per feature adjustment for minimum.
            scale_: Per feature relative scaling of the data.
            data_min_: Per feature minimum seen in the data.
            data_max_: Per feature maximum seen in the data.
            data_range_: Per feature range ``(data_max_ - data_min_)`` seen in the data.
        """
        self.feature_range = feature_range
        self.clip = clip

        self.min_: dict[str, float] = {}
        self.scale_: dict[str, float] = {}
        self.data_min_: dict[str, float] = {}
        self.data_max_: dict[str, float] = {}
        self.data_range_: dict[str, float] = {}

        self.custom_states: list[str] = [_utils.NumericStatistics.MIN, _utils.NumericStatistics.MAX]

        super().__init__(drop_input_cols=drop_input_cols, custom_states=self.custom_states)

        self.set_input_cols(input_cols)
        self.set_output_cols(output_cols)
        self.set_passthrough_cols(passthrough_cols)

    def _reset(self) -> None:
        """
        Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        super()._reset()
        # check one attribute is enough, because they are all set together.
        if hasattr(self, "scale_"):
            self.min_ = {}
            self.scale_ = {}
            self.data_min_ = {}
            self.data_max_ = {}
            self.data_range_ = {}

    def _check_input_column_types(self, dataset: snowpark.DataFrame) -> None:
        for field in dataset.schema.fields:
            if field.name in self.input_cols:
                if not issubclass(type(field.datatype), T._NumericType):
                    raise exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_DATA_TYPE,
                        original_exception=TypeError(
                            f"Non-numeric input column {field.name} datatype {field.datatype} "
                            "is not supported by the MinMaxScaler."
                        ),
                    )

    def _fit(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> "MinMaxScaler":
        """
        Compute min and max values of the dataset.

        Validates the transformer arguments and derives the scaling factors and ranges from the data, making
        dictionaries of both available as attributes of the transformer instance (see Attributes).

        Returns the transformer instance.

        Args:
            dataset: Input dataset.

        Returns:
            Fitted scaler.
        """
        super()._check_input_cols()
        super()._check_dataset_type(dataset)
        self._reset()

        if isinstance(dataset, pd.DataFrame):
            self._fit_sklearn(dataset)
        else:
            self._fit_snowpark(dataset)

        self._is_fitted = True
        return self

    def _fit_sklearn(self, dataset: pd.DataFrame) -> None:
        dataset = self._use_input_cols_only(dataset)
        sklearn_scaler = self._create_unfitted_sklearn_object()
        sklearn_scaler.fit(dataset[self.input_cols])

        for i, input_col in enumerate(self.input_cols):
            self.min_[input_col] = _utils.to_float_if_valid(sklearn_scaler.min_[i], input_col, "min_")
            self.scale_[input_col] = _utils.to_float_if_valid(sklearn_scaler.scale_[i], input_col, "scale_")
            self.data_min_[input_col] = _utils.to_float_if_valid(sklearn_scaler.data_min_[i], input_col, "data_min_")
            self.data_max_[input_col] = _utils.to_float_if_valid(sklearn_scaler.data_max_[i], input_col, "data_max_")
            self.data_range_[input_col] = _utils.to_float_if_valid(
                sklearn_scaler.data_range_[i], input_col, "data_range_"
            )

    def _fit_snowpark(self, dataset: snowpark.DataFrame) -> None:
        self._check_input_column_types(dataset)
        computed_states = self._compute(dataset, self.input_cols, self.custom_states)

        # assign states to the object
        for input_col in self.input_cols:
            numeric_stats = computed_states[input_col]

            data_min = _utils.to_float_if_valid(numeric_stats[_utils.NumericStatistics.MIN], input_col, "data_min_")
            data_max = _utils.to_float_if_valid(numeric_stats[_utils.NumericStatistics.MAX], input_col, "data_max_")
            data_range = data_max - data_min
            self.scale_[input_col] = (
                self.feature_range[1] - self.feature_range[0]
            ) / sklearn_preprocessing_data._handle_zeros_in_scale(data_range)
            self.min_[input_col] = self.feature_range[0] - data_min * self.scale_[input_col]
            self.data_min_[input_col] = data_min
            self.data_max_[input_col] = data_max
            self.data_range_[input_col] = data_range

    @telemetry.send_api_usage_telemetry(
        project=base.PROJECT,
        subproject=base.SUBPROJECT,
    )
    def transform(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> Union[snowpark.DataFrame, pd.DataFrame]:
        """
        Scale features according to feature_range.

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
        Scale features according to feature_range on
        Snowpark dataframe.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.
        """
        passthrough_columns = [c for c in dataset.columns if c not in self.output_cols]
        output_columns = []
        for _, input_col in enumerate(self.input_cols):
            output_column = dataset[input_col] * self.scale_[input_col] + self.min_[input_col]

            if self.clip:
                output_column = F.greatest(
                    output_column,
                    F.lit(self.feature_range[0]),
                )
                output_column = F.least(
                    output_column,
                    F.lit(self.feature_range[1]),
                )

            output_columns.append(output_column)

        transformed_dataset: snowpark.DataFrame = dataset.with_columns(self.output_cols, output_columns)
        # Reorder columns. Passthrough columns are added at the right to the output of the transformers.
        transformed_dataset = transformed_dataset[self.output_cols + passthrough_columns]
        return transformed_dataset

    def _create_unfitted_sklearn_object(self) -> preprocessing.MinMaxScaler:
        return preprocessing.MinMaxScaler(feature_range=self.feature_range, clip=self.clip)

    def _create_sklearn_object(self) -> preprocessing.MinMaxScaler:
        """
        Get an equivalent sklearn MinMaxScaler.

        Returns:
            Sklearn MinMaxScaler.
        """
        scaler = self._create_unfitted_sklearn_object()
        if self._is_fitted:
            scaler.min_ = self._convert_attribute_dict_to_ndarray(self.min_, np.float64)
            scaler.scale_ = self._convert_attribute_dict_to_ndarray(self.scale_, np.float64)
            scaler.data_min_ = self._convert_attribute_dict_to_ndarray(self.data_min_, np.float64)
            scaler.data_max_ = self._convert_attribute_dict_to_ndarray(self.data_max_, np.float64)
            scaler.data_range_ = self._convert_attribute_dict_to_ndarray(self.data_range_, np.float64)
        return scaler
