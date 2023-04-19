#!/usr/bin/env python3
#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import sklearn.preprocessing._data as sklearn_preprocessing_data
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler

import snowflake.snowpark.functions as F
from snowflake.ml.framework import utils
from snowflake.ml.framework.base import BaseEstimator, BaseTransformer
from snowflake.ml.utils import telemetry
from snowflake.snowpark import DataFrame

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "Preprocessing"


class MinMaxScaler(BaseEstimator, BaseTransformer):
    def __init__(
        self,
        *,
        feature_range: Tuple[float, float] = (0, 1),
        clip: bool = False,
        input_cols: Optional[Union[str, Iterable[str]]] = None,
        output_cols: Optional[Union[str, Iterable[str]]] = None,
        drop_input_cols: Optional[bool] = False,
    ) -> None:
        """
        Transform features by scaling each feature to a given range.

        Args:
            feature_range: Desired range of transformed data.
            clip: Set to True to clip transformed values of held-out data to provided `feature range`.
            input_cols: Single or multiple input columns.
            output_cols: Single or multiple output columns.
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

        self.min_: Dict[str, float] = {}
        self.scale_: Dict[str, float] = {}
        self.data_min_: Dict[str, float] = {}
        self.data_max_: Dict[str, float] = {}
        self.data_range_: Dict[str, float] = {}

        self.custom_state: List[str] = [utils.NumericStatistics.MIN.value, utils.NumericStatistics.MAX.value]

        BaseEstimator.__init__(self, custom_state=self.custom_state)
        BaseTransformer.__init__(self, drop_input_cols=drop_input_cols)

        self.set_input_cols(input_cols)
        self.set_output_cols(output_cols)

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

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def fit(self, dataset: Union[DataFrame, pd.DataFrame]) -> "MinMaxScaler":
        """
        Compute min and max values of the dataset.

        Args:
            dataset: Input dataset.

        Returns:
            Fitted scaler.

        Raises:
            TypeError: If the input dataset is neither a pandas or Snowpark DataFrame.
        """
        super()._check_input_cols()
        self._reset()

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

        for (i, input_col) in enumerate(self.input_cols):
            self.min_[input_col] = float(sklearn_encoder.min_[i])
            self.scale_[input_col] = float(sklearn_encoder.scale_[i])
            self.data_min_[input_col] = float(sklearn_encoder.data_min_[i])
            self.data_max_[input_col] = float(sklearn_encoder.data_max_[i])
            self.data_range_[input_col] = float(sklearn_encoder.data_range_[i])

    def _fit_snowpark(self, dataset: DataFrame) -> None:
        computed_states = self._compute(dataset, self.input_cols, self.custom_state)

        # assign states to the object
        for input_col in self.input_cols:
            numeric_stats = computed_states[input_col]

            data_min = float(numeric_stats[utils.NumericStatistics.MIN])
            data_max = float(numeric_stats[utils.NumericStatistics.MAX])
            data_range = data_max - data_min
            self.scale_[input_col] = (
                self.feature_range[1] - self.feature_range[0]
            ) / sklearn_preprocessing_data._handle_zeros_in_scale(data_range)
            self.min_[input_col] = self.feature_range[0] - data_min * self.scale_[input_col]
            self.data_min_[input_col] = data_min
            self.data_max_[input_col] = data_max
            self.data_range_[input_col] = data_range

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def transform(self, dataset: Union[DataFrame, pd.DataFrame]) -> Union[DataFrame, pd.DataFrame]:
        """
        Scale features according to feature_range.

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
        Scale features according to feature_range on
        Snowpark dataframe.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.
        """
        output_columns = []
        for _, input_col in enumerate(self.input_cols):
            output_column = dataset[input_col] * self.scale_[input_col] + self.min_[input_col]

            if self.clip:
                output_column = F.greatest(output_column, F.lit(self.feature_range[0]))
                output_column = F.least(output_column, F.lit(self.feature_range[1]))

            output_columns.append(output_column)

        transformed_dataset = dataset.with_columns(self.output_cols, output_columns)
        return transformed_dataset

    def _create_unfitted_sklearn_object(self) -> SklearnMinMaxScaler:
        return SklearnMinMaxScaler(feature_range=self.feature_range, clip=self.clip)

    def _create_sklearn_object(self) -> SklearnMinMaxScaler:
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
