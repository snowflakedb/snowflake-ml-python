#!/usr/bin/env python3
#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import sklearn.preprocessing._data as sklearn_preprocessing_data
from sklearn.preprocessing import MaxAbsScaler as SklearnMaxAbsScaler

from snowflake.ml.framework.base import BaseEstimator, BaseTransformer
from snowflake.ml.utils import telemetry
from snowflake.snowpark import DataFrame

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "Preprocessing"


class MaxAbsScaler(BaseEstimator, BaseTransformer):
    def __init__(
        self,
        *,
        input_cols: Optional[Union[str, Iterable[str]]] = None,
        output_cols: Optional[Union[str, Iterable[str]]] = None,
        drop_input_cols: Optional[bool] = False,
    ) -> None:
        """Scale each feature by its maximum absolute value.

        This transfomer scales and translates each feature individually such
        that the maximal absolute value of each feature in the
        training set will be 1.0. It does not shift/center the data, and
        thus does not destroy any sparsity.

        Args:
            input_cols: Single or multiple input columns.
            output_cols: Single or multiple output columns.
            drop_input_cols: Remove input columns from output if set True. False by default.

        Attributes:
            scale_: dict {column_name: value} or None
                Per feature relative scaling of the data.
            max_abs_: dict {column_name: value} or None
                Per feature maximum absolute value.
        """
        self.max_abs_: Dict[str, float] = {}
        self.scale_: Dict[str, float] = {}

        self.custom_state: List[str] = [
            "SQL>>>max(abs({col_name}))",
        ]

        BaseEstimator.__init__(self, custom_state=self.custom_state)
        BaseTransformer.__init__(self, drop_input_cols=drop_input_cols)

        self.set_input_cols(input_cols)
        self.set_output_cols(output_cols)

    def _reset(self) -> None:
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        super()._reset()
        self.scale_ = {}
        self.max_abs_ = {}

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def fit(self, dataset: Union[DataFrame, pd.DataFrame]) -> "MaxAbsScaler":
        """Compute the maximum absolute value to be used for later scaling.

        Args:
            dataset: Input dataset.

        Returns:
            Return self as fitted scaler.

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
            self.max_abs_[input_col] = float(sklearn_encoder.max_abs_[i])
            self.scale_[input_col] = float(sklearn_encoder.scale_[i])

    def _fit_snowpark(self, dataset: DataFrame) -> None:
        computed_states = self._compute(dataset, self.input_cols, self.custom_state)

        for input_col in self.input_cols:
            max_abs = float(computed_states[input_col]["SQL>>>max(abs({col_name}))"])
            self.max_abs_[input_col] = max_abs
            self.scale_[input_col] = sklearn_preprocessing_data._handle_zeros_in_scale(
                self.max_abs_[input_col], copy=True
            )

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def transform(self, dataset: Union[DataFrame, pd.DataFrame]) -> Union[DataFrame, pd.DataFrame]:
        """Scale the data.

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
        """Scale the data on snowflake DataFrame.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.
        """
        output_columns = []
        for _, input_col in enumerate(self.input_cols):
            col = dataset[input_col]
            col /= float(self.scale_[input_col])
            output_columns.append(col)

        transformed_dataset = dataset.with_columns(self.output_cols, output_columns)
        return transformed_dataset

    def _create_unfitted_sklearn_object(self) -> SklearnMaxAbsScaler:
        return SklearnMaxAbsScaler()

    def _create_sklearn_object(self) -> SklearnMaxAbsScaler:
        """Get an equivalent sklearn MaxAbsdScaler.

        Returns:
            Sklearn MaxAbsScaler.
        """
        scaler = self._create_unfitted_sklearn_object()
        scaler.scale_ = self._convert_attribute_dict_to_ndarray(self.scale_, np.float64)
        scaler.max_abs_ = self._convert_attribute_dict_to_ndarray(self.max_abs_, np.float64)
        return scaler
