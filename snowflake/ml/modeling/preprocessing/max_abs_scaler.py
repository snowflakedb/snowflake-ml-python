#!/usr/bin/env python3
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import _data as sklearn_preprocessing_data

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml.modeling.framework import base


class MaxAbsScaler(base.BaseTransformer):
    r"""Scale each feature by its maximum absolute value.

    This transformer scales and translates each feature individually such
    that the maximal absolute value of each feature in the
    training set will be 1.0. It does not shift/center the data, and
    thus does not destroy any sparsity.

    Values must be of float type. Each feature is scaled and transformed individually such that the maximal
    absolute value of each feature in the dataset is 1.0. This scaler does not shift or center the data,
    preserving sparsity.

    For more details on what this transformer does, see [sklearn.preprocessing.MaxAbsScaler]
    (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html).

    Args:
        input_cols: The name(s) of one or more columns in a DataFrame containing a feature to be scaled.
        output_cols: The name(s) of one or more columns in a DataFrame in which results will be stored. The number of
            columns specified must match the number of input columns.
        drop_input_cols: Remove input columns from output if set True. False by default.

    Attributes:
        scale_: dict {column_name: value} or None. Per-feature relative scaling factor.
        max_abs_: dict {column_name: value} or None. Per-feature maximum absolute value.
    """

    def __init__(
        self,
        *,
        input_cols: Optional[Union[str, Iterable[str]]] = None,
        output_cols: Optional[Union[str, Iterable[str]]] = None,
        drop_input_cols: Optional[bool] = False,
    ) -> None:
        """
        Scale each feature by its maximum absolute value.

        This transformer scales and translates each feature individually such
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

        self.custom_states: List[str] = [
            "SQL>>>max(abs({col_name}))",
        ]

        super().__init__(drop_input_cols=drop_input_cols, custom_states=self.custom_states)

        self.set_input_cols(input_cols)
        self.set_output_cols(output_cols)

    def _reset(self) -> None:
        """
        Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        super()._reset()
        self.scale_ = {}
        self.max_abs_ = {}

    @telemetry.send_api_usage_telemetry(
        project=base.PROJECT,
        subproject=base.SUBPROJECT,
    )
    def fit(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> "MaxAbsScaler":
        """
        Compute the maximum absolute value to be used for later scaling.

        Validates the transformer arguments and derives the scaling factors and maximum absolute values from the data,
        making dictionaries of both available as attributes of the transformer instance (see Attributes).

        Returns the transformer instance.

        Args:
            dataset: Input dataset.

        Returns:
            Return self as fitted scaler.
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
            self.max_abs_[input_col] = float(sklearn_scaler.max_abs_[i])
            self.scale_[input_col] = float(sklearn_scaler.scale_[i])

    def _fit_snowpark(self, dataset: snowpark.DataFrame) -> None:
        computed_states = self._compute(dataset, self.input_cols, self.custom_states)

        for input_col in self.input_cols:
            max_abs = float(computed_states[input_col]["SQL>>>max(abs({col_name}))"])
            self.max_abs_[input_col] = max_abs
            self.scale_[input_col] = sklearn_preprocessing_data._handle_zeros_in_scale(
                self.max_abs_[input_col], copy=True
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
        Scale the data.

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
        Scale the data on snowflake DataFrame.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.
        """
        passthrough_columns = [c for c in dataset.columns if c not in self.output_cols]
        output_columns = []
        for _, input_col in enumerate(self.input_cols):
            col = dataset[input_col]
            col /= float(self.scale_[input_col])
            output_columns.append(col)

        transformed_dataset: snowpark.DataFrame = dataset.with_columns(self.output_cols, output_columns)
        # Reorder columns. Passthrough columns are added at the right to the output of the transformers.
        transformed_dataset = transformed_dataset[self.output_cols + passthrough_columns]
        return transformed_dataset

    def _create_unfitted_sklearn_object(self) -> preprocessing.MaxAbsScaler:
        return preprocessing.MaxAbsScaler()

    def _create_sklearn_object(self) -> preprocessing.MaxAbsScaler:
        """
        Get an equivalent sklearn MaxAbsdScaler.

        Returns:
            Sklearn MaxAbsScaler.
        """
        scaler = self._create_unfitted_sklearn_object()
        if self._is_fitted:
            scaler.scale_ = self._convert_attribute_dict_to_ndarray(self.scale_, np.float64)
            scaler.max_abs_ = self._convert_attribute_dict_to_ndarray(self.max_abs_, np.float64)
        return scaler
