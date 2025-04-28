#!/usr/bin/env python3
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import _data as sklearn_preprocessing_data

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml.modeling.framework import _utils, base


class StandardScaler(base.BaseTransformer):
    r"""
    Standardizes features by removing the mean and scaling to unit variance. Values must be of float type.

    For more details on what this transformer does, see [sklearn.preprocessing.StandardScaler]
    (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

    Args:
        with_mean: bool, default=True
            If True, center the data before scaling.

        with_std: bool, default=True
            If True, scale the data unit variance (i.e. unit standard deviation).

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
        scale_: Optional[Dict[str, float]] = {}
            Dictionary mapping input column names to relative scaling factor to achieve zero mean and unit variance.
            If a variance is zero, unit variance could not be achieved, and the data is left as-is, giving a scaling
            factor of 1. None if with_std is False.

        mean_: Optional[Dict[str, float]] = {}
            Dictionary mapping input column name to the mean value for that feature. None if with_mean is False.

        var_: Optional[Dict[str, float]] = {}
            Dictionary mapping input column name to the variance for that feature. Used to compute scale_. None if
            with_std is False
    """

    def __init__(
        self,
        *,
        with_mean: bool = True,
        with_std: bool = True,
        input_cols: Optional[Union[str, Iterable[str]]] = None,
        output_cols: Optional[Union[str, Iterable[str]]] = None,
        passthrough_cols: Optional[Union[str, Iterable[str]]] = None,
        drop_input_cols: Optional[bool] = False,
    ) -> None:
        """
        Standardize features by removing the mean and scaling to unit variance.

        Args:
            with_mean: If True, center the data before scaling.
                This does not work (and will raise an exception) when attempted on
                sparse matrices, because centering them entails building a dense
                matrix which in common use cases is likely to be too large to fit in
                memory.
            with_std: If True, scale the data to unit variance (or equivalently,
                unit standard deviation).
            input_cols: Single or multiple input columns.
            output_cols: Single or multiple output columns.
            passthrough_cols: A string or a list of strings indicating column names to be excluded from any
                operations (such as train, transform, or inference). These specified column(s)
                will remain untouched throughout the process. This option is helful in scenarios
                requiring automatic input_cols inference, but need to avoid using specific
                columns, like index columns, during in training or inference.
            drop_input_cols: Remove input columns from output if set True. False by default.

        Attributes:
            scale_: dict {column_name: value} or None
                Per feature relative scaling of the data to achieve zero mean and unit
                variance. If a variance is zero, we can't achieve unit variance, and
                the data is left as-is, giving a scaling factor of 1. `scale_` is equal
                to `None` when `with_std=False`.
            mean_: dict {column_name: value} or None
                The mean value for each feature in the training set.
                Equal to ``None`` when ``with_mean=False``.
            var_: dict {column_name: value} or None
                The variance for each feature in the training set. Used to compute
                `scale_`. Equal to ``None`` when ``with_std=False``.
        """
        self.with_mean = with_mean
        self.with_std = with_std

        self.scale_: Optional[dict[str, float]] = {} if with_std else None
        self.mean_: Optional[dict[str, float]] = {} if with_mean else None
        self.var_: Optional[dict[str, float]] = {} if with_std else None

        self.custom_states: list[str] = []
        if with_mean:
            self.custom_states.append(_utils.NumericStatistics.MEAN)
        if with_std:
            self.custom_states.append(_utils.NumericStatistics.VAR_POP)
            self.custom_states.append(_utils.NumericStatistics.STDDEV_POP)

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
        if hasattr(self, "scale_"):
            self.scale_ = {} if self.with_std else None
        if hasattr(self, "mean_"):
            self.mean_ = {} if self.with_mean else None
        if hasattr(self, "var_"):
            self.var_ = {} if self.with_std else None

    def _fit(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> "StandardScaler":
        """
        Compute mean and std values of the dataset.

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
            if self.mean_ is not None:
                self.mean_[input_col] = _utils.to_float_if_valid(sklearn_scaler.mean_[i], input_col, "mean_")
            if self.scale_ is not None:
                self.scale_[input_col] = _utils.to_float_if_valid(sklearn_scaler.scale_[i], input_col, "scale_")
            if self.var_ is not None:
                self.var_[input_col] = _utils.to_float_if_valid(sklearn_scaler.var_[i], input_col, "var_")

    def _fit_snowpark(self, dataset: snowpark.DataFrame) -> None:
        computed_states = self._compute(dataset, self.input_cols, self.custom_states)

        # assign states to the object
        for input_col in self.input_cols:
            numeric_stats = computed_states[input_col]

            if self.mean_ is not None:
                self.mean_[input_col] = _utils.to_float_if_valid(
                    numeric_stats[_utils.NumericStatistics.MEAN], input_col, "mean_"
                )

            if self.var_ is not None:
                self.var_[input_col] = _utils.to_float_if_valid(
                    numeric_stats[_utils.NumericStatistics.VAR_POP], input_col, "var_"
                )

            if self.scale_ is not None:
                self.scale_[input_col] = sklearn_preprocessing_data._handle_zeros_in_scale(
                    _utils.to_float_if_valid(numeric_stats[_utils.NumericStatistics.STDDEV_POP], input_col, "scale_")
                )

    @telemetry.send_api_usage_telemetry(
        project=base.PROJECT,
        subproject=base.SUBPROJECT,
    )
    def transform(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> Union[snowpark.DataFrame, pd.DataFrame]:
        """
        Perform standardization by centering and scaling.

        Args:
            dataset: Input dataset.

        Returns:
            transformed_dataset: Output dataset.
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
        Perform standardization by centering and scaling on
        Snowpark dataframe.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.
        """
        passthrough_columns = [c for c in dataset.columns if c not in self.output_cols]
        output_columns = []
        for _, input_col in enumerate(self.input_cols):
            output_column = dataset[input_col]

            if self.mean_ is not None:
                output_column = output_column - self.mean_[input_col]
            if self.scale_ is not None:
                output_column = output_column / self.scale_[input_col]

            output_columns.append(output_column)

        transformed_dataset: snowpark.DataFrame = dataset.with_columns(self.output_cols, output_columns)
        # Reorder columns. Passthrough columns are added at the right to the output of the transformers.
        transformed_dataset = transformed_dataset[self.output_cols + passthrough_columns]
        return transformed_dataset

    def _create_unfitted_sklearn_object(self) -> preprocessing.StandardScaler:
        return preprocessing.StandardScaler(with_mean=self.with_mean, with_std=self.with_std)

    def _create_sklearn_object(self) -> preprocessing.StandardScaler:
        """
        Get an equivalent sklearn StandardScaler.

        Returns:
            The Sklearn StandardScaler.
        """
        scaler = self._create_unfitted_sklearn_object()
        if self._is_fitted:
            scaler.scale_ = self._convert_attribute_dict_to_ndarray(self.scale_, np.float64)
            scaler.mean_ = self._convert_attribute_dict_to_ndarray(self.mean_, np.float64)
            scaler.var_ = self._convert_attribute_dict_to_ndarray(self.var_, np.float64)
        return scaler
