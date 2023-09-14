#!/usr/bin/env python3
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import _data as sklearn_preprocessing_data

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml.modeling.framework import _utils, base


class RobustScaler(base.BaseTransformer):
    r"""Scales features using statistics that are robust to outliers. Values must be of float type.

    For more details on what this transformer does, see [sklearn.preprocessing.RobustScaler]
    (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html).

    Args:
        with_centering: If True, center the data around zero before scaling.
        with_scaling: If True, scale the data to interquartile range.
        quantile_range: tuple like (q_min, q_max), where 0.0 < q_min < q_max < 100.0, default=(25.0, 75.0). Quantile
            range used to calculate scale_. By default, this is equal to the IQR, i.e., q_min is the first quantile and
            q_max is the third quantile.
        unit_variance: If True, scale data so that normally-distributed features have a variance of 1. In general, if
            the difference between the x-values of q_max and q_min for a standard normal distribution is greater than 1,
            the dataset is scaled down. If less than 1, the dataset is scaled up.
        input_cols: The name(s) of one or more columns in a DataFrame containing a feature to be scaled.
        output_cols: The name(s) of one or more columns in a DataFrame in which results will be stored. The number of
            columns specified must match the number of input columns. For dense output, the column names specified are
            used as base names for the columns created for each category.
        drop_input_cols: Remove input columns from output if set True. False by default.

    Attributes:
        center_: Dictionary mapping input column name to the median value for that feature.
        scale_: Dictionary mapping input column name to the (scaled) interquartile range for that feature.
    """

    def __init__(
        self,
        *,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: Tuple[float, float] = (25.0, 75.0),
        unit_variance: bool = False,
        input_cols: Optional[Union[str, Iterable[str]]] = None,
        output_cols: Optional[Union[str, Iterable[str]]] = None,
        drop_input_cols: Optional[bool] = False,
    ) -> None:
        """
        Scale features using statistics that are robust to outliers.

        Args:
            with_centering: If True, center the data before scaling. This will cause transform
                to raise an exception when attempted on sparse matrices, because centering them
                entails building a dense matrix which in common use cases is likely to be too large
                to fit in memory.
            with_scaling: If True, scale the data to interquartile range.
            quantile_range: tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0, default=(25.0, 75.0)
                Quantile range used to calculate scale_. By default this is equal to the IQR, i.e.,
                q_min is the first quantile and q_max is the third quantile.
            unit_variance: If True, scale data so that normally distributed features have a variance
                of 1. In general, if the difference between the x-values of q_max and q_min for a
                standard normal distribution is greater than 1, the dataset will be scaled down.
                If less than 1, the dataset will be scaled up.
            input_cols: Single or multiple input columns.
            output_cols: Single or multiple output columns.
            drop_input_cols: Remove input columns from output if set True. False by default.

        Attributes:
            center_: dict {column_name: The median value for each feature in the training set}.
            scale_: The (scaled) interquartile range for each feature in the training set.
        """
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.unit_variance = unit_variance

        self._state_is_set = False
        self._center: Dict[str, float] = {}
        self._scale: Dict[str, float] = {}

        l_range = self.quantile_range[0] / 100.0
        r_range = self.quantile_range[1] / 100.0
        self.custom_states: List[str] = [
            _utils.NumericStatistics.MEDIAN,
            "SQL>>>percentile_cont(" + str(l_range) + ") within group (order by {col_name})",
            "SQL>>>percentile_cont(" + str(r_range) + ") within group (order by {col_name})",
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
        self._center = {}
        self._scale = {}
        self._state_is_set = False

    @property
    def center_(self) -> Optional[Dict[str, float]]:
        return None if (not self.with_centering or not self._state_is_set) else self._center

    @property
    def scale_(self) -> Optional[Dict[str, float]]:
        return None if (not self.with_scaling or not self._state_is_set) else self._scale

    @telemetry.send_api_usage_telemetry(
        project=base.PROJECT,
        subproject=base.SUBPROJECT,
    )
    def fit(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> "RobustScaler":
        """
        Compute center, scale and quantile values of the dataset.

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
        self._state_is_set = True
        return self

    def _fit_sklearn(self, dataset: pd.DataFrame) -> None:
        dataset = self._use_input_cols_only(dataset)
        sklearn_scaler = self._create_unfitted_sklearn_object()
        sklearn_scaler.fit(dataset[self.input_cols])

        for i, input_col in enumerate(self.input_cols):
            if self.with_centering:
                self._center[input_col] = float(sklearn_scaler.center_[i])
            if self.with_scaling:
                self._scale[input_col] = float(sklearn_scaler.scale_[i])

    def _fit_snowpark(self, dataset: snowpark.DataFrame) -> None:
        computed_states = self._compute(dataset, self.input_cols, self.custom_states)

        q_min, q_max = self.quantile_range
        if not 0 <= q_min <= q_max <= 100:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=ValueError("Invalid quantile range: %s" % str(self.quantile_range)),
            )

        pcont_left = self.custom_states[1]
        pcont_right = self.custom_states[2]

        for input_col in self.input_cols:
            numeric_stats = computed_states[input_col]
            if self.with_centering:
                self._center[input_col] = float(numeric_stats[_utils.NumericStatistics.MEDIAN])
            else:
                self._center[input_col] = 0

            if self.with_scaling:
                self._scale[input_col] = float(numeric_stats[pcont_right]) - float(numeric_stats[pcont_left])
                self._scale[input_col] = sklearn_preprocessing_data._handle_zeros_in_scale(
                    self._scale[input_col], copy=False
                )
                if self.unit_variance:
                    adjust = stats.norm.ppf(q_max / 100.0) - stats.norm.ppf(q_min / 100.0)
                    self._scale[input_col] = self._scale[input_col] / adjust
            else:
                self._scale[input_col] = 1

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
        Center and scale the data.

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
        Center and scale the data on snowflake DataFrame.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.
        """
        passthrough_columns = [c for c in dataset.columns if c not in self.output_cols]
        output_columns = []
        for _, input_col in enumerate(self.input_cols):
            col = dataset[input_col]
            if self.center_ is not None:
                col -= self.center_[input_col]
            if self.scale_ is not None:
                col /= float(self.scale_[input_col])
            output_columns.append(col)

        transformed_dataset: snowpark.DataFrame = dataset.with_columns(self.output_cols, output_columns)
        # Reorder columns. Passthrough columns are added at the right to the output of the transformers.
        transformed_dataset = transformed_dataset[self.output_cols + passthrough_columns]
        return transformed_dataset

    def _create_unfitted_sklearn_object(self) -> preprocessing.RobustScaler:
        return preprocessing.RobustScaler(
            with_centering=self.with_centering,
            with_scaling=self.with_scaling,
            quantile_range=self.quantile_range,
            copy=True,
            unit_variance=self.unit_variance,
        )

    def _create_sklearn_object(self) -> preprocessing.RobustScaler:
        """
        Get an equivalent sklearn RobustScaler.

        Returns:
            Sklearn RobustScaler.
        """
        scaler = self._create_unfitted_sklearn_object()
        if self._is_fitted:
            scaler.scale_ = self._convert_attribute_dict_to_ndarray(self.scale_, np.float64)
            scaler.center_ = self._convert_attribute_dict_to_ndarray(self.center_, np.float64)
        return scaler
