#!/usr/bin/env python3
#
# Copyright (c) 2012-2023 Snowflake Computing Inc. All rights reserved.
#
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder

from snowflake.ml.framework.base import BaseEstimator, BaseTransformer
from snowflake.ml.preprocessing.ordinal_encoder import OrdinalEncoder
from snowflake.ml.utils import telemetry
from snowflake.snowpark import DataFrame

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "Preprocessing"
_INDEX = "_INDEX"


class LabelEncoder(BaseEstimator, BaseTransformer):
    def __init__(
        self,
        input_cols: Optional[Union[str, Iterable[str]]] = None,
        output_cols: Optional[Union[str, Iterable[str]]] = None,
        drop_input_cols: Optional[bool] = False,
    ) -> None:
        """
        Encode target labels with integers between 0 and n_classes-1.

        Args:
            input_cols: One label column specified as a string or list with one member.
            output_cols: One output column specified as a string or list with one member.
            drop_input_cols: Remove input columns from output if set True. False by default.

        Attributes:
            classes_: A np.ndarray that holds the label for each class.

        """
        self._ordinal_encoder: Optional[OrdinalEncoder] = None
        self.classes_: Optional[np.ndarray] = None

        BaseEstimator.__init__(self)
        BaseTransformer.__init__(self, drop_input_cols=drop_input_cols)

        self.set_input_cols(input_cols)
        self.set_output_cols(output_cols)

    def _reset(self) -> None:
        super()._reset()
        if self._ordinal_encoder:
            self._ordinal_encoder = None
            self.classes_ = None

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def fit(self, dataset: Union[DataFrame, pd.DataFrame]) -> "LabelEncoder":
        """
        Fit label encoder with label column in dataset.

        Args:
            dataset: Input dataset.

        Returns:
            self

        Raises:
            ValueError: If length of input_cols is not 1 or length of output_cols is greater than 1.
        """
        if len(self.input_cols) != 1:
            raise ValueError("Label encoder must specify one input column.")
        input_col = self.input_cols[0]

        if len(self.output_cols) != 1:
            raise ValueError("Label encoder must specify one output column.")

        self._reset()

        # Use `OrdinalEncoder` to handle fits and transforms.
        self._ordinal_encoder = OrdinalEncoder(input_cols=self.input_cols, output_cols=self.output_cols)

        self._ordinal_encoder.fit(dataset)

        # Set `classes_` for compatibility with sklearn.
        self.classes_ = self._ordinal_encoder.categories_[input_col]

        self._is_fitted = True

        return self

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def transform(self, dataset: Union[DataFrame, pd.DataFrame]) -> Union[DataFrame, pd.DataFrame]:
        """
        Use fit result to transform snowpark dataframe or pandas dataframe. The original dataset with
        the transform result column added will be returned.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.

        Raises:
            RuntimeError: If transformer is not fitted first.
            TypeError: If the input dataset is neither a pandas or Snowpark DataFrame.
        """
        if not self._is_fitted or self._ordinal_encoder is None or self.classes_ is None:
            raise RuntimeError("Label encoder must be fitted before calling transform().")

        if isinstance(dataset, DataFrame):
            output_df = self._ordinal_encoder.transform(dataset).replace(
                float("nan"),
                len(self.classes_) - 1,
                subset=self.output_cols,
            )
        elif isinstance(dataset, pd.DataFrame):
            output_df = self._transform_sklearn(dataset)
        else:
            raise TypeError(
                f"Unexpected dataset type: {type(dataset)}."
                "Supported dataset types: snowpark.DataFrame, pandas.DataFrame."
            )

        return self._drop_input_columns(output_df) if self._drop_input_cols is True else output_df

    def _create_unfitted_sklearn_object(self) -> SklearnLabelEncoder:
        return SklearnLabelEncoder()

    def _create_sklearn_object(self) -> SklearnLabelEncoder:
        """
        Initialize and return the equivalent sklearn label encoder.

        Returns:
            Equivalent sklearn object.
        """
        label_encoder = self._create_unfitted_sklearn_object()
        label_encoder.classes_ = self.classes_
        return label_encoder
