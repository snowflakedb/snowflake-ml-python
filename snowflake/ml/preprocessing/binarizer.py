#!/usr/bin/env python3
#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
from typing import Iterable, Optional, Union

import pandas as pd
from sklearn import preprocessing

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml.framework import base
from snowflake.snowpark import functions as F, types as T


class Binarizer(base.BaseEstimator, base.BaseTransformer):
    def __init__(
        self,
        *,
        threshold: float = 0.0,
        input_cols: Optional[Union[str, Iterable[str]]] = None,
        output_cols: Optional[Union[str, Iterable[str]]] = None,
        drop_input_cols: Optional[bool] = False,
    ) -> None:
        """
        Map each feature to a binary value.

        Values greater than the threshold map to 1, while values less than or equal to
        the threshold map to 0. With the default threshold of 0, only positive values map to 1.

        Data must be float-valued.

        Args:
            threshold: Feature values below or equal to this are replaced by 0, above it by 1.
            input_cols: Single or multiple input columns.
            output_cols: Single or multiple output columns.
            drop_input_cols: Remove input columns from output if set True. False by default.
        """
        self.threshold = threshold

        base.BaseEstimator.__init__(self)
        base.BaseTransformer.__init__(self, drop_input_cols=drop_input_cols)

        self.set_input_cols(input_cols)
        self.set_output_cols(output_cols)

    def _reset(self) -> None:
        """
        Reset internal data-dependent state of the binarizer.
        __init__ parameters are not touched.

        This is a stateless transformer, so there is nothing to reset.
        """
        super()._reset()

    @telemetry.send_api_usage_telemetry(
        project=base.PROJECT,
        subproject=base.SUBPROJECT,
    )
    def fit(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> "Binarizer":
        """
        This is a stateless transformer, so there is nothing to fit.

        Args:
            dataset: Input dataset.

        Returns:
            self

        Raises:
            TypeError: If the threshold is not a float.
        """
        if not isinstance(self.threshold, float):
            raise TypeError(f"Binarizer threshold must be a float, but got {type(self.threshold)}.")

        self._is_fitted = True
        return self

    @telemetry.send_api_usage_telemetry(
        project=base.PROJECT,
        subproject=base.SUBPROJECT,
    )
    def transform(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> Union[snowpark.DataFrame, pd.DataFrame]:
        """
        Binarize the data. Map to 1 if it is strictly greater than the threshold, otherwise 0.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.

        Raises:
            TypeError: If the input dataset is neither a pandas nor Snowpark DataFrame.
        """
        super()._check_input_cols()
        super()._check_output_cols()

        if isinstance(dataset, snowpark.DataFrame):
            output_df = self._transform_snowpark(dataset)
        elif isinstance(dataset, pd.DataFrame):
            output_df = self._transform_sklearn(dataset)
        else:
            raise TypeError(
                f"Unexpected dataset type: {type(dataset)}."
                "Supported dataset types: snowpark.DataFrame, pandas.DataFrame."
            )

        return self._drop_input_columns(output_df) if self._drop_input_cols is True else output_df

    def _transform_snowpark(self, dataset: snowpark.DataFrame) -> snowpark.DataFrame:
        self._validate_data_has_no_nulls(dataset)
        output_columns = []
        for input_col in self.input_cols:
            col = F.iff(dataset[input_col] > self.threshold, 1.0, 0.0).cast(T.FloatType())  # type: ignore[arg-type]
            output_columns.append(col)

        transformed_dataset: snowpark.DataFrame = dataset.with_columns(self.output_cols, output_columns)
        return transformed_dataset

    def _create_unfitted_sklearn_object(self) -> preprocessing.Binarizer:
        return preprocessing.Binarizer(threshold=self.threshold)

    def _create_sklearn_object(self) -> preprocessing.Binarizer:
        """
        Get an equivalent sklearn Binarizer.

        Returns:
            Sklearn Binarizer.
        """
        return self._create_unfitted_sklearn_object()
