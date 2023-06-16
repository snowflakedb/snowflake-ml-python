#!/usr/bin/env python3
#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
from typing import Iterable, Optional, Union

import pandas as pd
from sklearn import preprocessing

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml.modeling.framework import base
from snowflake.snowpark import functions as F, types as T


class Binarizer(base.BaseTransformer):
    r"""Binarizes data (sets feature values to 0 or 1) according to the given threshold.

    Values must be of float type. Values greater than the threshold map to 1, while values less than or equal to
    the threshold map to 0. The default threshold of 0.0 maps only positive values to 1.

    For more details on what this transformer does, see [sklearn.preprocessing.Binarizer]
    (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html).

    Args:
        threshold: Feature values below or equal to this are replaced by 0, above it by 1. Default values is 0.0.
        input_cols: The name(s) of one or more columns in a DataFrame containing a feature to be binarized.
        output_cols: The name(s) of one or more columns in a DataFrame in which results will be stored. The number of
            columns specified must match the number of input columns.
        drop_input_cols: Remove input columns from output if set True. False by default.
    """

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
            threshold: Feature values below or equal to this are replaced by 0, above it by 1. Default values is 0.0.
            input_cols: The name(s) of one or more columns in a DataFrame containing a feature to be binarized.
            output_cols: The name(s) of one or more columns in a DataFrame in which results will be stored. The number
                of columns specified must match the number of input columns.
            drop_input_cols: Remove input columns from output if set True. False by default.
        """
        super().__init__(drop_input_cols=drop_input_cols)
        self.threshold = threshold
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
        This is a stateless transformer, so there is nothing to fit. Validates the transformer arguments.
        Returns the transformer instance.

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
    @telemetry.add_stmt_params_to_df(
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
        passthrough_columns = [c for c in dataset.columns if c not in self.output_cols]
        self._validate_data_has_no_nulls(dataset)
        output_columns = []
        for input_col in self.input_cols:
            col = F.iff(dataset[input_col] > self.threshold, 1.0, 0.0).cast(T.FloatType())  # type: ignore[arg-type]
            output_columns.append(col)

        transformed_dataset: snowpark.DataFrame = dataset.with_columns(self.output_cols, output_columns)
        # Reorder columns. Passthrough columns are added at the right to the output of the transformers.
        transformed_dataset = transformed_dataset[self.output_cols + passthrough_columns]
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
