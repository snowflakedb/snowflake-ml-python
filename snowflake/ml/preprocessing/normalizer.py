#!/usr/bin/env python3
#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
from typing import Iterable, Optional, Union

import pandas as pd
from sklearn import preprocessing

from snowflake import snowpark
from snowflake.ml.framework import base
from snowflake.ml.utils import telemetry
from snowflake.snowpark import functions, types

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "Preprocessing"
_VALID_NORMS = ["l1", "l2", "max"]


class Normalizer(base.BaseEstimator, base.BaseTransformer):
    def __init__(
        self,
        *,
        norm: str = "l2",
        input_cols: Optional[Union[str, Iterable[str]]] = None,
        output_cols: Optional[Union[str, Iterable[str]]] = None,
        drop_input_cols: Optional[bool] = False,
    ) -> None:
        """
        Normalize samples individually to each row's unit norm.

        Each sample (i.e. each row of the data matrix) with at least one
        nonzero component is rescaled independently of other samples so
        that its norm (l1, l2 or inf) equals one.

        Args:
            norm: The norm to use to normalize each non zero sample. If norm='max'
                is used, values will be rescaled by the maximum of the absolute
                values. It must be one of 'l1', 'l2', or 'max'.
            input_cols: Single or multiple input columns.
            output_cols: Single or multiple output columns.
            drop_input_cols: Remove input columns from output if set True. False by default.
        """
        self.norm = norm
        self._is_fitted = False

        base.BaseEstimator.__init__(self)
        base.BaseTransformer.__init__(self, drop_input_cols=drop_input_cols)

        self.set_input_cols(input_cols)
        self.set_output_cols(output_cols)

    def _reset(self) -> None:
        """
        Reset internal data-dependent state of the normalizer.
        __init__ parameters are not touched.

        This is a stateless transformer, so there is nothing to reset.
        """
        pass

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def fit(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> "Normalizer":
        """
        Does nothing, because the normalizer is a stateless transformer.

        Args:
            dataset: Input dataset.

        Returns:
            Fitted normalizer.
        """
        self._is_fitted = True
        return self

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def transform(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> Union[snowpark.DataFrame, pd.DataFrame]:
        """
        Scale each nonzero row of the input dataset to the unit norm.

        Args:
            dataset: Input dataset.

        Returns:
            transformed_dataset: Output dataset.

        Raises:
            ValueError: If the dataset contains nulls, or if the supplied norm is invalid.
            TypeError: If the input dataset is neither a pandas nor Snowpark DataFrame.
        """
        if self.norm not in _VALID_NORMS:
            raise ValueError(f"'{self.norm}' is not a supported norm.")

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
        if len(self.input_cols) == 0:
            raise ValueError("Found array with 0 columns, but a minimum of 1 is required.")

        if self.norm == "l1":
            norm = functions.lit("0")  # type: ignore[arg-type]
            for input_col in self.input_cols:
                norm += functions.abs(dataset[input_col])  # type: ignore[operator]

        elif self.norm == "l2":
            norm = functions.lit("0")  # type: ignore[arg-type]
            for input_col in self.input_cols:
                norm += dataset[input_col] * dataset[input_col]
            norm = functions.sqrt(norm)  # type: ignore[arg-type]

        elif self.norm == "max":
            norm = functions.greatest(
                *[functions.abs(dataset[input_col]) for input_col in self.input_cols]  # type: ignore[arg-type]
            )

        else:
            raise ValueError(f"'{self.norm}' is not a supported norm.")

        output_columns = []
        for input_col in self.input_cols:
            # Set the entry to 0 if the norm is 0, because the norm is 0 only when all entries are 0.
            output_column = functions.div0(
                dataset[input_col].cast(types.FloatType()),
                norm,  # type: ignore[arg-type]
            )
            output_columns.append(output_column)

        transformed_dataset: snowpark.DataFrame = dataset.with_columns(self.output_cols, output_columns)
        return transformed_dataset

    def _create_unfitted_sklearn_object(self) -> preprocessing.Normalizer:
        return preprocessing.Normalizer(norm=self.norm)

    def _create_sklearn_object(self) -> preprocessing.Normalizer:
        """
        Get an equivalent sklearn Normalizer.

        Returns:
            Sklearn Normalizer.
        """
        return self._create_unfitted_sklearn_object()
