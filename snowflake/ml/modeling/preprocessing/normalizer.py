#!/usr/bin/env python3
from typing import Iterable, Optional, Union

import pandas as pd
from sklearn import preprocessing

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml.modeling.framework import base
from snowflake.snowpark import functions as F, types as T

_VALID_NORMS = ["l1", "l2", "max"]


class Normalizer(base.BaseTransformer):
    """
    Normalize samples individually to each row's unit norm.

    Each sample (i.e. each row of the data matrix) with at least one
    non-zero component is rescaled independently of other samples so
    that its norm (l1, l2 or inf) equals one.

    Args:
        norm: str, default="l2"
            The norm to use to normalize each non-zero sample. If norm='max'
            is used, values will be rescaled by the maximum of the absolute
            values. It must be one of 'l1', 'l2', or 'max'.

        input_cols: Optional[Union[str, List[str]]]
            The name(s) of one or more columns in the input DataFrame containing feature(s) to be normalized. Input
            columns must be specified before transform with this argument or after initialization with the
            `set_input_cols` method. This argument is optional for API consistency.

        output_cols: Optional[Union[str, List[str]]]
            The name(s) to assign output columns in the output DataFrame. The number of
            columns specified must equal the number of input columns. Output columns must be specified before transform
            with this argument or after initialization with the `set_output_cols` method. This argument is optional for
            API consistency.

        passthrough_cols: Optional[Union[str, List[str]]]
            A string or a list of strings indicating column names to be excluded from any
            operations (such as train, transform, or inference). These specified column(s)
            will remain untouched throughout the process. This option is helpful in scenarios
            requiring automatic input_cols inference, but need to avoid using specific
            columns, like index columns, during training or inference.

        drop_input_cols: bool, default=False
            Remove input columns from output if set `True`.
    """

    def __init__(
        self,
        *,
        norm: str = "l2",
        input_cols: Optional[Union[str, Iterable[str]]] = None,
        output_cols: Optional[Union[str, Iterable[str]]] = None,
        passthrough_cols: Optional[Union[str, Iterable[str]]] = None,
        drop_input_cols: Optional[bool] = False,
    ) -> None:
        super().__init__(drop_input_cols=drop_input_cols)
        self.norm = norm
        self._is_fitted = False
        self.set_input_cols(input_cols)
        self.set_output_cols(output_cols)
        self.set_passthrough_cols(passthrough_cols)

    def _reset(self) -> None:
        """
        Reset internal data-dependent state of the normalizer.
        __init__ parameters are not touched.

        This is a stateless transformer, so there is nothing to reset.
        """

    def _fit(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> "Normalizer":
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
        project=base.PROJECT,
        subproject=base.SUBPROJECT,
    )
    def transform(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> Union[snowpark.DataFrame, pd.DataFrame]:
        """
        Scale each non-zero row of the input dataset to the unit norm.

        Args:
            dataset: Input dataset.

        Returns:
            transformed_dataset: Output dataset.

        Raises:
            SnowflakeMLException: If the dataset contains nulls, or if the supplied norm is invalid.
        """
        if self.norm not in _VALID_NORMS:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=ValueError(f"'{self.norm}' is not a supported norm."),
            )

        super()._check_input_cols()
        super()._check_output_cols()
        super()._check_dataset_type(dataset)

        if isinstance(dataset, snowpark.DataFrame):
            output_df = self._transform_snowpark(dataset)
        else:
            output_df = self._transform_sklearn(dataset)

        return self._drop_input_columns(output_df) if self._drop_input_cols is True else output_df

    def _transform_snowpark(self, dataset: snowpark.DataFrame) -> snowpark.DataFrame:
        passthrough_columns = [c for c in dataset.columns if c not in self.output_cols]
        self._validate_data_has_no_nulls(dataset)
        if len(self.input_cols) == 0:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=ValueError("Found array with 0 columns, but a minimum of 1 is required."),
            )

        if self.norm == "l1":
            norm = F.lit("0")
            for input_col in self.input_cols:
                norm += F.abs(dataset[input_col])

        elif self.norm == "l2":
            norm = F.lit("0")
            for input_col in self.input_cols:
                norm += dataset[input_col] * dataset[input_col]
            norm = F.sqrt(norm)

        elif self.norm == "max":
            norm = F.greatest(*[F.abs(dataset[input_col]) for input_col in self.input_cols])

        else:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=ValueError(f"'{self.norm}' is not a supported norm."),
            )

        output_columns = []
        for input_col in self.input_cols:
            # Set the entry to 0 if the norm is 0, because the norm is 0 only when all entries are 0.
            output_column = F.div0(
                dataset[input_col].cast(T.FloatType()),
                norm,
            )
            output_columns.append(output_column)

        transformed_dataset: snowpark.DataFrame = dataset.with_columns(self.output_cols, output_columns)
        # Reorder columns. Passthrough columns are added at the right to the output of the transformers.
        transformed_dataset = transformed_dataset[self.output_cols + passthrough_columns]
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
