#!/usr/bin/env python3
#
# Copyright (c) 2012-2023 Snowflake Computing Inc. All rights reserved.
#
from typing import Iterable, Optional, Union

import pandas as pd
from sklearn import preprocessing

from snowflake import snowpark
from snowflake.ml._internal import telemetry, type_utils
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml.modeling.framework import base
from snowflake.ml.modeling.preprocessing import ordinal_encoder


class LabelEncoder(base.BaseTransformer):
    r"""Encodes target labels with values between 0 and n_classes-1.

    In other words, each class (i.e., distinct numeric or string) is assigned an integer value, starting with zero.
    LabelEncoder is a specialization of OrdinalEncoder for 1-dimensional data.

    For more details on what this transformer does, see [sklearn.preprocessing.LabelEncoder]
    (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html).

    Args:
        input_cols: Optional[Union[str, List[str]]]
            The name of a column or a list containing one column name to be encoded in the input DataFrame. There must
            be exactly one input column specified before fit. This argument is optional for API consistency.

        output_cols: Optional[Union[str, List[str]]]
            The name of a column or a list containing one column name where the results will be stored. There must be
            exactly one output column specified before trainsform. This argument is optional for API consistency.

        passthrough_cols: Optional[Union[str, List[str]]]
            A string or a list of strings indicating column names to be excluded from any
            operations (such as train, transform, or inference). These specified column(s)
            will remain untouched throughout the process. This option is helpful in scenarios
            requiring automatic input_cols inference, but need to avoid using specific
            columns, like index columns, during training or inference.

        drop_input_cols: Optional[bool], default=False
            Remove input columns from output if set True. False by default.
    """

    def __init__(
        self,
        input_cols: Optional[Union[str, Iterable[str]]] = None,
        output_cols: Optional[Union[str, Iterable[str]]] = None,
        passthrough_cols: Optional[Union[str, Iterable[str]]] = None,
        drop_input_cols: Optional[bool] = False,
    ) -> None:
        """
        Encode target labels with integers between 0 and n_classes-1.

        Args:
            input_cols: Optional[Union[str, List[str]]]
                The name of a column or a list containing one column name to be encoded in the input DataFrame. There
                must be exactly one input column specified before fit. This argument is optional for API consistency.
            output_cols: Optional[Union[str, List[str]]]
                The name of a column or a list containing one column name where the results will be stored. There must
                be exactly one output column specified before transform. This argument is optional for API consistency.
            passthrough_cols: Optional[Union[str, List[str]]]
                A string or a list of strings indicating column names to be excluded from any
                operations (such as train, transform, or inference). These specified column(s)
                will remain untouched throughout the process. This option is helful in scenarios
                requiring automatic input_cols inference, but need to avoid using specific
                columns, like index columns, during in training or inference.
            drop_input_cols: Optional[bool], default=False
                Remove input columns from output if set True. False by default.

        Attributes:
            classes_: Optional[type_utils.LiteralNDArrayType]
                A np.ndarray that holds the label for each class.
                Attributes are valid only after fit() has been called.

        """
        super().__init__(drop_input_cols=drop_input_cols)
        self._ordinal_encoder: Optional[ordinal_encoder.OrdinalEncoder] = None
        self.classes_: Optional[type_utils.LiteralNDArrayType] = None
        self.set_input_cols(input_cols)
        self.set_output_cols(output_cols)
        self.set_passthrough_cols(passthrough_cols)

    def _reset(self) -> None:
        super()._reset()
        if self._ordinal_encoder:
            self._ordinal_encoder = None
            self.classes_ = None

    def _fit(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> "LabelEncoder":
        """
        Fit label encoder with label column in dataset.

        Validates the transformer arguments and derives the list of classes (distinct values) from the data, making
        this list available as an attribute of the transformer instance (see Attributes).
        Returns the transformer instance.

        Args:
            dataset: Input dataset.

        Returns:
            self

        Raises:
            SnowflakeMLException: If length of input_cols is not 1 or length of output_cols is greater than 1.
        """
        if len(self.input_cols) != 1:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=ValueError("Label encoder must specify one input column."),
            )
        input_col = self.input_cols[0]

        if len(self.output_cols) != 1:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=ValueError("Label encoder must specify one output column."),
            )

        self._reset()

        # Use `OrdinalEncoder` to handle fits and transforms.
        self._ordinal_encoder = ordinal_encoder.OrdinalEncoder(input_cols=self.input_cols, output_cols=self.output_cols)

        self._ordinal_encoder.fit(dataset)

        # Set `classes_` for compatibility with sklearn.
        self.classes_ = self._ordinal_encoder.categories_[input_col]

        self._is_fitted = True
        return self

    @telemetry.send_api_usage_telemetry(
        project=base.PROJECT,
        subproject=base.SUBPROJECT,
    )
    def transform(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> Union[snowpark.DataFrame, pd.DataFrame]:
        """
        Use fit result to transform snowpark dataframe or pandas dataframe. The original dataset with
        the transform result column added will be returned.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.
        """
        self._enforce_fit()
        super()._check_dataset_type(dataset)

        if isinstance(dataset, snowpark.DataFrame):
            # [SNOW-802691] Support for mypy type checking
            assert self._ordinal_encoder is not None
            output_df = self._ordinal_encoder.transform(dataset).na.replace(
                float("nan"),
                len(self.classes_) - 1,  # type: ignore[arg-type]
                subset=self.output_cols,
            )
        else:
            output_df = self._transform_sklearn(dataset)

        return self._drop_input_columns(output_df) if self._drop_input_cols is True else output_df

    def _create_unfitted_sklearn_object(self) -> preprocessing.LabelEncoder:
        return preprocessing.LabelEncoder()

    def _create_sklearn_object(self) -> preprocessing.LabelEncoder:
        """
        Initialize and return the equivalent sklearn label encoder.

        Returns:
            Equivalent sklearn object.
        """
        label_encoder = self._create_unfitted_sklearn_object()
        label_encoder.classes_ = self.classes_
        return label_encoder
