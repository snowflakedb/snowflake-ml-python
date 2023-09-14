#!/usr/bin/env python3
from __future__ import annotations

from itertools import chain
from typing import Dict, Iterable, List, Optional, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import KBinsDiscretizer as SK_KBinsDiscretizer

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml.modeling.framework import base
from snowflake.snowpark import functions as F, types as T
from snowflake.snowpark._internal.utils import (
    TempObjectType,
    random_name_for_temp_object,
)

# constants used to validate the compatibility of the kwargs passed to the sklearn
# transformer with the sklearn version
_SKLEARN_INITIAL_KEYWORDS = [
    "n_bins",
    "encode",
    "strategy",
]  # initial keywords in sklearn

# TODO(tbao): decide if we want to support subsample
_SKLEARN_UNUSED_KEYWORDS = [
    "dtype",
    "subsample",
    "random_state",
]  # sklearn keywords that are unused in snowml

_SNOWML_ONLY_KEYWORDS = [
    "input_cols",
    "output_cols",
]  # snowml only keywords not present in sklearn

_VALID_ENCODING_SCHEME = ["onehot", "onehot-dense", "ordinal"]
_VALID_STRATEGY = ["uniform", "quantile", "kmeans"]


# NB: numpy doesn't work with decimal type, so converting to float
def decimal_to_float(data: npt.NDArray[np.generic]) -> npt.NDArray[np.float32]:
    return np.array([float(x) for x in data])


# TODO(tbao): support kmeans with snowpark if needed
class KBinsDiscretizer(base.BaseTransformer):
    """
    Bin continuous data into intervals.

    Args:
        n_bins: int or array-like of shape (n_features,), default=5
            The number of bins to produce. Raises ValueError if n_bins < 2.

        encode: {'onehot', 'onehot-dense', 'ordinal'}, default='onehot'
            Method used to encode the transformed result.

            - 'onehot': Encode the transformed result with one-hot encoding and return a sparse representation.
            - 'onehot-dense': Encode the transformed result with one-hot encoding and return separate column for
                each encoded value.
            - 'ordinal': Return the bin identifier encoded as an integer value.

        strategy: {'uniform', 'quantile'}, default='quantile'
            Strategy used to define the widths of the bins.

            - 'uniform': All bins in each feature have identical widths.
            - 'quantile': All bins in each feature have the same number of points.

        input_cols: str or Iterable [column_name], default=None
            Single or multiple input columns.

        output_cols: str or Iterable [column_name], default=None
            Single or multiple output columns.

        drop_input_cols: boolean, default=False
            Remove input columns from output if set True.

    Attributes:
        bin_edges_: ndarray of ndarray of shape (n_features,)
            The edges of each bin. Contain arrays of varying shapes (n_bins_, )

        n_bins_: ndarray of shape (n_features,), dtype=np.int_
            Number of bins per feature.
    """

    def __init__(
        self,
        *,
        n_bins: Union[int, List[int]] = 5,
        encode: str = "onehot",
        strategy: str = "quantile",
        input_cols: Optional[Union[str, Iterable[str]]] = None,
        output_cols: Optional[Union[str, Iterable[str]]] = None,
        drop_input_cols: Optional[bool] = False,
    ) -> None:
        super().__init__(drop_input_cols=drop_input_cols)
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.set_input_cols(input_cols)
        self.set_output_cols(output_cols)

    def _enforce_params(self) -> None:
        self.n_bins = self.n_bins if isinstance(self.n_bins, Iterable) else [self.n_bins] * len(self.input_cols)
        if len(self.n_bins) != len(self.input_cols):
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=ValueError(
                    f"n_bins must have same size as input_cols, got: {self.n_bins} vs {self.input_cols}"
                ),
            )
        for idx, b in enumerate(self.n_bins):
            if b < 2:
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ATTRIBUTE,
                    original_exception=ValueError(f"n_bins cannot be less than 2, got: {b} at index {idx}"),
                )
        if self.encode not in _VALID_ENCODING_SCHEME:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=ValueError(f"encode must be one of f{_VALID_ENCODING_SCHEME}, got: {self.encode}"),
            )
        if self.strategy not in _VALID_STRATEGY:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=ValueError(f"strategy must be one of f{_VALID_STRATEGY}, got: {self.strategy}"),
            )

    def _reset(self) -> None:
        super()._reset()
        self.bin_edges_: Optional[npt.NDArray[np.float32]] = None
        self.n_bins_: Optional[npt.NDArray[np.int32]] = None

    @telemetry.send_api_usage_telemetry(
        project=base.PROJECT,
        subproject=base.SUBPROJECT,
    )
    def fit(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> KBinsDiscretizer:
        """
        Fit KBinsDiscretizer with dataset.

        Args:
            dataset: Input dataset.

        Returns:
            Fitted self instance.
        """
        self._reset()
        self._enforce_params()
        super()._check_input_cols()
        super()._check_dataset_type(dataset)

        if isinstance(dataset, pd.DataFrame):
            self._fit_sklearn(dataset)
        else:
            self._fit_snowpark(dataset)

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
    def transform(
        self, dataset: Union[snowpark.DataFrame, pd.DataFrame]
    ) -> Union[snowpark.DataFrame, pd.DataFrame, sparse.csr_matrix]:
        """
        Discretize the data.

        Args:
            dataset: Input dataset.

        Returns:
            Discretized output data based on input type.
            - If input is snowpark DataFrame, returns snowpark DataFrame
            - If input is a pd.DataFrame and 'self.encdoe=onehot', returns 'csr_matrix'
            - If input is a pd.DataFrame and 'self.encode in ['ordinal', 'onehot-dense']', returns 'pd.DataFrame'
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

    def _fit_snowpark(self, dataset: snowpark.DataFrame) -> None:
        if self.strategy == "quantile":
            self._handle_quantile(dataset)
        elif self.strategy == "uniform":
            self._handle_uniform(dataset)
        elif self.strategy == "kmeans":
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_IMPLEMENTED,
                original_exception=NotImplementedError("kmeans not supported yet"),
            )

    def _handle_quantile(self, dataset: snowpark.DataFrame) -> None:
        """
        Compute bins with percentile values of the feature.
        All bins in each feature will have the same number of points.

        Args:
            dataset: Input dataset.
        """
        # 1. Collect percentiles for each feature column
        # NB: if SQL compilation ever take too long on wide schema, consider applying optimization mentioned in
        # https://docs.google.com/document/d/1cilfCCtKYv6HvHqaqdZxfHAvQ0gg-t1AM8KYCQtJiLE/edit
        agg_queries = []
        for idx, col_name in enumerate(self.input_cols):
            percentiles = np.linspace(0, 1, cast(List[int], self.n_bins)[idx] + 1)
            for i, pct in enumerate(percentiles.tolist()):
                agg_queries.append(F.percentile_cont(pct).within_group(col_name).alias(f"{col_name}_pct_{i}"))
        state_df = dataset.agg(agg_queries)
        state = (
            state_df.to_pandas(
                statement_params=telemetry.get_statement_params(base.PROJECT, base.SUBPROJECT, self.__class__.__name__)
            )
            .to_numpy()
            .ravel()
        )

        # 2. Populate internal state variables
        # TODO(tbao): Remove bins whose width are too small (i.e., <= 1e-8)
        self.bin_edges_ = np.zeros(len(self.input_cols), dtype=object)
        self.n_bins_ = np.zeros(len(self.input_cols), dtype=np.int_)
        start = 0
        for i, b in enumerate(cast(List[int], self.n_bins)):
            self.bin_edges_[i] = decimal_to_float(state[start : start + b + 1])
            start += b + 1
            self.n_bins_[i] = len(self.bin_edges_[i]) - 1

    def _handle_uniform(self, dataset: snowpark.DataFrame) -> None:
        """
        Compute bins with min and max value of the feature.
        All bins in each feature will have identical widths.

        Args:
            dataset: Input dataset.
        """
        # 1. Collect min and max for each feature column
        agg_queries = list(
            chain.from_iterable((F.min(x).alias(f"{x}_min"), F.max(x).alias(f"{x}_max")) for x in self.input_cols)
        )
        state_df = dataset.select(*agg_queries)
        state = (
            state_df.to_pandas(
                statement_params=telemetry.get_statement_params(base.PROJECT, base.SUBPROJECT, self.__class__.__name__)
            )
            .to_numpy()
            .ravel()
        )

        # 2. Populate internal state variables
        self.bin_edges_ = np.zeros(len(self.input_cols), dtype=object)
        self.n_bins_ = np.zeros(len(self.input_cols), dtype=np.int_)
        for i, b in enumerate(cast(List[int], self.n_bins)):
            self.bin_edges_[i] = np.linspace(state[i * 2], state[i * 2 + 1], b + 1)
            self.n_bins_[i] = len(self.bin_edges_[i]) - 1

    def _fit_sklearn(self, dataset: pd.DataFrame) -> None:
        dataset = self._use_input_cols_only(dataset)
        sklearn_discretizer = self._create_unfitted_sklearn_object()
        sklearn_discretizer.fit(dataset[self.input_cols])

        self.bin_edges_ = sklearn_discretizer.bin_edges_
        self.n_bins_ = sklearn_discretizer.n_bins_
        if "onehot" in self.encode:
            self._encoder = sklearn_discretizer._encoder

    def _create_unfitted_sklearn_object(self) -> SK_KBinsDiscretizer:
        sklearn_args = self.get_sklearn_args(
            default_sklearn_obj=SK_KBinsDiscretizer(),
            sklearn_initial_keywords=_SKLEARN_INITIAL_KEYWORDS,
            sklearn_unused_keywords=_SKLEARN_UNUSED_KEYWORDS,
            snowml_only_keywords=_SNOWML_ONLY_KEYWORDS,
        )
        return SK_KBinsDiscretizer(**sklearn_args)

    def _create_sklearn_object(self) -> SK_KBinsDiscretizer:
        sklearn_discretizer = self._create_unfitted_sklearn_object()
        if self._is_fitted:
            sklearn_discretizer.bin_edges_ = self.bin_edges_
            sklearn_discretizer.n_bins_ = self.n_bins_
            if "onehot" in self.encode:
                sklearn_discretizer._encoder = self._encoder
        return sklearn_discretizer

    def _transform_snowpark(self, dataset: snowpark.DataFrame) -> snowpark.DataFrame:
        if self.encode == "ordinal":
            return self._handle_ordinal(dataset)
        elif self.encode == "onehot":
            return self._handle_onehot(dataset)
        elif self.encode == "onehot-dense":
            return self._handle_onehot_dense(dataset)
        else:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=ValueError(f"{self.encode} is not a valid encoding scheme."),
            )

    def _handle_ordinal(self, dataset: snowpark.DataFrame) -> snowpark.DataFrame:
        """
        Transform dataset with bucketization and output as ordinal encoding.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset with ordinal encoding.
        """
        passthrough_columns = [c for c in dataset.columns if c not in self.output_cols]
        # NB: the reason we need to generate a random UDF name each time is because the UDF registration
        # is centralized per database, so if there are multiple sessions with same UDF name, there might be
        # a conflict and some parties could fail to fetch the UDF.
        udf_name = random_name_for_temp_object(TempObjectType.FUNCTION)

        # 1. Register vec_bucketize UDF
        @F.pandas_udf(  # type: ignore[arg-type, misc]
            is_permanent=False,
            name=udf_name,
            replace=True,
            packages=["numpy"],
            session=dataset._session,
            statement_params=telemetry.get_statement_params(base.PROJECT, base.SUBPROJECT, self.__class__.__name__),
        )
        def vec_bucketize_temp(x: T.PandasSeries[float], boarders: T.PandasSeries[List[float]]) -> T.PandasSeries[int]:
            # NB: vectorized udf doesn't work well with const array arg, so we pass it in as a list via PandasSeries
            boarders = boarders[0]
            res = np.searchsorted(boarders[1:-1], x, side="right")
            return pd.Series(res)  # type: ignore[no-any-return]

        # 2. compute bucket per feature column
        for idx, input_col in enumerate(self.input_cols):
            output_col = self.output_cols[idx]
            assert self.bin_edges_ is not None
            boarders = [F.lit(float(x)) for x in self.bin_edges_[idx]]
            dataset = dataset.select(
                *dataset.columns,
                F.call_udf(udf_name, F.col(input_col), F.array_construct(*boarders)).alias(output_col),
            )
        # Reorder columns. Passthrough columns are added at the right to the output of the transformers.
        dataset = dataset[self.output_cols + passthrough_columns]
        return dataset

    def _handle_onehot(self, dataset: snowpark.DataFrame) -> snowpark.DataFrame:
        """
        Transform dataset with bucketization and output as sparse representation:
        {"{bucket_id}": 1.0, "array_length": {num_buckets}}

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset in sparse representation.
        """
        passthrough_columns = [c for c in dataset.columns if c not in self.output_cols]
        udf_name = random_name_for_temp_object(TempObjectType.FUNCTION)

        @F.pandas_udf(  # type: ignore[arg-type, misc]
            is_permanent=False,
            name=udf_name,
            replace=True,
            packages=["numpy"],
            session=dataset._session,
            statement_params=telemetry.get_statement_params(base.PROJECT, base.SUBPROJECT, self.__class__.__name__),
        )
        def vec_bucketize_sparse_output_temp(
            x: T.PandasSeries[float], boarders: T.PandasSeries[List[float]]
        ) -> T.PandasSeries[Dict[str, int]]:
            res: List[Dict[str, int]] = []
            boarders = boarders[0]
            buckets = np.searchsorted(boarders[1:-1], x, side="right")
            assert isinstance(buckets, np.ndarray), f"expecting buckets to be numpy ndarray, got {type(buckets)}"
            array_length = len(boarders) - 1
            for bucket in buckets:
                res.append({str(bucket): 1, "array_length": array_length})
            return pd.Series(res)

        for idx, input_col in enumerate(self.input_cols):
            assert self.bin_edges_ is not None
            output_col = self.output_cols[idx]
            boarders = [F.lit(float(x)) for x in self.bin_edges_[idx]]
            dataset = dataset.select(
                *dataset.columns,
                F.call_udf(udf_name, F.col(input_col), F.array_construct(*boarders)).alias(output_col),
            )
        # Reorder columns. Passthrough columns are added at the right to the output of the transformers.
        dataset = dataset[self.output_cols + passthrough_columns]
        return dataset

    def _handle_onehot_dense(self, dataset: snowpark.DataFrame) -> snowpark.DataFrame:
        """
        Transform dataset with bucketization and output as onehot dense representation:
        Each category will be reprensented in its own output column.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset in dense representation.
        """
        original_dataset_columns = dataset.columns[:]
        all_output_cols = []

        udf_name = random_name_for_temp_object(TempObjectType.FUNCTION)

        @F.pandas_udf(  # type: ignore[arg-type, misc]
            name=udf_name,
            replace=True,
            packages=["numpy"],
            session=dataset._session,
            statement_params=telemetry.get_statement_params(base.PROJECT, base.SUBPROJECT, self.__class__.__name__),
        )
        def vec_bucketize_dense_output_temp(
            x: T.PandasSeries[float], boarders: T.PandasSeries[List[float]]
        ) -> T.PandasSeries[List[int]]:
            res: List[npt.NDArray[np.int32]] = []
            boarders = boarders[0]
            buckets = np.searchsorted(boarders[1:-1], x, side="right")
            assert isinstance(buckets, np.ndarray), f"expecting buckets to be numpy ndarray, got {type(buckets)}"
            for bucket in buckets:
                encoded = np.zeros(len(boarders), dtype=int)
                encoded[bucket] = 1
                res.append(encoded)
            return pd.Series(res)

        for idx, input_col in enumerate(self.input_cols):
            assert self.bin_edges_ is not None
            output_col = self.output_cols[idx]
            boarders = [F.lit(float(x)) for x in self.bin_edges_[idx]]
            dataset = dataset.select(
                *dataset.columns,
                F.call_udf(udf_name, F.col(input_col), F.array_construct(*boarders)).alias(output_col),
            )
            dataset = dataset.with_columns(
                [f"{output_col}_{i}" for i in range(len(boarders) - 1)],
                [F.col(output_col)[i].cast(T.IntegerType()) for i in range(len(boarders) - 1)],
            ).drop(output_col)
            all_output_cols += [f"{output_col}_{i}" for i in range(len(boarders) - 1)]

        # Reorder columns. Passthrough columns are added at the right to the output of the transformers.
        dataset = dataset[all_output_cols + original_dataset_columns]
        return dataset

    def _transform_sklearn(self, dataset: pd.DataFrame) -> Union[pd.DataFrame, sparse.csr_matrix]:
        """
        Transform pandas dataframe using sklearn KBinsDiscretizer.
        Output be a sparse matrix if self.encode='onehot' and pandas dataframe otherwise.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.
        """
        self._enforce_fit()
        encoder_sklearn = self.to_sklearn()

        transformed_dataset = encoder_sklearn.transform(dataset[self.input_cols])

        if self.encode == "ordinal":
            dataset = dataset.copy()
            dataset[self.output_cols] = transformed_dataset
            return dataset
        elif self.encode == "onehot-dense":
            dataset = dataset.copy()
            dataset[self.get_output_cols()] = transformed_dataset
            return dataset
        else:
            return transformed_dataset

    def get_output_cols(self) -> List[str]:
        """
        Get output column names.
        Expand output column names for 'onehot-dense' encoding.

        Returns:
            Output column names.
        """

        if self.encode == "onehot-dense":
            output_cols = []
            for idx, col in enumerate(self.output_cols):
                output_cols.extend([f"{col}_{i}" for i in range(self.n_bins_[idx])])  # type: ignore[index]
            return output_cols
        else:
            return self.output_cols
