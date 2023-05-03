#!/usr/bin/env python3
#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
from __future__ import annotations

from itertools import chain
from typing import Iterable, List, Optional, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer as SK_KBinsDiscretizer

import snowflake.snowpark.functions as F
from snowflake import snowpark
from snowflake.ml.framework import base

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


# TODO(tbao): add doc string
# TODO(tbao): add telemetry
class KBinsDiscretizer(base.BaseEstimator, base.BaseTransformer):
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
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy

        base.BaseEstimator.__init__(self)
        base.BaseTransformer.__init__(self, drop_input_cols=drop_input_cols)

        self.set_input_cols(input_cols)
        self.set_output_cols(output_cols)

    def _enforce_params(self) -> None:
        self.n_bins = self.n_bins if isinstance(self.n_bins, Iterable) else [self.n_bins] * len(self.input_cols)
        if len(self.n_bins) != len(self.input_cols):
            raise ValueError(f"n_bins must have same size as input_cols, got: {self.n_bins} vs {self.input_cols}")
        for idx, b in enumerate(self.n_bins):
            if b < 2:
                raise ValueError(f"n_bins cannot be less than 2, got: {b} at index {idx}")
        if self.encode not in _VALID_ENCODING_SCHEME:
            raise ValueError(f"encode must be one of f{_VALID_ENCODING_SCHEME}, got: {self.encode}")
        if self.strategy not in _VALID_STRATEGY:
            raise ValueError(f"strategy must be one of f{_VALID_STRATEGY}, got: {self.strategy}")

    def _reset(self) -> None:
        super()._reset()
        self.bin_edges_: Optional[npt.NDArray[np.float32]] = None
        self.n_bins_: Optional[npt.NDArray[np.int32]] = None

    def fit(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> KBinsDiscretizer:
        self._reset()
        self._enforce_params()
        super()._check_input_cols()

        if isinstance(dataset, pd.DataFrame):
            self._fit_sklearn(dataset)
        elif isinstance(dataset, snowpark.DataFrame):
            self._fit_snowpark(dataset)
        else:
            raise TypeError(
                f"Unexpected dataset type: {type(dataset)}."
                "Supported dataset types: snowpark.DataFrame, pandas.DataFrame."
            )

        self._is_fitted = True
        return self

    def transform(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> Union[snowpark.DataFrame, pd.DataFrame]:
        self.enforce_fit()
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

    def _fit_snowpark(self, dataset: snowpark.DataFrame) -> None:
        if self.strategy == "quantile":
            self._handle_quantile(dataset)
        elif self.strategy == "uniform":
            self._handle_uniform(dataset)
        elif self.strategy == "kmeans":
            raise NotImplementedError("kmeans not supported yet")

    def _handle_quantile(self, dataset: snowpark.DataFrame) -> None:
        # 1. Collect percentiles for each feature column
        # NB: if SQL compilation ever take too long on wide schema, consider applying optimization mentioned in
        # https://docs.google.com/document/d/1cilfCCtKYv6HvHqaqdZxfHAvQ0gg-t1AM8KYCQtJiLE/edit
        agg_queries = []
        for idx, col_name in enumerate(self.input_cols):
            percentiles = np.linspace(0, 1, cast(List[int], self.n_bins)[idx] + 1)
            for i, pct in enumerate(percentiles.tolist()):
                agg_queries.append(
                    F.percentile_cont(pct).within_group(col_name).alias(f"{col_name}_pct_{i}")  # type: ignore[arg-type]
                )
        state_df = dataset.agg(agg_queries)
        state = state_df.to_pandas().to_numpy().ravel()

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
        # 1. Collect min and max for each feature column
        agg_queries = list(
            chain.from_iterable(
                (F.min(x).alias(f"{x}_min"), F.max(x).alias(f"{x}_max"))  # type: ignore[arg-type]
                for x in self.input_cols
            )
        )
        state_df = dataset.select(*agg_queries)
        state = state_df.to_pandas().to_numpy().ravel()

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
        # 1. Register bucketize UDF
        # TODO(tbao): look into vectorized UDF
        @F.udf(  # type: ignore[arg-type, misc]
            name="bucketize", replace=True, packages=["numpy"], session=dataset._session
        )
        def bucketize(x: float, boarders: List[float]) -> int:
            import numpy as np

            return int(np.searchsorted(boarders[1:-1], x, side="right"))

        # 2. compute bucket per feature column
        if self.encode == "ordinal":
            for idx, input_col in enumerate(self.input_cols):
                output_col = self.output_cols[idx]
                dataset = dataset.select(
                    *dataset.columns,
                    F.call_udf(
                        "bucketize", F.col(input_col), self.bin_edges_[idx].tolist()  # type: ignore[arg-type, index]
                    ).alias(output_col),
                )
            return dataset
        else:
            raise NotImplementedError(f"{self.encode} is not supported yet.")
