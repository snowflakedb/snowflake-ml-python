import uuid
from typing import Any, Iterable

import pandas as pd

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml._internal.utils import identifier
from snowflake.ml.modeling.framework import _utils, base
from snowflake.snowpark import functions as F, types as T
from snowflake.snowpark._internal import utils as snowpark_utils

_COLUMN_NAME = "_COLUMN_NAME"
_CATEGORY = "_CATEGORY"
_ENCODING = "_ENCODING"

# Local (pandas) encoding is delegated to category_encoders, which is an optional dependency. The
# Snowpark path never imports it. Keep this constraint in sync with requirements.yml.
_CATEGORY_ENCODERS_REQUIREMENT = "category_encoders>=2.6.0,<3"


class FrequencyEncoder(base.BaseTransformer):
    r"""Encodes each category with the frequency (or count) of that category observed during fit.

    Frequency encoding is a compact alternative to one-hot or ordinal encoding for high-cardinality
    categorical features: every distinct value is replaced by a single numeric column holding how often
    that value occurred in the fitting data, so the output dimensionality stays at one column per input
    column regardless of the number of distinct categories. Unlike an ordinal encoding, the assigned
    numbers carry information (relative prevalence) rather than an arbitrary rank.

    On a Snowpark DataFrame the encoding is computed and applied entirely in Snowflake via SQL, so it
    scales to columns with very large numbers of distinct values without pulling data to the client. On
    a pandas DataFrame the encoding is delegated to ``category_encoders.CountEncoder``, which is an
    optional dependency installed only when encoding locally.

    Args:
        normalize: bool, default=True
            When True, each category is encoded with its relative frequency (``count / number_of_rows``),
            a value in ``[0, 1]``. When False, each category is encoded with its raw integer count.

        unknown_value: int | float, default=0
            Value used to encode categories that were not seen during fit. Defaults to 0, matching the
            intuition that an unseen category has zero observed frequency.

        input_cols: Optional[Union[str, Iterable[str]]], default=None
            The name(s) of one or more columns in the input DataFrame containing categorical feature(s) to
            be encoded. Input columns must be specified before fit with this argument or after
            initialization with the `set_input_cols` method. This argument is optional for API consistency.

        output_cols: Optional[Union[str, Iterable[str]]], default=None
            The name(s) to assign to the encoded output columns. The number of output columns specified
            must equal the number of input columns. Output columns must be specified before transform with
            this argument or after initialization with the `set_output_cols` method. This argument is
            optional for API consistency.

        passthrough_cols: Optional[Union[str, Iterable[str]]], default=None
            A string or a list of strings indicating column names to be excluded from any operations (such
            as train, transform, or inference). These specified column(s) will remain untouched throughout
            the process. This option is helpful in scenarios requiring automatic input_cols inference, but
            need to avoid using specific columns, like index columns, during training or inference.

        drop_input_cols: Optional[bool], default=False
            Remove input columns from output if set True. False by default.

    Attributes:
        frequencies_ (dict[str, dict[Any, float]]): Maps each input column name to a mapping of category
            to its encoded value (relative frequency when ``normalize`` is True, otherwise raw count). A
            null category, when present in the fitting data, is represented by a ``None`` key. Valid only
            after fit() has been called.
    """

    def __init__(
        self,
        *,
        normalize: bool = True,
        unknown_value: int | float = 0,
        input_cols: str | Iterable[str] | None = None,
        output_cols: str | Iterable[str] | None = None,
        passthrough_cols: str | Iterable[str] | None = None,
        drop_input_cols: bool | None = False,
    ) -> None:
        super().__init__(drop_input_cols=drop_input_cols)
        self.normalize = normalize
        self.unknown_value = unknown_value

        self.frequencies_: dict[str, dict[Any, float]] = {}
        self._state_pandas: pd.DataFrame | None = None
        self._state_table_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.TABLE)

        self.set_input_cols(input_cols)
        self.set_output_cols(output_cols)
        self.set_passthrough_cols(passthrough_cols)

    def _reset(self) -> None:
        super()._reset()
        self.frequencies_ = {}
        self._state_pandas = None

    def _fit(self, dataset: snowpark.DataFrame | pd.DataFrame) -> "FrequencyEncoder":
        """Fit the encoder, deriving the per-category frequencies from the data.

        Args:
            dataset: Input dataset.

        Returns:
            Fitted encoder.
        """
        self._reset()
        super()._check_input_cols()
        super()._check_dataset_type(dataset)

        if isinstance(dataset, pd.DataFrame):
            self._fit_pandas(dataset)
        else:
            self._fit_snowpark(dataset)

        self._is_fitted = True
        return self

    def _fit_snowpark(self, dataset: snowpark.DataFrame) -> None:
        statement_params = telemetry.get_statement_params(base.PROJECT, base.SUBPROJECT, self.__class__.__name__)
        n_samples = dataset.count(statement_params=statement_params)

        state_df: snowpark.DataFrame | None = None
        for input_col in self.input_cols:
            count_col = F.count(F.lit(1))
            # NUMBER / NUMBER division in Snowflake SQL truncates to a fixed scale of 6 decimal
            # digits; casting the numerator to DoubleType forces float division so the result
            # matches the full double precision produced by category_encoders.CountEncoder.
            encoding = count_col.cast(T.DoubleType()) / F.lit(n_samples) if self.normalize else count_col
            col_state = (
                dataset.group_by(input_col)
                .agg(encoding.cast(T.FloatType()).alias(_ENCODING))
                .select(
                    F.lit(input_col).alias(_COLUMN_NAME),
                    F.col(input_col).cast(T.StringType()).alias(_CATEGORY),
                    F.col(_ENCODING),
                )
            )
            state_df = col_state if state_df is None else state_df.union_by_name(col_state)

        assert state_df is not None
        state_df.write.save_as_table(
            self._state_table_name,
            mode="overwrite",
            table_type="temporary",
            statement_params=statement_params,
        )
        self._state_pandas = state_df.to_pandas(statement_params=statement_params)
        self._assign_frequencies_from_state()

    def _assign_frequencies_from_state(self) -> None:
        """Populate `frequencies_` from `_state_pandas` (used after a Snowpark fit)."""
        assert self._state_pandas is not None
        self.frequencies_ = {}
        for input_col in self.input_cols:
            col_state = self._state_pandas[self._state_pandas[_COLUMN_NAME] == input_col]
            col_freqs: dict[Any, float] = {}
            for category, encoding in zip(col_state[_CATEGORY], col_state[_ENCODING]):
                col_freqs[None if pd.isna(category) else category] = float(encoding)
            self.frequencies_[input_col] = col_freqs

    def _fit_pandas(self, dataset: pd.DataFrame) -> None:
        dataset = self._use_input_cols_only(dataset)
        encoder = self._create_unfitted_sklearn_object()
        encoder.fit(dataset[self.input_cols])
        self._sklearn_object = encoder

        rows: list[dict[str, Any]] = []
        self.frequencies_ = {}
        for input_col in self.input_cols:
            mapping = encoder.mapping[input_col]
            col_freqs: dict[Any, float] = {}
            for category, encoding in mapping.items():
                is_missing = pd.isna(category)
                col_freqs[None if is_missing else category] = float(encoding)
                rows.append(
                    {
                        _COLUMN_NAME: input_col,
                        _CATEGORY: None if is_missing else str(category),
                        _ENCODING: float(encoding),
                    }
                )
            self.frequencies_[input_col] = col_freqs
        self._state_pandas = pd.DataFrame(rows, columns=[_COLUMN_NAME, _CATEGORY, _ENCODING])

    @telemetry.send_api_usage_telemetry(
        project=base.PROJECT,
        subproject=base.SUBPROJECT,
    )
    def transform(self, dataset: snowpark.DataFrame | pd.DataFrame) -> snowpark.DataFrame | pd.DataFrame:
        """Transform dataset, replacing each category with its fitted frequency (or count).

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
            output_df = self._transform_pandas(dataset)

        return self._drop_input_columns(output_df) if self._drop_input_cols is True else output_df

    def _transform_snowpark(self, dataset: snowpark.DataFrame) -> snowpark.DataFrame:
        passthrough_columns = [c for c in dataset.columns if c not in self.output_cols]
        assert dataset._session is not None
        statement_params = telemetry.get_statement_params(base.PROJECT, base.SUBPROJECT, self.__class__.__name__)
        state_df = (
            dataset._session.table(self._state_table_name)
            if _utils.table_exists(dataset._session, self._state_table_name, statement_params)
            else dataset._session.create_dataframe(self._state_pandas)
        )

        suffix = "_" + uuid.uuid4().hex.upper()
        transformed_dataset = dataset
        for input_col, output_col in zip(self.input_cols, self.output_cols):
            category_col = identifier.concat_names([_CATEGORY, suffix])
            encoding_col = identifier.concat_names([_ENCODING, suffix])
            input_col_state_df = state_df.filter(F.col(_COLUMN_NAME) == input_col).select(
                F.col(_CATEGORY).alias(category_col),
                F.col(_ENCODING).alias(encoding_col),
            )
            # A left join matches each row's category (compared as a string, with NULL matching the null
            # category via equal_null) to its fitted encoding; unmatched rows are unknown categories.
            joined = transformed_dataset.join(
                input_col_state_df,
                on=transformed_dataset[input_col].cast(T.StringType()).equal_null(input_col_state_df[category_col]),
                how="left",
            )
            encoded = F.coalesce(joined[encoding_col], F.lit(self.unknown_value)).cast(T.FloatType())
            transformed_dataset = joined.with_column(output_col, encoded).drop(category_col, encoding_col)

        transformed_dataset = transformed_dataset[self.output_cols + passthrough_columns]
        return transformed_dataset  # type: ignore[no-any-return]

    def _transform_pandas(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if self._sklearn_object is not None:
            return self._transform_sklearn(dataset)

        # Fitted on a Snowpark DataFrame: apply the derived frequencies directly, which avoids requiring
        # category_encoders locally just to reuse state that was already computed in Snowflake.
        dataset = dataset.copy()
        for input_col, output_col in zip(self.input_cols, self.output_cols):
            col_freqs = self.frequencies_[input_col]
            missing_encoding = col_freqs.get(None)
            known = {category: encoding for category, encoding in col_freqs.items() if category is not None}
            mapped = dataset[input_col].map(known)
            if missing_encoding is not None:
                mapped = mapped.mask(dataset[input_col].isna(), missing_encoding)
            dataset[output_col] = mapped.fillna(self.unknown_value).astype(float)
        return dataset

    def _create_unfitted_sklearn_object(self) -> Any:
        category_encoders = _import_category_encoders(self.__class__.__name__)
        return category_encoders.CountEncoder(
            cols=self.input_cols,
            normalize=self.normalize,
            handle_unknown=self.unknown_value,
            handle_missing="value",
            return_df=True,
        )

    def _create_sklearn_object(self) -> Any:
        """Return the equivalent fitted ``category_encoders.CountEncoder``.

        Returns:
            The fitted CountEncoder produced during a pandas fit.

        Raises:
            SnowflakeMLException: If the encoder was fitted on a Snowpark DataFrame, in which case no
                CountEncoder object exists.
        """
        if self._sklearn_object is not None:
            return self._sklearn_object
        raise exceptions.SnowflakeMLException(
            error_code=error_codes.METHOD_NOT_ALLOWED,
            original_exception=RuntimeError(
                "to_sklearn() returns a category_encoders.CountEncoder and is available after fitting "
                f"{self.__class__.__name__} on a pandas DataFrame. This encoder was fitted on a Snowpark "
                "DataFrame; call transform() instead, which accepts both Snowpark and pandas inputs."
            ),
        )


def _import_category_encoders(class_name: str) -> Any:
    try:
        import category_encoders

        return category_encoders
    except ImportError as e:
        raise exceptions.SnowflakeMLException(
            error_code=error_codes.CLIENT_DEPENDENCY_MISSING_ERROR,
            original_exception=ImportError(
                f"category_encoders is required to fit or transform {class_name} on a pandas DataFrame "
                "locally. Snowpark DataFrames are encoded entirely in Snowflake and do not require it. "
                f"To encode locally, install it with: pip install '{_CATEGORY_ENCODERS_REQUIREMENT}' "
                f"(or conda install -c https://repo.anaconda.com/pkgs/snowflake '{_CATEGORY_ENCODERS_REQUIREMENT}')."
            ),
        ) from e
