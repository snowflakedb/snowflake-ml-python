# mypy: disable-error-code="import"
"""Native handler for tabular foundation models served unbound (in-context-learning).

A tabular foundation model (TabICL, ...) does not learn task weights during
``fit`` -- ``fit`` only prepares in-context-learning (ICL) state from a labeled context.
This handler serves such models in their natural *unbound* mode: only the foundation
weights and constructor params are packaged, and the context (labeled rows) is supplied
at inference time inside the input relation and bound per call. This is the "log once,
use for many contexts" path, as opposed to baking a single fitted context into each
model version.

The machinery here (signature, partition split, fit-on-call caching, packaging layout)
is model-agnostic. The per-model specifics -- which estimator classes to claim, how to
persist/restore the foundation weights, and the runtime dependencies -- live in a small
``ModelAdapter`` (one per foundation model, in the ``tabular_foundation_model_adapters``
package) selected by estimator type. Adding a new foundation model is a new adapter, not a
new handler.

Packaging (pickle-free) under ``model/``:
  - ``foundation/``     -- the foundation weights (adapter-specific layout)
  - ``init_params.json``-- constructor params (minus the foundation reference) plus
                           estimator type, model type, device, and default target column

Heterogeneous, variable-width input:
  - Features are packed into a single ``FEATURES`` object column (SQL ``OBJECT`` /
    ``MapType(String, Variant)``) holding ``{feature_name: value}`` per row, and the
    label rides in a scalar ``TARGET`` column. Because ``OBJECT`` values are ``VARIANT``,
    a row can mix numeric, string (categorical), and boolean features, and different
    use-cases can carry different feature *names* and *counts*. The logged signature is
    therefore fixed and *generic* (``FEATURES: OBJECT`` + ``TARGET``); the polymorphism
    lives inside the object cell rather than in named columns. Each partition must be
    internally consistent (its rows share the same feature keys/types).
  - The ``TARGET`` and ``PREDICTION`` types default by estimator kind (regressor ->
    ``DOUBLE``; classifier -> ``STRING``, since labels may be non-numeric). A sample, when
    provided, keeps the generic input but refines those scalar types to what it implies;
    ``prediction_type`` or a fully user-supplied ``signatures`` overrides them.

Serving:
  - ``predict`` is a partitioned table function. Each partition arrives complete
    (context + query rows). The handler splits the partition into context (non-null
    target, or an explicit ``context_filter``) and query rows (null target, or
    ``prediction_filter``), reconstructs a typed DataFrame from the ``FEATURES`` objects,
    fits the estimator on the context (cached by a context fingerprint), and scores the
    query rows, returning only those rows (M:N).
  - A *pre-fitted* estimator is instead served *bound* (auto-detected from its fitted state, or
    forced with the ``bound`` option) with the concrete training frame passed as ``sample_input_data``:
    the signature is static over named feature columns, ``predict`` is a plain scalar function (one
    prediction per row, no table function), and the training context is stored beside the foundation
    weights and re-fit onto the estimator at load.
"""

import ast
import hashlib
import json
import logging
import os
from typing import Any, Callable, Iterable, cast, final

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import TypeGuard, Unpack

from snowflake.ml.model import custom_model, model_signature, type_hints as model_types
from snowflake.ml.model._packager.model_handlers import _base, _utils as handlers_utils
from snowflake.ml.model._packager.model_handlers.tabular_foundation_model_adapters import (
    base as adapter_base,
    tabicl,
)
from snowflake.ml.model._packager.model_handlers_migrator import base_migrator
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta as model_meta_api,
    model_meta_schema,
)
from snowflake.ml.model._signatures import utils as model_signature_utils

# Estimator instances are duck-typed (sklearn interface); the concrete classes are
# resolved lazily inside adapters so no estimator library (tabicl) is
# imported at module load -- only the lightweight adapter modules are.
_TabularModel = Any

logger = logging.getLogger(__name__)

# Registration order is the resolution order for ``_find_adapter``. Each foundation model
# contributes one adapter module; adding a model is a new adapter here, not a new handler.
_ADAPTERS: tuple[type[adapter_base.ModelAdapter], ...] = (tabicl.TabICLAdapter,)


def _find_adapter(model: Any) -> type[adapter_base.ModelAdapter] | None:
    """Return the first adapter whose claimed estimator types match ``model``, or None."""
    for adapter in _ADAPTERS:
        try:
            if adapter.matches(model):
                return adapter
        except Exception:
            logger.debug("adapter %s failed to match model %s", adapter.ADAPTER_ID, type(model), exc_info=True)
    return None


def _get_adapter(adapter_id: str) -> type[adapter_base.ModelAdapter]:
    """Return the adapter registered under ``adapter_id`` (persisted in the model blob options)."""
    for adapter in _ADAPTERS:
        if adapter.ADAPTER_ID == adapter_id:
            return adapter
    raise ValueError(f"Unknown tabular foundation model adapter: {adapter_id!r}")


_INIT_PARAMS_FILE = "init_params.json"
_FOUNDATION_DIR = "foundation"

# Bound mode only: the pre-fitted training context (features + target), re-fit at load time. Stored as
# pandas JSON Table Schema (dtype-faithful, pyarrow-free) to avoid the parquet engine's arrow extension
# registration clashing with snowpark's in the serving environment.
_CONTEXT_FILE = "context.json"

# Single object column carrying a ``{feature_name: value}`` map per row. Packing
# features into one OBJECT (VARIANT values) keeps the logged signature generic across
# use-cases and preserves mixed dtypes (numeric / string / bool).
FEATURES_COLUMN = "FEATURES"

# Default scalar label column. Non-null marks a context (labeled) row.
DEFAULT_TARGET_COLUMN = "TARGET"

# Output column carrying the scored value. Kept stable so the logged signature does
# not change between calls.
PREDICTION_COLUMN = "PREDICTION"

# Output column echoing each scored (query) row's input features so predictions can be joined back to
# their source row. It must not reuse the input name FEATURES: the Warehouse table-function invocation
# runs `SELECT * FROM input, TABLE(predict(...))`, which raises "ambiguous column name 'FEATURES'" when
# an output column shares an input column name.
OUTPUT_FEATURES_COLUMN = "OUTPUT_FEATURES"

# Inference parameters exposed on the signature (passed via run_batch InputSpec.params
# or a run() call, arriving as method kwargs at serving time).
_PARAM_TARGET_COLUMN = "target_column"
_PARAM_CONTEXT_FILTER = "context_filter"
_PARAM_PREDICTION_FILTER = "prediction_filter"

# init_params.json metadata keys (namespaced to avoid clashing with sklearn params).
_KEY_MODEL_TYPE = "__model_type__"
_KEY_DEVICE = "__device__"
_KEY_ESTIMATOR_CLASS = "__estimator_class__"
_KEY_TARGET_COLUMN = "__target_column__"


def _normalize_device(device: Any) -> str:
    if device is None:
        return "cpu"
    return str(device)


def _resolve_load_device(*, requested_device: Any, saved_device: str, use_gpu: bool) -> str:
    """Explicit ``device`` wins; otherwise SPCS GPU serving forces CUDA; else saved device."""
    if requested_device is not None:
        return _normalize_device(requested_device)
    if use_gpu:
        return "cuda"
    return _normalize_device(saved_device)


def _get_model_device(model: Any) -> str:
    foundation = getattr(model, "model", None)
    if foundation is None:
        return "cpu"
    try:
        return _normalize_device(next(foundation.parameters()).device)
    except (StopIteration, AttributeError, TypeError):
        return "cpu"


def _object_features_to_frame(feature_series: pd.Series, *, min_rows: int = 0) -> pd.DataFrame:
    """Reconstruct a typed feature DataFrame from a ``FEATURES`` object column.

    Each cell is a ``{feature_name: value}`` mapping (a SQL ``OBJECT`` decoded to a
    Python dict). Records are expanded into columns and pandas infers a dtype per
    column, so numeric features stay numeric and string features stay categorical --
    which is what lets the foundation model see the true per-column types. Columns are
    sorted by name for a deterministic, order-independent layout across context and
    query rows.

    Args:
        feature_series: A column whose cells are per-row ``{name: value}`` mappings.
        min_rows: Minimum number of rows required; 0 skips the check. Query frames may
            legitimately hold a single row, so callers only pass this for context rows,
            which need enough examples for in-context learning.

    Returns:
        A ``(n_rows, n_features)`` DataFrame with one column per feature key. Empty
        input yields an empty DataFrame.

    Raises:
        ValueError: If ``min_rows`` is set and the frame has fewer rows than that.
    """
    records = [dict(feature_map) if feature_map is not None else {} for feature_map in feature_series.to_numpy()]
    frame = pd.DataFrame.from_records(records)
    if min_rows and frame.shape[0] < min_rows:
        raise ValueError(f"At least {min_rows} context rows are required for in-context learning, got {len(frame)}.")
    if frame.shape[1] == 0:
        return frame
    return frame.reindex(sorted(frame.columns), axis=1).infer_objects()


def _infer_datatype(series: pd.Series) -> model_signature.DataType:
    """Best-effort map of a pandas column to a ``DataType`` for signature building.

    Numeric columns collapse to ``DOUBLE`` (class labels are frequently integer but
    round-trip safely as double), booleans to ``BOOL``, and anything else (object /
    string / categorical) to ``STRING``. Empty / all-null columns fall back to ``STRING``.

    Args:
        series: The column to inspect (e.g. the sample target or a smoke prediction).

    Returns:
        The inferred ``DataType``.
    """
    non_null = series.dropna()
    if non_null.empty:
        return model_signature.DataType.STRING
    kind = np.asarray(non_null.to_numpy()).dtype.kind
    if kind in ("f", "c", "i", "u"):
        return model_signature.DataType.DOUBLE
    if kind == "b":
        return model_signature.DataType.BOOL
    return model_signature.DataType.STRING


def _default_scalar_dtype(model_type: str) -> model_signature.DataType:
    """Default TARGET / PREDICTION type when no sample is available to infer from.

    The estimator kind is known at log time, so the default is chosen accordingly rather
    than a blanket ``DOUBLE``:

    - Regressors always emit continuous values -> ``DOUBLE``.
    - Classifier labels may be numeric OR categorical, so ``STRING`` is the widest safe
      default (numeric labels round-trip as their string form); a blanket ``DOUBLE`` would
      fail outright on string labels.

    Args:
        model_type: Either ``"regression"`` or ``"classification"``.

    Returns:
        The default scalar ``DataType`` for the label/scored columns.
    """
    if model_type == "regression":
        return model_signature.DataType.DOUBLE
    return model_signature.DataType.STRING


def _dump_init_params(
    model: Any,
    *,
    model_type: str,
    device: str,
    target_column: str,
    excluded_params: tuple[str, ...],
) -> dict[str, Any]:
    params = {key: value for key, value in model.get_params(deep=False).items() if key not in excluded_params}
    try:
        # Ensure params are JSON-serializable before shipping in the artifact.
        json.dumps(params)
    except (TypeError, ValueError) as ex:
        raise ValueError(
            "Tabular foundation model constructor parameters must be JSON-serializable; "
            f"found a value that is not: {ex}"
        ) from ex
    params[_KEY_MODEL_TYPE] = model_type
    params[_KEY_DEVICE] = device
    params[_KEY_ESTIMATOR_CLASS] = model.__class__.__name__
    params[_KEY_TARGET_COLUMN] = target_column
    return params


def _context_fingerprint(x_context: pd.DataFrame, y_context: npt.NDArray[Any]) -> str:
    """Stable fingerprint of a (possibly mixed-type) context set, used as the cache key.

    ``pd.util.hash_pandas_object`` handles heterogeneous values (numeric / string / bool)
    that a raw ``tobytes`` would not; the column layout is folded in so contexts with the
    same values but different feature keys do not collide.

    Args:
        x_context: The reconstructed context feature frame.
        y_context: The context labels aligned with ``x_context``.

    Returns:
        A hex digest uniquely identifying the context set.
    """
    hasher = hashlib.sha256()
    hasher.update(pd.util.hash_pandas_object(x_context, index=False).to_numpy().tobytes())
    hasher.update(pd.util.hash_pandas_object(pd.Series(y_context), index=False).to_numpy().tobytes())
    hasher.update(",".join(map(str, x_context.columns)).encode())
    hasher.update(repr(x_context.shape).encode())
    return hasher.hexdigest()


def _new_unfitted_like(model: _TabularModel) -> _TabularModel:
    """Construct a fresh unfitted estimator from an existing one's constructor params.

    ``sklearn.base.clone`` is intentionally avoided because it deep-copies non-estimator
    params, which would clone the (multi-GB) foundation reference. Re-instantiating from
    ``get_params`` passes the foundation reference (module object or on-disk path) through
    unchanged instead.

    Args:
        model: A loaded (unfitted) estimator whose foundation reference is reused.

    Returns:
        A new unfitted estimator sharing the same foundation reference.
    """
    return type(model)(**model.get_params(deep=False))


# Node types a context_filter/prediction_filter expression may use: boolean/comparison
# logic over column names and literals only. Anything else (attribute access, subscripts,
# calls, comprehensions, ...) is rejected before the string ever reaches ``df.query``,
# which otherwise evaluates it as a Python/numexpr expression.
_ALLOWED_FILTER_NODE_TYPES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.UAdd,
    ast.USub,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.List,
    ast.Tuple,
)


def _validate_filter_expression(filter_expression: str, *, allowed_columns: Iterable[str], param_name: str) -> None:
    """Restrict a ``context_filter``/``prediction_filter`` string to safe column comparisons.

    Args:
        filter_expression: The raw filter string supplied by the caller.
        allowed_columns: Column names the expression may reference.
        param_name: The parameter name, used to phrase the error.

    Raises:
        ValueError: If the expression does not parse, references a name outside
            ``allowed_columns``, or uses a construct other than boolean/comparison logic
            over column names and literals.
    """
    try:
        tree = ast.parse(filter_expression, mode="eval")
    except SyntaxError:
        raise ValueError(f"'{param_name}' is not a valid filter expression.") from None

    allowed_names = set(allowed_columns)
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_FILTER_NODE_TYPES):
            raise ValueError(f"'{param_name}' may only compare column values; unsupported expression.")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"'{param_name}' references an unknown column.")


def _split_context_and_query(
    df: pd.DataFrame,
    *,
    target_column: str,
    context_filter: str,
    prediction_filter: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition one input relation into context rows and query rows.

    Explicit filters (pandas ``query`` strings) win; otherwise context rows are those
    with a non-null target and query rows are those with a null target. Filters are
    restricted to column comparisons (see ``_validate_filter_expression``) since they
    carry user-controlled text through to ``df.query``.

    Args:
        df: A single input relation containing both context and query rows.
        target_column: Column holding the label (non-null for context rows).
        context_filter: Optional pandas ``query`` predicate selecting context rows.
        prediction_filter: Optional pandas ``query`` predicate selecting query rows.

    Returns:
        A ``(context_df, query_df)`` tuple.
    """
    if context_filter:
        _validate_filter_expression(context_filter, allowed_columns=df.columns, param_name="context_filter")
        context_df = df.query(context_filter)
    else:
        context_df = df[df[target_column].notna()]

    if prediction_filter:
        _validate_filter_expression(prediction_filter, allowed_columns=df.columns, param_name="prediction_filter")
        query_df = df.query(prediction_filter)
    else:
        query_df = df[df[target_column].isna()]

    return context_df, query_df


def _unbound_predict(
    model: _TabularModel,
    df: pd.DataFrame,
    *,
    target_column: str,
    context_filter: str,
    prediction_filter: str,
    fitted_cache: dict[str, _TabularModel],
) -> pd.DataFrame:
    """Fit on the partition's context rows and score its query rows.

    Args:
        model: The loaded (unfitted) foundation estimator.
        df: One partition holding both context (labeled) and query (unlabeled) rows.
        target_column: Column whose non-null values mark context rows.
        context_filter: Optional pandas ``query`` predicate selecting context rows.
        prediction_filter: Optional pandas ``query`` predicate selecting query rows.
        fitted_cache: Per-serving-instance cache of fitted estimators keyed by context
            fingerprint, so an identical context is not re-fit across calls.

    Returns:
        A DataFrame with one row per query row: an ``OUTPUT_FEATURES`` object column
        echoing the query features and a scalar ``PREDICTION`` column.

    Raises:
        ValueError: If the ``FEATURES`` column is missing, or the partition has no
            context rows or no query rows.
    """
    if FEATURES_COLUMN not in df.columns:
        raise ValueError(f"Input relation is missing the required '{FEATURES_COLUMN}' object column.")

    context_df, query_df = _split_context_and_query(
        df,
        target_column=target_column,
        context_filter=context_filter,
        prediction_filter=prediction_filter,
    )

    if len(context_df) == 0:
        raise ValueError(f"No context rows found (target column '{target_column}' is entirely null/filtered).")
    if len(query_df) == 0:
        raise ValueError(f"No query rows found (no rows with a null '{target_column}' / matching prediction_filter).")

    x_context = _object_features_to_frame(context_df[FEATURES_COLUMN], min_rows=2)
    y_context = context_df[target_column].to_numpy()
    # Align query columns to the context layout so the model sees a consistent feature
    # space; keys absent from a query row become NaN rather than an error.
    x_query = _object_features_to_frame(query_df[FEATURES_COLUMN]).reindex(columns=x_context.columns)

    fingerprint = _context_fingerprint(x_context, y_context)
    estimator = fitted_cache.get(fingerprint)
    if estimator is None:
        estimator = _new_unfitted_like(model)
        estimator.fit(x_context, y_context)
        fitted_cache[fingerprint] = estimator

    predictions = estimator.predict(x_query)

    result = query_df[[FEATURES_COLUMN]].rename(columns={FEATURES_COLUMN: OUTPUT_FEATURES_COLUMN})
    result = result.reset_index(drop=True)
    result[PREDICTION_COLUMN] = np.asarray(predictions).reshape(-1)
    return result


def _bound_predict(model: _TabularModel, x_query: pd.DataFrame) -> pd.DataFrame:
    """Score concrete feature rows with a pre-fitted (bound) estimator.

    Unlike the unbound path, the context is already baked into ``model`` (fit before logging and
    re-fit from the stored context at load), so this is a plain row-wise scalar predict: the input
    columns are the concrete features and the output is a single ``PREDICTION`` column (1:1 rows).

    Args:
        model: The loaded, already-fitted estimator.
        x_query: A batch of query rows carrying the concrete feature columns.

    Returns:
        A DataFrame with a single scalar ``PREDICTION`` column, one row per input row.
    """
    predictions = model.predict(x_query)
    return pd.DataFrame({PREDICTION_COLUMN: np.asarray(predictions).reshape(-1)})


@final
class TabularFoundationModelHandler(_base.BaseModelHandler[_TabularModel]):
    """Handler for tabular foundation models (TabICL) served in unbound (ICL) mode."""

    HANDLER_TYPE = "tabular_foundation_model"
    HANDLER_VERSION = "2026-09-24"
    _MIN_SNOWPARK_ML_VERSION = "2.0.0"
    _HANDLER_MIGRATOR_PLANS: dict[str, type[base_migrator.BaseModelHandlerMigrator]] = {}

    MODEL_BLOB_FILE_OR_DIR = "model"
    INIT_PARAMS_FILE = _INIT_PARAMS_FILE
    FOUNDATION_DIR = _FOUNDATION_DIR
    CONTEXT_FILE = _CONTEXT_FILE
    DEFAULT_TARGET_METHODS = ["predict"]

    # The generic signature can be produced from the estimator kind alone, so sample data
    # is optional; when given it refines the TARGET/PREDICTION types and smoke-tests fit.
    IS_AUTO_SIGNATURE = True

    @classmethod
    def can_handle(cls, model: model_types.SupportedModelType) -> TypeGuard[_TabularModel]:
        adapter = _find_adapter(model)
        if adapter is None:
            return False
        return any(
            (hasattr(model, method) and callable(getattr(model, method, None))) for method in cls.DEFAULT_TARGET_METHODS
        )

    @classmethod
    def cast_model(cls, model: model_types.SupportedModelType) -> _TabularModel:
        return model

    @classmethod
    def get_supported_target_platforms(
        cls, model: model_types.SupportedModelType
    ) -> list[model_types.TargetPlatform] | None:
        adapter = _find_adapter(model)
        if adapter is None:
            return None
        return adapter.supported_target_platforms()

    @classmethod
    def save_model(
        cls,
        name: str,
        model: _TabularModel,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input_data: model_types.SupportedDataType | None = None,
        is_sub_model: bool | None = False,
        **kwargs: Unpack[model_types.TabularFoundationModelSaveOptions],
    ) -> None:
        adapter = _find_adapter(model)
        if adapter is None:
            raise ValueError(f"No tabular foundation model adapter claims {type(model)}.")

        if kwargs.get("enable_explainability", False):
            raise ValueError("Explainability is not supported for tabular foundation models.")

        # A fitted estimator already carries its context, so serve it bound; a bare estimator serves
        # unbound. The mode follows the estimator's fitted state -- there is no explicit toggle.
        bound = adapter.is_fitted(model)
        target_column_option = kwargs.pop("target_column", None)
        prediction_type_option = kwargs.pop("prediction_type", None)
        model_type = adapter.estimator_kind(model)

        # When bound, the pre-fitted context captured here is written beside the foundation weights
        # below and re-fit at load; None keeps the default unbound (in-context) path.
        bound_context: tuple[pd.DataFrame, npt.NDArray[Any]] | None = None

        if not is_sub_model:
            default_target_column = target_column_option or DEFAULT_TARGET_COLUMN
            if bound:
                # (a) Bound: serve a pre-fitted estimator as a static, scalar signature.
                # sample_input_data is the concrete training frame; it both drives signature
                # inference and is stored as the context the estimator is re-fit on at load.
                bound_context = cls._prepare_bound_signature(
                    model,
                    model_meta,
                    sample_input_data,
                    target_column=default_target_column,
                    prediction_type=prediction_type_option,
                )
            elif model_meta.signatures:
                # (b) Full override: the user supplied `signatures` explicitly, so honor it
                # verbatim (they own the FEATURES/TARGET/PREDICTION types and any params). This
                # is the escape hatch for a fully bound / concrete interface.
                if "predict" not in model_meta.signatures:
                    raise ValueError("Tabular foundation models require a 'predict' entry in the provided signatures.")
            else:
                # (c) Keep the generic OBJECT input; only the scalar TARGET/PREDICTION types are
                # specialized. They default by estimator kind and are refined by a sample if given.
                target_dtype = _default_scalar_dtype(model_type)
                prediction_dtype = _default_scalar_dtype(model_type)
                if isinstance(sample_input_data, pd.DataFrame):
                    if FEATURES_COLUMN not in sample_input_data.columns:
                        raise ValueError(
                            f"sample_input_data is missing the required '{FEATURES_COLUMN}' object column."
                        )
                    if default_target_column not in sample_input_data.columns:
                        raise ValueError(f"sample_input_data is missing the target column '{default_target_column}'.")

                    # Smoke-test the real fit-on-call path once (validates the foundation weights
                    # load and fit/predict works) and let the sample seed the scalar types.
                    smoke_result = _unbound_predict(
                        model,
                        sample_input_data,
                        target_column=default_target_column,
                        context_filter="",
                        prediction_filter="",
                        fitted_cache={},
                    )
                    target_dtype = _infer_datatype(sample_input_data[default_target_column])
                    prediction_dtype = _infer_datatype(smoke_result[PREDICTION_COLUMN])
                elif sample_input_data is not None:
                    raise ValueError(
                        "sample_input_data must be a pandas DataFrame with a "
                        f"'{FEATURES_COLUMN}' object column plus a scalar target column."
                    )
                else:
                    logger.info(
                        "No sample_input_data provided; skipping the fit/predict smoke test and "
                        "defaulting TARGET/PREDICTION to %s from the estimator kind (%s). Provide "
                        "sample_input_data or `prediction_type` to refine the output type.",
                        prediction_dtype.name,
                        model_type,
                    )

                # An explicit `prediction_type` always wins for the scored output column.
                if prediction_type_option is not None:
                    prediction_dtype = model_signature.DataType.from_python_type(prediction_type_option)

                # The signature stays fixed and generic (FEATURES: OBJECT + TARGET): it encodes
                # neither a single use-case's feature width nor its feature names.
                model_meta.signatures["predict"] = cls._build_predict_signature(
                    target_column=default_target_column,
                    target_dtype=target_dtype,
                    prediction_dtype=prediction_dtype,
                )
            model_meta.task = model_types.Task.UNKNOWN
        else:
            default_target_column = target_column_option or ""

        serving_mode = "bound" if bound_context is not None else "unbound"

        device = _get_model_device(model)

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        artifact_dir = os.path.join(model_blob_path, cls.MODEL_BLOB_FILE_OR_DIR)
        foundation_dir = os.path.join(artifact_dir, cls.FOUNDATION_DIR)
        os.makedirs(foundation_dir, exist_ok=True)

        # Package the foundation weights; the fitted context (bound mode only) is stored separately.
        adapter.save_foundation(model, foundation_dir)
        init_params = _dump_init_params(
            model,
            model_type=model_type,
            device=device,
            target_column=default_target_column,
            excluded_params=adapter.EXCLUDED_INIT_PARAMS,
        )
        with open(os.path.join(artifact_dir, cls.INIT_PARAMS_FILE), "w", encoding="utf-8") as init_params_file:
            json.dump(init_params, init_params_file)

        blob_options: model_meta_schema.TabularFoundationModelBlobOptions = {
            "adapter": adapter.ADAPTER_ID,
            "estimator_type": model.__class__.__name__,
            "init_params_file": cls.INIT_PARAMS_FILE,
            "foundation_dir": cls.FOUNDATION_DIR,
            "device": device,
            "model_type": model_type,
            "serving_mode": serving_mode,
            "target_column": default_target_column,
        }
        if bound_context is not None:
            # Persist the training context so the estimator can be re-fit (bound) at load time.
            context_path = adapter_base.assert_path_within_directory(
                os.path.join(artifact_dir, cls.CONTEXT_FILE), artifact_dir
            )
            cls._write_context(bound_context, context_path, target_column=default_target_column)
            blob_options["context_file"] = cls.CONTEXT_FILE

        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODEL_BLOB_FILE_OR_DIR,
            options=blob_options,
        )
        model_meta.models[name] = base_meta
        model_meta.min_snowpark_ml_version = cls._MIN_SNOWPARK_ML_VERSION
        if serving_mode != "bound":
            # Unbound predict receives the whole partition (context + query) in one call; bound serves
            # a plain scalar predict, so it stays a regular (non-partitioned) function.
            model_meta.function_properties["predict"] = {model_meta_schema.FunctionProperties.PARTITIONED.value: True}

        adapter.add_dependencies(model_meta)
        # Tabular foundation models are torch models served on SPCS GPU compute pools, so always emit a
        # `gpu` runtime in model.yaml regardless of the local save device (e.g. logging from a CPU laptop).
        # The `cpu` runtime is still emitted for CPU/warehouse paths. Without this, a GPU model-build would
        # fail with `KeyError: 'gpu'` on `runtimes["gpu"]`.
        model_meta.env.cuda_version = kwargs.get("cuda_version", handlers_utils.get_default_cuda_version())

    @classmethod
    def _build_predict_signature(
        cls,
        *,
        target_column: str,
        target_dtype: model_signature.DataType,
        prediction_dtype: model_signature.DataType,
    ) -> model_signature.ModelSignature:
        """Build the fixed, generic predict signature.

        Inputs are a single ``FEATURES`` object column (``OBJECT`` / ``MapType(String,
        Variant)``) carrying ``{feature_name: value}`` per row plus a nullable scalar
        target column; outputs echo ``FEATURES`` and add a scalar ``PREDICTION``. The
        object cell absorbs both variable feature *width* and mixed value *types*, so the
        signature never changes across use-cases. The unbound inference params
        (target/filters) ride on the signature.

        Args:
            target_column: Name of the scalar label column (non-null marks context rows).
            target_dtype: Inferred/declared type of the label column (e.g. string labels).
            prediction_dtype: Inferred/declared type of the scored output column.

        Returns:
            The generic, mixed-type predict signature.
        """
        object_type = model_signature.DataType.OBJECT
        str_param_shape = model_signature.DataType.shape_from_python_type(str)
        str_param_type = model_signature.DataType.from_python_type(str)

        inputs = [
            model_signature.FeatureSpec(name=FEATURES_COLUMN, dtype=object_type, nullable=False),
            model_signature.FeatureSpec(name=target_column, dtype=target_dtype, nullable=True),
        ]
        outputs = [
            model_signature.FeatureSpec(name=OUTPUT_FEATURES_COLUMN, dtype=object_type, nullable=False),
            model_signature.FeatureSpec(name=PREDICTION_COLUMN, dtype=prediction_dtype, nullable=False),
        ]
        params = [
            model_signature.ParamSpec(
                name=_PARAM_TARGET_COLUMN,
                dtype=str_param_type,
                default_value=target_column,
                shape=str_param_shape,
            ),
            model_signature.ParamSpec(
                name=_PARAM_CONTEXT_FILTER,
                dtype=str_param_type,
                default_value="",
                shape=str_param_shape,
            ),
            model_signature.ParamSpec(
                name=_PARAM_PREDICTION_FILTER,
                dtype=str_param_type,
                default_value="",
                shape=str_param_shape,
            ),
        ]
        return model_signature.ModelSignature(inputs=inputs, outputs=outputs, params=params)

    @classmethod
    def _prepare_bound_signature(
        cls,
        model: _TabularModel,
        model_meta: model_meta_api.ModelMetadata,
        sample_input_data: model_types.SupportedDataType | None,
        *,
        target_column: str,
        prediction_type: type | None,
    ) -> tuple[pd.DataFrame, npt.NDArray[Any]]:
        """Infer the static scalar predict signature and return the context to store.

        Bound serving needs a concrete training frame: feature columns plus a scalar
        ``target_column``. The features drive a standard signature inference (named input columns
        -> a single ``PREDICTION`` output) through the shared ``validate_signature`` helper, which
        also validates a user-supplied ``signatures`` instead of inferring. The (features, labels)
        split is returned so the caller can persist it as the context the estimator is re-fit on at
        load.

        Args:
            model: The pre-fitted estimator (its ``predict`` is exercised on the sample).
            model_meta: Metadata whose ``signatures['predict']`` is populated here.
            sample_input_data: The concrete training frame (features + ``target_column``).
            target_column: Name of the scalar label column to split off as the labels.
            prediction_type: Optional override for the scored output type (ignored when the user
                supplied ``signatures``).

        Returns:
            A ``(x_context, y_context)`` tuple: the feature frame and aligned labels.

        Raises:
            ValueError: If ``sample_input_data`` is not a DataFrame, is missing ``target_column``,
                or carries no feature columns.
        """
        if not isinstance(sample_input_data, pd.DataFrame):
            raise ValueError(
                "Bound tabular foundation models require sample_input_data: a pandas DataFrame holding the "
                "training context (feature columns plus the target column)."
            )
        if target_column not in sample_input_data.columns:
            raise ValueError(f"sample_input_data is missing the target column '{target_column}'.")
        x_context = sample_input_data.drop(columns=[target_column])
        if x_context.shape[1] == 0:
            raise ValueError("sample_input_data must contain at least one feature column besides the target column.")
        y_context = sample_input_data[target_column].to_numpy()

        had_user_signature = bool(model_meta.signatures)

        def get_prediction(
            target_method_name: str, local_sample_input: model_types.SupportedLocalDataType
        ) -> model_types.SupportedLocalDataType:
            predictions = model.predict(local_sample_input)
            return pd.DataFrame({PREDICTION_COLUMN: np.asarray(predictions).reshape(-1)})

        handlers_utils.validate_signature(
            model=model,
            model_meta=model_meta,
            target_methods=["predict"],
            sample_input_data=x_context,
            get_prediction_fn=get_prediction,
        )

        # An explicit `prediction_type` overrides the inferred scored type unless the user owns the
        # whole signature.
        if prediction_type is not None and not had_user_signature:
            predict_signature = model_meta.signatures["predict"]
            overridden_output = model_signature.FeatureSpec(
                name=predict_signature.outputs[0].name,
                dtype=model_signature.DataType.from_python_type(prediction_type),
            )
            model_meta.signatures["predict"] = model_signature.ModelSignature(
                inputs=list(predict_signature.inputs), outputs=[overridden_output]
            )

        return x_context, y_context

    @classmethod
    def _write_context(
        cls,
        context: tuple[pd.DataFrame, npt.NDArray[Any]],
        path: str,
        *,
        target_column: str,
    ) -> None:
        """Persist the bound training context as one JSON Table Schema file (features + target column).

        The labels ride as a named column (``target_column``, recorded in the blob options) so the
        load side splits them back out without a reserved sentinel that could collide with a real
        feature name. The Table Schema orient embeds per-column dtypes for a faithful re-fit and keeps
        the write pyarrow-free (the parquet engine's arrow extension registration clashes with
        snowpark's in the serving environment).

        Args:
            context: The ``(x_context, y_context)`` split captured at save time.
            path: Destination path (already confined to the artifact directory).
            target_column: Name under which to store the labels.
        """
        x_context, y_context = context
        frame = x_context.copy()
        frame[target_column] = y_context
        frame.reset_index(drop=True).to_json(path, orient="table", double_precision=15)

    @classmethod
    def load_model(
        cls,
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.TabularFoundationModelLoadOptions],
    ) -> _TabularModel:
        model_blob_path = os.path.join(model_blobs_dir_path, name)
        model_blob_metadata = model_meta.models[name]
        model_blob_options = cast(model_meta_schema.TabularFoundationModelBlobOptions, model_blob_metadata.options)

        artifact_dir = adapter_base.assert_path_within_directory(
            os.path.join(model_blob_path, cls.MODEL_BLOB_FILE_OR_DIR), model_blob_path
        )
        foundation_dir = adapter_base.assert_path_within_directory(
            os.path.join(artifact_dir, cls.FOUNDATION_DIR), artifact_dir
        )
        init_params_path = adapter_base.assert_path_within_directory(
            os.path.join(artifact_dir, cls.INIT_PARAMS_FILE), artifact_dir
        )

        if not os.path.isdir(foundation_dir):
            raise FileNotFoundError(f"Foundation weights not found at {foundation_dir}")
        if not os.path.isfile(init_params_path):
            raise FileNotFoundError(f"init params not found at {init_params_path}")

        with open(init_params_path, encoding="utf-8") as init_params_file:
            init_params = json.load(init_params_file)
        init_params.pop(_KEY_MODEL_TYPE, None)
        init_params.pop(_KEY_ESTIMATOR_CLASS, None)
        init_params.pop(_KEY_TARGET_COLUMN, None)
        saved_device = init_params.pop(_KEY_DEVICE, model_blob_options["device"])
        device = _resolve_load_device(
            requested_device=kwargs.get("device"),
            saved_device=saved_device,
            use_gpu=bool(kwargs.get("use_gpu", False)),
        )

        adapter = _get_adapter(model_blob_options["adapter"])
        estimator = adapter.load_estimator(
            estimator_type=model_blob_options["estimator_type"],
            model_type=model_blob_options["model_type"],
            foundation_dir=foundation_dir,
            init_params=init_params,
            device=device,
        )

        if model_blob_options.get("serving_mode") == "bound":
            # Bound: re-fit the estimator on the stored context so it is ready for scalar predict.
            # (Unbound leaves it unfitted; the context is bound per call there instead.)
            context_file = model_blob_options.get("context_file", "")
            if not context_file:
                raise FileNotFoundError("Bound tabular foundation model is missing its stored context reference.")
            context_path = adapter_base.assert_path_within_directory(
                os.path.join(artifact_dir, context_file), artifact_dir
            )
            if not os.path.isfile(context_path):
                raise FileNotFoundError(f"Bound tabular foundation model context not found at {context_path}")
            target_column = model_blob_options.get("target_column", "") or DEFAULT_TARGET_COLUMN
            context_frame = pd.read_json(context_path, orient="table").reset_index(drop=True)
            if target_column not in context_frame.columns:
                raise ValueError(f"Stored context is missing the target column '{target_column}'.")
            y_context = context_frame[target_column].to_numpy()
            x_context = context_frame.drop(columns=[target_column])
            estimator.fit(x_context, y_context)

        return estimator

    @classmethod
    def convert_as_custom_model(
        cls,
        raw_model: _TabularModel,
        model_meta: model_meta_api.ModelMetadata,
        background_data: pd.DataFrame | None = None,
        **kwargs: Unpack[model_types.TabularFoundationModelLoadOptions],
    ) -> custom_model.CustomModel:
        from snowflake.ml.model import custom_model as custom_model_api

        model_blob_metadata = next(iter(model_meta.models.values()))
        blob_options = model_blob_metadata.options
        default_target_column = cast(str, blob_options.get("target_column", "") or "")
        serving_mode = cast(str, blob_options.get("serving_mode", "unbound") or "unbound")

        def _create_custom_model(
            raw_model: _TabularModel,
            model_meta: model_meta_api.ModelMetadata,
        ) -> type[custom_model.CustomModel]:
            # One fit-on-call cache shared across calls on this serving instance (unbound only).
            fitted_cache: dict[str, _TabularModel] = {}

            def unbound_fn_factory(
                raw_model: _TabularModel,
                signature: model_signature.ModelSignature,
                target_method: str,
            ) -> Callable[[custom_model.CustomModel, pd.DataFrame], pd.DataFrame]:
                @custom_model_api._internal_inference_api
                def fn(self: custom_model.CustomModel, X: pd.DataFrame, **method_kwargs: Any) -> pd.DataFrame:
                    target_column = method_kwargs.get(_PARAM_TARGET_COLUMN) or default_target_column
                    result = _unbound_predict(
                        raw_model,
                        X,
                        target_column=target_column,
                        context_filter=method_kwargs.get(_PARAM_CONTEXT_FILTER, "") or "",
                        prediction_filter=method_kwargs.get(_PARAM_PREDICTION_FILTER, "") or "",
                        fitted_cache=fitted_cache,
                    )
                    return model_signature_utils.rename_pandas_df(result, signature.outputs)

                return fn

            def bound_fn_factory(
                raw_model: _TabularModel,
                signature: model_signature.ModelSignature,
                target_method: str,
            ) -> Callable[[custom_model.CustomModel, pd.DataFrame], pd.DataFrame]:
                @custom_model_api._internal_inference_api
                def fn(self: custom_model.CustomModel, X: pd.DataFrame, **method_kwargs: Any) -> pd.DataFrame:
                    result = _bound_predict(raw_model, X)
                    return model_signature_utils.rename_pandas_df(result, signature.outputs)

                return fn

            fn_factory = bound_fn_factory if serving_mode == "bound" else unbound_fn_factory

            type_method_dict: dict[str, Any] = {"_raw_model": raw_model, "_allows_kwargs": True}
            for target_method_name, sig in model_meta.signatures.items():
                type_method_dict[target_method_name] = fn_factory(raw_model, sig, target_method_name)

            return type("_TabularFoundationModel", (custom_model_api.CustomModel,), type_method_dict)

        tabular_model_cls = _create_custom_model(raw_model, model_meta)
        return tabular_model_cls(custom_model_api.ModelContext())
