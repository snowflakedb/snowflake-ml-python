# mypy: disable-error-code="import"
"""Base class and shared helpers for tabular foundation model adapters.

The generic ``TabularFoundationModelHandler`` owns the signature, partition split,
fit-on-call caching, and the ``foundation/`` + ``init_params.json`` layout. A
``ModelAdapter`` supplies only what differs per foundation model: which estimator classes
it claims, how to persist and restore the foundation weights, which constructor params
must not be serialized, and the runtime dependencies. Adding a new foundation model is a
new adapter in its own submodule, not a new handler.
"""

import abc
import os
from typing import Any

from snowflake.ml._internal import type_utils
from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._packager.model_meta import model_meta as model_meta_api

# Estimator instances are duck-typed (sklearn interface); the concrete classes are
# resolved lazily inside adapters so no estimator library is imported at module load.
TabularModel = Any


def assert_path_within_directory(path: str, root: str) -> str:
    """Resolve ``path`` and reject values that escape ``root``."""
    resolved_path = os.path.realpath(path)
    resolved_root = os.path.realpath(root)
    if os.path.commonpath([resolved_path, resolved_root]) != resolved_root:
        raise ValueError("Invalid model artifact path.")
    return resolved_path


class ModelAdapter(abc.ABC):
    """Per-foundation-model hooks the generic handler dispatches to.

    The generic handler owns the signature, partition split, fit-on-call caching, and the
    ``foundation/`` + ``init_params.json`` layout. An adapter supplies only what differs
    per model: which estimator classes it claims, how to persist and restore the
    foundation weights, which constructor params must not be serialized (a foundation
    module object, or a path the load side overrides), and the runtime dependencies.
    """

    # Stable id persisted in the blob options; selects the adapter at load time.
    ADAPTER_ID: str
    # Estimator module and the classifier/regressor class names it exposes.
    MODULE: str
    CLASSIFIER_CLASS: str
    REGRESSOR_CLASS: str
    # Constructor params dropped from init_params.json (a non-serializable foundation
    # reference, or one the load side rebinds explicitly).
    EXCLUDED_INIT_PARAMS: tuple[str, ...] = ()
    # Whether this foundation model can feasibly serve in a Snowflake Warehouse. Multi-GB
    # foundations are too large for a Warehouse and serve only on Snowpark Container Services;
    # small-checkpoint models (e.g. TabICL) run in both.
    WAREHOUSE_FEASIBLE: bool = True

    @classmethod
    def supported_target_platforms(cls) -> list[model_types.TargetPlatform] | None:
        """The platforms this model is restricted to, or None when it runs anywhere."""
        if cls.WAREHOUSE_FEASIBLE:
            return None
        return [model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES]

    @classmethod
    def claimed_types(cls) -> tuple[str, ...]:
        return (f"{cls.MODULE}.{cls.CLASSIFIER_CLASS}", f"{cls.MODULE}.{cls.REGRESSOR_CLASS}")

    @classmethod
    def matches(cls, model: Any) -> bool:
        return any(type_utils.LazyType(claimed).isinstance(model) for claimed in cls.claimed_types())

    @classmethod
    def estimator_kind(cls, model: Any) -> str:
        """Return ``"classification"`` or ``"regression"`` from the estimator class name."""
        return "classification" if type(model).__name__ == cls.CLASSIFIER_CLASS else "regression"

    @classmethod
    def is_fitted(cls, model: Any) -> bool:
        """Whether ``model`` is already fit (serve bound) vs. a bare estimator (serve unbound).

        Tabular foundation estimators are sklearn-compatible, so the shared probe applies sklearn's
        own fitted convention without importing it: an estimator is fit if it exposes an
        ``__sklearn_is_fitted__`` hook that returns True, or sets at least one instance attribute with
        a trailing underscore during ``fit``. An adapter whose estimator deviates overrides this.

        Args:
            model: The estimator instance being logged.

        Returns:
            True if the estimator appears fit, False otherwise.
        """
        is_fitted_hook = getattr(model, "__sklearn_is_fitted__", None)
        if callable(is_fitted_hook):
            try:
                return bool(is_fitted_hook())
            except Exception:
                return False
        return any(name.endswith("_") and not name.startswith("__") for name in vars(model))

    @classmethod
    @abc.abstractmethod
    def save_foundation(cls, model: Any, foundation_dir: str) -> None:
        """Persist the foundation weights under ``foundation_dir`` (no fitted context)."""

    @classmethod
    @abc.abstractmethod
    def load_estimator(
        cls,
        *,
        estimator_type: str,
        model_type: str,
        foundation_dir: str,
        init_params: dict[str, Any],
        device: str,
    ) -> TabularModel:
        """Reconstruct an unfitted estimator bound to the packaged foundation weights."""

    @classmethod
    @abc.abstractmethod
    def add_dependencies(cls, model_meta: model_meta_api.ModelMetadata) -> None:
        """Declare the runtime pip dependencies for this foundation model."""
