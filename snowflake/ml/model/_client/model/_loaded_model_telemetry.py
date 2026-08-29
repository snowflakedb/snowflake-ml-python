"""Client telemetry for native objects returned by ModelVersion.load().

Each call to an allowlisted method emits a usage event. Instrumentation is
best-effort and must not change user-visible behavior.
"""

import functools
import logging
import os
from typing import Any

from snowflake.ml._internal import telemetry

_logger = logging.getLogger(__name__)

_TELEMETRY_PROJECT = "MLOps"
_TELEMETRY_SUBPROJECT = "ModelManagement"
_DISABLE_ENV_VAR = "SNOWFLAKE_ML_DISABLE_LOADED_MODEL_TELEMETRY"

_METHOD_ALLOWLIST = (
    "predict",  # sklearn, xgboost, lightgbm, catboost, prophet, keras, mlflow pyfunc
    "predict_proba",  # sklearn classifiers
    "predict_log_proba",  # sklearn classifiers
    "score",  # sklearn estimators
    "score_samples",  # sklearn density / outlier estimators
    "decision_function",  # sklearn classifiers / outlier
    "transform",  # sklearn transformers, sentence-transformers
    "fit",  # sklearn / xgboost estimators
    "partial_fit",  # sklearn incremental estimators
    "get_params",  # sklearn estimators
    "set_params",  # sklearn estimators
    "forward",  # torch.nn.Module
    "eval",  # torch.nn.Module
    "train",  # torch.nn.Module
    "encode",  # sentence-transformers
    "__call__",  # HF Pipeline, Keras Model, tf.Module
)


def _make_wrapper(
    original: Any,
    method_name: str,
    *,
    model_name: str,
    version_name: str,
    model_class_module: str,
    model_class_name: str,
) -> Any:
    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            telemetry.send_custom_usage(
                project=_TELEMETRY_PROJECT,
                telemetry_type=telemetry.TelemetryField.TYPE_SNOWML_LOADED_MODEL_USAGE.value,
                subproject=_TELEMETRY_SUBPROJECT,
                data={
                    "method_name": method_name,
                    "model_name": model_name,
                    "version_name": version_name,
                    "model_class_module": model_class_module,
                    "model_class_name": model_class_name,
                },
            )
        except Exception:
            _logger.debug("loaded-model telemetry emit failed", exc_info=True)
        return original(self, *args, **kwargs)

    return wrapper


def instrument_for_telemetry(model: Any, *, model_name: str, version_name: str) -> Any:
    if os.environ.get(_DISABLE_ENV_VAR):
        return model

    cls = type(model)
    overrides: dict[str, Any] = {}

    for name in _METHOD_ALLOWLIST:
        if name == "__call__":
            # getattr(cls, "__call__") returns the metaclass's __call__ (i.e. type.__call__,
            # used to instantiate the class) when the class itself does not define __call__.
            # Gate on callable(model) instead, which is True only when the instance is
            # actually invocable, and walk the MRO for the user-defined __call__.
            if not callable(model):
                continue
            original = None
            for base in cls.__mro__:
                if base is object:
                    break
                if "__call__" in base.__dict__:
                    original = base.__dict__["__call__"]
                    break
            if original is None:
                continue
        else:
            original = getattr(cls, name, None)
            if original is None or not callable(original):
                continue
        overrides[name] = _make_wrapper(
            original,
            name,
            model_name=model_name,
            version_name=version_name,
            model_class_module=cls.__module__,
            model_class_name=cls.__name__,
        )

    if not overrides:
        return model

    # Pickling under the dynamic subclass would fail because pickle looks the class
    # up by qualified name. Build a reduce that bypasses pickle's __newobj__ sanity
    # check (which requires args[0] == type(obj)) by routing through the top-level
    # _reconstruct helper. Unpickled instances are of the original class, so they
    # also lose instrumentation, which is the desired behavior.
    def _reduce_ex(self: Any, protocol: int) -> Any:
        instrumented = type(self)
        original_class = instrumented.__bases__[0]
        self.__class__ = original_class
        try:
            if hasattr(self, "__getstate__"):
                state = self.__getstate__()
            else:
                state = self.__dict__.copy()
        finally:
            self.__class__ = instrumented
        return (_reconstruct, (original_class,), state)

    overrides["__reduce_ex__"] = _reduce_ex

    instrumented_cls = type(f"_SnowMLInstrumented_{cls.__name__}", (cls,), overrides)
    try:
        model.__class__ = instrumented_cls
    except TypeError:
        _logger.debug("Cannot instrument %s for client telemetry: __class__ assignment forbidden", cls)
    return model


def _reconstruct(cls: type) -> Any:
    """Top-level reconstructor used by pickled instrumented models.

    Defined at module scope so pickle can locate it by qualified name. Bypasses
    pickle's __newobj__ sanity check by being a plain function rather than the
    sentinel copyreg.__newobj__.

    Args:
        cls: The original (uninstrumented) class to reconstruct an empty instance of.

    Returns:
        An empty instance of cls; pickle restores the state via __setstate__ or __dict__.
    """
    return object.__new__(cls)
