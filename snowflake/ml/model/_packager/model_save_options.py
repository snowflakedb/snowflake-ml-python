"""Validation for ``options`` passed to :meth:`~snowflake.ml.model._packager.model_packager.ModelPackager.save`.

User-facing keys are derived from :mod:`snowflake.ml.model.type_hints` TypedDicts. Trusted SnowML code paths
may additionally pass keys listed in ``INTERNAL_MODEL_SAVE_OPTION_KEYS`` and
``INTERNAL_MODEL_METHOD_SAVE_OPTION_KEYS`` when validating with ``include_internal_option_keys=True`` (the
default for :class:`~snowflake.ml.model._packager.model_packager.ModelPackager`).
"""

from __future__ import annotations

from typing import Any, Mapping

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml.model import type_hints as model_types

# Keys trusted SnowML layers may place on ``options``; not part of the public ``log_model`` contract.
INTERNAL_MODEL_SAVE_OPTION_KEYS: frozenset[str] = frozenset()

# Keys trusted SnowML may place inside ``method_options`` method entries; not public.
INTERNAL_MODEL_METHOD_SAVE_OPTION_KEYS: frozenset[str] = frozenset()


def _typed_dict_annotation_keys(typed_dict_cls: type) -> frozenset[str]:
    """Return all keys declared on ``typed_dict_cls`` or its TypedDict bases (via ``__mro__``)."""
    names: set[str] = set()
    for cls in typed_dict_cls.__mro__:
        if cls is dict or cls is object:
            continue
        ann = getattr(cls, "__annotations__", None)
        if ann:
            names.update(ann.keys())
    return frozenset(names)


_BASE_MODEL_SAVE_OPTION_KEYS: frozenset[str] = _typed_dict_annotation_keys(model_types.BaseModelSaveOption)

_HANDLER_OPTION_TYPED_DICT: dict[model_types.SupportedModelHandlerType, type] = {
    "catboost": model_types.CatBoostModelSaveOptions,
    "custom": model_types.CustomModelSaveOption,
    "huggingface_pipeline": model_types.HuggingFaceSaveOptions,
    "lightgbm": model_types.LGBMModelSaveOptions,
    "mlflow": model_types.MLFlowSaveOptions,
    "prophet": model_types.ProphetSaveOptions,
    "pytorch": model_types.PyTorchSaveOptions,
    "sentence_transformers": model_types.SentenceTransformersSaveOptions,
    "sklearn": model_types.SKLModelSaveOptions,
    "snowml": model_types.SNOWModelSaveOptions,
    "tensorflow": model_types.TensorflowSaveOptions,
    "torchscript": model_types.TorchScriptSaveOptions,
    "xgboost": model_types.XGBModelSaveOptions,
    "keras": model_types.KerasSaveOptions,
}

_HANDLER_SPECIFIC_SAVE_OPTION_KEYS: dict[model_types.SupportedModelHandlerType, frozenset[str]] = {
    ht: frozenset(_typed_dict_annotation_keys(td) - _BASE_MODEL_SAVE_OPTION_KEYS)
    for ht, td in _HANDLER_OPTION_TYPED_DICT.items()
}

_MODEL_METHOD_SAVE_OPTION_KEYS: frozenset[str] = _typed_dict_annotation_keys(model_types.ModelMethodSaveOptions)


def _validate_method_options(
    *,
    options: Mapping[str, Any],
    include_internal_option_keys: bool,
) -> None:
    """Raise if ``method_options`` is malformed or contains unknown per-method keys."""
    raw_method_options = options.get("method_options")
    if raw_method_options is None:
        return
    if not isinstance(raw_method_options, dict):
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=TypeError(
                "model save: method_options must be a dictionary mapping method names to their option dictionaries."
            ),
        )
    allowed_method_keys = _MODEL_METHOD_SAVE_OPTION_KEYS
    if include_internal_option_keys:
        allowed_method_keys = allowed_method_keys | INTERNAL_MODEL_METHOD_SAVE_OPTION_KEYS

    for method_name, raw_opts in raw_method_options.items():
        if not isinstance(raw_opts, dict):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=TypeError(
                    "model save: each method_options value must be a dictionary of options for that method."
                ),
            )
        unknown = frozenset(raw_opts.keys()) - allowed_method_keys
        if unknown:
            sorted_unknown = ", ".join(sorted(unknown))
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    "model save: unrecognized method_options keys for one or more methods "
                    f"(method {method_name!r}): {sorted_unknown}. "
                    f"Supported per-method keys are {', '.join(sorted(allowed_method_keys))}."
                ),
            )


def validate_model_save_option_keys(
    *,
    handler_type: model_types.SupportedModelHandlerType,
    options: Mapping[str, Any],
    include_internal_option_keys: bool = True,
) -> None:
    """Raise if ``options`` contains keys not supported for ``handler_type``.

    Args:
        handler_type: Handler resolved for the model being saved.
        options: Save options dict (same mapping passed to the packager).
        include_internal_option_keys: When False, only keys declared on public TypedDicts are allowed (use
            for user-provided ``log_model`` ``options`` before reconciliation). When True, also allow
            ``INTERNAL_MODEL_SAVE_OPTION_KEYS`` (default for
            :class:`~snowflake.ml.model._packager.model_packager.ModelPackager`).

    Raises:
        SnowflakeMLException: If an unknown option key is present (``INVALID_ARGUMENT``), or if
            ``method_options`` is invalid or contains unknown per-method keys.
    """
    allowed = _BASE_MODEL_SAVE_OPTION_KEYS | _HANDLER_SPECIFIC_SAVE_OPTION_KEYS[handler_type]
    if include_internal_option_keys:
        allowed = allowed | INTERNAL_MODEL_SAVE_OPTION_KEYS

    unknown = frozenset(options.keys()) - allowed
    if unknown:
        sorted_unknown = ", ".join(sorted(unknown))
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_ARGUMENT,
            original_exception=ValueError(
                "model save: unrecognized option keys for this model type: "
                f"{sorted_unknown}. Check spelling and supported keys for the framework you are logging."
            ),
        )
    _validate_method_options(options=options, include_internal_option_keys=include_internal_option_keys)
