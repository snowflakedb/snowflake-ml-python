import inspect
import json
import logging
import os
import shutil
from importlib import metadata as importlib_metadata
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Union, cast, final

import cloudpickle
import pandas as pd
from packaging import version
from typing_extensions import TypeGuard, Unpack

from snowflake.ml._internal import type_utils
from snowflake.ml.model import custom_model, model_signature, type_hints as model_types
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_handlers import _base, _utils as handlers_utils
from snowflake.ml.model._packager.model_handlers_migrator import base_migrator
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta as model_meta_api,
    model_meta_schema,
)
from snowflake.ml.model.models import huggingface as snowml_huggingface
from snowflake.snowpark._internal import utils as snowpark_utils

if TYPE_CHECKING:
    import sentence_transformers

logger = logging.getLogger(__name__)

# Allowlist of supported target methods for SentenceTransformer models.
# Note: Not all methods are available in all sentence-transformers versions.
_ALLOWED_TARGET_METHODS = ["encode", "encode_query", "encode_document", "encode_queries", "encode_documents"]
_DEFAULT_WRAPPER_TARGET_METHODS = ["encode", "encode_query", "encode_document"]
_MIN_INIT_TRUNCATE_DIM_VERSION = version.parse("2.7.0")
_MIN_ENCODE_TRUNCATE_DIM_PARAM_VERSION = version.parse("5.0.0")

# All potential default methods to check for availability on the model
_POTENTIAL_DEFAULT_METHODS = [
    "encode",  # Always check encode first (always available)
    "encode_query",  # Singular (newer versions)
    "encode_document",  # Singular (newer versions)
    "encode_queries",  # Plural (older versions)
    "encode_documents",  # Plural (older versions)
]


def _get_available_default_methods(model: "sentence_transformers.SentenceTransformer") -> Sequence[str]:
    """Get default target methods that are actually available on the model.

    This function checks which methods actually exist and are callable on the model
    instance, rather than relying on version number checks.

    Args:
        model: The SentenceTransformer model instance.

    Returns:
        List of method names that are available on the model.
    """
    available_methods = []
    for method_name in _POTENTIAL_DEFAULT_METHODS:
        method = getattr(model, method_name, None)
        if method is not None and callable(method):
            available_methods.append(method_name)
    return available_methods


def _get_sentence_transformers_version() -> Optional[version.Version]:
    """Return the installed sentence-transformers version, or None if not installed."""
    try:
        return version.parse(importlib_metadata.version("sentence-transformers"))
    except importlib_metadata.PackageNotFoundError:
        return None


def _supports_init_truncate_dim() -> bool:
    """Whether the installed sentence-transformers supports truncate_dim in __init__ (>= 2.7.0)."""
    sentence_transformers_version = _get_sentence_transformers_version()
    return sentence_transformers_version is not None and sentence_transformers_version >= _MIN_INIT_TRUNCATE_DIM_VERSION


def _supports_encode_truncate_dim_param() -> bool:
    """Whether the installed sentence-transformers supports truncate_dim as an encode() parameter (>= 5.0.0)."""
    sentence_transformers_version = _get_sentence_transformers_version()
    return (
        sentence_transformers_version is not None
        and sentence_transformers_version >= _MIN_ENCODE_TRUNCATE_DIM_PARAM_VERSION
    )


def _capture_model_truncate_dim(
    model: Union["sentence_transformers.SentenceTransformer", snowml_huggingface.SentenceTransformer],
) -> Optional[int]:
    """Read and validate the model's truncate_dim attribute, returning None if unset.

    Args:
        model: The SentenceTransformer model instance.

    Returns:
        The model's truncate_dim attribute, or None if unset.

    Raises:
        ValueError: If truncate_dim is set but is not a positive integer.
    """
    model_truncate_dim = getattr(model, "truncate_dim", None)
    if model_truncate_dim is None:
        return None
    if not isinstance(model_truncate_dim, int) or model_truncate_dim <= 0:
        raise ValueError("truncate_dim must be a positive integer")
    return model_truncate_dim


def _capture_pretrained_model_name(model: "sentence_transformers.SentenceTransformer") -> Optional[str]:
    """Read the pretrained model identifier from a SentenceTransformer instance.

    Args:
        model: The SentenceTransformer model instance.

    Returns:
        The hub model id or local path if available, otherwise None.
    """
    transformers_model = getattr(model, "transformers_model", None)
    if transformers_model is not None:
        name_or_path = getattr(transformers_model, "name_or_path", None)
        if isinstance(name_or_path, str) and name_or_path:
            return name_or_path

    return None


def _auto_infer_signature(
    target_method: str,
    embedding_dim: int,
    batch_size: Optional[int] = None,
    *,
    include_truncate_dim_param: bool = False,
) -> Optional[model_signature.ModelSignature]:
    """Auto-infer signature for SentenceTransformer models.

    SentenceTransformer models have a simple signature: they take a string input
    and return an embedding vector (array of floats).

    Args:
        target_method: The target method name (e.g., "encode", "encode_query").
        embedding_dim: The dimension of the embedding vector output by the model.
        batch_size: Default batch size for inference, exposed as a runtime param.
        include_truncate_dim_param: Whether to add truncate_dim as a runtime param (default None).

    Returns:
        A ModelSignature for the target method, or None if the method is not supported.
    """
    if target_method not in _ALLOWED_TARGET_METHODS:
        return None

    params = [
        model_signature.ParamSpec(
            name="batch_size",
            dtype=model_signature.DataType.INT64,
            default_value=batch_size,
        ),
    ]
    if include_truncate_dim_param:
        params.append(
            model_signature.ParamSpec(
                name="truncate_dim",
                dtype=model_signature.DataType.INT64,
                default_value=None,
            )
        )

    return model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(name="sentence", dtype=model_signature.DataType.STRING),
        ],
        outputs=[
            model_signature.FeatureSpec(
                name="output",
                dtype=model_signature.DataType.DOUBLE,
                shape=(embedding_dim,),
            ),
        ],
        params=params,
    )


_POOLING_MODES = frozenset(
    {
        "pooling_mode_cls_token",
        "pooling_mode_max_tokens",
        "pooling_mode_mean_tokens",
        "pooling_mode_mean_sqrt_len_tokens",
        "pooling_mode_weightedmean_tokens",
        "pooling_mode_lasttoken",
    }
)


def _get_module_embedding_dim_from_config(module_type: str, module_config: dict[str, Any]) -> Optional[int]:
    """Extract embedding dimension from a single sentence-transformers module config."""
    if "Dense" in module_type:
        out_features = module_config.get("out_features")
        return int(out_features) if out_features is not None else None

    if "Pooling" in module_type:
        word_embedding_dim = module_config.get("word_embedding_dimension")
        if word_embedding_dim is None:
            word_embedding_dim = module_config.get("embedding_dimension")
        if word_embedding_dim is None:
            return None

        pooling_mode = module_config.get("pooling_mode")
        if pooling_mode is not None:
            if isinstance(pooling_mode, str):
                multiplier = 1
            elif isinstance(pooling_mode, (list, tuple)):
                multiplier = len(pooling_mode)
            else:
                multiplier = 1
            return int(word_embedding_dim) * multiplier

        enabled_modes = sum(1 for mode in _POOLING_MODES if module_config.get(mode))
        multiplier = max(enabled_modes, 1)
        return int(word_embedding_dim) * multiplier

    if "LayerNorm" in module_type:
        dimension = module_config.get("dimension")
        return int(dimension) if dimension is not None else None

    if "WeightedLayerPooling" in module_type:
        embedding_dimension = module_config.get("embedding_dimension")
        if embedding_dimension is None:
            embedding_dimension = module_config.get("word_embedding_dimension")
        return int(embedding_dimension) if embedding_dimension is not None else None

    if "LSTM" in module_type or "CNN" in module_type:
        embeddings_dimension = module_config.get("embeddings_dimension")
        return int(embeddings_dimension) if embeddings_dimension is not None else None

    if "BoW" in module_type:
        sentence_embedding_dimension = module_config.get("sentence_embedding_dimension")
        return int(sentence_embedding_dimension) if sentence_embedding_dimension is not None else None

    return None


def _get_embedding_dim_from_config(repo_snapshot_dir: str) -> Optional[int]:
    """Extract the embedding dimension from a sentence-transformers snapshot.

    Walks ``modules.json`` in reverse and reads each module's ``config.json``,
    matching the logic of ``SentenceTransformer.get_sentence_embedding_dimension()``.

    Args:
        repo_snapshot_dir: Path to the downloaded HuggingFace snapshot.

    Returns:
        The embedding dimension, or ``None`` if it cannot be determined.
    """
    modules_json_path = os.path.join(repo_snapshot_dir, "modules.json")
    if not os.path.isfile(modules_json_path):
        return None

    try:
        with open(modules_json_path, encoding="utf-8") as f:
            modules = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    for module_entry in reversed(modules):
        module_type = module_entry.get("type", "")
        module_path = module_entry.get("path", "")
        if module_path:
            module_config_path = os.path.join(repo_snapshot_dir, module_path, "config.json")
        else:
            module_config_path = os.path.join(repo_snapshot_dir, "config.json")

        if not os.path.isfile(module_config_path):
            continue

        try:
            with open(module_config_path, encoding="utf-8") as f:
                module_config = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        embedding_dim = _get_module_embedding_dim_from_config(module_type, module_config)
        if embedding_dim is not None:
            return embedding_dim

    return None


def _resolve_wrapper_target_methods(target_methods: Optional[Sequence[str]]) -> Sequence[str]:
    """Resolve target methods for wrapper local-mode signature inference.

    Defaults to ``_DEFAULT_WRAPPER_TARGET_METHODS`` (sentence-transformers 5.x
    singular API). Uses the allowlist only; does not import sentence_transformers.

    Args:
        target_methods: Optional user-provided method names.

    Returns:
        Resolved method names to register.

    Raises:
        ValueError: If target_methods is empty or contains unsupported names.
    """
    if target_methods is None:
        return _DEFAULT_WRAPPER_TARGET_METHODS
    if not target_methods:
        raise ValueError("At least one target method must be specified.")
    unsupported = set(target_methods) - set(_ALLOWED_TARGET_METHODS)
    if unsupported:
        raise ValueError(
            f"Unsupported model methods: {sorted(unsupported)}. "
            f"SentenceTransformer model methods must be one of: {_ALLOWED_TARGET_METHODS}."
        )
    return target_methods


def _infer_signatures_for_target_methods(
    model_meta: model_meta_api.ModelMetadata,
    *,
    embedding_dim: int,
    target_methods: Sequence[str],
    batch_size: Optional[int],
    include_truncate_dim_param: bool = False,
) -> None:
    """Auto-infer and register signatures for each target method.

    Args:
        model_meta: Model metadata to update.
        embedding_dim: Output embedding dimension.
        target_methods: Method names to register.
        batch_size: Default batch size for inference.
        include_truncate_dim_param: Whether to add truncate_dim as a runtime param.

    Raises:
        ValueError: If signature inference fails for any method.
    """
    for target_method in target_methods:
        inferred_sig = _auto_infer_signature(
            target_method=target_method,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            include_truncate_dim_param=include_truncate_dim_param,
        )
        if inferred_sig is None:
            raise ValueError(
                f"Unable to auto-infer signature for method '{target_method}'. "
                "Please provide sample_input_data or signatures explicitly."
            )
        model_meta.signatures[target_method] = inferred_sig

    _validate_sentence_transformers_signatures(model_meta.signatures)


def _add_inference_params(
    model_meta: model_meta_api.ModelMetadata,
    batch_size: Optional[int],
    *,
    include_truncate_dim_param: bool = False,
) -> None:
    """Add batch_size and optionally truncate_dim as runtime params to all signatures in model_meta."""
    inference_params = [
        model_signature.ParamSpec(
            name="batch_size",
            dtype=model_signature.DataType.INT64,
            default_value=batch_size,
        ),
    ]
    if include_truncate_dim_param:
        inference_params.append(
            model_signature.ParamSpec(
                name="truncate_dim",
                dtype=model_signature.DataType.INT64,
                default_value=None,
            )
        )
    for method_name, sig in model_meta.signatures.items():
        params_to_add = [param for param in inference_params if not any(p.name == param.name for p in sig.params)]
        if not params_to_add:
            continue
        combined_params: list[model_signature.BaseParamSpec] = list(sig.params)
        combined_params.extend(params_to_add)
        model_meta.signatures[method_name] = model_signature.ModelSignature(
            inputs=sig.inputs,
            outputs=sig.outputs,
            params=combined_params,
        )


def _validate_sentence_transformers_signatures(
    sigs: dict[str, model_signature.ModelSignature],
) -> None:
    """Validate signatures for SentenceTransformer models.

    Args:
        sigs: Dictionary mapping method names to their signatures.

    Raises:
        ValueError: If signatures are empty, contain unsupported methods, or violate per-method constraints.
    """
    # Check that signatures are non-empty
    if not sigs:
        raise ValueError("At least one signature must be provided.")

    # Check that all methods are in the allowlist
    unsupported_methods = set(sigs.keys()) - set(_ALLOWED_TARGET_METHODS)
    if unsupported_methods:
        raise ValueError(
            f"Unsupported target methods: {sorted(unsupported_methods)}. "
            f"Supported methods are: {_ALLOWED_TARGET_METHODS}."
        )

    # Validate per-method constraints
    for method_name, sig in sigs.items():
        if len(sig.inputs) != 1:
            raise ValueError(f"SentenceTransformer method '{method_name}' must have exactly 1 input column.")

        if len(sig.outputs) != 1:
            raise ValueError(f"SentenceTransformer method '{method_name}' must have exactly 1 output column.")

        # FeatureSpec is expected here; FeatureGroupSpec would indicate a nested/grouped input
        # which SentenceTransformer does not support.
        if not isinstance(sig.inputs[0], model_signature.FeatureSpec):
            raise ValueError(
                f"SentenceTransformer method '{method_name}' requires a FeatureSpec input, "
                f"got {type(sig.inputs[0]).__name__}."
            )

        if sig.inputs[0]._shape is not None:
            raise ValueError(f"SentenceTransformer method '{method_name}' does not support input shape.")

        if sig.inputs[0]._dtype != model_signature.DataType.STRING:
            raise ValueError(
                f"SentenceTransformer method '{method_name}' only accepts STRING input, "
                f"got {sig.inputs[0]._dtype.name}."
            )


@final
class SentenceTransformerHandler(_base.BaseModelHandler["sentence_transformers.SentenceTransformer"]):
    HANDLER_TYPE = "sentence_transformers"
    HANDLER_VERSION = "2024-03-15"
    _MIN_SNOWPARK_ML_VERSION = "1.43.0"
    _HANDLER_MIGRATOR_PLANS: dict[str, type[base_migrator.BaseModelHandlerMigrator]] = {}

    MODEL_PICKLE_FILE = "snowml_sentence_transformer.pkl"
    MODEL_BLOB_FILE_OR_DIR = "model"
    IS_AUTO_SIGNATURE = True

    @classmethod
    def can_handle(  # type: ignore[override]
        cls,
        model: model_types.SupportedModelType,
    ) -> TypeGuard[Union["sentence_transformers.SentenceTransformer", snowml_huggingface.SentenceTransformer,]]:
        if type_utils.LazyType("sentence_transformers.SentenceTransformer").isinstance(model):
            return True
        if isinstance(model, snowml_huggingface.SentenceTransformer):
            return True
        return False

    @classmethod
    def cast_model(  # type: ignore[override]
        cls,
        model: model_types.SupportedModelType,
    ) -> Union["sentence_transformers.SentenceTransformer", snowml_huggingface.SentenceTransformer]:
        if isinstance(model, snowml_huggingface.SentenceTransformer):
            return model
        import sentence_transformers

        assert isinstance(model, sentence_transformers.SentenceTransformer)
        return model

    @classmethod
    def save_model(
        cls,
        name: str,
        model: Union["sentence_transformers.SentenceTransformer", snowml_huggingface.SentenceTransformer],
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.SentenceTransformersSaveOptions],
    ) -> None:
        enable_explainability = kwargs.get("enable_explainability", False)
        if enable_explainability:
            raise NotImplementedError("Explainability is not supported for Sentence Transformer model.")

        user_defined_batch_size: Optional[int] = kwargs.get("batch_size", None)
        if user_defined_batch_size is not None and (
            not isinstance(user_defined_batch_size, int) or user_defined_batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        batch_size = user_defined_batch_size
        model_truncate_dim = _capture_model_truncate_dim(model)
        include_truncate_dim_param = _supports_encode_truncate_dim_param()
        target_methods = kwargs.pop("target_methods", None)
        is_wrapper = isinstance(model, snowml_huggingface.SentenceTransformer)

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)

        blob_options: model_meta_schema.SentenceTransformersModelBlobOptions = {}

        if is_wrapper:
            assert isinstance(model, snowml_huggingface.SentenceTransformer)
            cls._save_wrapper_model(
                model=model,
                model_meta=model_meta,
                model_blob_path=model_blob_path,
                batch_size=batch_size,
                is_sub_model=is_sub_model,
                target_methods=target_methods,
            )
            blob_options["model"] = model.model
            if model.compute_pool_for_log is None and model.repo_snapshot_dir is not None:
                blob_options["is_repo_downloaded"] = True
            if batch_size is not None:
                blob_options["batch_size"] = batch_size

        if not is_wrapper:
            import sentence_transformers

            assert isinstance(model, sentence_transformers.SentenceTransformer)
            real_model = model
            if not is_sub_model:
                model_meta = cls._validate_and_set_signatures(
                    model=real_model,
                    model_meta=model_meta,
                    sample_input_data=sample_input_data,
                    batch_size=batch_size,
                    include_truncate_dim_param=include_truncate_dim_param,
                    has_user_defined_batch_size=user_defined_batch_size is not None,
                    target_methods=target_methods,
                )

            save_path = os.path.join(model_blob_path, cls.MODEL_BLOB_FILE_OR_DIR)
            real_model.save(save_path)
            handlers_utils.save_transformers_config_with_auto_map(save_path)

            pretrained_model_name = _capture_pretrained_model_name(real_model)
            if pretrained_model_name is not None:
                blob_options["model"] = pretrained_model_name
            if model_truncate_dim is not None:
                blob_options["truncate_dim"] = model_truncate_dim
            if batch_size is not None:
                blob_options["batch_size"] = batch_size
        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODEL_BLOB_FILE_OR_DIR,
            options=blob_options,
        )
        model_meta.models[name] = base_meta
        model_meta.min_snowpark_ml_version = cls._MIN_SNOWPARK_ML_VERSION

        model_meta.env.include_if_absent(
            [
                model_env.ModelDependency(
                    requirement="sentence-transformers",
                    pip_name="sentence-transformers",
                ),
                model_env.ModelDependency(requirement="transformers", pip_name="transformers"),
                model_env.ModelDependency(requirement="tokenizers", pip_name="tokenizers"),
                model_env.ModelDependency(requirement="pytorch", pip_name="torch"),
            ],
            check_local_version=not is_wrapper,
        )
        model_meta.env.cuda_version = kwargs.get("cuda_version", handlers_utils.get_default_cuda_version())

    @classmethod
    def _save_wrapper_model(
        cls,
        model: snowml_huggingface.SentenceTransformer,
        model_meta: model_meta_api.ModelMetadata,
        model_blob_path: str,
        batch_size: Optional[int],
        *,
        is_sub_model: Optional[bool] = False,
        target_methods: Optional[Sequence[str]] = None,
    ) -> None:
        """Save a SentenceTransformer wrapper using pickle + copytree.

        Args:
            model: The wrapper instance.
            model_meta: Model metadata to update with signatures.
            model_blob_path: Base path for model blobs.
            batch_size: Batch size for inference.
            is_sub_model: Whether this is a sub-model.
            target_methods: Optional list of target methods for signature inference.
        """
        model_blob_file_or_dir = os.path.join(model_blob_path, cls.MODEL_BLOB_FILE_OR_DIR)
        os.makedirs(model_blob_file_or_dir, exist_ok=True)

        pickle_file = os.path.join(model_blob_file_or_dir, cls.MODEL_PICKLE_FILE)
        with open(pickle_file, "wb") as f:
            cloudpickle.dump(model, f)

        if model.compute_pool_for_log is None and model.repo_snapshot_dir:
            logger.info("Wrapper repo_snapshot_dir is available, copying snapshot")
            shutil.copytree(
                model.repo_snapshot_dir,
                model_blob_file_or_dir,
                dirs_exist_ok=True,
            )

        if not is_sub_model and model.compute_pool_for_log is None:
            cls._set_signatures_from_config(
                model_meta=model_meta,
                repo_snapshot_dir=model.repo_snapshot_dir,
                batch_size=batch_size,
                target_methods=target_methods,
            )

    @classmethod
    def _set_signatures_from_config(
        cls,
        model_meta: model_meta_api.ModelMetadata,
        repo_snapshot_dir: Optional[str],
        batch_size: Optional[int],
        *,
        target_methods: Optional[Sequence[str]] = None,
    ) -> None:
        """Set auto-inferred signatures using config files from a local snapshot.

        Used when saving a ``SentenceTransformer`` wrapper in local logging mode
        (``compute_pool_for_log=None``) where we cannot call methods on a live
        model instance.

        Args:
            model_meta: Model metadata to update.
            repo_snapshot_dir: Path to the locally downloaded snapshot.
            batch_size: Batch size for inference.
            target_methods: Optional list of target methods to register. Defaults to
                ``_DEFAULT_WRAPPER_TARGET_METHODS``.

        Raises:
            ValueError: If the embedding dimension cannot be determined.
        """
        if model_meta.signatures:
            return

        if repo_snapshot_dir is None:
            raise ValueError(
                "Unable to determine the model's embedding dimension from snapshot config files. "
                "Please provide sample_input_data or signatures explicitly."
            )

        embedding_dim = _get_embedding_dim_from_config(repo_snapshot_dir)
        if embedding_dim is None:
            raise ValueError(
                "Unable to determine the model's embedding dimension from snapshot config files. "
                "Please provide sample_input_data or signatures explicitly."
            )

        resolved_target_methods = _resolve_wrapper_target_methods(target_methods)
        _infer_signatures_for_target_methods(
            model_meta,
            embedding_dim=embedding_dim,
            target_methods=resolved_target_methods,
            batch_size=batch_size,
            include_truncate_dim_param=True,
        )

    @classmethod
    def _validate_and_set_signatures(
        cls,
        model: "sentence_transformers.SentenceTransformer",
        model_meta: model_meta_api.ModelMetadata,
        sample_input_data: Optional[model_types.SupportedDataType],
        batch_size: Optional[int],
        target_methods: Optional[Sequence[str]],
        *,
        include_truncate_dim_param: bool = False,
        has_user_defined_batch_size: bool = False,
    ) -> model_meta_api.ModelMetadata:
        """Validate target methods and set signatures on model_meta.

        This method handles three cases:
        1. User provided explicit signatures - validate them
        2. User provided sample_input_data - infer signatures from data
        3. Neither provided - auto-infer signatures from model embedding dimension

        Args:
            model: The SentenceTransformer model.
            model_meta: Model metadata to update with signatures.
            sample_input_data: Optional sample input data for signature inference.
            batch_size: Batch size for model inference.
            target_methods: Optional list of target methods to use.
            include_truncate_dim_param: Whether to add truncate_dim as a runtime param (default None).
            has_user_defined_batch_size: Whether batch_size was explicitly provided by the caller.

        Returns:
            Updated model metadata with validated signatures.

        Raises:
            ValueError: If no target methods are specified, if target methods are not in the
                allowed list, if a target method is not callable on the model, or if
                auto-inference fails.
        """
        # Get available default methods by checking which methods exist on the model
        available_default_methods = _get_available_default_methods(model)

        # get_target_methods will filter to only callable methods
        resolved_target_methods = handlers_utils.get_target_methods(
            model=model,
            target_methods=target_methods,
            default_target_methods=available_default_methods,
        )

        if not resolved_target_methods:
            raise ValueError("At least one target method must be specified.")

        if not set(resolved_target_methods).issubset(_ALLOWED_TARGET_METHODS):
            unsupported = set(resolved_target_methods) - set(_ALLOWED_TARGET_METHODS)
            raise ValueError(
                f"Unsupported model methods: {sorted(unsupported)}. "
                f"SentenceTransformer model methods must be one of: {_ALLOWED_TARGET_METHODS}."
            )

        embedding_dim = model.get_sentence_embedding_dimension()

        def get_prediction(
            target_method_name: str,
            input_data: model_types.SupportedLocalDataType,
        ) -> model_types.SupportedLocalDataType:
            if not isinstance(input_data, pd.DataFrame):
                input_data = model_signature._convert_local_data_to_df(data=input_data)

            if input_data.shape[1] != 1:
                raise ValueError("SentenceTransformer can only accept 1 input column when converted to pd.DataFrame")
            X_list = input_data.iloc[:, 0].tolist()

            method_to_call = getattr(model, target_method_name, None)
            if not callable(method_to_call):
                raise ValueError(f"SentenceTransformer model does not have a callable method '{target_method_name}'.")
            encode_kwargs: dict[str, Any] = {}
            if batch_size is not None:
                encode_kwargs["batch_size"] = batch_size
            return pd.DataFrame({0: method_to_call(X_list, **encode_kwargs).tolist()})

        # Case 1: User provided explicit signatures
        if model_meta.signatures:
            if has_user_defined_batch_size:
                logger.warning(
                    "batch_size option will not be added as a configurable parameter to explicit signatures. "
                    "To make batch_size configurable at runtime, add a batch_size ParamSpec to your signature."
                )
            handlers_utils.validate_target_methods(model, list(model_meta.signatures.keys()))
            model_meta = handlers_utils.validate_signature(
                model=model,
                model_meta=model_meta,
                target_methods=resolved_target_methods,
                sample_input_data=sample_input_data,
                get_prediction_fn=get_prediction,
            )
            _validate_sentence_transformers_signatures(model_meta.signatures)
            return model_meta

        # Case 2: User provided sample_input_data - infer from data
        handlers_utils.validate_target_methods(model, resolved_target_methods)
        if sample_input_data is not None:
            model_meta = handlers_utils.validate_signature(
                model=model,
                model_meta=model_meta,
                target_methods=resolved_target_methods,
                sample_input_data=sample_input_data,
                get_prediction_fn=get_prediction,
            )
            _validate_sentence_transformers_signatures(model_meta.signatures)
            _add_inference_params(
                model_meta,
                batch_size,
                include_truncate_dim_param=include_truncate_dim_param,
            )
            return model_meta

        # Case 3: Auto-infer signature from model embedding dimension
        if embedding_dim is None:
            raise ValueError(
                "Unable to determine the model's embedding dimension. "
                "Please provide sample_input_data or signatures explicitly."
            )

        if not resolved_target_methods:
            raise ValueError(
                "No valid target methods found on the model. "
                "Please provide sample_input_data or signatures explicitly, "
                "or specify target_methods that exist on your model."
            )

        _infer_signatures_for_target_methods(
            model_meta,
            embedding_dim=embedding_dim,
            target_methods=resolved_target_methods,
            batch_size=batch_size,
            include_truncate_dim_param=include_truncate_dim_param,
        )
        return model_meta

    @staticmethod
    def _get_device_config(
        **kwargs: Unpack[model_types.SentenceTransformersLoadOptions],
    ) -> Optional[str]:
        if kwargs.get("device", None) is not None:
            return kwargs["device"]
        elif kwargs.get("use_gpu", False):
            return "cuda"

        return None

    @classmethod
    def load_model(
        cls,
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.SentenceTransformersLoadOptions],
    ) -> "sentence_transformers.SentenceTransformer":
        import sentence_transformers

        if snowpark_utils.is_in_stored_procedure():  # type: ignore[no-untyped-call]
            os.environ["TRANSFORMERS_CACHE"] = "/tmp"
            os.environ["HF_HOME"] = "/tmp"

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        model_blob_metadata = model_meta.models[name]
        model_blob_filename = model_blob_metadata.path
        model_blob_file_or_dir_path = os.path.join(model_blob_path, model_blob_filename)

        model_blob_options = cast(
            model_meta_schema.SentenceTransformersModelBlobOptions,
            model_blob_metadata.options,
        )
        is_repo_downloaded = model_blob_options.get("is_repo_downloaded", False)

        if is_repo_downloaded:
            return cls._load_from_wrapper_snapshot(
                model_blob_file_or_dir_path=model_blob_file_or_dir_path,
                **kwargs,
            )

        additional_kwargs: dict[str, Any] = {}
        if "trust_remote_code" in inspect.signature(sentence_transformers.SentenceTransformer).parameters:
            additional_kwargs["trust_remote_code"] = True

        load_truncate_dim = kwargs.get("truncate_dim", model_blob_options.get("truncate_dim"))
        if load_truncate_dim is not None and _supports_init_truncate_dim():
            additional_kwargs["truncate_dim"] = load_truncate_dim

        model = sentence_transformers.SentenceTransformer(
            model_blob_file_or_dir_path,
            device=cls._get_device_config(**kwargs),
            **additional_kwargs,
        )
        return model

    @classmethod
    def _load_from_wrapper_snapshot(
        cls,
        model_blob_file_or_dir_path: str,
        **kwargs: Unpack[model_types.SentenceTransformersLoadOptions],
    ) -> "sentence_transformers.SentenceTransformer":
        """Load a SentenceTransformer from a pickled wrapper + snapshot.

        Unpickles the wrapper to recover metadata (e.g. ``trust_remote_code``),
        then constructs a real ``SentenceTransformer`` from the snapshot
        directory.

        Args:
            model_blob_file_or_dir_path: Path to the blob directory containing
                the pickle file and snapshot.
            kwargs: Load options forwarded for device config.

        Returns:
            A live ``SentenceTransformer`` instance.
        """
        import sentence_transformers

        pickle_file = os.path.join(model_blob_file_or_dir_path, cls.MODEL_PICKLE_FILE)
        with open(pickle_file, "rb") as f:
            wrapper: snowml_huggingface.SentenceTransformer = cloudpickle.load(f)

        additional_kwargs: dict[str, object] = {}
        if "trust_remote_code" in inspect.signature(sentence_transformers.SentenceTransformer).parameters:
            additional_kwargs["trust_remote_code"] = bool(wrapper.trust_remote_code)

        return sentence_transformers.SentenceTransformer(
            model_blob_file_or_dir_path,
            device=cls._get_device_config(**kwargs),
            **additional_kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def convert_as_custom_model(
        cls,
        raw_model: "sentence_transformers.SentenceTransformer",
        model_meta: model_meta_api.ModelMetadata,
        background_data: Optional[pd.DataFrame] = None,
        **kwargs: Unpack[model_types.SentenceTransformersLoadOptions],
    ) -> custom_model.CustomModel:
        import sentence_transformers

        from snowflake.ml.model import custom_model

        def _create_custom_model(
            raw_model: "sentence_transformers.SentenceTransformer",
            model_meta: model_meta_api.ModelMetadata,
        ) -> type[custom_model.CustomModel]:
            blob_default_batch_size = cast(
                model_meta_schema.SentenceTransformersModelBlobOptions,
                model_meta.models[model_meta.name].options,
            ).get("batch_size", None)

            def get_prediction(
                raw_model: "sentence_transformers.SentenceTransformer",
                signature: model_signature.ModelSignature,
                target_method: str,
            ) -> Callable[..., pd.DataFrame]:
                # Capture target_method in closure to call the correct model method
                method_to_call = getattr(raw_model, target_method, None)
                if not callable(method_to_call):
                    raise ValueError(
                        f"SentenceTransformer model does not have a callable method '{target_method}'. "
                        f"This method may not be available in your version of sentence-transformers."
                    )

                # Prefer the signature's ParamSpec defaults (matches server-side resolution),
                # fall back to blob options for old models without a batch_size ParamSpec.
                param_defaults = {param.name: param.default_value for param in signature.params}
                method_batch_size = param_defaults.get("batch_size", blob_default_batch_size)
                has_truncate_dim_param = "truncate_dim" in param_defaults
                method_truncate_dim = param_defaults["truncate_dim"] if has_truncate_dim_param else None
                output_name = signature.outputs[0].name

                @custom_model._internal_inference_api
                def fn(
                    self: custom_model.CustomModel,
                    X: pd.DataFrame,
                    **kwargs: Any,
                ) -> pd.DataFrame:
                    X_list = X.iloc[:, 0].tolist()
                    encode_kwargs: dict[str, Any] = {}
                    # The handler imposes no batch-size policy of its own. When `batch_size` is unset
                    # it is omitted here so sentence-transformers applies its own default; callers that
                    # need a specific size set it via the `batch_size` log_model option, an
                    # inference-time `batch_size` parameter, or the serving layer (online/batch) that
                    # knows the resource bounds.
                    batch_size = kwargs.get("batch_size", method_batch_size)
                    if batch_size is not None:
                        encode_kwargs["batch_size"] = batch_size
                    if has_truncate_dim_param:
                        truncate_dim = kwargs.get("truncate_dim", method_truncate_dim)
                        if truncate_dim is not None:
                            encode_kwargs["truncate_dim"] = truncate_dim

                    return pd.DataFrame({output_name: method_to_call(X_list, **encode_kwargs).tolist()})

                return fn

            type_method_dict: dict[str, Any] = {"_allows_kwargs": True}
            for target_method_name, sig in model_meta.signatures.items():
                if target_method_name in _ALLOWED_TARGET_METHODS:
                    type_method_dict[target_method_name] = get_prediction(raw_model, sig, target_method_name)
                else:
                    raise ValueError(f"{target_method_name} is currently not supported.")

            _SentenceTransformer = type(
                "_SentenceTransformer",
                (custom_model.CustomModel,),
                type_method_dict,
            )
            return _SentenceTransformer

        assert isinstance(raw_model, sentence_transformers.SentenceTransformer)
        model = raw_model

        _SentenceTransformer = _create_custom_model(model, model_meta)
        sentence_transformers_model = _SentenceTransformer(custom_model.ModelContext())

        return sentence_transformers_model
