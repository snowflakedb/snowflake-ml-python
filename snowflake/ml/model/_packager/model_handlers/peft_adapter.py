import json
import os
import shutil
import tempfile
from typing import NoReturn, Optional, cast, final

import pandas as pd
from typing_extensions import TypeGuard, Unpack

from snowflake.ml._internal import platform_capabilities
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml.model import custom_model, type_hints as model_types
from snowflake.ml.model._packager.model_handlers import _base
from snowflake.ml.model._packager.model_handlers_migrator import base_migrator
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta as model_meta_api,
    model_meta_schema,
)
from snowflake.ml.model.models import huggingface as peft_adapter_type

_ADAPTER_CONFIG_FILE = "adapter_config.json"
_STAGE_OR_SNOW_PREFIXES = ("@", "snow://")


def _invalid_argument(message: str) -> NoReturn:
    raise snowml_exceptions.SnowflakeMLException(
        error_code=error_codes.INVALID_ARGUMENT,
        original_exception=ValueError(message),
    )


def _require_lora_adapters_enabled() -> None:
    if not platform_capabilities.PlatformCapabilities.get_instance().is_lora_adapters_enabled():
        _invalid_argument("PEFT adapter logging is unavailable because ENABLE_LORA_ADAPTERS is not enabled.")


def _blob_options_from_config(config_path: str) -> model_meta_schema.PeftAdapterModelBlobOptions:
    options = cast(model_meta_schema.PeftAdapterModelBlobOptions, {})
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return options
    if not isinstance(config, dict):
        return options

    peft_type = config.get("peft_type")
    if isinstance(peft_type, str) and peft_type:
        options["peft_type"] = peft_type.lower()

    rank = config.get("r")
    if isinstance(rank, int) and not isinstance(rank, bool) and rank > 0:
        options["lora_rank"] = rank

    if "lora_extra_vocab_size" in config:
        extra_vocab = config["lora_extra_vocab_size"]
        if isinstance(extra_vocab, int) and not isinstance(extra_vocab, bool):
            options["lora_extra_vocab_size"] = extra_vocab
    return options


def _join_subfolder(root: str, subfolder: Optional[str]) -> str:
    if subfolder is None:
        return root
    return os.path.join(root, subfolder)


def _is_regular_file(path: str) -> bool:
    return os.path.isfile(path) and not os.path.islink(path)


def _copy_top_level_files(*, source_dir: str, dest_dir: str) -> None:
    config_path = os.path.join(source_dir, _ADAPTER_CONFIG_FILE)
    if not _is_regular_file(config_path):
        _invalid_argument(
            f"Adapter directory is missing required file {_ADAPTER_CONFIG_FILE}. "
            f"Point adapter_path at the directory that contains {_ADAPTER_CONFIG_FILE}, "
            f"or set subfolder= to that directory."
        )

    os.makedirs(dest_dir, exist_ok=True)
    for filename in os.listdir(source_dir):
        src_path = os.path.join(source_dir, filename)
        if not _is_regular_file(src_path):
            continue
        shutil.copy2(src_path, os.path.join(dest_dir, filename))


def _snapshot_download_adapter_repo(
    *,
    repo_id: str,
    revision: Optional[str],
    token: Optional[str],
    local_dir: str,
    allow_patterns: Optional[str] = None,
) -> str:
    import huggingface_hub

    return huggingface_hub.snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
    )


@final
class PeftAdapterModelHandler(_base.BaseModelHandler["peft_adapter_type.PeftAdapter"]):
    """Handler for PEFT adapters logged via ``PeftAdapter``."""

    HANDLER_TYPE = "peft_adapter"
    HANDLER_VERSION = "2026-08-01"
    # TODO(SNOW-4000665): set to the first snowflake-ml-python release that ships PeftAdapter.
    _MIN_SNOWPARK_ML_VERSION = "1.52.0"
    _HANDLER_MIGRATOR_PLANS: dict[str, type[base_migrator.BaseModelHandlerMigrator]] = {}

    MODEL_BLOB_FILE_OR_DIR = "adapter"
    DEFAULT_TARGET_METHODS: list[str] = []

    @classmethod
    def can_handle(cls, model: model_types.SupportedModelType) -> TypeGuard["peft_adapter_type.PeftAdapter"]:
        return isinstance(model, peft_adapter_type.PeftAdapter)

    @classmethod
    def cast_model(cls, model: model_types.SupportedModelType) -> "peft_adapter_type.PeftAdapter":
        if not isinstance(model, peft_adapter_type.PeftAdapter):
            raise ValueError(f"Expected PeftAdapter, got {type(model).__name__}.")
        return model

    @classmethod
    def save_model(
        cls,
        name: str,
        model: "peft_adapter_type.PeftAdapter",
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.PeftAdapterSaveOptions],
    ) -> None:
        _require_lora_adapters_enabled()

        if sample_input_data is not None:
            _invalid_argument(
                "sample_input_data is not supported when logging a PEFT adapter; "
                "signatures are copied at commit from the pin."
            )
        if model_meta._signatures_supplied:
            _invalid_argument(
                "signatures= is not supported when logging a PEFT adapter; "
                "signatures are copied at commit from the pin."
            )

        dest_dir = os.path.join(model_blobs_dir_path, name, cls.MODEL_BLOB_FILE_OR_DIR)
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = cls._materialize_adapter_dir(model, tmpdir)
            _copy_top_level_files(source_dir=source_dir, dest_dir=dest_dir)
            config_path = os.path.join(source_dir, _ADAPTER_CONFIG_FILE)
            config_options = _blob_options_from_config(config_path)

        blob_options: model_meta_schema.PeftAdapterModelBlobOptions = {
            "base_model_name": model.base_model.fully_qualified_model_name,
            "base_model_version": model.base_model.version_name,
        }
        blob_options.update(config_options)

        model_meta.models[name] = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODEL_BLOB_FILE_OR_DIR,
            options=blob_options,
        )
        model_meta.min_snowpark_ml_version = cls._MIN_SNOWPARK_ML_VERSION

    @classmethod
    def _materialize_adapter_dir(cls, model: peft_adapter_type.PeftAdapter, tmpdir: str) -> str:
        if model.adapter_path is not None:
            if model.adapter_path.startswith(_STAGE_OR_SNOW_PREFIXES):
                _invalid_argument("adapter_path starting with '@' or 'snow://' is not supported yet.")
            selected_root = _join_subfolder(model.adapter_path, model.subfolder)
            if not os.path.isdir(selected_root):
                _invalid_argument(f"adapter_path {selected_root!r} is not a local directory.")
            return selected_root

        if model.adapter_repo is not None:
            allow_patterns = f"{model.subfolder}/*" if model.subfolder is not None else None
            _snapshot_download_adapter_repo(
                repo_id=model.adapter_repo,
                revision=model.revision,
                token=model.token,
                local_dir=tmpdir,
                allow_patterns=allow_patterns,
            )
            return _join_subfolder(tmpdir, model.subfolder)

        assert model.adapter is not None
        model.adapter.save_pretrained(tmpdir)
        return tmpdir

    @classmethod
    def load_model(
        cls,
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.PeftAdapterLoadOptions],
    ) -> "peft_adapter_type.PeftAdapter":
        _invalid_argument("Adapters cannot be loaded into memory for warehouse run.")

    @classmethod
    def convert_as_custom_model(
        cls,
        raw_model: "peft_adapter_type.PeftAdapter",
        model_meta: model_meta_api.ModelMetadata,
        background_data: Optional[pd.DataFrame] = None,
        **kwargs: Unpack[model_types.PeftAdapterLoadOptions],
    ) -> custom_model.CustomModel:
        _invalid_argument("Adapters cannot be converted for warehouse run.")
