import inspect
import logging
import os
from typing import TYPE_CHECKING, Callable, Optional, cast, final

import pandas as pd
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
from snowflake.snowpark._internal import utils as snowpark_utils

if TYPE_CHECKING:
    import sentence_transformers

logger = logging.getLogger(__name__)


def _validate_sentence_transformers_signatures(sigs: dict[str, model_signature.ModelSignature]) -> None:
    if list(sigs.keys()) != ["encode"]:
        raise ValueError("target_methods can only be ['encode']")

    if len(sigs["encode"].inputs) != 1:
        raise ValueError("SentenceTransformer can only accept 1 input column")

    if len(sigs["encode"].outputs) != 1:
        raise ValueError("SentenceTransformer can only return 1 output column")

    assert isinstance(sigs["encode"].inputs[0], model_signature.FeatureSpec)

    if sigs["encode"].inputs[0]._shape is not None:
        raise ValueError("SentenceTransformer does not support input shape")

    if sigs["encode"].inputs[0]._dtype != model_signature.DataType.STRING:
        raise ValueError("SentenceTransformer only accepts string input")


@final
class SentenceTransformerHandler(_base.BaseModelHandler["sentence_transformers.SentenceTransformer"]):
    HANDLER_TYPE = "sentence_transformers"
    HANDLER_VERSION = "2024-03-15"
    _MIN_SNOWPARK_ML_VERSION = "1.3.1"
    _HANDLER_MIGRATOR_PLANS: dict[str, type[base_migrator.BaseModelHandlerMigrator]] = {}

    MODEL_BLOB_FILE_OR_DIR = "model"
    DEFAULT_TARGET_METHODS = ["encode"]

    @classmethod
    def can_handle(
        cls,
        model: model_types.SupportedModelType,
    ) -> TypeGuard["sentence_transformers.SentenceTransformer"]:
        if type_utils.LazyType("sentence_transformers.SentenceTransformer").isinstance(model):
            return True
        return False

    @classmethod
    def cast_model(
        cls,
        model: model_types.SupportedModelType,
    ) -> "sentence_transformers.SentenceTransformer":
        import sentence_transformers

        assert isinstance(model, sentence_transformers.SentenceTransformer)
        return cast(sentence_transformers.SentenceTransformer, model)

    @classmethod
    def save_model(
        cls,
        name: str,
        model: "sentence_transformers.SentenceTransformer",
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.SentenceTransformersSaveOptions],  # registry.log_model(options={...})
    ) -> None:
        enable_explainability = kwargs.get("enable_explainability", False)
        if enable_explainability:
            raise NotImplementedError("Explainability is not supported for Sentence Transformer model.")

        batch_size = kwargs.get("batch_size", 32)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        # Validate target methods and signature (if possible)
        if not is_sub_model:
            target_methods = handlers_utils.get_target_methods(
                model=model,
                target_methods=kwargs.pop("target_methods", None),
                default_target_methods=cls.DEFAULT_TARGET_METHODS,
            )
            if target_methods != ["encode"]:
                raise ValueError("target_methods can only be ['encode']")

            def get_prediction(
                target_method_name: str, sample_input_data: model_types.SupportedLocalDataType
            ) -> model_types.SupportedLocalDataType:
                if not isinstance(sample_input_data, pd.DataFrame):
                    sample_input_data = model_signature._convert_local_data_to_df(data=sample_input_data)

                if sample_input_data.shape[1] != 1:
                    raise ValueError(
                        "SentenceTransformer can only accept 1 input column when converted to pd.DataFrame"
                    )
                X_list = sample_input_data.iloc[:, 0].tolist()

                assert callable(getattr(model, "encode", None))
                return pd.DataFrame({0: model.encode(X_list, batch_size=batch_size).tolist()})

            if model_meta.signatures:
                handlers_utils.validate_target_methods(model, list(model_meta.signatures.keys()))
                model_meta = handlers_utils.validate_signature(
                    model=model,
                    model_meta=model_meta,
                    target_methods=target_methods,
                    sample_input_data=sample_input_data,
                    get_prediction_fn=get_prediction,
                )
            else:
                handlers_utils.validate_target_methods(model, target_methods)  # DEFAULT_TARGET_METHODS only
                if sample_input_data is not None:
                    model_meta = handlers_utils.validate_signature(
                        model=model,
                        model_meta=model_meta,
                        target_methods=target_methods,
                        sample_input_data=sample_input_data,
                        get_prediction_fn=get_prediction,
                    )

            _validate_sentence_transformers_signatures(model_meta.signatures)

        # save model
        model_blob_path = os.path.join(model_blobs_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)
        save_path = os.path.join(model_blob_path, cls.MODEL_BLOB_FILE_OR_DIR)
        model.save(save_path)
        handlers_utils.save_transformers_config_with_auto_map(
            save_path,
        )

        # save model metadata
        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODEL_BLOB_FILE_OR_DIR,
            options=model_meta_schema.SentenceTransformersModelBlobOptions(batch_size=batch_size),
        )
        model_meta.models[name] = base_meta
        model_meta.min_snowpark_ml_version = cls._MIN_SNOWPARK_ML_VERSION

        model_meta.env.include_if_absent(
            [
                model_env.ModelDependency(requirement="sentence-transformers", pip_name="sentence-transformers"),
                model_env.ModelDependency(requirement="transformers", pip_name="transformers"),
                model_env.ModelDependency(requirement="pytorch", pip_name="torch"),
            ],
            check_local_version=True,
        )
        model_meta.env.cuda_version = kwargs.get("cuda_version", handlers_utils.get_default_cuda_version())

    @staticmethod
    def _get_device_config(**kwargs: Unpack[model_types.SentenceTransformersLoadOptions]) -> Optional[str]:
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
        **kwargs: Unpack[model_types.SentenceTransformersLoadOptions],  # use_gpu
    ) -> "sentence_transformers.SentenceTransformer":
        import sentence_transformers

        if snowpark_utils.is_in_stored_procedure():  # type: ignore[no-untyped-call]
            # We need to redirect the same folders to a writable location in the sandbox.
            os.environ["TRANSFORMERS_CACHE"] = "/tmp"
            os.environ["HF_HOME"] = "/tmp"

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        model_blobs_metadata = model_meta.models
        model_blob_metadata = model_blobs_metadata[name]
        model_blob_filename = model_blob_metadata.path
        model_blob_file_or_dir_path = os.path.join(model_blob_path, model_blob_filename)

        additional_kwargs = {}
        if "trust_remote_code" in inspect.signature(sentence_transformers.SentenceTransformer).parameters:
            additional_kwargs["trust_remote_code"] = True

        model = sentence_transformers.SentenceTransformer(
            model_blob_file_or_dir_path,
            device=cls._get_device_config(**kwargs),
            **additional_kwargs,
        )
        return model

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
            batch_size = cast(
                model_meta_schema.SentenceTransformersModelBlobOptions, model_meta.models[model_meta.name].options
            ).get("batch_size", None)

            def get_prediction(
                raw_model: "sentence_transformers.SentenceTransformer",
                signature: model_signature.ModelSignature,
                target_method: str,
            ) -> Callable[[custom_model.CustomModel, pd.DataFrame], pd.DataFrame]:
                @custom_model.inference_api
                def fn(self: custom_model.CustomModel, X: pd.DataFrame) -> pd.DataFrame:
                    X_list = X.iloc[:, 0].tolist()

                    return pd.DataFrame(
                        {signature.outputs[0].name: raw_model.encode(X_list, batch_size=batch_size).tolist()}
                    )

                return fn

            type_method_dict = {}
            for target_method_name, sig in model_meta.signatures.items():
                if target_method_name == "encode":
                    type_method_dict[target_method_name] = get_prediction(raw_model, sig, target_method_name)
                else:
                    ValueError(f"{target_method_name} is currently not supported.")

            _SentenceTransformer = type(
                "_SentenceTransformer",
                (custom_model.CustomModel,),
                type_method_dict,
            )
            return _SentenceTransformer

        assert isinstance(raw_model, sentence_transformers.SentenceTransformer)
        model = raw_model

        _SentenceTransformer = _create_custom_model(model, model_meta)
        sentence_transformers_SentenceTransformer_model = _SentenceTransformer(custom_model.ModelContext())
        predict_method = getattr(sentence_transformers_SentenceTransformer_model, "encode", None)
        assert callable(predict_method)
        return sentence_transformers_SentenceTransformer_model
