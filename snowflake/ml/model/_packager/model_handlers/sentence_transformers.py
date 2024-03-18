import logging
import os
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type, cast, final

import cloudpickle
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
)
from snowflake.ml.model._signatures import utils as model_signature_utils
from snowflake.snowpark._internal import utils as snowpark_utils

if TYPE_CHECKING:
    import sentence_transformers

logger = logging.getLogger(__name__)


@final
class SentenceTransformerHandler(_base.BaseModelHandler["sentence_transformers.SentenceTransformer"]):
    HANDLER_TYPE = "sentence_transformers"
    HANDLER_VERSION = "2024-03-15"
    _MIN_SNOWPARK_ML_VERSION = "1.3.1"
    _HANDLER_MIGRATOR_PLANS: Dict[str, Type[base_migrator.BaseModelHandlerMigrator]] = {}

    MODELE_BLOB_FILE_OR_DIR = "model"
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
        # Validate target methods and signature (if possible)
        if not is_sub_model:
            target_methods = handlers_utils.get_target_methods(
                model=model,
                target_methods=kwargs.pop("target_methods", None),
                default_target_methods=cls.DEFAULT_TARGET_METHODS,
            )
            assert target_methods == ["encode"], "target_methods can only be ['encode']"

            def get_prediction(
                target_method_name: str, sample_input_data: model_types.SupportedLocalDataType
            ) -> model_types.SupportedLocalDataType:
                return _sentence_transformer_encode(model, sample_input_data)

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

        # save model
        model_blob_path = os.path.join(model_blobs_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)
        model.save(os.path.join(model_blob_path, cls.MODELE_BLOB_FILE_OR_DIR))

        # save model metadata
        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODELE_BLOB_FILE_OR_DIR,
        )
        model_meta.models[name] = base_meta
        model_meta.min_snowpark_ml_version = cls._MIN_SNOWPARK_ML_VERSION

        model_meta.env.include_if_absent(
            [
                model_env.ModelDependency(requirement="sentence-transformers", pip_name="sentence-transformers"),
            ],
            check_local_version=True,
        )

    @classmethod
    def load_model(
        cls,
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.ModelLoadOption],  # use_gpu
    ) -> "sentence_transformers.SentenceTransformer":
        import sentence_transformers

        if snowpark_utils.is_in_stored_procedure():  # type: ignore[no-untyped-call]
            # We need to redirect the same folders to a writable location in the sandbox.
            os.environ["TRANSFORMERS_CACHE"] = "/tmp"

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        model_blobs_metadata = model_meta.models
        model_blob_metadata = model_blobs_metadata[name]
        model_blob_filename = model_blob_metadata.path
        model_blob_file_or_dir_path = os.path.join(model_blob_path, model_blob_filename)

        if os.path.isdir(model_blob_file_or_dir_path):  # if the saved model is a directory
            model = sentence_transformers.SentenceTransformer(model_blob_file_or_dir_path)
        else:
            assert os.path.isfile(model_blob_file_or_dir_path)  # if the saved model is a file
            with open(model_blob_file_or_dir_path, "rb") as f:
                model = cloudpickle.load(f)
            assert isinstance(model, sentence_transformers.SentenceTransformer)
        return model

    @classmethod
    def convert_as_custom_model(
        cls,
        raw_model: "sentence_transformers.SentenceTransformer",
        model_meta: model_meta_api.ModelMetadata,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> custom_model.CustomModel:
        import sentence_transformers

        from snowflake.ml.model import custom_model

        def _create_custom_model(
            raw_model: "sentence_transformers.SentenceTransformer",
            model_meta: model_meta_api.ModelMetadata,
        ) -> Type[custom_model.CustomModel]:
            def get_prediction(
                raw_model: "sentence_transformers.SentenceTransformer",
                signature: model_signature.ModelSignature,
                target_method: str,
            ) -> Callable[[custom_model.CustomModel, pd.DataFrame], pd.DataFrame]:
                @custom_model.inference_api
                def fn(self: custom_model.CustomModel, X: pd.DataFrame) -> pd.DataFrame:
                    predictions_df = _sentence_transformer_encode(raw_model, X)
                    return model_signature_utils.rename_pandas_df(predictions_df, signature.outputs)

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


def _sentence_transformer_encode(
    model: "sentence_transformers.SentenceTransformer", X: model_types.SupportedLocalDataType
) -> model_types.SupportedLocalDataType:

    if not isinstance(X, pd.DataFrame):
        X = model_signature._convert_local_data_to_df(X)

    assert X.shape[1] == 1, "SentenceTransformer can only accept 1 input column when converted to pd.DataFrame"
    X_list = X.iloc[:, 0].tolist()

    assert callable(getattr(model, "encode", None))
    return pd.DataFrame({0: model.encode(X_list, batch_size=X.shape[0]).tolist()})
