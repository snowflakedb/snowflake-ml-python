import os
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type, cast, final

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
from snowflake.ml.model._signatures import (
    pytorch_handler,
    utils as model_signature_utils,
)

if TYPE_CHECKING:
    import torch


@final
class TorchScriptHandler(_base.BaseModelHandler["torch.jit.ScriptModule"]):  # type:ignore[name-defined]
    """Handler for PyTorch JIT based model.

    Currently torch.jit.ScriptModule based classes are supported.
    """

    HANDLER_TYPE = "torchscript"
    HANDLER_VERSION = "2023-12-01"
    _MIN_SNOWPARK_ML_VERSION = "1.0.12"
    _HANDLER_MIGRATOR_PLANS: Dict[str, Type[base_migrator.BaseModelHandlerMigrator]] = {}

    MODELE_BLOB_FILE_OR_DIR = "model.pt"
    DEFAULT_TARGET_METHODS = ["forward"]

    @classmethod
    def can_handle(
        cls,
        model: model_types.SupportedModelType,
    ) -> TypeGuard["torch.jit.ScriptModule"]:  # type:ignore[name-defined]
        return type_utils.LazyType("torch.jit.ScriptModule").isinstance(model)

    @classmethod
    def cast_model(
        cls,
        model: model_types.SupportedModelType,
    ) -> "torch.jit.ScriptModule":  # type:ignore[name-defined]
        import torch

        assert isinstance(model, torch.jit.ScriptModule)  # type:ignore[attr-defined]

        return cast(torch.jit.ScriptModule, model)  # type:ignore[name-defined]

    @classmethod
    def save_model(
        cls,
        name: str,
        model: "torch.jit.ScriptModule",  # type:ignore[name-defined]
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.TorchScriptSaveOptions],
    ) -> None:
        import torch

        assert isinstance(model, torch.jit.ScriptModule)  # type:ignore[attr-defined]

        if not is_sub_model:
            target_methods = handlers_utils.get_target_methods(
                model=model,
                target_methods=kwargs.pop("target_methods", None),
                default_target_methods=cls.DEFAULT_TARGET_METHODS,
            )

            def get_prediction(
                target_method_name: str, sample_input: "model_types.SupportedLocalDataType"
            ) -> model_types.SupportedLocalDataType:
                if not pytorch_handler.SeqOfPyTorchTensorHandler.can_handle(sample_input):
                    sample_input = pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(
                        model_signature._convert_local_data_to_df(sample_input)
                    )

                model.eval()
                target_method = getattr(model, target_method_name, None)
                assert callable(target_method)
                with torch.no_grad():
                    predictions_df = target_method(*sample_input)

                if isinstance(predictions_df, torch.Tensor):
                    predictions_df = [predictions_df]

                return predictions_df

            model_meta = handlers_utils.validate_signature(
                model=model,
                model_meta=model_meta,
                target_methods=target_methods,
                sample_input=sample_input,
                get_prediction_fn=get_prediction,
            )

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)
        with open(os.path.join(model_blob_path, cls.MODELE_BLOB_FILE_OR_DIR), "wb") as f:
            torch.jit.save(model, f)  # type:ignore[attr-defined]
        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODELE_BLOB_FILE_OR_DIR,
        )
        model_meta.models[name] = base_meta
        model_meta.min_snowpark_ml_version = cls._MIN_SNOWPARK_ML_VERSION

        model_meta.env.include_if_absent(
            [model_env.ModelDependency(requirement="pytorch", pip_name="torch")], check_local_version=True
        )
        model_meta.env.cuda_version = kwargs.get("cuda_version", model_env.DEFAULT_CUDA_VERSION)

    @classmethod
    def load_model(
        cls,
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> "torch.jit.ScriptModule":  # type:ignore[name-defined]
        import torch

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        model_blobs_metadata = model_meta.models
        model_blob_metadata = model_blobs_metadata[name]
        model_blob_filename = model_blob_metadata.path
        with open(os.path.join(model_blob_path, model_blob_filename), "rb") as f:
            m = torch.jit.load(  # type:ignore[attr-defined]
                f, map_location="cuda" if kwargs.get("use_gpu", False) else "cpu"
            )
        assert isinstance(m, torch.jit.ScriptModule)  # type:ignore[attr-defined]

        if kwargs.get("use_gpu", False):
            m = m.cuda()

        return m

    @classmethod
    def convert_as_custom_model(
        cls,
        raw_model: "torch.jit.ScriptModule",  # type:ignore[name-defined]
        model_meta: model_meta_api.ModelMetadata,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> custom_model.CustomModel:
        from snowflake.ml.model import custom_model

        def _create_custom_model(
            raw_model: "torch.jit.ScriptModule",  # type:ignore[name-defined]
            model_meta: model_meta_api.ModelMetadata,
        ) -> Type[custom_model.CustomModel]:
            def fn_factory(
                raw_model: "torch.jit.ScriptModule",  # type:ignore[name-defined]
                signature: model_signature.ModelSignature,
                target_method: str,
            ) -> Callable[[custom_model.CustomModel, pd.DataFrame], pd.DataFrame]:
                @custom_model.inference_api
                def fn(self: custom_model.CustomModel, X: pd.DataFrame) -> pd.DataFrame:
                    if X.isnull().any(axis=None):
                        raise ValueError("Tensor cannot handle null values.")

                    import torch

                    raw_model.eval()

                    t = pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(X, signature.inputs)

                    if kwargs.get("use_gpu", False):
                        t = [element.cuda() for element in t]

                    with torch.no_grad():
                        res = getattr(raw_model, target_method)(*t)

                    if isinstance(res, torch.Tensor):
                        res = [res]

                    return model_signature_utils.rename_pandas_df(
                        data=pytorch_handler.SeqOfPyTorchTensorHandler.convert_to_df(res), features=signature.outputs
                    )

                return fn

            type_method_dict = {}
            for target_method_name, sig in model_meta.signatures.items():
                type_method_dict[target_method_name] = fn_factory(raw_model, sig, target_method_name)

            _TorchScriptModel = type(
                "_TorchScriptModel",
                (custom_model.CustomModel,),
                type_method_dict,
            )

            return _TorchScriptModel

        _TorchScriptModel = _create_custom_model(raw_model, model_meta)
        torchscript_model = _TorchScriptModel(custom_model.ModelContext())

        return torchscript_model
