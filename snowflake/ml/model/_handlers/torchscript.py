import os
from typing import TYPE_CHECKING, Callable, Optional, Type, cast

import pandas as pd
from typing_extensions import TypeGuard, Unpack

from snowflake.ml._internal import type_utils
from snowflake.ml.model import (
    _model_meta as model_meta_api,
    custom_model,
    model_signature,
    type_hints as model_types,
)
from snowflake.ml.model._handlers import _base
from snowflake.ml.model._signatures import (
    pytorch_handler,
    utils as model_signature_utils,
)

if TYPE_CHECKING:
    import torch


class _TorchScriptHandler(_base._ModelHandler["torch.jit.ScriptModule"]):  # type:ignore[name-defined]
    """Handler for PyTorch JIT based model.

    Currently torch.jit.ScriptModule based classes are supported.
    """

    handler_type = "torchscript"
    MODEL_BLOB_FILE = "model.pt"
    DEFAULT_TARGET_METHODS = ["forward"]

    @staticmethod
    def can_handle(
        model: model_types.SupportedModelType,
    ) -> TypeGuard["torch.jit.ScriptModule"]:  # type:ignore[name-defined]
        return type_utils.LazyType("torch.jit.ScriptModule").isinstance(model)

    @staticmethod
    def cast_model(
        model: model_types.SupportedModelType,
    ) -> "torch.jit.ScriptModule":  # type:ignore[name-defined]
        import torch

        assert isinstance(model, torch.jit.ScriptModule)  # type:ignore[attr-defined]

        return cast(torch.jit.ScriptModule, model)  # type:ignore[name-defined]

    @staticmethod
    def _save_model(
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
            target_methods = model_meta_api._get_target_methods(
                model=model,
                target_methods=kwargs.pop("target_methods", None),
                default_target_methods=_TorchScriptHandler.DEFAULT_TARGET_METHODS,
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

            model_meta = model_meta_api._validate_signature(
                model=model,
                model_meta=model_meta,
                target_methods=target_methods,
                sample_input=sample_input,
                get_prediction_fn=get_prediction,
            )

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)
        with open(os.path.join(model_blob_path, _TorchScriptHandler.MODEL_BLOB_FILE), "wb") as f:
            torch.jit.save(model, f)  # type:ignore[attr-defined]
        base_meta = model_meta_api._ModelBlobMetadata(
            name=name, model_type=_TorchScriptHandler.handler_type, path=_TorchScriptHandler.MODEL_BLOB_FILE
        )
        model_meta.models[name] = base_meta
        model_meta._include_if_absent([model_meta_api.Dependency(conda_name="pytorch", pip_name="torch")])

        model_meta.cuda_version = kwargs.get("cuda_version", model_meta_api._DEFAULT_CUDA_VERSION)

    @staticmethod
    def _load_model(
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> "torch.jit.ScriptModule":  # type:ignore[name-defined]
        import torch

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        if not hasattr(model_meta, "models"):
            raise ValueError("Ill model metadata found.")
        model_blobs_metadata = model_meta.models
        if name not in model_blobs_metadata:
            raise ValueError(f"Blob of model {name} does not exist.")
        model_blob_metadata = model_blobs_metadata[name]
        model_blob_filename = model_blob_metadata.path
        with open(os.path.join(model_blob_path, model_blob_filename), "rb") as f:
            m = torch.jit.load(f)  # type:ignore[attr-defined]
        assert isinstance(m, torch.jit.ScriptModule)  # type:ignore[attr-defined]

        if kwargs.get("use_gpu", False):
            m = m.cuda()

        return m

    @staticmethod
    def _load_as_custom_model(
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> custom_model.CustomModel:
        """Create a custom model class wrap for unified interface when being deployed. The predict method will be
        re-targeted based on target_method metadata.

        Args:
            name: Name of the model.
            model_meta: The model metadata.
            model_blobs_dir_path: Directory path to the whole model.
            kwargs: Options when loading the model.

        Returns:
            The model object as a custom model.
        """
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

        raw_model = _TorchScriptHandler._load_model(name, model_meta, model_blobs_dir_path, **kwargs)
        _TorchScriptModel = _create_custom_model(raw_model, model_meta)
        torchscript_model = _TorchScriptModel(custom_model.ModelContext())

        return torchscript_model
