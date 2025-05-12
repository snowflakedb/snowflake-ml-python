import os
from typing import TYPE_CHECKING, Callable, Optional, cast, final

import pandas as pd
from typing_extensions import TypeGuard, Unpack

from snowflake.ml._internal import type_utils
from snowflake.ml.model import custom_model, model_signature, type_hints as model_types
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_handlers import _base, _utils as handlers_utils
from snowflake.ml.model._packager.model_handlers_migrator import (
    base_migrator,
    torchscript_migrator_2023_12_01,
)
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta as model_meta_api,
    model_meta_schema,
)
from snowflake.ml.model._signatures import (
    pytorch_handler,
    utils as model_signature_utils,
)

if TYPE_CHECKING:
    import torch


@final
class TorchScriptHandler(_base.BaseModelHandler["torch.jit.ScriptModule"]):
    """Handler for PyTorch JIT based model.

    Currently torch.jit.ScriptModule based classes are supported.
    """

    HANDLER_TYPE = "torchscript"
    HANDLER_VERSION = "2025-03-01"
    _MIN_SNOWPARK_ML_VERSION = "1.8.0"
    _HANDLER_MIGRATOR_PLANS: dict[str, type[base_migrator.BaseModelHandlerMigrator]] = {
        "2023-12-01": torchscript_migrator_2023_12_01.TorchScriptHandlerMigrator20231201
    }

    MODEL_BLOB_FILE_OR_DIR = "model.pt"
    DEFAULT_TARGET_METHODS = ["forward"]

    @classmethod
    def can_handle(
        cls,
        model: model_types.SupportedModelType,
    ) -> TypeGuard["torch.jit.ScriptModule"]:
        return type_utils.LazyType("torch.jit.ScriptModule").isinstance(model)

    @classmethod
    def cast_model(
        cls,
        model: model_types.SupportedModelType,
    ) -> "torch.jit.ScriptModule":
        import torch

        assert isinstance(model, torch.jit.ScriptModule)

        return cast(torch.jit.ScriptModule, model)

    @classmethod
    def save_model(
        cls,
        name: str,
        model: "torch.jit.ScriptModule",
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.TorchScriptSaveOptions],
    ) -> None:
        enable_explainability = kwargs.get("enable_explainability", False)
        if enable_explainability:
            raise NotImplementedError("Explainability is not supported for Torch Script model.")

        multiple_inputs = kwargs.get("multiple_inputs", False)

        import torch

        assert isinstance(model, torch.jit.ScriptModule)

        if not is_sub_model:
            target_methods = handlers_utils.get_target_methods(
                model=model,
                target_methods=kwargs.pop("target_methods", None),
                default_target_methods=cls.DEFAULT_TARGET_METHODS,
            )

            def get_prediction(
                target_method_name: str, sample_input_data: "model_types.SupportedLocalDataType"
            ) -> model_types.SupportedLocalDataType:
                if multiple_inputs:
                    if not pytorch_handler.SeqOfPyTorchTensorHandler.can_handle(sample_input_data):
                        sample_input_data = pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(
                            model_signature._convert_local_data_to_df(sample_input_data)
                        )
                else:
                    if not pytorch_handler.PyTorchTensorHandler.can_handle(sample_input_data):
                        sample_input_data = pytorch_handler.PyTorchTensorHandler.convert_from_df(
                            model_signature._convert_local_data_to_df(sample_input_data)
                        )

                model.eval()
                target_method = getattr(model, target_method_name, None)
                assert callable(target_method)
                with torch.no_grad():
                    if multiple_inputs:
                        predictions_df = target_method(*sample_input_data)
                        if not isinstance(predictions_df, tuple):
                            predictions_df = [predictions_df]
                    else:
                        predictions_df = target_method(sample_input_data)

                return predictions_df

            model_meta = handlers_utils.validate_signature(
                model=model,
                model_meta=model_meta,
                target_methods=target_methods,
                sample_input_data=sample_input_data,
                get_prediction_fn=get_prediction,
            )

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)
        with open(os.path.join(model_blob_path, cls.MODEL_BLOB_FILE_OR_DIR), "wb") as f:
            torch.jit.save(model, f)  # type:ignore[no-untyped-call]
        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODEL_BLOB_FILE_OR_DIR,
            options=model_meta_schema.TorchScriptModelBlobOptions(multiple_inputs=multiple_inputs),
        )
        model_meta.models[name] = base_meta
        model_meta.min_snowpark_ml_version = cls._MIN_SNOWPARK_ML_VERSION

        model_meta.env.include_if_absent(
            [model_env.ModelDependency(requirement="pytorch", pip_name="torch")], check_local_version=True
        )
        model_meta.env.cuda_version = kwargs.get("cuda_version", handlers_utils.get_default_cuda_version())

    @classmethod
    def load_model(
        cls,
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.TorchScriptLoadOptions],
    ) -> "torch.jit.ScriptModule":
        import torch

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        model_blobs_metadata = model_meta.models
        model_blob_metadata = model_blobs_metadata[name]
        model_blob_filename = model_blob_metadata.path
        with open(os.path.join(model_blob_path, model_blob_filename), "rb") as f:
            m = torch.jit.load(  # type:ignore[no-untyped-call]
                f, map_location="cuda" if kwargs.get("use_gpu", False) else "cpu"
            )
        assert isinstance(m, torch.jit.ScriptModule)

        if kwargs.get("use_gpu", False):
            m = m.cuda()

        return m

    @classmethod
    def convert_as_custom_model(
        cls,
        raw_model: "torch.jit.ScriptModule",
        model_meta: model_meta_api.ModelMetadata,
        background_data: Optional[pd.DataFrame] = None,
        **kwargs: Unpack[model_types.TorchScriptLoadOptions],
    ) -> custom_model.CustomModel:
        from snowflake.ml.model import custom_model

        def _create_custom_model(
            raw_model: "torch.jit.ScriptModule",
            model_meta: model_meta_api.ModelMetadata,
        ) -> type[custom_model.CustomModel]:
            def fn_factory(
                raw_model: "torch.jit.ScriptModule",
                signature: model_signature.ModelSignature,
                target_method: str,
            ) -> Callable[[custom_model.CustomModel, pd.DataFrame], pd.DataFrame]:
                multiple_inputs = cast(
                    model_meta_schema.TorchScriptModelBlobOptions, model_meta.models[model_meta.name].options
                )["multiple_inputs"]

                @custom_model.inference_api
                def fn(self: custom_model.CustomModel, X: pd.DataFrame) -> pd.DataFrame:
                    if X.isnull().any(axis=None):
                        raise ValueError("Tensor cannot handle null values.")

                    import torch

                    raw_model.eval()

                    if multiple_inputs:
                        st = pytorch_handler.SeqOfPyTorchTensorHandler.convert_from_df(X, signature.inputs)

                        if kwargs.get("use_gpu", False):
                            st = [element.cuda() for element in st]

                        with torch.no_grad():
                            res = getattr(raw_model, target_method)(*st)

                        if not isinstance(res, tuple):
                            res = [res]
                    else:
                        t = pytorch_handler.PyTorchTensorHandler.convert_from_df(X, signature.inputs)
                        if kwargs.get("use_gpu", False):
                            t = t.cuda()

                        with torch.no_grad():
                            res = getattr(raw_model, target_method)(t)
                    return model_signature_utils.rename_pandas_df(
                        model_signature._convert_local_data_to_df(res, ensure_serializable=True),
                        features=signature.outputs,
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
