import os
from typing import Optional, cast

import cloudpickle
import pandas as pd
from packaging import requirements
from typing_extensions import TypeGuard, Unpack

from snowflake.ml._internal import env_utils, file_utils
from snowflake.ml.model import (
    _model_meta as model_meta_api,
    custom_model,
    type_hints as model_types,
)
from snowflake.ml.model._handlers import _base
from snowflake.ml.model._signatures import core
from snowflake.ml.model.models import llm


class _LLMHandler(_base._ModelHandler[llm.LLM]):
    handler_type = "llm"
    MODEL_BLOB_DIR = "model"
    LLM_META = "llm_meta"
    is_auto_signature = True

    @staticmethod
    def can_handle(
        model: model_types.SupportedModelType,
    ) -> TypeGuard[llm.LLM]:
        return isinstance(model, llm.LLM)

    @staticmethod
    def cast_model(
        model: model_types.SupportedModelType,
    ) -> llm.LLM:
        assert isinstance(model, llm.LLM)
        return cast(llm.LLM, model)

    @staticmethod
    def _save_model(
        name: str,
        model: llm.LLM,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.BaseModelSaveOption],
    ) -> None:
        assert not is_sub_model, "LLM can not be sub-model."
        model_blob_path = os.path.join(model_blobs_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)
        model_blob_dir_path = os.path.join(model_blob_path, _LLMHandler.MODEL_BLOB_DIR)
        model_meta.cuda_version = model_meta_api._DEFAULT_CUDA_VERSION
        sig = core.ModelSignature(
            inputs=[
                core.FeatureSpec(name="input", dtype=core.DataType.STRING),
            ],
            outputs=[
                core.FeatureSpec(name="generated_text", dtype=core.DataType.STRING),
            ],
        )
        model_meta._signatures = {"infer": sig}
        assert os.path.isdir(model.model_id_or_path), "Only model dir is supported for now."
        file_utils.copytree(model.model_id_or_path, model_blob_dir_path)
        with open(
            os.path.join(model_blob_dir_path, _LLMHandler.LLM_META),
            "wb",
        ) as f:
            cloudpickle.dump(model, f)

        base_meta = model_meta_api._ModelBlobMetadata(
            name=name,
            model_type=_LLMHandler.handler_type,
            path=_LLMHandler.MODEL_BLOB_DIR,
            options={
                "batch_size": str(model.max_batch_size),
            },
        )
        model_meta.models[name] = base_meta
        pkgs_requirements = [
            model_meta_api.Dependency(conda_name="transformers", pip_req="transformers"),
            model_meta_api.Dependency(conda_name="pytorch", pip_req="torch==2.0.1"),
        ]
        if model.model_type == llm.SupportedLLMType.LLAMA_MODEL_TYPE:
            pkgs_requirements = [
                model_meta_api.Dependency(conda_name="sentencepiece", pip_req="sentencepiece"),
                model_meta_api.Dependency(conda_name="protobuf", pip_req="protobuf"),
                *pkgs_requirements,
            ]
        model_meta._include_if_absent(pkgs_requirements)
        # Recent peft versions are only available in PYPI.
        env_utils.append_requirement_list(
            model_meta._pip_requirements,
            requirements.Requirement("peft==0.5.0"),
        )

    @staticmethod
    def _load_model(
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> llm.LLM:
        model_blob_path = os.path.join(model_blobs_dir_path, name)
        if not hasattr(model_meta, "models"):
            raise ValueError("Ill model metadata found.")
        model_blobs_metadata = model_meta.models
        if name not in model_blobs_metadata:
            raise ValueError(f"Blob of model {name} does not exist.")
        model_blob_metadata = model_blobs_metadata[name]
        model_blob_filename = model_blob_metadata.path
        model_blob_dir_path = os.path.join(model_blob_path, model_blob_filename)
        assert model_blob_dir_path, "It must be a directory."
        with open(os.path.join(model_blob_dir_path, _LLMHandler.LLM_META), "rb") as f:
            m = cloudpickle.load(f)
        assert isinstance(m, llm.LLM)
        # Switch to local path
        m.model_id_or_path = model_blob_dir_path
        return m

    @staticmethod
    def _load_as_custom_model(
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> custom_model.CustomModel:
        raw_model = _LLMHandler._load_model(
            name,
            model_meta,
            model_blobs_dir_path,
            **kwargs,
        )
        import peft
        import transformers

        hub_kwargs = {
            "revision": raw_model.revision,
            "token": raw_model.token,
        }
        model_dir_path = raw_model.model_id_or_path
        hf_model = peft.AutoPeftModelForCausalLM.from_pretrained(  # type: ignore[attr-defined]
            model_dir_path,
            device_map="auto",
            torch_dtype="auto",
            **hub_kwargs,
        )
        peft_config = peft.PeftConfig.from_pretrained(model_dir_path)  # type: ignore[attr-defined]
        base_model_path = peft_config.base_model_name_or_path
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_path,
            padding_side="right",
            use_fast=False,
            **hub_kwargs,
        )
        hf_model.eval()

        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        # TODO(lhw): migrate away from hf pipeline
        pipe = transformers.pipeline(
            task="text-generation",
            model=hf_model,
            tokenizer=tokenizer,
            batch_size=raw_model.max_batch_size,
        )

        class _LLMCustomModel(custom_model.CustomModel):
            @custom_model.inference_api
            def infer(self, X: pd.DataFrame) -> pd.DataFrame:
                input_data = X.to_dict("list")["input"]
                res = pipe(input_data, return_full_text=False)
                # TODO(lhw): Assume single beam only.
                return pd.DataFrame({"generated_text": [output[0]["generated_text"] for output in res]})

        llm_custom = _LLMCustomModel(custom_model.ModelContext())

        return llm_custom
