import logging
import os
from typing import Dict, Optional, Type, cast, final

import cloudpickle
import pandas as pd
from typing_extensions import TypeGuard, Unpack

from snowflake.ml._internal import file_utils
from snowflake.ml.model import custom_model, model_signature, type_hints as model_types
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_handlers import _base
from snowflake.ml.model._packager.model_handlers_migrator import base_migrator
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta as model_meta_api,
    model_meta_schema,
)
from snowflake.ml.model.models import llm

logger = logging.getLogger(__name__)


@final
class LLMHandler(_base.BaseModelHandler[llm.LLM]):
    HANDLER_TYPE = "llm"
    HANDLER_VERSION = "2023-12-01"
    _MIN_SNOWPARK_ML_VERSION = "1.0.12"
    _HANDLER_MIGRATOR_PLANS: Dict[str, Type[base_migrator.BaseModelHandlerMigrator]] = {}

    MODELE_BLOB_FILE_OR_DIR = "model"
    LLM_META = "llm_meta"
    IS_AUTO_SIGNATURE = True

    @classmethod
    def can_handle(
        cls,
        model: model_types.SupportedModelType,
    ) -> TypeGuard[llm.LLM]:
        return isinstance(model, llm.LLM)

    @classmethod
    def cast_model(
        cls,
        model: model_types.SupportedModelType,
    ) -> llm.LLM:
        assert isinstance(model, llm.LLM)
        return cast(llm.LLM, model)

    @classmethod
    def save_model(
        cls,
        name: str,
        model: llm.LLM,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.LLMSaveOptions],
    ) -> None:
        assert not is_sub_model, "LLM can not be sub-model."
        model_blob_path = os.path.join(model_blobs_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)
        model_blob_dir_path = os.path.join(model_blob_path, cls.MODELE_BLOB_FILE_OR_DIR)

        sig = model_signature.ModelSignature(
            inputs=[
                model_signature.FeatureSpec(name="input", dtype=model_signature.DataType.STRING),
            ],
            outputs=[
                model_signature.FeatureSpec(name="generated_text", dtype=model_signature.DataType.STRING),
            ],
        )
        model_meta.signatures = {"infer": sig}
        if os.path.isdir(model.model_id_or_path):
            file_utils.copytree(model.model_id_or_path, model_blob_dir_path)

        os.makedirs(model_blob_dir_path, exist_ok=True)
        with open(
            os.path.join(model_blob_dir_path, cls.LLM_META),
            "wb",
        ) as f:
            cloudpickle.dump(model, f)

        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODELE_BLOB_FILE_OR_DIR,
            options=model_meta_schema.LLMModelBlobOptions(
                {
                    "batch_size": model.max_batch_size,
                }
            ),
        )
        model_meta.models[name] = base_meta
        model_meta.min_snowpark_ml_version = cls._MIN_SNOWPARK_ML_VERSION

        pkgs_requirements = [
            model_env.ModelDependency(requirement="transformers>=4.32.1", pip_name="transformers"),
            model_env.ModelDependency(requirement="pytorch==2.0.1", pip_name="torch"),
        ]
        if model.model_type == llm.SupportedLLMType.LLAMA_MODEL_TYPE.value:
            pkgs_requirements = [
                model_env.ModelDependency(requirement="sentencepiece", pip_name="sentencepiece"),
                model_env.ModelDependency(requirement="protobuf", pip_name="protobuf"),
                *pkgs_requirements,
            ]
        model_meta.env.include_if_absent(pkgs_requirements, check_local_version=True)
        # Recent peft versions are only available in PYPI.
        model_meta.env.include_if_absent_pip(["peft==0.5.0", "vllm==0.2.1.post1"])

        model_meta.env.cuda_version = kwargs.get("cuda_version", model_env.DEFAULT_CUDA_VERSION)

    @classmethod
    def load_model(
        cls,
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
        with open(os.path.join(model_blob_dir_path, cls.LLM_META), "rb") as f:
            m = cloudpickle.load(f)
        assert isinstance(m, llm.LLM)
        if m.mode == llm.LLM.Mode.LOCAL_LORA:
            # Switch to local path
            m.model_id_or_path = model_blob_dir_path
        return m

    @classmethod
    def convert_as_custom_model(
        cls,
        raw_model: llm.LLM,
        model_meta: model_meta_api.ModelMetadata,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> custom_model.CustomModel:
        import gc
        import tempfile

        import torch
        import transformers
        import vllm

        assert torch.cuda.is_available(), "LLM inference only works on GPUs."
        device_count = torch.cuda.device_count()
        logger.warning(f"There's total {device_count} GPUs visible to use.")

        class _LLMCustomModel(custom_model.CustomModel):
            def _memory_stats(self, msg: str) -> None:
                logger.warning(msg)
                logger.warning(f"Torch VRAM {torch.cuda.memory_allocated()/1024**2} MB allocated.")
                logger.warning(f"Torch VRAM {torch.cuda.memory_reserved()/1024**2} MB reserved.")

            def _prepare_for_pretrain(self) -> None:
                hub_kwargs = {
                    "revision": raw_model.revision,
                    "token": raw_model.token,
                }
                model_dir_path = raw_model.model_id_or_path
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_dir_path,
                    padding_side="right",
                    use_fast=False,
                    **hub_kwargs,
                )
                if not tokenizer.pad_token:
                    tokenizer.pad_token = tokenizer.eos_token
                tokenizer.save_pretrained(self.local_model_dir)
                hf_model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_dir_path,
                    device_map="auto",
                    torch_dtype="auto",
                    **hub_kwargs,
                )
                hf_model.eval()
                hf_model.save_pretrained(self.local_model_dir)
                logger.warning(f"Model state is saved to {self.local_model_dir}.")
                del tokenizer
                del hf_model
                gc.collect()
                torch.cuda.empty_cache()
                self._memory_stats("After GC on model.")

            def _prepare_for_lora(self) -> None:
                self._memory_stats("Before model load & merge.")
                import peft

                hub_kwargs = {
                    "revision": raw_model.revision,
                    "token": raw_model.token,
                }
                model_dir_path = raw_model.model_id_or_path
                peft_config = peft.PeftConfig.from_pretrained(model_dir_path)  # type: ignore[attr-defined]
                base_model_path = peft_config.base_model_name_or_path
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    base_model_path,
                    padding_side="right",
                    use_fast=False,
                    **hub_kwargs,
                )
                if not tokenizer.pad_token:
                    tokenizer.pad_token = tokenizer.eos_token
                tokenizer.save_pretrained(self.local_model_dir)
                logger.warning(f"Tokenizer state is saved to {self.local_model_dir}.")
                hf_model = peft.AutoPeftModelForCausalLM.from_pretrained(  # type: ignore[attr-defined]
                    model_dir_path,
                    device_map="auto",
                    torch_dtype="auto",
                    **hub_kwargs,
                )
                hf_model.eval()
                hf_model = hf_model.merge_and_unload()
                hf_model.save_pretrained(self.local_model_dir)
                logger.warning(f"Merged model state is saved to {self.local_model_dir}.")
                self._memory_stats("After model load & merge.")
                del hf_model
                gc.collect()
                torch.cuda.empty_cache()
                self._memory_stats("After GC on model.")

            def __init__(self, context: custom_model.ModelContext) -> None:
                self.local_tmp_holder = tempfile.TemporaryDirectory()
                self.local_model_dir = self.local_tmp_holder.name
                if raw_model.mode == llm.LLM.Mode.LOCAL_LORA:
                    self._prepare_for_lora()
                elif raw_model.mode == llm.LLM.Mode.REMOTE_PRETRAIN:
                    self._prepare_for_pretrain()
                self.sampling_params = vllm.SamplingParams(
                    temperature=raw_model.temperature,
                    top_p=raw_model.top_p,
                    max_tokens=raw_model.max_tokens,
                )
                self._init_engine()

            # This has to have same lifetime as main thread
            # in order to avoid pre-maturely terminate ray.
            def _init_engine(self) -> None:
                tp_size = torch.cuda.device_count() if raw_model.enable_tp else 1
                self.llm_engine = vllm.LLM(
                    model=self.local_model_dir,
                    tensor_parallel_size=tp_size,
                )

            @custom_model.inference_api
            def infer(self, X: pd.DataFrame) -> pd.DataFrame:
                input_data = X.to_dict("list")["input"]
                res = self.llm_engine.generate(input_data, self.sampling_params)
                return pd.DataFrame({"generated_text": [o.outputs[0].text for o in res]})

        llm_custom = _LLMCustomModel(custom_model.ModelContext())

        return llm_custom
