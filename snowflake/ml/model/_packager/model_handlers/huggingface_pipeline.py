import json
import os
import warnings
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    cast,
    final,
)

import cloudpickle
import numpy as np
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
from snowflake.ml.model._signatures import (
    builtins_handler,
    utils as model_signature_utils,
)
from snowflake.ml.model.models import huggingface_pipeline
from snowflake.snowpark._internal import utils as snowpark_utils

if TYPE_CHECKING:
    import transformers


def get_requirements_from_task(task: str, spcs_only: bool = False) -> List[model_env.ModelDependency]:
    # Text
    if task in [
        "conversational",
        "fill-mask",
        "ner",
        "token-classification",
        "question-answering",
        "summarization",
        "table-question-answering",
        "text-classification",
        "sentiment-analysis",
        "text-generation",
        "text2text-generation",
        "zero-shot-classification",
    ] or task.startswith("translation"):
        return (
            [model_env.ModelDependency(requirement="tokenizers>=0.13.3", pip_name="tokenizers")]
            if spcs_only
            else [model_env.ModelDependency(requirement="tokenizers", pip_name="tokenizers")]
        )

    return []


class NumpyEncoder(json.JSONEncoder):
    # This is a JSON encoder class to ensure the output from Huggingface pipeline is JSON serializable.
    # What it covers is numpy object.
    def default(self, z: object) -> object:
        if isinstance(z, np.number):
            if np.can_cast(z, np.int64, casting="safe"):
                return int(z)
            elif np.can_cast(z, np.float64, casting="safe"):
                return z.astype(np.float64)
        return super().default(z)


@final
class HuggingFacePipelineHandler(
    _base.BaseModelHandler[Union[huggingface_pipeline.HuggingFacePipelineModel, "transformers.Pipeline"]]
):
    """Handler for custom model."""

    HANDLER_TYPE = "huggingface_pipeline"
    HANDLER_VERSION = "2023-12-01"
    _MIN_SNOWPARK_ML_VERSION = "1.0.12"
    _HANDLER_MIGRATOR_PLANS: Dict[str, Type[base_migrator.BaseModelHandlerMigrator]] = {}

    MODELE_BLOB_FILE_OR_DIR = "model"
    ADDITIONAL_CONFIG_FILE = "pipeline_config.pt"
    DEFAULT_TARGET_METHODS = ["__call__"]
    IS_AUTO_SIGNATURE = True

    @classmethod
    def can_handle(
        cls,
        model: model_types.SupportedModelType,
    ) -> TypeGuard[Union[huggingface_pipeline.HuggingFacePipelineModel, "transformers.Pipeline"]]:
        if type_utils.LazyType("transformers.Pipeline").isinstance(model):
            return True
        if isinstance(model, huggingface_pipeline.HuggingFacePipelineModel):
            return True
        return False

    @classmethod
    def cast_model(
        cls,
        model: model_types.SupportedModelType,
    ) -> Union[huggingface_pipeline.HuggingFacePipelineModel, "transformers.Pipeline"]:
        try:
            if isinstance(model, huggingface_pipeline.HuggingFacePipelineModel):
                raise ImportError
            else:
                import transformers
        except ImportError:
            assert isinstance(model, huggingface_pipeline.HuggingFacePipelineModel)
            return model
        else:
            assert isinstance(model, transformers.Pipeline)
            return model

    @classmethod
    def save_model(
        cls,
        name: str,
        model: Union[huggingface_pipeline.HuggingFacePipelineModel, "transformers.Pipeline"],
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        sample_input: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.HuggingFaceSaveOptions],
    ) -> None:
        if type_utils.LazyType("transformers.Pipeline").isinstance(model):
            task = model.task  # type:ignore[attr-defined]
            framework = model.framework  # type:ignore[attr-defined]
            batch_size = model._batch_size  # type:ignore[attr-defined]
        else:
            assert isinstance(model, huggingface_pipeline.HuggingFacePipelineModel)
            task = model.task
            framework = getattr(model, "framework", None)
            batch_size = getattr(model, "batch_size", None)

        if type_utils.LazyType("transformers.Pipeline").isinstance(model):
            params = {
                **model._preprocess_params,  # type:ignore[attr-defined]
                **model._forward_params,  # type:ignore[attr-defined]
                **model._postprocess_params,  # type:ignore[attr-defined]
            }
        else:
            assert isinstance(model, huggingface_pipeline.HuggingFacePipelineModel)
            params = {**model.__dict__, **model.model_kwargs}

        inferred_pipe_sig = model_signature_utils.huggingface_pipeline_signature_auto_infer(task, params=params)

        if not is_sub_model:
            target_methods = handlers_utils.get_target_methods(
                model=model,
                target_methods=kwargs.pop("target_methods", None),
                default_target_methods=cls.DEFAULT_TARGET_METHODS,
            )

            if model_meta.signatures:
                if type_utils.LazyType("transformers.Pipeline").isinstance(model):
                    handlers_utils.validate_target_methods(model, list(model_meta.signatures.keys()))
                else:
                    warnings.warn(
                        "It is impossible to validate your model signatures when using a"
                        " `snowflake.ml.model.models.huggingface_pipeline.HuggingFacePipelineModel` object. "
                        "Please make sure you are providing correct model signatures.",
                        UserWarning,
                        stacklevel=2,
                    )
            else:
                handlers_utils.validate_target_methods(model, target_methods)
                if sample_input is not None:
                    warnings.warn(
                        "Inferring model signature from sample input for hugggingface pipeline is not supported. "
                        + "Model signature will automatically be inferred from pipeline task. "
                        + "Or, you could specify model signature manually.",
                        UserWarning,
                        stacklevel=2,
                    )
                if inferred_pipe_sig is None:
                    raise NotImplementedError(f"Cannot auto infer the signature of pipeline for task {task}")

                model_meta.signatures = {"__call__": inferred_pipe_sig}

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        os.makedirs(model_blob_path, exist_ok=True)

        if type_utils.LazyType("transformers.Pipeline").isinstance(model):
            model.save_pretrained(  # type:ignore[attr-defined]
                os.path.join(model_blob_path, cls.MODELE_BLOB_FILE_OR_DIR)
            )
            pipeline_params = {
                "_batch_size": model._batch_size,  # type:ignore[attr-defined]
                "_num_workers": model._num_workers,  # type:ignore[attr-defined]
                "_preprocess_params": model._preprocess_params,  # type:ignore[attr-defined]
                "_forward_params": model._forward_params,  # type:ignore[attr-defined]
                "_postprocess_params": model._postprocess_params,  # type:ignore[attr-defined]
            }
            with open(
                os.path.join(
                    model_blob_path,
                    cls.MODELE_BLOB_FILE_OR_DIR,
                    cls.ADDITIONAL_CONFIG_FILE,
                ),
                "wb",
            ) as f:
                cloudpickle.dump(pipeline_params, f)
        else:
            with open(
                os.path.join(model_blob_path, cls.MODELE_BLOB_FILE_OR_DIR),
                "wb",
            ) as f:
                cloudpickle.dump(model, f)

        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODELE_BLOB_FILE_OR_DIR,
            options=model_meta_schema.HuggingFacePipelineModelBlobOptions(
                {
                    "task": task,
                    "batch_size": batch_size if batch_size is not None else 1,
                }
            ),
        )
        model_meta.models[name] = base_meta
        model_meta.min_snowpark_ml_version = cls._MIN_SNOWPARK_ML_VERSION

        pkgs_requirements = [
            model_env.ModelDependency(requirement="transformers>=4.32.1", pip_name="transformers"),
        ] + get_requirements_from_task(
            task, spcs_only=(not type_utils.LazyType("transformers.Pipeline").isinstance(model))
        )
        if framework is None or framework == "pt":
            # Since we set default cuda version to be 11.7, to make sure it works with GPU, we need to have a default
            # Pytorch version that works with CUDA 11.7 as well. This is required for huggingface pipelines only as
            # users are not required to install pytorch locally if they are using the wrapper.
            pkgs_requirements.append(model_env.ModelDependency(requirement="pytorch==2.0.1", pip_name="torch"))
        elif framework == "tf":
            pkgs_requirements.append(model_env.ModelDependency(requirement="tensorflow", pip_name="tensorflow"))
        model_meta.env.include_if_absent(
            pkgs_requirements, check_local_version=(type_utils.LazyType("transformers.Pipeline").isinstance(model))
        )
        model_meta.env.cuda_version = kwargs.get("cuda_version", model_env.DEFAULT_CUDA_VERSION)

    @staticmethod
    def _get_device_config() -> Dict[str, str]:
        device_config = {}
        device_config["device_map"] = "auto"

        return device_config

    @classmethod
    def load_model(
        cls,
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> Union[huggingface_pipeline.HuggingFacePipelineModel, "transformers.Pipeline"]:
        if snowpark_utils.is_in_stored_procedure():  # type: ignore[no-untyped-call]
            # We need to redirect the some folders to a writable location in the sandbox.
            os.environ["HF_HOME"] = "/tmp"
            os.environ["XDG_CACHE_HOME"] = "/tmp"

        model_blob_path = os.path.join(model_blobs_dir_path, name)
        model_blobs_metadata = model_meta.models
        model_blob_metadata = model_blobs_metadata[name]
        model_blob_filename = model_blob_metadata.path
        model_blob_options = cast(model_meta_schema.HuggingFacePipelineModelBlobOptions, model_blob_metadata.options)
        if "task" not in model_blob_options:
            raise ValueError("Missing field `task` in model blob metadata for type `huggingface_pipeline`")
        if "batch_size" not in model_blob_options:
            raise ValueError("Missing field `batch_size` in model blob metadata for type `huggingface_pipeline`")

        model_blob_file_or_dir_path = os.path.join(model_blob_path, model_blob_filename)
        if os.path.isdir(model_blob_file_or_dir_path):
            import transformers

            with open(
                os.path.join(
                    model_blob_file_or_dir_path,
                    cls.ADDITIONAL_CONFIG_FILE,
                ),
                "rb",
            ) as f:
                pipeline_params = cloudpickle.load(f)

            if kwargs.get("use_gpu", False):
                device_config = cls._get_device_config()
            else:
                device_config = {}

            m = transformers.pipeline(
                model_blob_options["task"],
                model=model_blob_file_or_dir_path,
                **device_config,
            )

            m.__dict__.update(pipeline_params)

        else:
            assert os.path.isfile(model_blob_file_or_dir_path)
            with open(model_blob_file_or_dir_path, "rb") as f:
                m = cloudpickle.load(f)
            assert isinstance(m, huggingface_pipeline.HuggingFacePipelineModel)
            if (
                getattr(m, "device", None) is None
                and getattr(m, "device_map", None) is None
                and kwargs.get("use_gpu", False)
            ):
                m.__dict__.update(cls._get_device_config())

            if getattr(m, "torch_dtype", None) is None and kwargs.get("use_gpu", False):
                m.__dict__.update(torch_dtype="auto")
        return m

    @classmethod
    def convert_as_custom_model(
        cls,
        raw_model: Union[huggingface_pipeline.HuggingFacePipelineModel, "transformers.Pipeline"],
        model_meta: model_meta_api.ModelMetadata,
        **kwargs: Unpack[model_types.ModelLoadOption],
    ) -> custom_model.CustomModel:
        import transformers

        from snowflake.ml.model import custom_model

        def _create_custom_model(
            raw_model: "transformers.Pipeline",
            model_meta: model_meta_api.ModelMetadata,
        ) -> Type[custom_model.CustomModel]:
            def fn_factory(
                raw_model: "transformers.Pipeline",
                signature: model_signature.ModelSignature,
                target_method: str,
            ) -> Callable[[custom_model.CustomModel, pd.DataFrame], pd.DataFrame]:
                @custom_model.inference_api
                def fn(self: custom_model.CustomModel, X: pd.DataFrame) -> pd.DataFrame:
                    # These 3 zero-shot classification cannot take a list of dict as input like other multi input
                    # pipelines, thus dealing separately.
                    if isinstance(
                        raw_model,
                        (
                            transformers.ZeroShotAudioClassificationPipeline,
                            transformers.ZeroShotClassificationPipeline,
                            transformers.ZeroShotImageClassificationPipeline,
                        ),
                    ):
                        temp_res = X.apply(
                            lambda row: getattr(raw_model, target_method)(
                                row[signature.inputs[0].name], row["candidate_labels"]
                            ),
                            axis=1,
                        ).to_list()
                    else:
                        # For others, we could offer the whole dataframe as a list.
                        # Some of them may need some conversion
                        if isinstance(raw_model, transformers.ConversationalPipeline):
                            input_data = [
                                transformers.Conversation(
                                    text=conv_data["user_inputs"][0],
                                    past_user_inputs=conv_data["user_inputs"][1:],
                                    generated_responses=conv_data["generated_responses"],
                                )
                                for conv_data in X.to_dict("records")
                            ]
                        elif len(signature.inputs) == 1:
                            input_data = X.to_dict("list")[signature.inputs[0].name]
                        else:
                            if isinstance(raw_model, transformers.TableQuestionAnsweringPipeline):
                                X["table"] = X["table"].apply(json.loads)

                            input_data = X.to_dict("records")
                        temp_res = getattr(raw_model, target_method)(input_data)

                    # Some huggingface pipeline will omit the outer list when there is only 1 input.
                    # Making it not aligned with the auto-inferred signature.
                    # If the output is a dict, we could blindly create a list containing that.
                    # Otherwise, creating pandas DataFrame won't succeed.
                    if isinstance(temp_res, (dict, transformers.Conversation)) or (
                        # For some pipeline that is expected to generate a list of dict per input
                        # When it omit outer list, it becomes list of dict instead of list of list of dict.
                        # We need to distinguish them from those pipelines that designed to output a dict per input
                        # So we need to check the pipeline type.
                        isinstance(
                            raw_model,
                            (
                                transformers.FillMaskPipeline,
                                transformers.QuestionAnsweringPipeline,
                            ),
                        )
                        and X.shape[0] == 1
                        and isinstance(temp_res[0], dict)
                    ):
                        temp_res = [temp_res]

                    if len(temp_res) == 0:
                        return pd.DataFrame()

                    if isinstance(raw_model, transformers.ConversationalPipeline):
                        temp_res = [[conv.generated_responses] for conv in temp_res]

                    # To concat those who outputs a list with one input.
                    if builtins_handler.ListOfBuiltinHandler.can_handle(temp_res):
                        res = builtins_handler.ListOfBuiltinHandler.convert_to_df(temp_res)
                    elif isinstance(temp_res[0], dict):
                        res = pd.DataFrame(temp_res)
                    elif isinstance(temp_res[0], list):
                        res = pd.DataFrame([json.dumps(output, cls=NumpyEncoder) for output in temp_res])
                    else:
                        raise ValueError(f"Cannot parse output {temp_res} from pipeline object")

                    return model_signature_utils.rename_pandas_df(data=res, features=signature.outputs)

                return fn

            type_method_dict = {}
            for target_method_name, sig in model_meta.signatures.items():
                type_method_dict[target_method_name] = fn_factory(raw_model, sig, target_method_name)

            _HFPipelineModel = type(
                "_HFPipelineModel",
                (custom_model.CustomModel,),
                type_method_dict,
            )

            return _HFPipelineModel

        if isinstance(raw_model, huggingface_pipeline.HuggingFacePipelineModel):
            if version.parse(transformers.__version__) < version.parse("4.32.0"):
                # Backward compatibility since HF interface change.
                raw_model.__dict__["use_auth_token"] = raw_model.__dict__["token"]
                del raw_model.__dict__["token"]
            pipe = transformers.pipeline(**raw_model.__dict__)
        else:
            pipe = raw_model

        pipe.binary_output = False

        # To enable batch_size > 1 for LLM
        # Pipe might not have tokenizer, but should always have a model, and model should always have a config.
        if getattr(pipe, "tokenizer", None) is not None and pipe.tokenizer.pad_token_id is None:
            pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

        _HFPipelineModel = _create_custom_model(pipe, model_meta)
        hg_pipe_model = _HFPipelineModel(custom_model.ModelContext())

        return hg_pipe_model
