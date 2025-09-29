import json
import logging
import os
import shutil
import time
import uuid
import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, cast, final

import cloudpickle
import numpy as np
import pandas as pd
from packaging import version
from typing_extensions import TypeGuard, Unpack

from snowflake.ml._internal import type_utils
from snowflake.ml.model import (
    custom_model,
    model_signature,
    openai_signatures,
    type_hints as model_types,
)
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_handlers import _base, _utils as handlers_utils
from snowflake.ml.model._packager.model_handlers_migrator import base_migrator
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta as model_meta_api,
    model_meta_schema,
)
from snowflake.ml.model._signatures import utils as model_signature_utils
from snowflake.ml.model.models import huggingface_pipeline
from snowflake.snowpark._internal import utils as snowpark_utils

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import transformers

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"  # noqa: E501


def get_requirements_from_task(task: str, spcs_only: bool = False) -> list[model_env.ModelDependency]:
    # Text
    if task in [
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


def sanitize_output(data: Any) -> Any:
    if isinstance(data, np.number):
        return data.item()
    if isinstance(data, np.ndarray):
        return sanitize_output(data.tolist())
    if isinstance(data, list):
        return [sanitize_output(x) for x in data]
    if isinstance(data, dict):
        return {k: sanitize_output(v) for k, v in data.items()}
    return data


@final
class HuggingFacePipelineHandler(
    _base.BaseModelHandler[Union[huggingface_pipeline.HuggingFacePipelineModel, "transformers.Pipeline"]]
):
    """Handler for custom model."""

    HANDLER_TYPE = "huggingface_pipeline"
    HANDLER_VERSION = "2023-12-01"
    _MIN_SNOWPARK_ML_VERSION = "1.0.12"
    _HANDLER_MIGRATOR_PLANS: dict[str, type[base_migrator.BaseModelHandlerMigrator]] = {}

    MODEL_BLOB_FILE_OR_DIR = "model"
    MODEL_PICKLE_FILE = "snowml_huggingface_pipeline.pkl"
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
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        is_sub_model: Optional[bool] = False,
        **kwargs: Unpack[model_types.HuggingFaceSaveOptions],
    ) -> None:
        enable_explainability = kwargs.get("enable_explainability", False)
        if enable_explainability:
            raise NotImplementedError("Explainability is not supported for huggingface model.")
        if type_utils.LazyType("transformers.Pipeline").isinstance(model):
            task = model.task  # type:ignore[attr-defined]
            framework = model.framework  # type:ignore[attr-defined]
            batch_size = model._batch_size  # type:ignore[attr-defined]
        else:
            assert isinstance(model, huggingface_pipeline.HuggingFacePipelineModel)
            task = model.task
            framework = getattr(model, "framework", None)
            batch_size = getattr(model, "batch_size", None)

        has_tokenizer = getattr(model, "tokenizer", None) is not None
        has_feature_extractor = getattr(model, "feature_extractor", None) is not None
        has_image_preprocessor = getattr(model, "image_preprocessor", None) is not None

        if type_utils.LazyType("transformers.Pipeline").isinstance(model):
            params = {
                **model._preprocess_params,  # type:ignore[attr-defined]
                **model._forward_params,  # type:ignore[attr-defined]
                **model._postprocess_params,  # type:ignore[attr-defined]
            }
        else:
            assert isinstance(model, huggingface_pipeline.HuggingFacePipelineModel)
            params = {**model.__dict__, **model.model_kwargs}

        inferred_pipe_sig = model_signature_utils.huggingface_pipeline_signature_auto_infer(
            task,
            params=params,
        )

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
                if sample_input_data is not None:
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

        is_repo_downloaded = False
        if type_utils.LazyType("transformers.Pipeline").isinstance(model):
            save_path = os.path.join(model_blob_path, cls.MODEL_BLOB_FILE_OR_DIR)
            model.save_pretrained(  # type:ignore[attr-defined]
                save_path
            )
            handlers_utils.save_transformers_config_with_auto_map(
                save_path,
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
                    cls.MODEL_BLOB_FILE_OR_DIR,
                    cls.ADDITIONAL_CONFIG_FILE,
                ),
                "wb",
            ) as f:
                cloudpickle.dump(pipeline_params, f)
        else:
            model_blob_file_or_dir = os.path.join(model_blob_path, cls.MODEL_BLOB_FILE_OR_DIR)
            model_blob_pickle_file = os.path.join(model_blob_file_or_dir, cls.MODEL_PICKLE_FILE)
            os.makedirs(model_blob_file_or_dir, exist_ok=True)
            with open(
                model_blob_pickle_file,
                "wb",
            ) as f:
                cloudpickle.dump(model, f)
            if model.repo_snapshot_dir:
                logger.info("model's repo_snapshot_dir is available, copying snapshot")
                shutil.copytree(
                    model.repo_snapshot_dir,
                    model_blob_file_or_dir,
                    dirs_exist_ok=True,
                )
                is_repo_downloaded = True

        base_meta = model_blob_meta.ModelBlobMeta(
            name=name,
            model_type=cls.HANDLER_TYPE,
            handler_version=cls.HANDLER_VERSION,
            path=cls.MODEL_BLOB_FILE_OR_DIR,
            options=model_meta_schema.HuggingFacePipelineModelBlobOptions(
                task=task,
                batch_size=batch_size if batch_size is not None else 1,
                has_tokenizer=has_tokenizer,
                has_feature_extractor=has_feature_extractor,
                has_image_preprocessor=has_image_preprocessor,
                is_repo_downloaded=is_repo_downloaded,
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
            pkgs_requirements.append(model_env.ModelDependency(requirement="pytorch", pip_name="torch"))
        elif framework == "tf":
            pkgs_requirements.append(model_env.ModelDependency(requirement="tensorflow", pip_name="tensorflow"))
        model_meta.env.include_if_absent(
            pkgs_requirements, check_local_version=(type_utils.LazyType("transformers.Pipeline").isinstance(model))
        )
        model_meta.env.cuda_version = kwargs.get("cuda_version", handlers_utils.get_default_cuda_version())

    @staticmethod
    def _get_device_config(**kwargs: Unpack[model_types.HuggingFaceLoadOptions]) -> dict[str, str]:
        device_config: dict[str, Any] = {}
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        gpu_nums = 0
        if cuda_visible_devices is not None:
            gpu_nums = len(cuda_visible_devices.split(","))
        if (
            kwargs.get("use_gpu", False)
            and kwargs.get("device_map", None) is None
            and kwargs.get("device", None) is None
        ):
            if gpu_nums == 0 or gpu_nums > 1:
                # Use accelerator if there are multiple GPUs or no GPU
                device_config["device_map"] = "auto"
            else:
                device_config["device"] = "cuda"
        elif kwargs.get("device_map", None) is not None:
            device_config["device_map"] = kwargs["device_map"]
        elif kwargs.get("device", None) is not None:
            device_config["device"] = kwargs["device"]

        return device_config

    @staticmethod
    def _load_pickle_model(
        pickle_file: str,
        **kwargs: Unpack[model_types.HuggingFaceLoadOptions],
    ) -> huggingface_pipeline.HuggingFacePipelineModel:
        with open(pickle_file, "rb") as f:
            m = cloudpickle.load(f)
        assert isinstance(m, huggingface_pipeline.HuggingFacePipelineModel)
        torch_dtype: Optional[str] = None
        device_config = None
        if getattr(m, "device", None) is None and getattr(m, "device_map", None) is None:
            device_config = HuggingFacePipelineHandler._get_device_config(**kwargs)
            m.__dict__.update(device_config)

        if getattr(m, "torch_dtype", None) is None and kwargs.get("use_gpu", False):
            torch_dtype = "auto"
            m.__dict__.update(torch_dtype=torch_dtype)
        else:
            m.__dict__.update(torch_dtype=None)
        return m

    @classmethod
    def load_model(
        cls,
        name: str,
        model_meta: model_meta_api.ModelMetadata,
        model_blobs_dir_path: str,
        **kwargs: Unpack[model_types.HuggingFaceLoadOptions],
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
        is_repo_downloaded = model_blob_options.get("is_repo_downloaded", False)

        def _create_pipeline_from_dir(
            model_blob_file_or_dir_path: str,
            model_blob_options: model_meta_schema.HuggingFacePipelineModelBlobOptions,
            **kwargs: Unpack[model_types.HuggingFaceLoadOptions],
        ) -> "transformers.Pipeline":
            import transformers

            additional_pipeline_params = {}
            if model_blob_options.get("has_tokenizer", False):
                additional_pipeline_params["tokenizer"] = model_blob_file_or_dir_path
            if model_blob_options.get("has_feature_extractor", False):
                additional_pipeline_params["feature_extractor"] = model_blob_file_or_dir_path
            if model_blob_options.get("has_image_preprocessor", False):
                additional_pipeline_params["image_preprocessor"] = model_blob_file_or_dir_path

            with open(
                os.path.join(
                    model_blob_file_or_dir_path,
                    cls.ADDITIONAL_CONFIG_FILE,
                ),
                "rb",
            ) as f:
                pipeline_params = cloudpickle.load(f)

            device_config = HuggingFacePipelineHandler._get_device_config(**kwargs)

            m = transformers.pipeline(
                model_blob_options["task"],
                model=model_blob_file_or_dir_path,
                trust_remote_code=True,
                torch_dtype="auto",
                **additional_pipeline_params,
                **device_config,
            )

            # If the task is text-generation, and the tokenizer does not have a chat_template,
            # set the default chat template.
            if (
                hasattr(m, "task")
                and m.task == "text-generation"
                and hasattr(m.tokenizer, "chat_template")
                and not m.tokenizer.chat_template
            ):
                warnings.warn(
                    "The tokenizer does not have default chat_template. "
                    "Setting the chat_template to default ChatML template.",
                    UserWarning,
                    stacklevel=1,
                )
                logger.info(DEFAULT_CHAT_TEMPLATE)
                m.tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

            m.__dict__.update(pipeline_params)
            return m

        def _create_pipeline_from_model(
            model_blob_file_or_dir_path: str,
            m: huggingface_pipeline.HuggingFacePipelineModel,
            **kwargs: Unpack[model_types.HuggingFaceLoadOptions],
        ) -> "transformers.Pipeline":
            import transformers

            return transformers.pipeline(
                m.task,
                model=model_blob_file_or_dir_path,
                trust_remote_code=m.trust_remote_code,
                torch_dtype=getattr(m, "torch_dtype", None),
                revision=m.revision,
                # pass device or device_map when creating the pipeline
                **HuggingFacePipelineHandler._get_device_config(**kwargs),
                # pass other model_kwargs to transformers.pipeline.from_pretrained method
                **m.model_kwargs,
            )

        if os.path.isdir(model_blob_file_or_dir_path) and not is_repo_downloaded:
            # the logged model is a transformers.Pipeline object
            # weights of the model are saved in the directory
            return _create_pipeline_from_dir(model_blob_file_or_dir_path, model_blob_options, **kwargs)
        else:
            # case 1: LEGACY logging, repo snapshot is not logged
            if os.path.isfile(model_blob_file_or_dir_path):
                # LEGACY logging that had model as a pickle file in the model blob directory
                # the logged model is a huggingface_pipeline.HuggingFacePipelineModel object
                # the model_blob_file_or_dir_path is the pickle file that holds
                # the huggingface_pipeline.HuggingFacePipelineModel object
                # the snapshot of the repo is not logged
                return cls._load_pickle_model(model_blob_file_or_dir_path)
            else:
                assert os.path.isdir(model_blob_file_or_dir_path)
                # the logged model is a huggingface_pipeline.HuggingFacePipelineModel object
                # the pickle_file holds the huggingface_pipeline.HuggingFacePipelineModel object
                pickle_file = os.path.join(model_blob_file_or_dir_path, cls.MODEL_PICKLE_FILE)
                m = cls._load_pickle_model(pickle_file)

                # case 2: logging without the snapshot of the repo
                if not is_repo_downloaded:
                    # we return the huggingface_pipeline.HuggingFacePipelineModel object
                    return m
                # case 3: logging with the snapshot of the repo
                else:
                    # the model_blob_file_or_dir_path is the directory that holds
                    # weights of the model from `huggingface_hub.snapshot_download`
                    # the huggingface_pipeline.HuggingFacePipelineModel object is logged
                    # with a snapshot of the repo, we create a transformers.Pipeline object
                    # by reading the snapshot directory
                    return _create_pipeline_from_model(model_blob_file_or_dir_path, m, **kwargs)

    @classmethod
    def convert_as_custom_model(
        cls,
        raw_model: Union[huggingface_pipeline.HuggingFacePipelineModel, "transformers.Pipeline"],
        model_meta: model_meta_api.ModelMetadata,
        background_data: Optional[pd.DataFrame] = None,
        **kwargs: Unpack[model_types.HuggingFaceLoadOptions],
    ) -> custom_model.CustomModel:
        import transformers

        from snowflake.ml.model import custom_model

        def _create_custom_model(
            raw_model: "transformers.Pipeline",
            model_meta: model_meta_api.ModelMetadata,
        ) -> type[custom_model.CustomModel]:
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
                    elif raw_model.task == "text-generation":
                        # verify when the target method is __call__ and
                        # if the signature is default text-generation signature
                        # then use the HuggingFaceOpenAICompatibleModel to wrap the pipeline
                        if signature == openai_signatures._OPENAI_CHAT_SIGNATURE_SPEC:
                            wrapped_model = HuggingFaceOpenAICompatibleModel(pipeline=raw_model)

                            temp_res = X.apply(
                                lambda row: wrapped_model.generate_chat_completion(
                                    messages=row["messages"],
                                    max_completion_tokens=row.get("max_completion_tokens", None),
                                    temperature=row.get("temperature", None),
                                    stop_strings=row.get("stop", None),
                                    n=row.get("n", 1),
                                    stream=row.get("stream", False),
                                    top_p=row.get("top_p", 1.0),
                                    frequency_penalty=row.get("frequency_penalty", None),
                                    presence_penalty=row.get("presence_penalty", None),
                                ),
                                axis=1,
                            ).to_list()
                        else:
                            if len(signature.inputs) > 1:
                                input_data = X.to_dict("records")
                            # If it is only expecting one argument, Then it is expecting a list of something.
                            else:
                                input_data = X[signature.inputs[0].name].to_list()
                            temp_res = getattr(raw_model, target_method)(input_data)
                    else:
                        # TODO: remove conversational pipeline code
                        # For others, we could offer the whole dataframe as a list.
                        # Some of them may need some conversion
                        if hasattr(transformers, "ConversationalPipeline") and isinstance(
                            raw_model, transformers.ConversationalPipeline
                        ):
                            input_data = [
                                transformers.Conversation(
                                    text=conv_data["user_inputs"][0],
                                    past_user_inputs=conv_data["user_inputs"][1:],
                                    generated_responses=conv_data["generated_responses"],
                                )
                                for conv_data in X.to_dict("records")
                            ]
                        else:
                            if isinstance(raw_model, transformers.TableQuestionAnsweringPipeline):
                                X["table"] = X["table"].apply(json.loads)

                            # Most pipelines if it is expecting more than one arguments,
                            # it is expecting a list of dict, where each dict has keys corresponding to the argument.
                            if len(signature.inputs) > 1:
                                input_data = X.to_dict("records")
                            # If it is only expecting one argument, Then it is expecting a list of something.
                            else:
                                input_data = X[signature.inputs[0].name].to_list()
                        temp_res = getattr(raw_model, target_method)(input_data)

                    # Some huggingface pipeline will omit the outer list when there is only 1 input.
                    # Making it not aligned with the auto-inferred signature.
                    # If the output is a dict, we could blindly create a list containing that.
                    # Otherwise, creating pandas DataFrame won't succeed.
                    if (
                        (hasattr(transformers, "Conversation") and isinstance(temp_res, transformers.Conversation))
                        or isinstance(temp_res, dict)
                        or (
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
                        )
                    ):
                        temp_res = [temp_res]

                    if len(temp_res) == 0:
                        return pd.DataFrame()

                    if hasattr(transformers, "ConversationalPipeline") and isinstance(
                        raw_model, transformers.ConversationalPipeline
                    ):
                        temp_res = [[conv.generated_responses] for conv in temp_res]

                    # To concat those who outputs a list with one input.
                    if isinstance(temp_res[0], list):
                        if isinstance(temp_res[0][0], dict):
                            res = pd.DataFrame({0: temp_res})
                        else:
                            res = pd.DataFrame(temp_res)
                    else:
                        res = pd.DataFrame(temp_res)

                    if hasattr(res, "map"):
                        res = res.map(sanitize_output)
                    else:
                        res = res.applymap(sanitize_output)

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
        if (
            getattr(pipe, "tokenizer", None) is not None
            and pipe.tokenizer.pad_token_id is None
            and hasattr(pipe.model.config, "eos_token_id")
        ):
            if isinstance(pipe.model.config.eos_token_id, int):
                pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
            elif (
                isinstance(pipe.model.config.eos_token_id, list)
                and len(pipe.model.config.eos_token_id) > 0
                and isinstance(pipe.model.config.eos_token_id[0], int)
            ):
                pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id[0]
            else:
                warnings.warn(
                    f"Unexpected type of eos_token_id: {type(pipe.model.config.eos_token_id)}. "
                    "Not setting pad_token_id to eos_token_id.",
                    stacklevel=2,
                )

        _HFPipelineModel = _create_custom_model(pipe, model_meta)
        hg_pipe_model = _HFPipelineModel(custom_model.ModelContext())

        return hg_pipe_model


class HuggingFaceOpenAICompatibleModel:
    """
    A class to wrap a Hugging Face text generation model and provide an
    OpenAI-compatible chat completion interface.
    """

    def __init__(self, pipeline: "transformers.Pipeline") -> None:
        """
        Initializes the model and tokenizer.

        Args:
            pipeline (transformers.pipeline): The Hugging Face pipeline to wrap.
        """

        self.pipeline = pipeline
        self.model = self.pipeline.model
        self.tokenizer = self.pipeline.tokenizer

        self.model_name = self.pipeline.model.name_or_path

    def _apply_chat_template(self, messages: list[dict[str, Any]]) -> str:
        """
        Applies a chat template to a list of messages.
        If the tokenizer has a chat template, it uses that.
        Otherwise, it falls back to a simple concatenation.

        Args:
            messages (list[dict]): A list of message dictionaries, e.g.,
                                   [{"role": "user", "content": "Hello!"}, ...]

        Returns:
            The formatted prompt string ready for model input.
        """

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            # Use the tokenizer's built-in chat template if available
            # `tokenize=False` means it returns a string, not token IDs
            return self.tokenizer.apply_chat_template(  # type: ignore[no-any-return]
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback to a simple concatenation for models without a specific chat template
            # This is a basic example; real chat models often need specific formatting.
            prompt = ""
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                if role == "system":
                    prompt += f"System: {content}\n"
                elif role == "user":
                    prompt += f"User: {content}\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n"
            prompt += "Assistant:"  # Indicate that the assistant should respond
            return prompt

    def generate_chat_completion(
        self,
        messages: list[dict[str, Any]],
        max_completion_tokens: Optional[int] = None,
        stream: Optional[bool] = False,
        stop_strings: Optional[list[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        n: int = 1,
    ) -> dict[str, Any]:
        """
        Generates a chat completion response in an OpenAI-compatible format.

        Args:
            messages (list[dict]): A list of message dictionaries, e.g.,
                                   [{"role": "system", "content": "You are a helpful assistant."},
                                    {"role": "user", "content": "What is deep learning?"}]
            max_completion_tokens (int): The maximum number of completion tokens to generate.
            stop_strings (list[str]): A list of strings to stop generation.
            temperature (float): The temperature for sampling.
            top_p (float): The top-p value for sampling.
            stream (bool): Whether to stream the generation.
            frequency_penalty (float): The frequency penalty for sampling.
            presence_penalty (float): The presence penalty for sampling.
            n (int): The number of samples to generate.

        Returns:
            dict: An OpenAI-compatible dictionary representing the chat completion.
        """
        # Apply chat template to convert messages into a single prompt string

        prompt_text = self._apply_chat_template(messages)

        # Tokenize the prompt
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        prompt_tokens = inputs.input_ids.shape[1]

        from transformers import GenerationConfig

        generation_config = GenerationConfig(
            max_new_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            stop_strings=stop_strings,
            stream=stream,
            num_return_sequences=n,
            num_beams=max(1, n),  # must be >1
            repetition_penalty=frequency_penalty,
            # TODO: Handle diversity_penalty and num_beam_groups
            # not all models support them making it hard to support any huggingface model
            # diversity_penalty=presence_penalty if n > 1 else None,
            # num_beam_groups=max(2, n) if presence_penalty else 1,
            do_sample=False,
        )

        # Generate text
        output_ids = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            generation_config=generation_config,
        )

        generated_texts = []
        completion_tokens = 0
        total_tokens = prompt_tokens
        for output_id in output_ids:
            # The output_ids include the input prompt
            # Decode the generated text, excluding the input prompt
            # so we slice to get only new tokens
            generated_tokens = output_id[prompt_tokens:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)

            # Calculate completion tokens
            completion_tokens += len(generated_tokens)
            total_tokens += len(generated_tokens)

        choices = []
        for i, generated_text in enumerate(generated_texts):
            choices.append(
                {
                    "index": i,
                    "message": {"role": "assistant", "content": generated_text},
                    "logprobs": None,  # Not directly supported in this basic implementation
                    "finish_reason": "stop",  # Assuming stop for simplicity
                }
            )

        # Construct OpenAI-compatible response
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }
        return response
