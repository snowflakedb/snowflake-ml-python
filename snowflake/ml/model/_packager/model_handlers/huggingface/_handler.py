import io
import json
import logging
import os
import shutil
import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, cast, final

import cloudpickle
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
from snowflake.ml.model._packager.model_handlers.huggingface import (
    _openai_chat_wrapper,
    _utils as _hf_utils,
)
from snowflake.ml.model._packager.model_handlers_migrator import base_migrator
from snowflake.ml.model._packager.model_meta import (
    model_blob_meta,
    model_meta as model_meta_api,
    model_meta_schema,
)
from snowflake.ml.model._signatures import (
    core as model_signature_core,
    utils as model_signature_utils,
)
from snowflake.ml.model.models import (
    huggingface as huggingface_base,
    huggingface_pipeline,
)
from snowflake.snowpark._internal import utils as snowpark_utils

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import transformers


def _is_transformers_type(obj: Any, class_name: str) -> bool:
    """Safely check isinstance against a transformers class that may not exist in all versions."""
    import transformers

    cls = getattr(transformers, class_name, None)
    return cls is not None and isinstance(obj, cls)


@final
class TransformersPipelineHandler(
    _base.BaseModelHandler[
        Union[
            huggingface_base.TransformersPipeline,
            huggingface_pipeline.HuggingFacePipelineModel,
            "transformers.Pipeline",
        ]
    ]
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
    ) -> TypeGuard[
        Union[
            huggingface_base.TransformersPipeline,
            huggingface_pipeline.HuggingFacePipelineModel,
            "transformers.Pipeline",
        ]
    ]:
        if type_utils.LazyType("transformers.Pipeline").isinstance(model):
            return True
        if isinstance(model, huggingface_pipeline.HuggingFacePipelineModel):
            return True
        if isinstance(model, huggingface_base.TransformersPipeline):
            return True
        return False

    @classmethod
    def cast_model(
        cls,
        model: model_types.SupportedModelType,
    ) -> Union[
        huggingface_base.TransformersPipeline,
        huggingface_pipeline.HuggingFacePipelineModel,
        "transformers.Pipeline",
    ]:
        if type_utils.LazyType("transformers.Pipeline").isinstance(model):
            return model
        elif isinstance(model, huggingface_pipeline.HuggingFacePipelineModel) or isinstance(
            model, huggingface_base.TransformersPipeline
        ):
            return model
        else:
            raise ValueError(f"Model {model} is not a valid Hugging Face model.")

    @classmethod
    def save_model(
        cls,
        name: str,
        model: Union[
            huggingface_base.TransformersPipeline,
            huggingface_pipeline.HuggingFacePipelineModel,
            "transformers.Pipeline",
        ],
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
            framework = getattr(model, "framework", None)
            batch_size = getattr(model, "_batch_size", None) or getattr(model, "batch_size", None)
            tokenizer = getattr(model, "tokenizer", None)
            if tokenizer:
                has_chat_template = bool(getattr(tokenizer, "chat_template", None))
            else:
                has_chat_template = False
        else:
            assert isinstance(model, huggingface_pipeline.HuggingFacePipelineModel) or isinstance(
                model, huggingface_base.TransformersPipeline
            )
            task = model.task
            framework = getattr(model, "framework", None)
            batch_size = getattr(model, "batch_size", None)
            has_chat_template = getattr(model, "has_chat_template", False)

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
            assert isinstance(model, huggingface_pipeline.HuggingFacePipelineModel) or isinstance(
                model, huggingface_base.TransformersPipeline
            )
            params = {**model.__dict__, **model.model_kwargs}

        inferred_pipe_sig = model_signature_utils.huggingface_pipeline_signature_auto_infer(
            task,
            params=params,
            has_chat_template=has_chat_template,
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
                        f" {type(model).__name__} object. "
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
            # transformers>=5 removed the `modelcard` attribute from Pipeline,
            # but save_pretrained still references it. Setting to None avoids AttributeError.
            if not hasattr(model, "modelcard"):
                model.modelcard = None  # type:ignore[attr-defined]
            model.save_pretrained(  # type:ignore[attr-defined]
                save_path,
                safe_serialization=True,  # creates safetensors instead of pytorch binaries or pt files
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
        ] + _hf_utils.get_requirements_from_task(
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
    ) -> Union[huggingface_pipeline.HuggingFacePipelineModel, huggingface_base.TransformersPipeline]:
        with open(pickle_file, "rb") as f:
            m = cloudpickle.load(f)
        assert isinstance(m, huggingface_pipeline.HuggingFacePipelineModel) or isinstance(
            m, huggingface_base.TransformersPipeline
        )
        torch_dtype: Optional[str] = None
        device_config = None
        if getattr(m, "device", None) is None and getattr(m, "device_map", None) is None:
            device_config = TransformersPipelineHandler._get_device_config(**kwargs)
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
    ) -> Union[
        huggingface_pipeline.HuggingFacePipelineModel, huggingface_base.TransformersPipeline, "transformers.Pipeline"
    ]:
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

            device_config = TransformersPipelineHandler._get_device_config(**kwargs)

            m = transformers.pipeline(
                model_blob_options["task"],
                model=model_blob_file_or_dir_path,
                trust_remote_code=True,
                torch_dtype="auto",
                **additional_pipeline_params,
                **device_config,
            )

            m.__dict__.update(pipeline_params)
            return m

        def _create_pipeline_from_model(
            model_blob_file_or_dir_path: str,
            m: Union[huggingface_pipeline.HuggingFacePipelineModel, huggingface_base.TransformersPipeline],
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
                **TransformersPipelineHandler._get_device_config(**kwargs),
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
        raw_model: Union[
            huggingface_pipeline.HuggingFacePipelineModel,
            huggingface_base.TransformersPipeline,
            "transformers.Pipeline",
        ],
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
            ) -> Callable[..., pd.DataFrame]:
                @custom_model._internal_inference_api
                def fn(self: custom_model.CustomModel, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
                    # These zero-shot pipelines cannot take a list of dict as input like other multi input
                    # pipelines, thus dealing separately.
                    if (
                        _is_transformers_type(raw_model, "ZeroShotAudioClassificationPipeline")
                        or _is_transformers_type(raw_model, "ZeroShotClassificationPipeline")
                        or _is_transformers_type(raw_model, "ZeroShotImageClassificationPipeline")
                        or _is_transformers_type(raw_model, "ZeroShotObjectDetectionPipeline")
                    ):

                        def process_zero_shot_row(row: pd.Series) -> Any:
                            input_val = row[signature.inputs[0].name]
                            # Convert bytes to PIL for image-based zero-shot pipelines
                            if _is_transformers_type(
                                raw_model, "ZeroShotImageClassificationPipeline"
                            ) or _is_transformers_type(raw_model, "ZeroShotObjectDetectionPipeline"):
                                from PIL import Image

                                input_val = Image.open(io.BytesIO(input_val))
                            return getattr(raw_model, target_method)(
                                input_val, candidate_labels=row["candidate_labels"]
                            )

                        temp_res = X.apply(process_zero_shot_row, axis=1).to_list()
                    elif raw_model.task == "text-generation":
                        # verify when the target method is __call__ and
                        # if the signature is default text-generation signature
                        # then use the HuggingFaceOpenAICompatibleModel to wrap the pipeline
                        if signature in openai_signatures._OPENAI_CHAT_SIGNATURE_SPECS:
                            wrapped_model = _openai_chat_wrapper.HuggingFaceOpenAICompatibleModel(pipeline=raw_model)

                            temp_res = X.apply(
                                lambda row: wrapped_model.generate_chat_completion(
                                    messages=row["messages"],
                                    **_hf_utils._resolve_chat_params(row, kwargs),
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
                    elif _is_transformers_type(raw_model, "ImageClassificationPipeline"):
                        # Image classification expects PIL Images. Convert bytes to PIL Images.
                        from PIL import Image

                        input_col = signature.inputs[0].name
                        images = [Image.open(io.BytesIO(img_bytes)) for img_bytes in X[input_col].to_list()]
                        temp_res = getattr(raw_model, target_method)(images)
                    elif (
                        _is_transformers_type(raw_model, "ImageToTextPipeline")
                        or _is_transformers_type(raw_model, "ImageFeatureExtractionPipeline")
                        or _is_transformers_type(raw_model, "ObjectDetectionPipeline")
                        or _is_transformers_type(raw_model, "DocumentQuestionAnsweringPipeline")
                        or _is_transformers_type(raw_model, "VisualQuestionAnsweringPipeline")
                    ):
                        # Image pipelines that need bytes→PIL conversion.
                        # HuggingFace's load_image() does not accept raw bytes.
                        from PIL import Image

                        input_col = signature.inputs[0].name
                        if len(signature.inputs) > 1:
                            # Multi-input: convert image bytes to PIL, pass other columns as-is
                            def process_image_row(row: pd.Series) -> Any:
                                pil_image = Image.open(io.BytesIO(row[input_col]))
                                kwargs = {k: row[k] for k in row.index if k != input_col}
                                return getattr(raw_model, target_method)(pil_image, **kwargs)

                            temp_res = X.apply(process_image_row, axis=1).to_list()
                        else:
                            # Single-input: convert all image bytes to PIL
                            images = [Image.open(io.BytesIO(img_bytes)) for img_bytes in X[input_col].to_list()]
                            temp_res = getattr(raw_model, target_method)(images)
                    elif _is_transformers_type(raw_model, "AutomaticSpeechRecognitionPipeline"):
                        # ASR pipeline accepts a single audio input (bytes, str, np.ndarray, or dict),
                        # not a list. Process each audio input individually.
                        input_col = signature.inputs[0].name
                        audio_inputs = X[input_col].to_list()
                        temp_res = [getattr(raw_model, target_method)(audio) for audio in audio_inputs]
                    elif _is_transformers_type(raw_model, "VideoClassificationPipeline"):
                        # Video classification expects file paths. Write bytes to temp files,
                        # process them, and clean up.
                        import tempfile

                        input_col = signature.inputs[0].name
                        video_bytes_list = X[input_col].to_list()
                        temp_file_paths = []
                        temp_files = []
                        try:
                            # TODO: parallelize this if needed
                            for video_bytes in video_bytes_list:
                                temp_file = tempfile.NamedTemporaryFile()
                                temp_file.write(video_bytes)
                                temp_file_paths.append(temp_file.name)
                                temp_files.append(temp_file)
                            temp_res = getattr(raw_model, target_method)(temp_file_paths)
                        finally:
                            for f in temp_files:
                                f.close()
                    else:
                        # TODO: remove conversational pipeline code
                        # For others, we could offer the whole dataframe as a list.
                        # Some of them may need some conversion
                        if _is_transformers_type(raw_model, "ConversationalPipeline"):
                            input_data = [
                                transformers.Conversation(
                                    text=conv_data["user_inputs"][0],
                                    past_user_inputs=conv_data["user_inputs"][1:],
                                    generated_responses=conv_data["generated_responses"],
                                )
                                for conv_data in X.to_dict("records")
                            ]
                        else:
                            if _is_transformers_type(raw_model, "TableQuestionAnsweringPipeline"):
                                X["table"] = X["table"].apply(json.loads)

                            # Most pipelines if it is expecting more than one arguments,
                            # it is expecting a list of dict, where each dict has keys corresponding to the argument.
                            if len(signature.inputs) > 1:
                                input_data = X.to_dict("records")
                            # If it is only expecting one argument, Then it is expecting a list of something.
                            else:
                                input_data = X[signature.inputs[0].name].to_list()
                        temp_res = getattr(raw_model, target_method)(input_data)

                    # Some huggingface pipelines omit the outer list when there is only 1 input,
                    # making the output not aligned with the auto-inferred signature.
                    #
                    # Expected output shape is list[result], one result per input row.
                    # When the outer list is dropped for a single input we need to re-wrap.
                    #
                    # Cases:
                    #   - bare dict
                    #       (e.g. text-classification → {"label": ..., "score": ...}) → wrap
                    #   - list[dict] with FeatureGroupSpec output
                    #       (e.g. fill-mask → [{"token": "a"}, {"token": "b"}]) → outer list was dropped, wrap
                    #   - Conversation object
                    #       (legacy transformers <4.42) → wrap
                    #   - list[list[dict]] with FeatureGroupSpec → already correct, don't wrap
                    _is_group_output = (
                        len(signature.outputs) == 1
                        and isinstance(signature.outputs[0], model_signature_core.FeatureGroupSpec)
                        and signature.outputs[0]._shape is not None
                    )
                    # TODO (SNOW-3107908): Create subclass handlers for all tasks to keep it manageable.
                    _needs_wrap = (
                        isinstance(temp_res, dict)
                        or _is_transformers_type(temp_res, "Conversation")
                        or (
                            X.shape[0] == 1
                            and _is_group_output
                            and isinstance(temp_res, list)
                            and len(temp_res) > 0
                            and not isinstance(temp_res[0], list)
                        )
                    )
                    if _needs_wrap:
                        temp_res = [temp_res]

                    if len(temp_res) == 0:
                        return pd.DataFrame()

                    if _is_transformers_type(raw_model, "ConversationalPipeline"):
                        temp_res = [[conv.generated_responses] for conv in temp_res]

                    # To concat those who outputs a list with one input.
                    # if `signature.outputs` is single valued and is a FeatureGroupSpec,
                    # we create a DataFrame with one column and the values are stored as a dictionary.
                    # Otherwise, we create a DataFrame with the output as the column.
                    if len(signature.outputs) == 1 and isinstance(
                        signature.outputs[0], model_signature_core.FeatureGroupSpec
                    ):
                        # creating a dataframe with one column
                        res = pd.DataFrame({signature.outputs[0].name: temp_res})
                    else:
                        res = pd.DataFrame(temp_res)

                    if hasattr(res, "map"):
                        res = res.map(_hf_utils.sanitize_output)
                    else:
                        res = res.applymap(_hf_utils.sanitize_output)

                    return model_signature_utils.rename_pandas_df(data=res, features=signature.outputs)

                return fn

            type_method_dict: dict[str, Any] = {"_allows_kwargs": True}
            for target_method_name, sig in model_meta.signatures.items():
                type_method_dict[target_method_name] = fn_factory(raw_model, sig, target_method_name)

            _HFPipelineModel = type(
                "_HFPipelineModel",
                (custom_model.CustomModel,),
                type_method_dict,
            )

            return _HFPipelineModel

        if isinstance(raw_model, huggingface_pipeline.HuggingFacePipelineModel) or isinstance(
            raw_model, huggingface_base.TransformersPipeline
        ):
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
