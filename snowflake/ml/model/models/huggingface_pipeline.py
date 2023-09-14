import warnings
from typing import Any, Dict, Optional

from packaging import version


class HuggingFacePipelineModel:
    def __init__(
        self,
        task: Optional[str] = None,
        model: Optional[str] = None,
        *,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Utility factory method to build a wrapper over transformers [`Pipeline`].
        When deploying, this wrapper will create a real pipeline object and loading tokenizers and models.

        For pipelines docs, please refer:
        https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline

        Args:
            task: The task that pipeline will be used. If None it would be inferred from model.
                For available tasks, please refer Transformers's documentation. Defaults to None.
            model: The model that will be used by the pipeline to make predictions. This can only be a model identifier
                currently. If not provided, the default for the `task` will be loaded. Defaults to None.
            revision: When passing a task name or a string model identifier: The specific model version to use. It can
                be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and
                other artifacts on huggingface.co, so `revision` can be any identifier allowed by git. Defaults to None.
            token: The token to use as HTTP bearer authorization for remote files. Defaults to None.
            trust_remote_code: Whether or not to allow for custom code defined on the Hub in their own modeling,
                configuration, tokenization or even pipeline files. This option should only be set to `True` for
                repositories you trust and in which you have read the code, as it will execute code present on the Hub.
                Defaults to None.
            model_kwargs: Additional dictionary of keyword arguments passed along to the model's `from_pretrained(...,`.
                Defaults to None.
            kwargs: Additional keyword arguments passed along to the specific pipeline init (see the documentation for
                the corresponding pipeline class for possible values).

        Return:
            A wrapper over transformers [`Pipeline`].

        Raises:
            RuntimeError: Raised when the input argument cannot determine the pipeline.
            ValueError: Raised when the pipeline contains remote code but trust_remote_code is not set or False.
            ValueError: Raised when having conflicting arguments.
        """
        import transformers

        config = kwargs.get("config", None)
        tokenizer = kwargs.get("tokenizer", None)
        framework = kwargs.get("framework", None)
        feature_extractor = kwargs.get("feature_extractor", None)

        # ==== Start pipeline logic from transformers ====
        if model_kwargs is None:
            model_kwargs = {}

        use_auth_token = model_kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        hub_kwargs = {
            "revision": revision,
            "token": token,
            "trust_remote_code": trust_remote_code,
            "_commit_hash": None,
        }

        # Backward compatibility since HF interface change.
        if version.parse(transformers.__version__) < version.parse("4.32.0"):
            # Backward compatibility since HF interface change.
            hub_kwargs["use_auth_token"] = hub_kwargs["token"]
            del hub_kwargs["token"]

        if task is None and model is None:
            raise RuntimeError(
                "Impossible to instantiate a pipeline without either a task or a model being specified. "
            )

        if model is None and tokenizer is not None:
            raise RuntimeError(
                "Impossible to instantiate a pipeline with tokenizer specified but not the model as the provided"
                " tokenizer may not be compatible with the default model. Please provide an identifier to a pretrained"
                " model when providing tokenizer."
            )
        if model is None and feature_extractor is not None:
            raise RuntimeError(
                "Impossible to instantiate a pipeline with feature_extractor specified but not the model as the "
                "provided feature_extractor may not be compatible with the default model. Please provide an identifier"
                " to a pretrained model when providing feature_extractor."
            )

        # ==== End pipeline logic from transformers ====

        # We only support string as model argument.

        if model is not None and not isinstance(model, str):
            raise RuntimeError(
                "Impossible to use non-string model as input for HuggingFacePipelineModel. Use transformers.Pipeline"
                " object if required."
            )

        # ==== Start pipeline logic (Config) from transformers ====

        # Config is the primordial information item.
        # Instantiate config if needed
        config_obj = None

        if isinstance(config, str):
            config_obj = transformers.AutoConfig.from_pretrained(
                config, _from_pipeline=task, **hub_kwargs, **model_kwargs
            )
            hub_kwargs["_commit_hash"] = config_obj._commit_hash
        elif config is None and isinstance(model, str):
            config_obj = transformers.AutoConfig.from_pretrained(
                model, _from_pipeline=task, **hub_kwargs, **model_kwargs
            )
            hub_kwargs["_commit_hash"] = config_obj._commit_hash
        # We only support string as config argument.
        elif config is not None and not isinstance(config, str):
            raise RuntimeError(
                "Impossible to use non-string config as input for HuggingFacePipelineModel. Use transformers.Pipeline"
                " object if required."
            )

        # ==== Start pipeline logic (Task) from transformers ====

        custom_tasks = {}
        if config_obj is not None and len(getattr(config_obj, "custom_pipelines", {})) > 0:
            custom_tasks = config_obj.custom_pipelines
            if task is None and trust_remote_code is not False:
                if len(custom_tasks) == 1:
                    task = list(custom_tasks.keys())[0]
                else:
                    raise RuntimeError(
                        "We can't infer the task automatically for this model as there are multiple tasks available. "
                        f"Pick one in {', '.join(custom_tasks.keys())}"
                    )

        if task is None and model is not None:
            task = transformers.pipelines.get_task(model, token)

        # Retrieve the task
        if task in custom_tasks:
            normalized_task = task
            targeted_task, task_options = transformers.pipelines.clean_custom_task(custom_tasks[task])
            if not trust_remote_code:
                raise ValueError(
                    "Loading this pipeline requires you to execute the code in the pipeline file in that"
                    " repo on your local machine. Make sure you have read the code there to avoid malicious use, then"
                    " set the option `trust_remote_code=True` to remove this error."
                )
        else:
            (
                normalized_task,
                targeted_task,
                task_options,
            ) = transformers.pipelines.check_task(task)

        # ==== Start pipeline logic (Model) from transformers ====

        # Use default model/config/tokenizer for the task if no model is provided
        if model is None:
            # At that point framework might still be undetermined
            (
                model,
                default_revision,
            ) = transformers.pipelines.get_default_model_and_revision(targeted_task, framework, task_options)
            revision = revision if revision is not None else default_revision
            warnings.warn(
                f"No model was supplied, defaulted to {model} and revision"
                f" {revision} ({transformers.pipelines.HUGGINGFACE_CO_RESOLVE_ENDPOINT}/{model}).\n"
                "Using a pipeline without specifying a model name and revision in production is not recommended."
            )
            if config is None and isinstance(model, str):
                config_obj = transformers.AutoConfig.from_pretrained(
                    model, _from_pipeline=task, **hub_kwargs, **model_kwargs
                )
                hub_kwargs["_commit_hash"] = config_obj._commit_hash

        if kwargs.get("device_map", None) is not None:
            if "device_map" in model_kwargs:
                raise ValueError(
                    'You cannot use both `pipeline(... device_map=..., model_kwargs={"device_map":...})` as those'
                    " arguments might conflict, use only one.)"
                )
            if kwargs.get("device", None) is not None:
                warnings.warn(
                    "Both `device` and `device_map` are specified. `device` will override `device_map`. You"
                    " will most likely encounter unexpected behavior. Please remove `device` and keep `device_map`."
                )

        # ==== End pipeline logic from transformers ====

        self.task = normalized_task
        self.model = model
        self.revision = revision
        self.token = token
        self.trust_remote_code = trust_remote_code
        self.model_kwargs = model_kwargs
        self.__dict__.update(kwargs)
