import logging
import warnings
from typing import Any, Optional, Union

from packaging import version

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model.compute_pool import DEFAULT_CPU_COMPUTE_POOL

logger = logging.getLogger(__name__)


_TELEMETRY_PROJECT = "MLOps"
_TELEMETRY_SUBPROJECT = "ModelManagement"


class TransformersPipeline:
    def __init__(
        self,
        task: Optional[str] = None,
        model: Optional[str] = None,
        *,
        revision: Optional[str] = None,
        token_or_secret: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
        compute_pool_for_log: Optional[str] = DEFAULT_CPU_COMPUTE_POOL,
        # repo snapshot download args
        allow_patterns: Optional[Union[list[str], str]] = None,
        ignore_patterns: Optional[Union[list[str], str]] = None,
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
            token_or_secret: The token to use as HTTP bearer authorization for remote files. Defaults to None.
              The token can be a token or a secret. If a secret is provided, it must a fully qualified secret name.
            trust_remote_code: Whether or not to allow for custom code defined on the Hub in their own modeling,
                configuration, tokenization or even pipeline files. This option should only be set to `True` for
                repositories you trust and in which you have read the code, as it will execute code present on the Hub.
                Defaults to None.
            model_kwargs: Additional dictionary of keyword arguments passed along to the model's `from_pretrained(...,`.
                Defaults to None.
            compute_pool_for_log: The compute pool to use for logging the model. Defaults to DEFAULT_CPU_COMPUTE_POOL.
                If a string is provided, it will be used as the compute pool name. This override allows for logging
                the model when there is no system compute pool available.
                If None is passed,
                    if `huggingface_hub` is installed, the model artifacts will be downloaded
                    from the HuggingFace repository.
                    otherwise, the only the metadata will be logged to snowflake.
            allow_patterns: If provided, only files matching at least one pattern are downloaded.
            ignore_patterns: If provided, files matching any of the patterns are not downloaded.
            kwargs: Additional keyword arguments passed along to the specific pipeline init (see the documentation for
                the corresponding pipeline class for possible values).

        Raises:
            RuntimeError: Raised when the input argument cannot determine the pipeline.
            ValueError: Raised when the pipeline contains remote code but trust_remote_code is not set or False.
            ValueError: Raised when having conflicting arguments.

        .. # noqa: DAR003
        """
        import transformers

        config = kwargs.get("config", None)
        tokenizer = kwargs.get("tokenizer", None)
        framework = kwargs.get("framework", None)
        feature_extractor = kwargs.get("feature_extractor", None)

        self.secret_identifier: Optional[str] = None
        uses_secret = False
        if token_or_secret is not None and isinstance(token_or_secret, str):
            db, schema, secret_name = sql_identifier.parse_fully_qualified_name(token_or_secret)
            if db is not None and schema is not None and secret_name is not None:
                self.secret_identifier = sql_identifier.get_fully_qualified_name(
                    db=db,
                    schema=schema,
                    object=secret_name,
                )
                uses_secret = True
            else:
                logger.info("The token_or_secret is not a fully qualified secret name. It will be used as is.")

        can_download_snapshot = False
        if compute_pool_for_log is None:
            try:
                import huggingface_hub as hf_hub

                can_download_snapshot = True
            except ImportError:
                pass

        if compute_pool_for_log is None and not can_download_snapshot:
            logger.info(
                "The model will be logged with metadata only. No model artifacts will be downloaded. "
                "During deployment, the model artifacts will be downloaded from the HuggingFace repository."
            )

        # ==== Start pipeline logic from transformers ====
        if model_kwargs is None:
            model_kwargs = {}

        use_auth_token = model_kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers.",
                FutureWarning,
                stacklevel=2,
            )
            if token_or_secret is not None:
                raise ValueError(
                    "`token_or_secret` and `use_auth_token` are both specified. "
                    "Please set only the argument `token_or_secret`."
                )
            token_or_secret = use_auth_token

        hub_kwargs = {
            "revision": revision,
            "token": token_or_secret,
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
            raise RuntimeError(f"Impossible to use non-string model as input for class {self.__class__.__name__}.")

        # ==== Start pipeline logic (Config) from transformers ====

        # Config is the primordial information item.
        # Instantiate config if needed
        config_obj = None

        if not can_download_snapshot:
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
                    f"Impossible to use non-string config as input for class {self.__class__.__name__}. "
                    "Use transformers.Pipeline object if required."
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
            task = transformers.pipelines.get_task(model, token_or_secret)

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
                "Using a pipeline without specifying a model name and revision in production is not recommended.",
                stacklevel=2,
            )
            if not can_download_snapshot and config is None and isinstance(model, str):
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
                    " will most likely encounter unexpected behavior. Please remove `device` and keep `device_map`.",
                    stacklevel=2,
                )

        repo_snapshot_dir: Optional[str] = None
        if can_download_snapshot and not uses_secret:
            try:
                repo_snapshot_dir = hf_hub.snapshot_download(
                    repo_id=model,
                    revision=revision,
                    token=token_or_secret,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                )
            except ImportError:
                logger.info("huggingface_hub package is not installed, skipping snapshot download")

        # ==== End pipeline logic from transformers ====

        self.model = model
        self.task = normalized_task
        self.revision = revision
        self.token_or_secret = token_or_secret
        self.trust_remote_code = trust_remote_code
        self.model_kwargs = model_kwargs
        self.tokenizer = tokenizer

        self.repo_snapshot_dir = repo_snapshot_dir
        self.compute_pool_for_log = compute_pool_for_log
        self.__dict__.update(kwargs)
