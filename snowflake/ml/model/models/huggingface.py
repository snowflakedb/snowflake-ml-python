import enum
import logging
import warnings
from typing import Any, Optional, Union

from packaging import version

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model.compute_pool import DEFAULT_CPU_COMPUTE_POOL

logger = logging.getLogger(__name__)


_TELEMETRY_PROJECT = "MLOps"
_TELEMETRY_SUBPROJECT = "ModelManagement"


class _LoggingMode(enum.Enum):
    """Defines how the model artifacts are handled during logging."""

    # REMOTE model logging, default behavior. The model metadata like model id, task etc are captured
    # and used to generate SYSTEM$IMPORT_MODEL statement when using `registry.log_model`.
    REMOTE = "remote"

    # SNAPSHOT_DOWNLOAD mode: The model artifacts are downloaded from the HuggingFace repository.
    # This mode is used when using `registry.log_model` with `compute_pool_for_log=None`.
    SNAPSHOT_DOWNLOAD = "snapshot_download"

    # CONFIG_ONLY mode: The model metadata like model id, task etc are captured along with model config
    # The model is logged similar to a regular model, without downloading the model artifacts.
    # During inference, the model artifacts are downloaded from the HuggingFace repository.
    # This model works only when the service has an EAI enabled to have egress to HuggingFace hosts.
    CONFIG_ONLY = "config_only"


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


        .. # noqa: DAR003
        """
        import transformers

        self.secret_identifier, uses_secret = self._parse_secret_identifier(token_or_secret)

        logging_mode = self._determine_logging_mode(compute_pool_for_log)

        if model_kwargs is None:
            model_kwargs = {}
        token_or_secret = self._handle_deprecated_auth_token(model_kwargs, token_or_secret)

        hub_kwargs = self._build_hub_kwargs(transformers, revision, token_or_secret, trust_remote_code)

        config = kwargs.get("config", None)
        tokenizer = kwargs.get("tokenizer", None)
        framework = kwargs.get("framework", None)
        feature_extractor = kwargs.get("feature_extractor", None)
        self._validate_inputs(task, model, config, tokenizer, feature_extractor)

        normalized_task, model, revision, config_obj = self._resolve_task_and_model(
            transformers=transformers,
            task=task,
            model=model,
            config=config,
            framework=framework,
            revision=revision,
            token_or_secret=token_or_secret,
            trust_remote_code=trust_remote_code,
            hub_kwargs=hub_kwargs,
            model_kwargs=model_kwargs,
            logging_mode=logging_mode,
        )

        self._validate_device_map(kwargs, model_kwargs)

        repo_snapshot_dir = self._download_snapshot_if_needed(
            logging_mode=logging_mode,
            uses_secret=uses_secret,
            model=model,
            revision=revision,
            token_or_secret=token_or_secret,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

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

    @staticmethod
    def _parse_secret_identifier(token_or_secret: Optional[str]) -> tuple[Optional[str], bool]:
        """Parse the token_or_secret to extract secret identifier if it's a fully qualified name.

        Args:
            token_or_secret: The token or secret string to parse.

        Returns:
            A tuple of (secret_identifier, uses_secret) where secret_identifier is the fully
            qualified secret name if parsed successfully, None otherwise. uses_secret indicates
            whether a secret is being used.
        """
        if token_or_secret is None or not isinstance(token_or_secret, str):
            return None, False

        db, schema, secret_name = sql_identifier.parse_fully_qualified_name(token_or_secret)
        if db is not None and schema is not None and secret_name is not None:
            secret_identifier = sql_identifier.get_fully_qualified_name(
                db=db,
                schema=schema,
                object=secret_name,
            )
            return secret_identifier, True

        logger.info("The token_or_secret is not a fully qualified secret name. It will be used as is.")
        return None, False

    @staticmethod
    def _determine_logging_mode(compute_pool_for_log: Optional[str]) -> _LoggingMode:
        """Determine the logging mode based on compute_pool_for_log and available packages.

        Args:
            compute_pool_for_log: The compute pool for logging, or None for local handling.

        Returns:
            The appropriate LoggingMode enum value.
        """
        if compute_pool_for_log is not None:
            return _LoggingMode.REMOTE

        # Check if huggingface_hub is available for snapshot download
        try:
            import huggingface_hub as _  # noqa: F401

            return _LoggingMode.SNAPSHOT_DOWNLOAD
        except ImportError:
            logger.info(
                "The model will be logged with metadata only. No model artifacts will be downloaded. "
                "During deployment, the model artifacts will be downloaded from the HuggingFace repository."
            )
            return _LoggingMode.CONFIG_ONLY

    @staticmethod
    def _handle_deprecated_auth_token(model_kwargs: dict[str, Any], token_or_secret: Optional[str]) -> Optional[str]:
        """Handle the deprecated use_auth_token argument.

        Args:
            model_kwargs: The model kwargs dict (will be modified in place).
            token_or_secret: The current token_or_secret value.

        Returns:
            The resolved token_or_secret value.

        Raises:
            ValueError: If both token_or_secret and use_auth_token are specified.
        """
        use_auth_token: Optional[str] = model_kwargs.pop("use_auth_token", None)
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
            return use_auth_token
        return token_or_secret

    @staticmethod
    def _build_hub_kwargs(
        transformers: Any,
        revision: Optional[str],
        token_or_secret: Optional[str],
        trust_remote_code: Optional[bool],
    ) -> dict[str, Any]:
        """Build the hub_kwargs dict for HuggingFace API calls.

        Args:
            transformers: The transformers module.
            revision: The model revision.
            token_or_secret: The auth token or secret.
            trust_remote_code: Whether to trust remote code.

        Returns:
            The hub_kwargs dictionary.
        """
        hub_kwargs: dict[str, Any] = {
            "revision": revision,
            "token": token_or_secret,
            "trust_remote_code": trust_remote_code,
            "_commit_hash": None,
        }

        # Backward compatibility since HF interface change.
        if version.parse(transformers.__version__) < version.parse("4.32.0"):
            hub_kwargs["use_auth_token"] = hub_kwargs["token"]
            del hub_kwargs["token"]

        return hub_kwargs

    def _validate_inputs(
        self,
        task: Optional[str],
        model: Optional[str],
        config: Optional[Any],
        tokenizer: Optional[Any],
        feature_extractor: Optional[Any],
    ) -> None:
        """Validate the input arguments for pipeline creation.

        Args:
            task: The pipeline task.
            model: The model identifier.
            config: The config argument.
            tokenizer: The tokenizer argument.
            feature_extractor: The feature extractor argument.

        Raises:
            RuntimeError: If input arguments are invalid.
        """
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

        if model is not None and not isinstance(model, str):
            raise RuntimeError(f"Impossible to use non-string model as input for class {self.__class__.__name__}.")

        if config is not None and not isinstance(config, str):
            raise RuntimeError(
                f"Impossible to use non-string config as input for class {self.__class__.__name__}. "
                "Use transformers.Pipeline object if required."
            )

    def _load_config(
        self,
        transformers: Any,
        config_or_model: str,
        task: Optional[str],
        hub_kwargs: dict[str, Any],
        model_kwargs: dict[str, Any],
    ) -> Any:
        """Load the config from HuggingFace.

        Args:
            transformers: The transformers module.
            config_or_model: The config name or model name to load config from.
            task: The pipeline task.
            hub_kwargs: The hub kwargs for HuggingFace API.
            model_kwargs: Additional model kwargs.

        Returns:
            The loaded config object.
        """
        config_obj = transformers.AutoConfig.from_pretrained(
            config_or_model, _from_pipeline=task, **hub_kwargs, **model_kwargs
        )
        hub_kwargs["_commit_hash"] = config_obj._commit_hash
        return config_obj

    def _resolve_task_and_model(
        self,
        transformers: Any,
        task: Optional[str],
        model: Optional[str],
        config: Optional[str],
        framework: Optional[str],
        revision: Optional[str],
        token_or_secret: Optional[str],
        trust_remote_code: Optional[bool],
        hub_kwargs: dict[str, Any],
        model_kwargs: dict[str, Any],
        logging_mode: _LoggingMode,
    ) -> tuple[str, str, Optional[str], Optional[Any]]:
        """Resolve the task and model, loading config if needed.

        Args:
            transformers: The transformers module.
            task: The pipeline task (may be None).
            model: The model identifier (may be None).
            config: The config identifier.
            framework: The framework to use.
            revision: The model revision.
            token_or_secret: The auth token or secret.
            trust_remote_code: Whether to trust remote code.
            hub_kwargs: The hub kwargs for HuggingFace API.
            model_kwargs: Additional model kwargs.
            logging_mode: The current logging mode.

        Returns:
            A tuple of (normalized_task, model, revision, config_obj).

        Raises:
            RuntimeError: Raised when the input argument cannot determine the pipeline.
            ValueError: Raised when the pipeline contains remote code but trust_remote_code is not set or False.
            ValueError: Raised when having conflicting arguments.
            KeyError: Raised when the task is not recognized.
        """
        config_obj = None

        # Load config only in CONFIG_ONLY mode (need metadata for task resolution)
        if logging_mode == _LoggingMode.CONFIG_ONLY:
            if isinstance(config, str):
                config_obj = self._load_config(transformers, config, task, hub_kwargs, model_kwargs)
            elif config is None and isinstance(model, str):
                config_obj = self._load_config(transformers, model, task, hub_kwargs, model_kwargs)

        # Handle custom tasks from config
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

        # Infer task from model if not provided
        if task is None and model is not None:
            if logging_mode == _LoggingMode.CONFIG_ONLY:
                task = transformers.pipelines.get_task(model, token_or_secret)
            else:
                raise ValueError("task must be explicitly provided when using remote logging or snapshot download mode")

        # Resolve task to normalized form
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
            try:
                (
                    normalized_task,
                    targeted_task,
                    task_options,
                ) = transformers.pipelines.check_task(task)
            except KeyError:
                # In non-CONFIG_ONLY modes, custom tasks from config are not loaded.
                # If the task is not recognized, assume it's a custom task that will be
                # validated at deployment time when the model config is loaded.
                # This is needed because the local transformer version may not have the task,
                # but remote version is installed from pip which may recognize the task.
                if logging_mode != _LoggingMode.CONFIG_ONLY and model is not None:
                    normalized_task = task
                    targeted_task = None
                    task_options = None
                else:
                    raise

        # Handle default model when none provided
        if model is None:
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
            # Load config for default model in CONFIG_ONLY mode
            if logging_mode == _LoggingMode.CONFIG_ONLY and config is None and isinstance(model, str):
                config_obj = self._load_config(transformers, model, task, hub_kwargs, model_kwargs)

        # At this point, normalized_task and model are guaranteed to be set:
        # - normalized_task is set via check_task() or custom_tasks
        # - model is either provided or set via get_default_model_and_revision()
        assert normalized_task is not None, "normalized_task should be set"
        assert model is not None, "model should be set"

        return normalized_task, model, revision, config_obj

    @staticmethod
    def _validate_device_map(kwargs: dict[str, Any], model_kwargs: dict[str, Any]) -> None:
        """Validate device_map arguments for conflicts.

        Args:
            kwargs: The kwargs dict.
            model_kwargs: The model kwargs dict.

        Raises:
            ValueError: If device_map is specified in both kwargs and model_kwargs.
        """
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

    @staticmethod
    def _download_snapshot_if_needed(
        logging_mode: _LoggingMode,
        uses_secret: bool,
        model: Optional[str],
        revision: Optional[str],
        token_or_secret: Optional[str],
        allow_patterns: Optional[Union[list[str], str]],
        ignore_patterns: Optional[Union[list[str], str]],
    ) -> Optional[str]:
        """Download the model snapshot if in SNAPSHOT_DOWNLOAD mode.

        Args:
            logging_mode: The current logging mode.
            uses_secret: Whether a secret is being used for authentication.
            model: The model identifier.
            revision: The model revision.
            token_or_secret: The auth token or secret.
            allow_patterns: Patterns of files to include.
            ignore_patterns: Patterns of files to exclude.

        Returns:
            The path to the downloaded snapshot directory, or None if not downloaded.
        """
        if logging_mode != _LoggingMode.SNAPSHOT_DOWNLOAD:
            return None

        if uses_secret:
            # Cannot use secret for local download, will download during deployment
            return None

        if model is None:
            return None

        try:
            import huggingface_hub as hf_hub

            return hf_hub.snapshot_download(
                repo_id=model,
                revision=revision,
                token=token_or_secret,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
        except ImportError:
            logger.info("huggingface_hub package is not installed, skipping snapshot download")
            return None
