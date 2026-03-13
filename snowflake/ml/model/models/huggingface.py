import enum
import json
import logging
import os
import warnings
from typing import Any, Optional, Union

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


class TransformersPipeline:
    def __init__(
        self,
        task: str,
        model: str,
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
            task: The task that pipeline will be used for. Must be explicitly provided.
                For available tasks, please refer Transformers's documentation.
            model: The model that will be used by the pipeline to make predictions. This can only be a model identifier
                currently. Must be explicitly provided.
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
                If None is passed, the `huggingface_hub` package must be installed and the model artifacts will be
                downloaded from the HuggingFace repository.
            allow_patterns: If provided, only files matching at least one pattern are downloaded.
            ignore_patterns: If provided, files matching any of the patterns are not downloaded.
            kwargs: Additional keyword arguments passed along to the specific pipeline init (see the documentation for
                the corresponding pipeline class for possible values).


        .. # noqa: DAR003
        """
        self.secret_identifier, uses_secret = self._parse_secret_identifier(token_or_secret)

        logging_mode = self._determine_logging_mode(compute_pool_for_log)

        if model_kwargs is None:
            model_kwargs = {}
        token_or_secret = self._handle_deprecated_auth_token(model_kwargs, token_or_secret)

        tokenizer = kwargs.get("tokenizer", None)
        self._validate_inputs(task, model)

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

        self.has_chat_template = self._has_chat_template(
            logging_mode=logging_mode,
            task=task,
            repo_snapshot_dir=repo_snapshot_dir,
        )

        self.model = model
        self.task = task
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

        Raises:
            ImportError: If huggingface_hub is not installed and compute_pool_for_log is None.
        """
        if compute_pool_for_log is not None:
            return _LoggingMode.REMOTE

        # Check if huggingface_hub is available for snapshot download
        try:
            import huggingface_hub as _  # noqa: F401

            return _LoggingMode.SNAPSHOT_DOWNLOAD
        except ImportError:
            raise ImportError(
                "The `huggingface_hub` package is required when `compute_pool_for_log` is None. "
                "Please install it with: pip install huggingface_hub"
            )

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

    def _validate_inputs(
        self,
        task: Optional[str],
        model: Optional[str],
    ) -> None:
        """Validate the input arguments for pipeline creation.

        Args:
            task: The pipeline task.
            model: The model identifier.

        Raises:
            RuntimeError: If input arguments are invalid.
        """
        if task is None:
            raise RuntimeError(
                "Impossible to instantiate a pipeline without a task being specified. "
                "Please provide a task name explicitly."
            )

        if model is None:
            raise RuntimeError(
                "Impossible to instantiate a pipeline without a model being specified. "
                "Please provide an identifier to a pretrained model."
            )

        if not isinstance(model, str):
            raise RuntimeError(f"Impossible to use non-string model as input for class {self.__class__.__name__}.")

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

    _CHAT_TEMPLATE_TASKS = frozenset(
        {
            "text-generation",
            "image-text-to-text",
            "video-text-to-text",
            "audio-text-to-text",
        }
    )

    def _has_chat_template(
        self,
        logging_mode: _LoggingMode,
        task: str,
        repo_snapshot_dir: Optional[str],
    ) -> Optional[bool]:
        """Determine whether the model has a chat template.

        Chat template detection only applies to snapshot-downloaded models
        with a task that supports chat-based inference.

        Args:
            logging_mode: The current logging mode.
            task: The normalized pipeline task.
            repo_snapshot_dir: Path to the downloaded snapshot, or None.

        Returns:
            True if a chat template is found, False otherwise.
            None for models that do not belong to the chat template tasks or are not in SNAPSHOT_DOWNLOAD mode.
        """
        if (
            logging_mode == _LoggingMode.SNAPSHOT_DOWNLOAD
            and task in self._CHAT_TEMPLATE_TASKS
            and repo_snapshot_dir is not None
        ):
            return self._detect_chat_template(repo_snapshot_dir)
        return None

    @staticmethod
    def _detect_chat_template(local_repo_path: str) -> Optional[bool]:
        """
        Checks if a local Hugging Face repository has a chat template defined.

        Args:
            local_repo_path: The path to the downloaded Hugging Face repository directory.

        Returns:
            True if a chat template is found, False otherwise.
            None if the repository does not have a chat template or the chat template is not valid JSON.
        """
        tokenizer_config_path = os.path.join(local_repo_path, "tokenizer_config.json")
        chat_template_jinja_path = os.path.join(local_repo_path, "chat_template.jinja")

        # 1. Check for the dedicated chat_template.jinja file
        if os.path.exists(chat_template_jinja_path):
            return True

        # 2. Check within the tokenizer_config.json file
        if os.path.exists(tokenizer_config_path):
            try:
                with open(tokenizer_config_path, encoding="utf-8") as f:
                    config_data = json.load(f)
                # The 'chat_template' attribute stores the Jinja template as a string
                if config_data.get("chat_template") is not None:
                    return True
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {tokenizer_config_path}")
                return None
            except Exception as e:
                logger.error(f"An error occurred: {e}")
                return None

        # If neither the file nor the config attribute is found, it likely uses a default
        # class-specific template or has none defined explicitly in the repo.
        return False
