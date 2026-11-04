from typing import Any, Callable
from unittest import mock

from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.registry import registry as model_registry


def log_model_embedding_local_ml_library(original: Callable[..., Any]) -> Callable[..., Any]:
    """Return a log_model wrapper that embeds the local snowflake-ml-python library.

    Pip-packaged models install snowflake-ml-python from PyPI. Integration tests run the workspace
    copy, whose version is not on PyPI until the release is published. An explicit
    ``embed_local_ml_library=False`` is preserved so a test can pin an already published build.
    Copying a ModelVersion does not accept save options, so those calls are left unchanged.

    Args:
        original: Registry.log_model implementation to invoke after adjusting options.

    Returns:
        A wrapper with the same calling convention as Registry.log_model.
    """

    def wrapper(self: Any, model: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(model, model_version_impl.ModelVersion):
            options = kwargs.get("options")
            options = {} if options is None else dict(options)
            options.setdefault("embed_local_ml_library", True)
            kwargs["options"] = options
        return original(self, model, *args, **kwargs)

    return wrapper


def patch_registry_log_model() -> Any:
    """Patch Registry.log_model so tests embed the workspace snowflake-ml-python.

    Returns:
        A started-on-enter patch context manager.
    """
    original_log_model = model_registry.Registry.log_model
    return mock.patch.object(
        model_registry.Registry,
        "log_model",
        new=log_model_embedding_local_ml_library(original_log_model),
    )
