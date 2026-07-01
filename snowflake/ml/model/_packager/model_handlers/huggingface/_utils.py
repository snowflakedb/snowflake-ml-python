import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_LAZY_UPLOAD_SECRET_ERROR = (
    "model upload: HuggingFace lazy upload cannot resolve auth from a Snowflake secret during local logging. "
    "Set the HF_TOKEN environment variable or pass a HuggingFace token when constructing the model."
)


def download_token_for_lazy_upload(model: object) -> Optional[str]:
    """Return the HuggingFace token to use for lazy upload from a wrapper model.

    Args:
        model: A HuggingFace wrapper with optional ``token_or_secret`` and ``secret_identifier`` attributes.

    Returns:
        The HuggingFace auth token when the wrapper was constructed with a plain token, otherwise None.

    Raises:
        ValueError: If the wrapper was constructed with a Snowflake secret reference.
    """
    if getattr(model, "secret_identifier", None) is not None:
        raise ValueError(_LAZY_UPLOAD_SECRET_ERROR)
    token_or_secret = getattr(model, "token_or_secret", None)
    if isinstance(token_or_secret, str):
        return token_or_secret
    return None


def _resolve_chat_params(row: pd.Series, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Resolve chat completion params from kwargs (ParamSpec) or DataFrame row columns."""
    src = kwargs if kwargs else row
    return {
        "max_completion_tokens": src.get("max_completion_tokens", None),
        "temperature": src.get("temperature", None),
        "stop_strings": src.get("stop", None),
        "n": src.get("n", 1),
        "stream": src.get("stream", False),
        "top_p": src.get("top_p", 1.0),
        "frequency_penalty": src.get("frequency_penalty", None),
        "presence_penalty": src.get("presence_penalty", None),
    }


def is_transformers_type(obj: Any, class_name: str) -> bool:
    """Safely check isinstance against a transformers class that may not exist in all versions."""
    import transformers

    cls = getattr(transformers, class_name, None)
    return cls is not None and isinstance(obj, cls)


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
