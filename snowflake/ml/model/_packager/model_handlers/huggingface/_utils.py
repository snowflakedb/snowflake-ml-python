import logging
from typing import Any

import numpy as np
import pandas as pd

from snowflake.ml.model._packager.model_env import model_env

logger = logging.getLogger(__name__)


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
