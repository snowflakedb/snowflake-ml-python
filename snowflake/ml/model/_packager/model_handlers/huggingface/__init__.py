"""HuggingFace Transformers pipeline model handler package.

Re-exports public symbols for backward-compatible imports:
  from snowflake.ml.model._packager.model_handlers.huggingface import TransformersPipelineHandler
  from snowflake.ml.model._packager.model_handlers import huggingface
"""

from snowflake.ml.model._packager.model_handlers.huggingface._handler import (
    TransformersPipelineHandler,
)
from snowflake.ml.model._packager.model_handlers.huggingface._openai_chat_wrapper import (
    HuggingFaceOpenAICompatibleModel,
)

__all__ = [
    "TransformersPipelineHandler",
    "HuggingFaceOpenAICompatibleModel",
]
