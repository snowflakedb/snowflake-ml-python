"""Pickle compatibility for models logged before ``HuggingFacePipelineModel`` was removed.

New code should use ``snowflake.ml.model.models.huggingface.TransformersPipeline``.
"""

from snowflake.ml.model.models.huggingface import (
    TransformersPipeline as HuggingFacePipelineModel,
)

__all__ = ["HuggingFacePipelineModel"]
