from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd
from packaging import version

from snowflake.ml.model import model_signature

if TYPE_CHECKING:
    import transformers


class HuggingFaceTaskHandler(ABC):
    """Abstract base class for task-specific HuggingFace pipeline inference logic.

    Each subclass encapsulates the input preprocessing, inference execution,
    result wrapping, and formatting for a specific HuggingFace pipeline task type.
    This enables clean separation of task-specific concerns while sharing common
    post-processing in the handler.
    """

    def get_transformers_upper_bound(self) -> Optional[version.Version]:
        """Return the maximum supported transformers version for this task, or None if unbounded.

        Override in subclasses that require a specific transformers version ceiling.

        Returns:
            A packaging.version.Version upper bound, or None if any version is supported.
        """
        return None

    @abstractmethod
    def run_inference(
        self,
        raw_model: "transformers.Pipeline",
        signature: model_signature.ModelSignature,
        target_method: str,
        X: pd.DataFrame,
        **kwargs: Any,
    ) -> Any:
        """Execute task-specific inference on the input DataFrame.

        Args:
            raw_model: The HuggingFace pipeline.
            signature: The model signature for this method.
            target_method: The pipeline method to call (e.g. "__call__").
            X: Input DataFrame.
            **kwargs: Additional keyword arguments (e.g. ParamSpec params).
        """
        ...

    def _needs_list_wrapping(
        self,
        raw_model: "transformers.Pipeline",
        signature: model_signature.ModelSignature,
        result: Any,
        input_size: int,
    ) -> bool:
        """Check if the result needs to be wrapped in a list for single-input cases.

        Some HuggingFace pipelines omit the outer list when there is only one input,
        which breaks DataFrame construction. Defaults to False; override in subclasses
        where the pipeline drops the outer list for single inputs.

        Args:
            raw_model: The HuggingFace pipeline.
            signature: The model signature for this method.
            result: The raw inference result to check.
            input_size: Number of input rows.

        Returns:
            True if the outer list should be re-added for single-input cases; False otherwise.
        """
        return False

    def _format_result(
        self,
        raw_model: "transformers.Pipeline",
        result: Any,
    ) -> list[Any]:
        """Apply task-specific result formatting after list wrapping.

        Override in subclasses that need to transform the result shape
        (e.g., conversational pipelines).

        Args:
            raw_model: The HuggingFace pipeline.
            result: The inference result to format.

        Returns:
            Formatted result as a list.
        """
        return list(result)
