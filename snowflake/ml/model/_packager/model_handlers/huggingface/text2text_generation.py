from typing import TYPE_CHECKING, Any, Optional

import pandas as pd
from packaging import version
from typing_extensions import override

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager.model_handlers.huggingface import _task_handler

if TYPE_CHECKING:
    import transformers


class Text2TextGenerationTaskHandler(_task_handler.HuggingFaceTaskHandler):
    """Handles text2text generation pipelines, removed in transformers 5.x."""

    @override
    def get_transformers_upper_bound(self) -> Optional[version.Version]:
        return version.Version("5")

    @override
    def run_inference(
        self,
        raw_model: "transformers.Pipeline",
        signature: model_signature.ModelSignature,
        target_method: str,
        X: pd.DataFrame,
        **kwargs: Any,
    ) -> Any:
        input_data = X[signature.inputs[0].name].to_list()
        # HF text2text-generation accepts generation params as flat kwargs.
        filtered_kwargs = _task_handler._filter_none_kwargs(kwargs)
        return getattr(raw_model, target_method)(input_data, **filtered_kwargs)

    @override
    def _needs_list_wrapping(
        self,
        raw_model: "transformers.Pipeline",
        signature: model_signature.ModelSignature,
        result: Any,
        input_size: int,
    ) -> bool:
        # Pipeline returns a bare dict instead of list[dict] for single inputs.
        return isinstance(result, dict)
