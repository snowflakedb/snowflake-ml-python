import json
from typing import TYPE_CHECKING, Any

import pandas as pd
from typing_extensions import override

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager.model_handlers.huggingface import _task_handler

if TYPE_CHECKING:
    import transformers


class TableQuestionAnsweringTaskHandler(_task_handler.HuggingFaceTaskHandler):
    """Handles table question answering pipelines."""

    @override
    def run_inference(
        self,
        raw_model: "transformers.Pipeline",
        signature: model_signature.ModelSignature,
        target_method: str,
        X: pd.DataFrame,
        **kwargs: Any,
    ) -> Any:
        X["table"] = X["table"].apply(json.loads)
        input_data = X.to_dict(orient="records")
        return getattr(raw_model, target_method)(input_data)

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
