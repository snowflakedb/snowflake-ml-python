from typing import TYPE_CHECKING, Any

import pandas as pd

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager.model_handlers.huggingface import (
    _task_handler,
    _utils as _hf_utils,
)

if TYPE_CHECKING:
    import transformers


class ConversationalTaskHandler(_task_handler.HuggingFaceTaskHandler):
    """Handles legacy conversational pipelines (transformers < 4.42)."""

    def run_inference(
        self,
        raw_model: "transformers.Pipeline",
        signature: model_signature.ModelSignature,
        target_method: str,
        X: pd.DataFrame,
        **kwargs: Any,
    ) -> Any:
        import transformers

        input_data = [
            transformers.Conversation(
                text=conv_data["user_inputs"][0],
                past_user_inputs=conv_data["user_inputs"][1:],
                generated_responses=conv_data["generated_responses"],
            )
            for conv_data in X.to_dict(orient="records")
        ]
        return getattr(raw_model, target_method)(input_data)

    def _needs_list_wrapping(
        self,
        raw_model: "transformers.Pipeline",
        signature: model_signature.ModelSignature,
        result: Any,
        input_size: int,
    ) -> bool:
        return _hf_utils.is_transformers_type(result, "Conversation")

    def _format_result(
        self,
        raw_model: "transformers.Pipeline",
        result: Any,
    ) -> list[Any]:
        return [[conv.generated_responses] for conv in result]
