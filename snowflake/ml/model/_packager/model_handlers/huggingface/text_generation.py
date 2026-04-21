from typing import TYPE_CHECKING, Any

import pandas as pd
from typing_extensions import override

from snowflake.ml.model import model_signature, openai_signatures
from snowflake.ml.model._packager.model_handlers.huggingface import (
    _openai_chat_wrapper,
    _task_handler,
    _utils as _hf_utils,
)

if TYPE_CHECKING:
    import transformers


class TextGenerationTaskHandler(_task_handler.HuggingFaceTaskHandler):
    """Handles text-generation pipelines, including OpenAI-compatible chat completions."""

    @override
    def run_inference(
        self,
        raw_model: "transformers.Pipeline",
        signature: model_signature.ModelSignature,
        target_method: str,
        X: pd.DataFrame,
        **kwargs: Any,
    ) -> Any:
        # Models logged with a chat-compatible signature (messages/choices) use the
        # OpenAI wrapper. Models without a chat template (e.g. plain GPT-2 text
        # completion) fall through to the standard pipeline call below.
        if signature in openai_signatures._OPENAI_CHAT_SIGNATURE_SPECS:
            wrapped_model = _openai_chat_wrapper.HuggingFaceOpenAICompatibleModel(pipeline=raw_model)

            return X.apply(
                lambda row: wrapped_model.generate_chat_completion(
                    messages=row["messages"],
                    **_hf_utils._resolve_chat_params(row, kwargs),
                ),
                axis=1,
            ).to_list()
        else:
            if len(signature.inputs) > 1:
                # Multi-input: pipeline([{"text": "prompt", "max_new_tokens": 50}, ...])
                input_data = X.to_dict(orient="records")
            else:
                # Single-input: pipeline(["prompt1", "prompt2", ...])
                input_data = X[signature.inputs[0].name].to_list()
            return getattr(raw_model, target_method)(input_data)

    @override
    def _needs_list_wrapping(
        self,
        raw_model: "transformers.Pipeline",
        signature: model_signature.ModelSignature,
        result: Any,
        input_size: int,
    ) -> bool:
        # Text-generation with OpenAI signatures returns a list per row, no wrapping needed.
        # Regular text-generation returns list[dict] or dict depending on input count.
        return isinstance(result, dict)

    @override
    def _format_result(
        self,
        raw_model: "transformers.Pipeline",
        result: Any,
    ) -> list[Any]:
        return list(result)
