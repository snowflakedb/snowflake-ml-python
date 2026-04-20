from typing import TYPE_CHECKING, Any

import pandas as pd
from typing_extensions import override

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager.model_handlers.huggingface import _task_handler
from snowflake.ml.model._signatures import core as model_signature_core

if TYPE_CHECKING:
    import transformers


class DefaultTaskHandler(_task_handler.HuggingFaceTaskHandler):
    """Minimal fallback handler for pipelines not covered by a dedicated handler."""

    @override
    def run_inference(
        self,
        raw_model: "transformers.Pipeline",
        signature: model_signature.ModelSignature,
        target_method: str,
        X: pd.DataFrame,
        **kwargs: Any,
    ) -> Any:
        # Most pipelines expecting more than one argument take a list of dicts,
        # where each dict has keys corresponding to the arguments.
        if len(signature.inputs) > 1:
            input_data = X.to_dict(orient="records")
        # If it is only expecting one argument, it expects a list of values.
        else:
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
        # Some huggingface pipelines omit the outer list when there is only 1 input,
        # making the output not aligned with the auto-inferred signature.
        #
        # Expected output shape is list[result], one result per input row.
        # When the outer list is dropped for a single input we need to re-wrap.
        #
        # Cases:
        #   - bare dict
        #       (e.g. text-classification -> {"label": ..., "score": ...}) -> wrap
        #   - list[dict] with FeatureGroupSpec output
        #       (e.g. fill-mask -> [{"token": "a"}, {"token": "b"}]) -> outer list was dropped, wrap
        #   - list[list[dict]] with FeatureGroupSpec -> already correct, don't wrap
        _is_group_output = (
            len(signature.outputs) == 1
            and isinstance(signature.outputs[0], model_signature_core.FeatureGroupSpec)
            and signature.outputs[0]._shape is not None
        )
        return isinstance(result, dict) or (
            input_size == 1
            and _is_group_output
            and isinstance(result, list)
            and len(result) > 0
            and not isinstance(result[0], list)
        )
