import io
from typing import TYPE_CHECKING, Any

import pandas as pd
from typing_extensions import override

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager.model_handlers.huggingface import _task_handler
from snowflake.ml.model._signatures import core as model_signature_core

if TYPE_CHECKING:
    import transformers


class DocumentQuestionAnsweringTaskHandler(_task_handler.HuggingFaceTaskHandler):
    """Handles document question answering pipelines."""

    @override
    def run_inference(
        self,
        raw_model: "transformers.Pipeline",
        signature: model_signature.ModelSignature,
        target_method: str,
        X: pd.DataFrame,
        **kwargs: Any,
    ) -> Any:
        from PIL import Image

        input_col = signature.inputs[0].name

        # Multi-input: convert image bytes to PIL, pass other columns as-is.
        def process_row(row: pd.Series) -> Any:
            pil_image = Image.open(io.BytesIO(row[input_col]))
            extra_kwargs = {k: row[k] for k in row.index if k != input_col}
            return getattr(raw_model, target_method)(pil_image, **extra_kwargs)

        return X.apply(process_row, axis=1).to_list()

    @override
    def _needs_list_wrapping(
        self,
        raw_model: "transformers.Pipeline",
        signature: model_signature.ModelSignature,
        result: Any,
        input_size: int,
    ) -> bool:
        # Single-input batch mode can return a bare dict. Also handle
        # FeatureGroupSpec outputs where the outer list is dropped for
        # single inputs.
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
