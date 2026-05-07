import io
from typing import TYPE_CHECKING, Any

import pandas as pd
from typing_extensions import override

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager.model_handlers.huggingface import _task_handler

if TYPE_CHECKING:
    import transformers


class ImageClassificationTaskHandler(_task_handler.HuggingFaceTaskHandler):
    """Handles image classification pipelines."""

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
        images = [Image.open(io.BytesIO(img_bytes)) for img_bytes in X[input_col].to_list()]
        return getattr(raw_model, target_method)(images, **kwargs)

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
