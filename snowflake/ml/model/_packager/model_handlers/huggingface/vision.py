import io
from typing import TYPE_CHECKING, Any

import pandas as pd

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager.model_handlers.huggingface import _task_handler
from snowflake.ml.model._signatures import core as model_signature_core

if TYPE_CHECKING:
    import transformers


class VisionTaskHandler(_task_handler.HuggingFaceTaskHandler):
    """Handles image/vision pipelines that need bytes-to-PIL conversion.

    Covers ImageClassification, ImageToText, ImageFeatureExtraction,
    ObjectDetection, DocumentQuestionAnswering, and VisualQuestionAnswering.
    """

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
        if len(signature.inputs) > 1:
            # Multi-input (e.g. DocumentQA: image + question): convert image bytes
            # to PIL, pass other columns as-is
            def process_image_row(row: pd.Series) -> Any:
                pil_image = Image.open(io.BytesIO(row[input_col]))
                extra_kwargs = {k: row[k] for k in row.index if k != input_col}
                return getattr(raw_model, target_method)(pil_image, **extra_kwargs)

            return X.apply(process_image_row, axis=1).to_list()
        else:
            images = [Image.open(io.BytesIO(img_bytes)) for img_bytes in X[input_col].to_list()]
            return getattr(raw_model, target_method)(images)

    def _needs_list_wrapping(
        self,
        raw_model: "transformers.Pipeline",
        signature: model_signature.ModelSignature,
        result: Any,
        input_size: int,
    ) -> bool:
        # Single-input batch mode can return a bare dict (e.g. image-classification
        # with 1 image). Also handle FeatureGroupSpec outputs where the outer list
        # is dropped for single inputs (e.g. object-detection → [{box1}, {box2}]).
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
