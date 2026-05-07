import io
from typing import TYPE_CHECKING, Any

import pandas as pd
from typing_extensions import override

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager.model_handlers.huggingface import _task_handler

if TYPE_CHECKING:
    import transformers


class ZeroShotObjectDetectionTaskHandler(_task_handler.HuggingFaceTaskHandler):
    """Handles zero-shot object detection pipelines."""

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

        def process_row(row: pd.Series) -> Any:
            pil_image = Image.open(io.BytesIO(row[signature.inputs[0].name]))
            return getattr(raw_model, target_method)(pil_image, candidate_labels=row["candidate_labels"], **kwargs)

        return X.apply(process_row, axis=1).to_list()
