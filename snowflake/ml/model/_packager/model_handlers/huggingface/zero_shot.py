import io
from typing import TYPE_CHECKING, Any

import pandas as pd

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager.model_handlers.huggingface import (
    _task_handler,
    _utils as _hf_utils,
)

if TYPE_CHECKING:
    import transformers


class ZeroShotTaskHandler(_task_handler.HuggingFaceTaskHandler):
    """Handles zero-shot classification and detection pipelines.

    These pipelines cannot take a list of dicts as input like other multi-input
    pipelines, so each row is processed individually with candidate_labels.
    """

    def run_inference(
        self,
        raw_model: "transformers.Pipeline",
        signature: model_signature.ModelSignature,
        target_method: str,
        X: pd.DataFrame,
        **kwargs: Any,
    ) -> Any:
        def process_zero_shot_row(row: pd.Series) -> Any:
            input_val = row[signature.inputs[0].name]
            # Convert bytes to PIL for image-based zero-shot pipelines
            if _hf_utils.is_transformers_type(
                raw_model, "ZeroShotImageClassificationPipeline"
            ) or _hf_utils.is_transformers_type(raw_model, "ZeroShotObjectDetectionPipeline"):
                from PIL import Image

                input_val = Image.open(io.BytesIO(input_val))
            return getattr(raw_model, target_method)(input_val, candidate_labels=row["candidate_labels"])

        return X.apply(process_zero_shot_row, axis=1).to_list()

    def _needs_list_wrapping(
        self,
        raw_model: "transformers.Pipeline",
        signature: model_signature.ModelSignature,
        result: Any,
        input_size: int,
    ) -> bool:
        # apply().to_list() always produces a properly-shaped list, no wrapping needed.
        return False
