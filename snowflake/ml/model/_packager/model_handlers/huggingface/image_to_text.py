import io
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd
from packaging import version
from typing_extensions import override

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager.model_handlers.huggingface import _task_handler
from snowflake.ml.model._signatures import core as model_signature_core

if TYPE_CHECKING:
    import transformers


class ImageToTextTaskHandler(_task_handler.HuggingFaceTaskHandler):
    """Handles image-to-text pipelines, removed in transformers 5.x."""

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
        from PIL import Image

        input_col = signature.inputs[0].name
        images = [Image.open(io.BytesIO(img_bytes)) for img_bytes in X[input_col].to_list()]
        return getattr(raw_model, target_method)(images)

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
