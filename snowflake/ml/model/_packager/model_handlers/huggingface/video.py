import tempfile
from typing import TYPE_CHECKING, Any

import pandas as pd

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager.model_handlers.huggingface import _task_handler
from snowflake.ml.model._signatures import core as model_signature_core

if TYPE_CHECKING:
    import transformers


class VideoTaskHandler(_task_handler.HuggingFaceTaskHandler):
    """Handles video classification pipelines.

    Video classification expects file paths. Bytes are written to temp files,
    processed, then cleaned up.
    """

    def run_inference(
        self,
        raw_model: "transformers.Pipeline",
        signature: model_signature.ModelSignature,
        target_method: str,
        X: pd.DataFrame,
        **kwargs: Any,
    ) -> Any:
        input_col = signature.inputs[0].name
        video_bytes_list = X[input_col].to_list()
        temp_file_paths: list[str] = []
        temp_files = []
        try:
            for video_bytes in video_bytes_list:
                temp_file = tempfile.NamedTemporaryFile()
                temp_file.write(video_bytes)
                temp_file.flush()
                temp_file_paths.append(temp_file.name)
                temp_files.append(temp_file)
            return getattr(raw_model, target_method)(temp_file_paths)
        finally:
            for f in temp_files:
                f.close()

    def _needs_list_wrapping(
        self,
        raw_model: "transformers.Pipeline",
        signature: model_signature.ModelSignature,
        result: Any,
        input_size: int,
    ) -> bool:
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
