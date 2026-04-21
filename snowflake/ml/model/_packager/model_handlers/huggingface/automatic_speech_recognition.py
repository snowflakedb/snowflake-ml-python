from typing import TYPE_CHECKING, Any

import pandas as pd
from typing_extensions import override

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager.model_handlers.huggingface import _task_handler

if TYPE_CHECKING:
    import transformers


class AutomaticSpeechRecognitionTaskHandler(_task_handler.HuggingFaceTaskHandler):
    """Handles automatic speech recognition pipelines.

    ASR pipelines accept a single audio input (bytes, str, np.ndarray, or dict),
    not a list. Each audio input is processed individually.
    """

    @override
    def run_inference(
        self,
        raw_model: "transformers.Pipeline",
        signature: model_signature.ModelSignature,
        target_method: str,
        X: pd.DataFrame,
        **kwargs: Any,
    ) -> Any:
        input_col = signature.inputs[0].name
        audio_inputs = X[input_col].to_list()
        return [getattr(raw_model, target_method)(audio) for audio in audio_inputs]
