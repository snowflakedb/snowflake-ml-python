from typing import TYPE_CHECKING, Any

import pandas as pd
from typing_extensions import override

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager.model_handlers.huggingface import _task_handler

if TYPE_CHECKING:
    import transformers


class ZeroShotAudioClassificationTaskHandler(_task_handler.HuggingFaceTaskHandler):
    """Handles zero-shot audio classification pipelines."""

    @override
    def run_inference(
        self,
        raw_model: "transformers.Pipeline",
        signature: model_signature.ModelSignature,
        target_method: str,
        X: pd.DataFrame,
        **kwargs: Any,
    ) -> Any:
        def process_row(row: pd.Series) -> Any:
            return getattr(raw_model, target_method)(
                row[signature.inputs[0].name], candidate_labels=row["candidate_labels"]
            )

        return X.apply(process_row, axis=1).to_list()
