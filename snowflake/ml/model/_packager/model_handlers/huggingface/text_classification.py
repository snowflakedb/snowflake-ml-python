from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from typing_extensions import override

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager.model_handlers.huggingface import _task_handler
from snowflake.ml.model._signatures import core as model_signature_core

if TYPE_CHECKING:
    import transformers


class TextClassificationTaskHandler(_task_handler.HuggingFaceTaskHandler):
    """Handles text classification pipelines."""

    @override
    def run_inference(
        self,
        raw_model: "transformers.Pipeline",
        signature: model_signature.ModelSignature,
        target_method: str,
        X: pd.DataFrame,
        **kwargs: Any,
    ) -> Any:
        input_data = X[signature.inputs[0].name].to_list()
        # TextClassificationPipeline._sanitize_parameters uses isinstance(top_k, int)
        # which rejects np.int64 values arriving from Snowflake UDF DataFrames.
        native_kwargs = {
            k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v
            for k, v in kwargs.items()
        }
        return getattr(raw_model, target_method)(input_data, **native_kwargs)

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
