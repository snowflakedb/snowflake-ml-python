from typing import Any

from snowflake.ml.model import inference_engine as inference_engine_module
from snowflake.ml.model._client.ops import service_ops


def _get_inference_engine_args(
    inference_engine_options: dict[str, Any] | None,
) -> service_ops.InferenceEngineArgs | None:

    if not inference_engine_options:
        return None

    if "engine" not in inference_engine_options:
        raise ValueError("'engine' field is required in inference_engine_options")

    return service_ops.InferenceEngineArgs(
        inference_engine=inference_engine_module.InferenceEngine.from_value(inference_engine_options["engine"]),
        inference_engine_args_override=inference_engine_options.get("engine_args_override"),
    )
