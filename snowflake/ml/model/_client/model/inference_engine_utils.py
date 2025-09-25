from typing import Any, Optional, Union

from snowflake.ml.model._client.ops import service_ops


def _get_inference_engine_args(
    experimental_options: Optional[dict[str, Any]],
) -> Optional[service_ops.InferenceEngineArgs]:

    if not experimental_options:
        return None

    if "inference_engine" not in experimental_options:
        raise ValueError("inference_engine is required in experimental_options")

    return service_ops.InferenceEngineArgs(
        inference_engine=experimental_options["inference_engine"],
        inference_engine_args_override=experimental_options.get("inference_engine_args_override"),
    )


def _enrich_inference_engine_args(
    inference_engine_args: service_ops.InferenceEngineArgs,
    gpu_requests: Optional[Union[str, int]] = None,
) -> Optional[service_ops.InferenceEngineArgs]:
    """Enrich inference engine args with model path and tensor parallelism settings.

    Args:
        inference_engine_args: The original inference engine args
        gpu_requests: The number of GPUs requested

    Returns:
        Enriched inference engine args

    Raises:
        ValueError: Invalid gpu_requests
    """
    if inference_engine_args.inference_engine_args_override is None:
        inference_engine_args.inference_engine_args_override = []

    gpu_count = None

    # Set tensor-parallelism if gpu_requests is specified
    if gpu_requests is not None:
        # assert gpu_requests is a string or an integer before casting to int
        try:
            gpu_count = int(gpu_requests)
            if gpu_count > 0:
                inference_engine_args.inference_engine_args_override.append(f"--tensor-parallel-size={gpu_count}")
            else:
                raise ValueError(f"GPU count must be greater than 0, got {gpu_count}")
        except ValueError:
            raise ValueError(f"Invalid gpu_requests: {gpu_requests} with type {type(gpu_requests).__name__}")

    return inference_engine_args
