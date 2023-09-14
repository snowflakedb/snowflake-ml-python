import asyncio
import http
import logging
import os
import sys
import tempfile
import threading
import traceback
import zipfile
from enum import Enum
from typing import Dict, List, Optional, cast

import pandas as pd
from gunicorn import arbiter
from starlette import applications, concurrency, requests, responses, routing


class _ModelLoadingState(Enum):
    """
    Enum class to represent various model loading state.
    """

    LOADING = "loading"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class CustomThread(threading.Thread):
    """
    Custom Thread implementation that overrides Thread.run.

    This is necessary because the default Thread implementation suppresses exceptions in child threads. The standard
    behavior involves the Thread class catching exceptions and throwing a SystemExit exception, which requires
    Thread.join to terminate the process. To address this, we overwrite Thread.run and use os._exit instead.

    We throw specific error code "Arbiter.APP_LOAD_ERROR" such that Gunicorn Arbiter master process will be killed,
    which then trigger the container to be marked as failed. This ensures the container becomes ready when all workers
    loaded the model successfully.
    """

    def run(self) -> None:
        try:
            super().run()
        except Exception:
            logger.error(traceback.format_exc())
            os._exit(arbiter.Arbiter.APP_LOAD_ERROR)


logger = logging.getLogger(__name__)
_LOADED_MODEL = None
_LOADED_META = None
_MODEL_CODE_DIR = "code"
_MODEL_LOADING_STATE = _ModelLoadingState.LOADING
_MODEL_LOADING_EVENT = threading.Event()
_CONCURRENT_REQUESTS_MAX: Optional[int] = None
_CONCURRENT_COUNTER = 0
_CONCURRENT_COUNTER_LOCK = asyncio.Lock()
TARGET_METHOD = None


def _run_setup() -> None:
    """Set up logging and load model into memory."""
    # Align the application logger's handler with Gunicorn's to capture logs from all processes.
    gunicorn_logger = logging.getLogger("gunicorn.error")
    logger.handlers = gunicorn_logger.handlers
    logger.setLevel(gunicorn_logger.level)

    logger.info(f"ENV: {os.environ}")

    global _LOADED_MODEL
    global _LOADED_META
    global _MODEL_LOADING_STATE
    global _MODEL_LOADING_EVENT
    global _CONCURRENT_REQUESTS_MAX
    global TARGET_METHOD

    try:
        MODEL_ZIP_STAGE_PATH = os.getenv("MODEL_ZIP_STAGE_PATH")
        assert MODEL_ZIP_STAGE_PATH, "Missing environment variable MODEL_ZIP_STAGE_PATH"

        TARGET_METHOD = os.getenv("TARGET_METHOD")

        _concurrent_requests_max_env = os.getenv("_CONCURRENT_REQUESTS_MAX", None)

        _CONCURRENT_REQUESTS_MAX = int(_concurrent_requests_max_env) if _concurrent_requests_max_env else None

        root_path = os.path.abspath(os.sep)
        model_zip_stage_path = os.path.join(root_path, MODEL_ZIP_STAGE_PATH)

        with tempfile.TemporaryDirectory() as tmp_dir:
            if zipfile.is_zipfile(model_zip_stage_path):
                extracted_dir = os.path.join(tmp_dir, "extracted_model_dir")
                logger.info(f"Extracting model zip from {model_zip_stage_path} to {extracted_dir}")
                with zipfile.ZipFile(model_zip_stage_path, "r") as model_zip:
                    if len(model_zip.namelist()) > 1:
                        model_zip.extractall(extracted_dir)
            else:
                raise RuntimeError(f"No model zip found at stage path: {model_zip_stage_path}")
            logger.info(f"Loading model from {extracted_dir} into memory")

            sys.path.insert(0, os.path.join(extracted_dir, _MODEL_CODE_DIR))
            from snowflake.ml.model import (
                _model as model_api,
                type_hints as model_types,
            )

            # Backward for <= 1.0.5
            if hasattr(model_api, "_load_model_for_deploy"):
                _LOADED_MODEL, _LOADED_META = model_api._load_model_for_deploy(extracted_dir)
            else:
                _LOADED_MODEL, _LOADED_META = model_api._load(
                    local_dir_path=extracted_dir,
                    as_custom_model=True,
                    meta_only=False,
                    options=model_types.ModelLoadOption(
                        {"use_gpu": cast(bool, os.environ.get("SNOWML_USE_GPU", False))}
                    ),
                )
            _MODEL_LOADING_STATE = _ModelLoadingState.SUCCEEDED
            logger.info("Successfully loaded model into memory")
            _MODEL_LOADING_EVENT.set()
    except Exception as e:
        _MODEL_LOADING_STATE = _ModelLoadingState.FAILED
        raise e


async def ready(request: requests.Request) -> responses.JSONResponse:
    """Check if the application is ready to serve requests.

    This endpoint is used to determine the readiness of the application to handle incoming requests. It returns an HTTP
    200 status code only when the model has been successfully loaded into memory. If the model has not yet been loaded,
    it responds with an HTTP 503 status code, which signals to the readiness probe to continue probing until the
    application becomes ready or until the client's timeout is reached.

    Args:
        request:
            The HTTP request object.

    Returns:
        A JSON response with status information:
        - HTTP 200 status code and {"status": "ready"} when the model is loaded and the application is ready.
        - HTTP 503 status code and {"status": "not ready"} when the model is not yet loaded.

    """
    if _MODEL_LOADING_STATE == _ModelLoadingState.SUCCEEDED:
        return responses.JSONResponse({"status": "ready"})
    return responses.JSONResponse({"status": "not ready"}, status_code=http.HTTPStatus.SERVICE_UNAVAILABLE)


def _do_predict(input_json: Dict[str, List[List[object]]]) -> responses.JSONResponse:
    from snowflake.ml.model.model_signature import FeatureSpec

    assert _LOADED_MODEL, "model is not loaded"
    assert _LOADED_META, "model metadata is not loaded"
    assert TARGET_METHOD, "Missing environment variable TARGET_METHOD"

    try:
        features = cast(List[FeatureSpec], _LOADED_META.signatures[TARGET_METHOD].inputs)
        dtype_map = {feature.name: feature.as_dtype() for feature in features}
        input_cols = [spec.name for spec in features]
        output_cols = [spec.name for spec in _LOADED_META.signatures[TARGET_METHOD].outputs]
        assert "data" in input_json, "missing data field in the request input"
        # The expression x[1:] is used to exclude the index of the data row.
        input_data = [x[1] for x in input_json["data"]]
        df = pd.json_normalize(input_data).astype(dtype=dtype_map)
        x = df[input_cols]
        assert len(input_data) != 0 and not all(not row for row in input_data), "empty data"
    except Exception as e:
        error_message = f"Input data malformed: {str(e)}\n{traceback.format_exc()}"
        return responses.JSONResponse({"error": error_message}, status_code=http.HTTPStatus.BAD_REQUEST)

    try:
        predictions_df = getattr(_LOADED_MODEL, TARGET_METHOD)(x)
        predictions_df.columns = output_cols
        # Use _ID to keep the order of prediction result and associated features.
        _KEEP_ORDER_COL_NAME = "_ID"
        if _KEEP_ORDER_COL_NAME in df.columns:
            predictions_df[_KEEP_ORDER_COL_NAME] = df[_KEEP_ORDER_COL_NAME]
        response = {"data": [[i, row] for i, row in enumerate(predictions_df.to_dict(orient="records"))]}
        return responses.JSONResponse(response)
    except Exception as e:
        error_message = f"Prediction failed: {str(e)}\n{traceback.format_exc()}"
        return responses.JSONResponse({"error": error_message}, status_code=http.HTTPStatus.BAD_REQUEST)


async def predict(request: requests.Request) -> responses.JSONResponse:
    """Endpoint to make predictions based on input data.

    Args:
        request: The input data is expected to be in the following JSON format:
            {
                "data": [
                    [0, {'_ID': 0, 'input_feature_0': 0.0, 'input_feature_1': 1.0}],
                    [1, {'_ID': 1, 'input_feature_0': 2.0, 'input_feature_1': 3.0}],
            }
            Each row is represented as a list, where the first element denotes the index of the row.

    Returns:
        Two possible responses:
        For success, return a JSON response
            {
                "data": [
                    [0, {'_ID': 0, 'output': 1}],
                    [1, {'_ID': 1, 'output': 2}]
                ]
            },
            The first element of each resulting list denotes the index of the row, and the rest of the elements
            represent the prediction results for that row.
        For an error, return {"error": error_message, "status_code": http_response_status_code}.
    """
    _MODEL_LOADING_EVENT.wait()  # Ensure model is indeed loaded into memory

    global _CONCURRENT_COUNTER
    global _CONCURRENT_COUNTER_LOCK

    input_json = await request.json()

    if _CONCURRENT_REQUESTS_MAX:
        async with _CONCURRENT_COUNTER_LOCK:
            if _CONCURRENT_COUNTER >= int(_CONCURRENT_REQUESTS_MAX):
                return responses.JSONResponse(
                    {"error": "Too many requests"}, status_code=http.HTTPStatus.TOO_MANY_REQUESTS
                )

    async with _CONCURRENT_COUNTER_LOCK:
        _CONCURRENT_COUNTER += 1

    resp = await concurrency.run_in_threadpool(_do_predict, input_json)

    async with _CONCURRENT_COUNTER_LOCK:
        _CONCURRENT_COUNTER -= 1

    return resp


def _in_test_mode() -> bool:
    """Check if the code is running in test mode.

    Specifically, it checks for the presence of
    - "PYTEST_CURRENT_TEST" environment variable, which is automatically set by Pytest when running tests, and
    - "TEST_WORKSPACE" environment variable, which is set by Bazel test, and
    - "TEST_SRCDIR" environment variable, which is set by the Absl test.

    Returns:
        True if in test mode; otherwise, returns False
    """
    is_running_under_py_test = "PYTEST_CURRENT_TEST" in os.environ
    is_running_under_bazel_test = "TEST_WORKSPACE" in os.environ
    is_running_under_absl_test = "TEST_SRCDIR" in os.environ
    return is_running_under_py_test or is_running_under_bazel_test or is_running_under_absl_test


def run_app() -> applications.Starlette:
    if _in_test_mode():
        _MODEL_LOADING_EVENT.set()
    else:
        # TODO[shchen]: SNOW-893654. Before SnowService supports Startup probe, or extends support for Readiness probe
        # with configurable failureThreshold, we will have to load the model in a separate thread in order to prevent
        # gunicorn worker timeout.
        model_loading_worker = CustomThread(target=_run_setup)
        model_loading_worker.start()

    routes = [
        routing.Route("/health", endpoint=ready, methods=["GET"]),
        routing.Route("/predict", endpoint=predict, methods=["POST"]),
    ]
    return applications.Starlette(routes=routes)


app = run_app()
