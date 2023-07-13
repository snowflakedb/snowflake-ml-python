import logging
import os
import tempfile
import zipfile

import pandas as pd
from starlette import applications, requests, responses, routing

logger = logging.getLogger(__name__)
loaded_model = None


def _run_setup() -> None:
    """Set up logging and load model into memory."""
    # Align the application logger's handler with Gunicorn's to capture logs from all processes.
    gunicorn_logger = logging.getLogger("gunicorn.error")
    logger.handlers = gunicorn_logger.handlers
    logger.setLevel(gunicorn_logger.level)

    from snowflake.ml.model import _model as model_api

    global loaded_model

    MODEL_ZIP_STAGE_PATH = os.getenv("MODEL_ZIP_STAGE_PATH")
    assert MODEL_ZIP_STAGE_PATH, "Missing environment variable MODEL_ZIP_STAGE_PATH"
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
        loaded_model, _ = model_api._load_model_for_deploy(model_dir_path=extracted_dir)
        logger.info("Successfully loaded model into memory")


async def ready(request: requests.Request) -> responses.JSONResponse:
    """Endpoint to check if the application is ready."""
    return responses.JSONResponse({"status": "ready"})


async def predict(request: requests.Request) -> responses.JSONResponse:
    """Endpoint to make predictions based on input data.

    Args:
        request: The input data is expected to be in the following JSON format:
            {
                "data": [
                    [0, 5.1, 3.5, 4.2, 1.3],
                    [1, 4.7, 3.2, 4.1, 4.2]
            }
            Each row is represented as a list, where the first element denotes the index of the row.

    Returns:
        Two possible responses:
        For success, return a JSON response {"data": [[0, 1], [1, 2]]}, where the first element of each resulting list
            denotes the index of the row, and the rest of the elements represent the prediction results for that row.
        For an error, return {"error": error_message, "status_code": http_response_status_code}.
    """
    try:
        input = await request.json()
        assert "data" in input, "missing data field in the request input"
        # The expression x[1:] is used to exclude the index of the data row.
        input_data = [x[1:] for x in input.get("data")]
        x = pd.DataFrame(input_data)
        assert len(input_data) != 0 and not all(not row for row in input_data), "empty data"
    except Exception as e:
        error_message = f"Input data malformed: {str(e)}"
        return responses.JSONResponse({"error": error_message}, status_code=400)

    assert loaded_model

    try:
        # TODO(shchen): SNOW-835369, Support target method in inference server (Multi-task model).
        # Mypy ignore will be fixed along with the above ticket.
        predictions = loaded_model.predict(x)  # type: ignore[attr-defined]
        result = predictions.to_records(index=True).tolist()
        response = {"data": result}
        return responses.JSONResponse(response)
    except Exception as e:
        error_message = f"Prediction failed: {str(e)}"
        return responses.JSONResponse({"error": error_message}, status_code=400)


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
    if not _in_test_mode():
        _run_setup()
    routes = [
        routing.Route("/health", endpoint=ready, methods=["GET"]),
        routing.Route("/predict", endpoint=predict, methods=["POST"]),
    ]
    return applications.Starlette(routes=routes)


app = run_app()
