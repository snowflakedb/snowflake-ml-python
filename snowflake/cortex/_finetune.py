import json
from dataclasses import dataclass
from typing import Any, Optional, Union, cast

from snowflake import snowpark
from snowflake.cortex._util import (
    CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
    call_sql_function_literals,
)
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import snowpark_dataframe_utils

_CORTEX_FINETUNE_SYSTEM_FUNCTION_NAME = "SNOWFLAKE.CORTEX.FINETUNE"
CORTEX_FINETUNE_TELEMETRY_SUBPROJECT = "FINETUNE"
CORTEX_FINETUNE_FIRST_VERSION = "1.7.0"
CORTEX_FINETUNE_DOCUMENTATION_URL = "https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-finetuning"


class FinetuneError(Exception):
    def __init__(self, message: str, original_exception: Optional[Exception] = None) -> None:
        """Finetuning Exception Class.

        Args:
            message: Error message to be reported.
            original_exception: Original exception. This is the exception raised to users by telemetry.

        Attributes:
            original_exception: Original exception with an error code in its message.
        """
        self.original_exception = original_exception
        self._pretty_msg = message + repr(self.original_exception) if self.original_exception is not None else ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._pretty_msg!r})"

    def __str__(self) -> str:
        return self._pretty_msg


@dataclass
class FinetuneStatus:
    """Fine-tuning job status."""

    id: Optional[str] = None
    """Workflow ID for the fine-tuning run."""

    status: Optional[str] = None
    """Status string, e.g. PENDING, RUNNING, SUCCESS, ERROR, CANCELLED."""

    base_model: Optional[str] = None
    """Name of the base model that is being fine-tuned."""

    created_on: Optional[int] = None
    """Creation timestamp of the Fine-tuning job in milliseconds."""

    error: Optional[dict[str, Any]] = None
    """Error message propagated from the job."""

    finished_on: Optional[int] = None
    """Completion timestamp of the Fine-tuning job in milliseconds."""

    progress: Optional[float] = None
    """Progress made as a fraction of total [0.0,1.0]."""

    training_result: Optional[list[dict[str, Any]]] = None
    """Detailed metrics report for a completed training."""

    trained_tokens: Optional[int] = None
    """Number of tokens trained on. If multiple epochs are run, this can be larger than number of tokens in the
    training data."""

    training_data: Optional[str] = None
    """Training data query."""

    validation_data: Optional[str] = None
    """Validation data query."""

    model: Optional[str] = None
    """Location of the fine-tuned model."""


class FinetuneJob:
    def __init__(self, session: Optional[snowpark.Session], status: FinetuneStatus) -> None:
        """Fine-tuning Job.

        Args:
            session: Snowpark session to use to communicate with Snowflake.
            status: FinetuneStatus for this job.
        """
        self._session = session
        self.status = status

    def __repr__(self) -> str:
        return self.status.__repr__()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FinetuneJob):
            raise NotImplementedError(
                f"Equality comparison of FinetuneJob with objects of type {type(other)} is not implemented."
            )
        return self.status == other.status

    @snowpark._internal.utils.experimental(version=CORTEX_FINETUNE_FIRST_VERSION)
    @telemetry.send_api_usage_telemetry(
        project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
        subproject=CORTEX_FINETUNE_TELEMETRY_SUBPROJECT,
    )
    def cancel(self) -> bool:
        """Cancel a fine-tuning run.

        No confirmation will be required.

        [Documentation](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-finetuning)

        Args:

        Returns:
            True if the cancellation was successful, False otherwise.
        """
        result = _finetune_impl(operation="CANCEL", session=self._session, function_args=[self.status.id])
        return result is not None and isinstance(result, str) and result.startswith("Canceled Cortex Fine-tuning job: ")

    @snowpark._internal.utils.experimental(version=CORTEX_FINETUNE_FIRST_VERSION)
    @telemetry.send_api_usage_telemetry(
        project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
        subproject=CORTEX_FINETUNE_TELEMETRY_SUBPROJECT,
    )
    def describe(self) -> FinetuneStatus:
        """Describe a fine-tuning run.

        Args:

        Returns:
            FinetuneStatus containing of attributes of the fine-tuning run.
        """
        result_string = _finetune_impl(operation="DESCRIBE", session=self._session, function_args=[self.status.id])

        result = FinetuneStatus(**cast(dict[str, Any], _try_load_json(result_string)))
        return result


class Finetune:
    @snowpark._internal.utils.experimental(version=CORTEX_FINETUNE_FIRST_VERSION)
    @telemetry.send_api_usage_telemetry(
        project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
        subproject=CORTEX_FINETUNE_TELEMETRY_SUBPROJECT,
    )
    def __init__(self, session: Optional[snowpark.Session] = None) -> None:
        """Cortex Fine-Tuning API.

        [Documentation](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-finetuning)

        Args:
            session: Snowpark session to be used. If none is given, we will attempt to
                use the currently active session.
        """
        self._session = session

    @snowpark._internal.utils.experimental(version=CORTEX_FINETUNE_FIRST_VERSION)
    @telemetry.send_api_usage_telemetry(
        project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
        subproject=CORTEX_FINETUNE_TELEMETRY_SUBPROJECT,
    )
    def create(
        self,
        name: str,
        base_model: str,
        training_data: Union[str, snowpark.DataFrame],
        validation_data: Optional[Union[str, snowpark.DataFrame]] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> FinetuneJob:
        """Create a new fine-tuning runs.

        The expected format of training and validation data is two fields or columns:
            "prompt": the input to the model and
            "completion": the output that the model is expected to generate.

        Both data parameters "training_data" and "validation_data" expect to be one of
            (1) stage path to JSONL-formatted data,
            (2) select-query string resulting in a table,
            (3) Snowpark DataFrame containing the data

        [Documentation](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-finetuning)

        Args:
            name: Name of the resulting fine-tuned model.
            base_model: The name of the base model to start fine-tuning from.
            training_data: Data used for fine-tuning the model.
            validation_data: Data used for validating the fine-tuned model (not used in training)
            options: Dictionary of additional options to be passed to the training procedure.
                Please refer to the official documentation for a list of available options.

        Returns:
            The identifier of the fine-tuning run.

        Raises:
            ValueError: If the Snowpark DataFrame used is incompatible with this API.
                This can happen if the DataFrame contains multiple queries.
        """

        # Handle data provided as snowpark dataframes
        if isinstance(training_data, snowpark.DataFrame):
            if snowpark_dataframe_utils.is_single_query_snowpark_dataframe(training_data):
                training_string = str(training_data.queries["queries"][0])
            else:
                raise ValueError(
                    "Snowpark DataFrame given in 'training_data' contains "
                    + f'{training_data.queries["queries"]} queries and '
                    + f'{training_data.queries["post_actions"]} post_actions. It needs '
                    "to contain exactly one query and no post_actions."
                )
        else:
            training_string = training_data

        validation_string: Optional[str] = None
        if isinstance(validation_data, snowpark.DataFrame):
            if snowpark_dataframe_utils.is_single_query_snowpark_dataframe(validation_data):
                validation_string = str(validation_data.queries["queries"][0])
            else:
                raise ValueError(
                    "Snowpark DataFrame given in 'validation_data' contains "
                    + f'{validation_data.queries["queries"]} queries and '
                    + f'{validation_data.queries["post_actions"]} post_actions. It needs '
                    "to contain exactly one query and no post_actions."
                )
        else:
            validation_string = validation_data

        result = _finetune_impl(
            operation="CREATE",
            session=self._session,
            function_args=[name, base_model, training_string, validation_string, options],
        )
        finetune_status = FinetuneStatus(id=result)
        finetune_run = FinetuneJob(self._session, finetune_status)
        return finetune_run

    @snowpark._internal.utils.experimental(version=CORTEX_FINETUNE_FIRST_VERSION)
    @telemetry.send_api_usage_telemetry(
        project=CORTEX_FUNCTIONS_TELEMETRY_PROJECT,
        subproject=CORTEX_FINETUNE_TELEMETRY_SUBPROJECT,
    )
    def list_jobs(self) -> list["FinetuneJob"]:
        """Show current and past fine-tuning runs.

        Returns:
            List of dictionaries of attributes of the fine-tuning runs. Please refer to the official documentation for a
                list of expected fields.
        """
        result_string = _finetune_impl(operation="SHOW", session=self._session, function_args=[])
        result = _try_load_json(result_string)

        return [FinetuneJob(session=self._session, status=FinetuneStatus(**run_status)) for run_status in result]


def _try_load_json(json_string: str) -> Union[dict[Any, Any], list[Any]]:
    try:
        result = json.loads(str(json_string))
    except json.JSONDecodeError as e:
        message = f"""Unable to parse JSON from: "{json_string}". """
        raise FinetuneError(message=message, original_exception=e)
    except Exception as e:
        message = f"""Unable to parse JSON from: "{json_string}". """
        raise FinetuneError(message=message, original_exception=e)
    else:
        if not isinstance(result, dict) and not isinstance(result, list):
            message = f"""Unable to parse JSON from: "{json_string}". Result was not a dictionary."""
            raise FinetuneError(message=message)
    return result


def _finetune_impl(operation: str, session: Optional[snowpark.Session], function_args: list[Any]) -> str:
    return call_sql_function_literals(_CORTEX_FINETUNE_SYSTEM_FUNCTION_NAME, session, operation, *function_args)
