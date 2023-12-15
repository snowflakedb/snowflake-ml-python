import logging
from datetime import datetime
from typing import Any, Dict, Optional

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import query_result_checker

logger = logging.getLogger(__name__)

_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f %z"
_COMPUTE_POOL = "compute_pool"
_CREATED_ON = "created_on"
_INSTANCE_FAMILY = "instance_family"
_NAME = "name"
_TELEMETRY_PROJECT = "MLOps"
_TELEMETRY_SUBPROJECT = "SpcsDeployment"
_SERVICE_START = "SPCS_SERVICE_START"
_SERVICE_END = "SPCS_SERVICE_END"


def _desc_compute_pool(session: snowpark.Session, compute_pool_name: str) -> Dict[str, Any]:
    sql = f"DESC COMPUTE POOL {compute_pool_name}"
    result = (
        query_result_checker.SqlResultValidator(
            session=session,
            query=sql,
        )
        .has_column(_INSTANCE_FAMILY)
        .has_column(_NAME)
        .has_dimensions(expected_rows=1)
        .validate()
    )
    return result[0].as_dict()


def _desc_service(session: snowpark.Session, fully_qualified_name: str) -> Dict[str, Any]:
    sql = f"DESC SERVICE {fully_qualified_name}"
    result = (
        query_result_checker.SqlResultValidator(
            session=session,
            query=sql,
        )
        .has_column(_COMPUTE_POOL)
        .has_dimensions(expected_rows=1)
        .validate()
    )
    return result[0].as_dict()


def _get_current_time() -> datetime:
    """
    This method exists to make it easier to mock datetime in test.

    Returns:
        current datetime
    """
    return datetime.now()


def _send_service_telemetry(
    fully_qualified_name: Optional[str] = None,
    compute_pool_name: Optional[str] = None,
    service_details: Optional[Dict[str, Any]] = None,
    compute_pool_details: Optional[Dict[str, Any]] = None,
    duration_in_seconds: Optional[int] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        telemetry.send_custom_usage(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
            telemetry_type=telemetry.TelemetryField.TYPE_SNOWML_SPCS_USAGE.value,
            data={
                "service_name": fully_qualified_name,
                "compute_pool_name": compute_pool_name,
                "service_details": service_details,
                "compute_pool_details": compute_pool_details,
                "duration_in_seconds": duration_in_seconds,
            },
            kwargs=kwargs,
        )
    except Exception as e:
        logger.error(f"Failed to send service telemetry: {e}")


def record_service_start(session: snowpark.Session, fully_qualified_name: str) -> None:
    service_details = _desc_service(session, fully_qualified_name)
    compute_pool_name = service_details[_COMPUTE_POOL]
    compute_pool_details = _desc_compute_pool(session, compute_pool_name)

    _send_service_telemetry(
        fully_qualified_name=fully_qualified_name,
        compute_pool_name=compute_pool_name,
        service_details=service_details,
        compute_pool_details=compute_pool_details,
        kwargs={telemetry.TelemetryField.KEY_CUSTOM_TAGS.value: _SERVICE_START},
    )

    logger.info(f"Service {fully_qualified_name} created with compute pool {compute_pool_name}.")


def record_service_end(session: snowpark.Session, fully_qualified_name: str) -> None:
    service_details = _desc_service(session, fully_qualified_name)
    compute_pool_details = _desc_compute_pool(session, service_details[_COMPUTE_POOL])
    compute_pool_name = service_details[_COMPUTE_POOL]

    created_on_datetime: datetime = service_details[_CREATED_ON]
    current_time: datetime = _get_current_time()
    current_time = current_time.replace(tzinfo=created_on_datetime.tzinfo)
    duration_in_seconds = int((current_time - created_on_datetime).total_seconds())

    _send_service_telemetry(
        fully_qualified_name=fully_qualified_name,
        compute_pool_name=compute_pool_name,
        service_details=service_details,
        compute_pool_details=compute_pool_details,
        duration_in_seconds=duration_in_seconds,
        kwargs={telemetry.TelemetryField.KEY_CUSTOM_TAGS.value: _SERVICE_END},
    )

    logger.info(f"Service {fully_qualified_name} deleted from compute pool {compute_pool_name}")
