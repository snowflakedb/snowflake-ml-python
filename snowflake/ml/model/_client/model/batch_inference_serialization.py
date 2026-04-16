import base64
import datetime
import json
from typing import Any, Optional

from pydantic import TypeAdapter

from snowflake.ml.model._client.model.batch_inference_specs import ColumnHandlingOptions

_UTF8_ENCODING = "utf-8"


def encode_params(params: Optional[dict[str, Any]]) -> Optional[str]:
    """Encode params dictionary to a base64 string.

    Args:
        params: Optional dictionary of model inference parameters.

    Returns:
        Base64 encoded JSON string of the params, or None if input is None.
    """
    if params is None:
        return None

    def serialize_value(v: Any) -> Any:
        """Convert non-JSON-serializable types to JSON-compatible formats."""
        if isinstance(v, bytes):
            return v.hex()
        if isinstance(v, datetime.datetime):
            return v.isoformat()
        if isinstance(v, list):
            return [serialize_value(item) for item in v]
        if isinstance(v, dict):
            return {k: serialize_value(val) for k, val in v.items()}
        return v

    serializable_params = {k: serialize_value(v) for k, v in params.items()}
    return base64.b64encode(json.dumps(serializable_params).encode(_UTF8_ENCODING)).decode(_UTF8_ENCODING)


def encode_column_handling(
    column_handling: Optional[dict[str, ColumnHandlingOptions]],
) -> Optional[str]:
    """Validate and encode column_handling to a base64 string.

    Args:
        column_handling: Optional dictionary mapping column names to file encoding options.

    Returns:
        Base64 encoded JSON string of the column handling options, or None if input is None.
    """
    if column_handling is None:
        return None
    adapter = TypeAdapter(dict[str, ColumnHandlingOptions])
    validated_input = adapter.validate_python(column_handling)
    return base64.b64encode(adapter.dump_json(validated_input)).decode(_UTF8_ENCODING)
