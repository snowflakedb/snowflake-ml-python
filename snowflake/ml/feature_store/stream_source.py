from typing import Any, Optional

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.snowpark.types import (
    BooleanType,
    DataType,
    DateType,
    DecimalType,
    FloatType,
    StringType,
    StructField,
    StructType,
    TimestampTimeZone,
    TimestampType,
    TimeType,
)

_STREAM_SOURCE_NAME_LENGTH_LIMIT = 32

_LIST_STREAM_SOURCE_SCHEMA = StructType(
    [
        StructField("NAME", StringType()),
        StructField("SCHEMA", StringType()),
        StructField("DESC", StringType()),
        StructField("OWNER", StringType()),
    ]
)

# Mapping from type class name to type class for schema serialization/deserialization.
_TYPE_NAME_TO_CLASS: dict[str, type] = {
    "StringType": StringType,  # VARCHAR(<n>)
    "FloatType": FloatType,  # FLOAT
    "DecimalType": DecimalType,  # NUMBER(<p>,<s>)
    "BooleanType": BooleanType,  # BOOLEAN
    "TimestampType": TimestampType,  # TIMESTAMP_NTZ(9)
    "DateType": DateType,  # DATE
    "TimeType": TimeType,  # TIME(9)
}


def _schema_to_dict(schema: StructType) -> list[dict[str, Any]]:
    """Convert a StructType schema to a JSON-serializable list of field descriptors."""
    fields = []
    for f in schema.fields:
        type_name = type(f.datatype).__name__
        if type_name not in _TYPE_NAME_TO_CLASS:
            raise ValueError(
                f"Unsupported type '{type_name}' for field '{f.name}'. "
                f"Supported types: {list(_TYPE_NAME_TO_CLASS.keys())}"
            )
        field_dict: dict[str, Any] = {
            "name": f.name,
            "type": type_name,
        }
        # Persist parameterized type attributes
        if isinstance(f.datatype, DecimalType):
            field_dict["precision"] = f.datatype.precision
            field_dict["scale"] = f.datatype.scale
        elif isinstance(f.datatype, StringType):
            if f.datatype.length is not None:
                field_dict["length"] = f.datatype.length
        # TimestampType: no extra attributes stored (always NTZ, validated at construction).
        fields.append(field_dict)
    return fields


def _schema_from_dict(fields_data: list[dict[str, Any]]) -> StructType:
    """Reconstruct a StructType schema from a list of field descriptors."""
    fields = []
    for fd in fields_data:
        type_cls = _TYPE_NAME_TO_CLASS.get(fd["type"])
        if type_cls is None:
            raise ValueError(
                f"Unsupported type '{fd['type']}' in stored schema. "
                f"Supported types: {list(_TYPE_NAME_TO_CLASS.keys())}"
            )
        # Reconstruct parameterized types with their attributes
        dtype: DataType
        if fd["type"] == "DecimalType":
            dtype = DecimalType(fd.get("precision", 38), fd.get("scale", 0))
        elif fd["type"] == "StringType":
            length = fd.get("length")  # None means max/unlimited
            dtype = StringType(length)
        elif fd["type"] == "TimestampType":
            # Always NTZ; any stored "tz" value is ignored for backward compatibility.
            dtype = TimestampType(TimestampTimeZone.NTZ)
        else:
            dtype = type_cls()
        fields.append(StructField(fd["name"], dtype))
    return StructType(fields)


class StreamSource:
    """
    A streaming data source for streaming feature views.

    StreamSource defines the schema and metadata for a streaming data source that can be
    referenced by streaming feature views. It is registered and managed in the FeatureStore
    similar to an Entity.
    """

    def __init__(self, name: str, schema: StructType, *, desc: str = "") -> None:
        """
        Creates a StreamSource instance.

        Args:
            name: Unique name for the stream source. Must not exceed 32 characters.
            schema: Expected schema of ingested data as a Snowpark StructType. All field types
                must be supported types: StringType (VARCHAR), FloatType (FLOAT),
                DecimalType (NUMBER), BooleanType (BOOLEAN), TimestampType (TIMESTAMP_NTZ),
                DateType (DATE), TimeType (TIME). Only TIMESTAMP_NTZ is supported for
                timestamps; TimestampType() and TimestampType(TimestampTimeZone.NTZ) are both
                accepted. All timestamps are stored as UTC.
            desc: Description of the stream source.

        Raises:
            ValueError: If name exceeds length limit, schema is empty, schema contains
                unsupported types.  # noqa: DAR402

        Example::

            >>> from snowflake.snowpark.types import (
            ...     StructType, StructField, StringType, FloatType, TimestampType,
            ... )
            >>> txn_source = StreamSource(
            ...     name="transaction_events",
            ...     schema=StructType([
            ...         StructField("user_id", StringType()),
            ...         StructField("amount", FloatType()),
            ...         StructField("event_time", TimestampType()),
            ...     ]),
            ...     desc="Real-time transaction events",
            ... )
            >>> fs.register_stream_source(txn_source)

        """
        self._validate(name, schema)

        self.name: SqlIdentifier = SqlIdentifier(name)
        self.schema: StructType = schema
        self.owner: Optional[str] = None
        self.desc: str = desc

    def _validate(self, name: str, schema: StructType) -> None:
        if len(name) > _STREAM_SOURCE_NAME_LENGTH_LIMIT:
            raise ValueError(f"StreamSource name `{name}` exceeds maximum length: {_STREAM_SOURCE_NAME_LENGTH_LIMIT}")

        if not schema.fields:
            raise ValueError("StreamSource schema must have at least one field.")

        # Validate all field types are supported
        for f in schema.fields:
            type_name = type(f.datatype).__name__
            if type_name not in _TYPE_NAME_TO_CLASS:
                raise ValueError(
                    f"Unsupported type '{type_name}' for field '{f.name}'. "
                    f"Supported types: {list(_TYPE_NAME_TO_CLASS.keys())}"
                )
            # Only TIMESTAMP_NTZ is supported (DEFAULT is accepted and treated as NTZ).
            if isinstance(f.datatype, TimestampType) and f.datatype.tz not in (
                TimestampTimeZone.NTZ,
                TimestampTimeZone.DEFAULT,
            ):
                raise ValueError(
                    f"Unsupported timezone '{f.datatype.tz.name}' for field '{f.name}'. "
                    f"Only TIMESTAMP_NTZ is supported. All timestamps are stored as UTC. "
                    f"Use TimestampType() or TimestampType(TimestampTimeZone.NTZ)."
                )

    def _to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary for backend storage."""
        return {
            "name": str(self.name),
            "schema": _schema_to_dict(self.schema),
            "desc": self.desc,
            "owner": self.owner or "",
            "ref_count": 0,
        }

    @staticmethod
    def _construct_stream_source(
        name: str,
        schema: StructType,
        desc: str,
        owner: Optional[str] = None,
    ) -> "StreamSource":
        """Construct a StreamSource with owner populated (for objects retrieved from backend)."""
        ss = StreamSource(name, schema, desc=desc)
        ss.owner = owner
        return ss

    @staticmethod
    def _from_dict(data: dict[str, Any]) -> "StreamSource":
        """Reconstruct a StreamSource from a stored metadata dictionary."""
        schema = _schema_from_dict(data["schema"])
        return StreamSource._construct_stream_source(
            name=data["name"],
            schema=schema,
            desc=data.get("desc", ""),
            owner=data.get("owner") or None,
        )

    def __repr__(self) -> str:
        states = (f"{k}={v}" for k, v in vars(self).items())
        return f"{type(self).__name__}({', '.join(states)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StreamSource):
            return False

        return (
            self.name == other.name
            and self.schema == other.schema
            and self.desc == other.desc
            and self.owner == other.owner
        )
