from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Discriminator, Tag, TypeAdapter, model_validator
from typing_extensions import Annotated, NotRequired, TypedDict


class BinaryManifest(TypedDict):
    """
    Binary data manifest schema.
    Contains one of: path, bytes, or base64 for the serialized data.
    """

    path: NotRequired[str]  # Path to file
    bytes: NotRequired[bytes]  # In-line byte string (not supported with JSON codec)
    base64: NotRequired[str]  # Base64 encoded string


class ParquetManifest(TypedDict):
    """Protocol manifest schema for parquet files."""

    paths: list[str]  # File paths


# Union type for all manifest types, including catch-all dict[str, Any] for backward compatibility
PayloadManifest = Union[BinaryManifest, ParquetManifest, dict[str, Any]]


class ProtocolInfo(BaseModel):
    """
    The protocol used to serialize the result and the manifest of the result.
    """

    name: str
    version: Optional[str] = None
    metadata: Optional[dict[str, str]] = None
    manifest: Optional[PayloadManifest] = None

    def __str__(self) -> str:
        result = self.name
        if self.version:
            result += f"-{self.version}"
        return result

    def with_manifest(self, manifest: PayloadManifest) -> "ProtocolInfo":
        """
        Return a new ProtocolInfo object with the manifest.
        """
        return ProtocolInfo(
            name=self.name,
            version=self.version,
            metadata=self.metadata,
            manifest=manifest,
        )


class ResultMetadata(BaseModel):
    """
    The metadata of a result.
    """

    type: str
    repr: str


class ExceptionMetadata(ResultMetadata):
    message: str
    traceback: str


class PayloadDTO(BaseModel):
    """
    Base class for serializable payloads.

    Args:
        kind: Discriminator field for DTO type dispatch.
        value: The payload value (if JSON-serializable).
        protocol: The protocol used to serialize the payload (if not JSON-serializable).
    """

    kind: Literal["base"] = "base"
    value: Optional[Any] = None
    protocol: Optional[ProtocolInfo] = None
    serialize_error: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_fields(cls, data: Any) -> Any:
        """Ensure at least one of value or protocol keys is specified."""
        if cls is PayloadDTO and isinstance(data, dict):
            required_fields = {"value", "protocol"}
            if not any(field in data for field in required_fields):
                raise ValueError("At least one of 'value' or 'protocol' must be specified")
        return data


class ResultDTO(PayloadDTO):
    """
    A JSON representation of an execution result.

    Args:
        kind: Discriminator field for DTO type dispatch.
        success: Whether the execution was successful.
        value: The value of the execution or the exception if the execution failed.
        protocol: The protocol used to serialize the result.
        metadata: The metadata of the result.
    """

    kind: Literal["result"] = "result"  # type: ignore[assignment]
    success: bool
    metadata: Optional[Union[ResultMetadata, ExceptionMetadata]] = None

    @model_validator(mode="before")
    @classmethod
    def validate_fields(cls, data: Any) -> Any:
        """Ensure at least one of value, protocol, or metadata keys is specified."""
        if isinstance(data, dict):
            required_fields = {"value", "protocol", "metadata"}
            if not any(field in data for field in required_fields):
                raise ValueError("At least one of 'value', 'protocol', or 'metadata' must be specified")
        return data


def _get_dto_kind(data: Any) -> str:
    """Extract the 'kind' discriminator from input, defaulting to 'result' for backward compatibility."""
    if isinstance(data, dict):
        kind = data.get("kind", "result")
    else:
        kind = getattr(data, "kind", "result")
    return str(kind)


AnyResultDTO = Annotated[
    Union[
        Annotated[ResultDTO, Tag("result")],
        Annotated[PayloadDTO, Tag("base")],
    ],
    Discriminator(_get_dto_kind),
]

ResultDTOAdapter: TypeAdapter[AnyResultDTO] = TypeAdapter(AnyResultDTO)
