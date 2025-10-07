from typing import Any, Optional, Union

from pydantic import BaseModel, model_validator
from typing_extensions import NotRequired, TypedDict


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


class ResultDTO(BaseModel):
    """
    A JSON representation of an execution result.

    Args:
        success: Whether the execution was successful.
        value: The value of the execution or the exception if the execution failed.
        protocol: The protocol used to serialize the result.
        metadata: The metadata of the result.
    """

    success: bool
    value: Optional[Any] = None
    protocol: Optional[ProtocolInfo] = None
    metadata: Optional[Union[ResultMetadata, ExceptionMetadata]] = None
    serialize_error: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_fields(cls, data: Any) -> Any:
        """Ensure at least one of value, protocol, or metadata keys is specified."""
        if isinstance(data, dict):
            required_fields = {"value", "protocol", "metadata"}
            if not any(field in data for field in required_fields):
                raise ValueError("At least one of 'value', 'protocol', or 'metadata' must be specified")
        return data
