import base64
import logging
import pickle
import posixpath
import sys
from typing import Any, Callable, Optional, Protocol, Union, cast, runtime_checkable

from snowflake import snowpark
from snowflake.ml.jobs._interop import data_utils
from snowflake.ml.jobs._interop.dto_schema import (
    BinaryManifest,
    ParquetManifest,
    ProtocolInfo,
)

Condition = Union[type, tuple[type, ...], Callable[[Any], bool], None]

logger = logging.getLogger(__name__)


class SerializationError(TypeError):
    """Exception raised when a serialization protocol fails."""


class DeserializationError(ValueError):
    """Exception raised when a serialization protocol fails."""


class InvalidPayloadError(DeserializationError):
    """Exception raised when the payload is invalid."""


class ProtocolMismatchError(DeserializationError):
    """Exception raised when the protocol of the serialization protocol is incompatible."""


class VersionMismatchError(ProtocolMismatchError):
    """Exception raised when the version of the serialization protocol is incompatible."""


class ProtocolNotFoundError(SerializationError):
    """Exception raised when no suitable serialization protocol is available."""


@runtime_checkable
class SerializationProtocol(Protocol):
    """
    More advanced protocol which supports more flexibility in how results are saved or loaded.
    Results can be saved as one or more files, or directly inline in the PayloadManifest.
    If saving as files, the PayloadManifest can save arbitrary "manifest" information.
    """

    @property
    def supported_types(self) -> Condition:
        """The types that the protocol supports."""

    @property
    def protocol_info(self) -> ProtocolInfo:
        """The information about the protocol."""

    def save(self, obj: Any, dest_dir: str, session: Optional[snowpark.Session] = None) -> ProtocolInfo:
        """Save the object to the destination directory."""

    def load(
        self,
        payload_info: ProtocolInfo,
        session: Optional[snowpark.Session] = None,
        path_transform: Optional[Callable[[str], str]] = None,
    ) -> Any:
        """Load the object from the source directory."""


class CloudPickleProtocol(SerializationProtocol):
    """
    CloudPickle serialization protocol.
    Uses BinaryManifest for manifest schema.
    """

    DEFAULT_PATH = "mljob_extra.pkl"

    def __init__(self) -> None:
        import cloudpickle as cp

        self._backend = cp

    def _get_compatibility_error(self, payload_info: ProtocolInfo) -> Optional[Exception]:
        """Check compatibility and attempt load, raising helpful errors on failure."""
        version_error = python_error = None

        # Check cloudpickle version compatibility
        if payload_info.version:
            try:
                from packaging import version

                payload_major, current_major = (
                    version.parse(payload_info.version).major,
                    version.parse(self._backend.__version__).major,
                )
                if payload_major != current_major:
                    version_error = "cloudpickle version mismatch: payload={}, current={}".format(
                        payload_info.version, self._backend.__version__
                    )
            except Exception:
                if payload_info.version != self.protocol_info.version:
                    version_error = "cloudpickle version mismatch: payload={}, current={}".format(
                        payload_info.version, self.protocol_info.version
                    )

        # Check Python version compatibility
        if payload_info.metadata and "python_version" in payload_info.metadata:
            payload_py, current_py = (
                payload_info.metadata["python_version"],
                f"{sys.version_info.major}.{sys.version_info.minor}",
            )
            if payload_py != current_py:
                python_error = f"Python version mismatch: payload={payload_py}, current={current_py}"

        if version_error or python_error:
            errors = [err for err in [version_error, python_error] if err]
            return VersionMismatchError(f"Load failed due to incompatibility: {'; '.join(errors)}")
        return None

    @property
    def supported_types(self) -> Condition:
        return None  # All types are supported

    @property
    def protocol_info(self) -> ProtocolInfo:
        return ProtocolInfo(
            name="cloudpickle",
            version=self._backend.__version__,
            metadata={
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            },
        )

    def save(self, obj: Any, dest_dir: str, session: Optional[snowpark.Session] = None) -> ProtocolInfo:
        """Save the object to the destination directory."""
        result_path = posixpath.join(dest_dir, self.DEFAULT_PATH)
        with data_utils.open_stream(result_path, "wb", session=session) as f:
            self._backend.dump(obj, f)
        manifest: BinaryManifest = {"path": result_path}
        return self.protocol_info.with_manifest(manifest)

    def load(
        self,
        payload_info: ProtocolInfo,
        session: Optional[snowpark.Session] = None,
        path_transform: Optional[Callable[[str], str]] = None,
    ) -> Any:
        """Load the object from the source directory."""
        if payload_info.name != self.protocol_info.name:
            raise ProtocolMismatchError(
                f"Invalid payload protocol: expected '{self.protocol_info.name}', got '{payload_info.name}'"
            )

        payload_manifest = cast(BinaryManifest, payload_info.manifest)
        try:
            if payload_bytes := payload_manifest.get("bytes"):
                return self._backend.loads(payload_bytes)
            if payload_b64 := payload_manifest.get("base64"):
                return self._backend.loads(base64.b64decode(payload_b64))
            result_path = path_transform(payload_manifest["path"]) if path_transform else payload_manifest["path"]
            with data_utils.open_stream(result_path, "rb", session=session) as f:
                return self._backend.load(f)
        except (
            pickle.UnpicklingError,
            TypeError,
            AttributeError,
            MemoryError,
        ) as pickle_error:
            if error := self._get_compatibility_error(payload_info):
                raise error from pickle_error
            raise


class ArrowTableProtocol(SerializationProtocol):
    """
    Arrow Table serialization protocol.
    Uses ParquetManifest for manifest schema.
    """

    DEFAULT_PATH_PATTERN = "mljob_extra_{0}.parquet"

    def __init__(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        self._pa = pa
        self._pq = pq

    @property
    def supported_types(self) -> Condition:
        return cast(type, self._pa.Table)

    @property
    def protocol_info(self) -> ProtocolInfo:
        return ProtocolInfo(
            name="pyarrow",
            version=self._pa.__version__,
        )

    def save(self, obj: Any, dest_dir: str, session: Optional[snowpark.Session] = None) -> ProtocolInfo:
        """Save the object to the destination directory."""
        if not isinstance(obj, self._pa.Table):
            raise SerializationError(f"Expected {self._pa.Table.__name__} object, got {type(obj).__name__}")

        # TODO: Support partitioned writes for large datasets
        result_path = posixpath.join(dest_dir, self.DEFAULT_PATH_PATTERN.format(0))
        with data_utils.open_stream(result_path, "wb", session=session) as stream:
            self._pq.write_table(obj, stream)

        manifest: ParquetManifest = {"paths": [result_path]}
        return self.protocol_info.with_manifest(manifest)

    def load(
        self,
        payload_info: ProtocolInfo,
        session: Optional[snowpark.Session] = None,
        path_transform: Optional[Callable[[str], str]] = None,
    ) -> Any:
        """Load the object from the source directory."""
        if payload_info.name != self.protocol_info.name:
            raise ProtocolMismatchError(
                f"Invalid payload protocol: expected '{self.protocol_info.name}', got '{payload_info.name}'"
            )

        payload_manifest = cast(ParquetManifest, payload_info.manifest)
        tables = []
        for path in payload_manifest["paths"]:
            transformed_path = path_transform(path) if path_transform else path
            with data_utils.open_stream(transformed_path, "rb", session=session) as f:
                table = self._pq.read_table(f)
                tables.append(table)
        return self._pa.concat_tables(tables) if len(tables) > 1 else tables[0]


class PandasDataFrameProtocol(SerializationProtocol):
    """
    Pandas DataFrame serialization protocol.
    Uses ParquetManifest for manifest schema.
    """

    DEFAULT_PATH_PATTERN = "mljob_extra_{0}.parquet"

    def __init__(self) -> None:
        import pandas as pd

        self._pd = pd

    @property
    def supported_types(self) -> Condition:
        return cast(type, self._pd.DataFrame)

    @property
    def protocol_info(self) -> ProtocolInfo:
        return ProtocolInfo(
            name="pandas",
            version=self._pd.__version__,
        )

    def save(self, obj: Any, dest_dir: str, session: Optional[snowpark.Session] = None) -> ProtocolInfo:
        """Save the object to the destination directory."""
        if not isinstance(obj, self._pd.DataFrame):
            raise SerializationError(f"Expected {self._pd.DataFrame.__name__} object, got {type(obj).__name__}")

        # TODO: Support partitioned writes for large datasets
        result_path = posixpath.join(dest_dir, self.DEFAULT_PATH_PATTERN.format(0))
        with data_utils.open_stream(result_path, "wb", session=session) as stream:
            obj.to_parquet(stream)

        manifest: ParquetManifest = {"paths": [result_path]}
        return self.protocol_info.with_manifest(manifest)

    def load(
        self,
        payload_info: ProtocolInfo,
        session: Optional[snowpark.Session] = None,
        path_transform: Optional[Callable[[str], str]] = None,
    ) -> Any:
        """Load the object from the source directory."""
        if payload_info.name != self.protocol_info.name:
            raise ProtocolMismatchError(
                f"Invalid payload protocol: expected '{self.protocol_info.name}', got '{payload_info.name}'"
            )

        payload_manifest = cast(ParquetManifest, payload_info.manifest)
        dfs = []
        for path in payload_manifest["paths"]:
            transformed_path = path_transform(path) if path_transform else path
            with data_utils.open_stream(transformed_path, "rb", session=session) as f:
                df = self._pd.read_parquet(f)
                dfs.append(df)
        return self._pd.concat(dfs) if len(dfs) > 1 else dfs[0]


class NumpyArrayProtocol(SerializationProtocol):
    """
    Numpy Array serialization protocol.
    Uses BinaryManifest for manifest schema.
    """

    DEFAULT_PATH_PATTERN = "mljob_extra.npy"

    def __init__(self) -> None:
        import numpy as np

        self._np = np

    @property
    def supported_types(self) -> Condition:
        return cast(type, self._np.ndarray)

    @property
    def protocol_info(self) -> ProtocolInfo:
        return ProtocolInfo(
            name="numpy",
            version=self._np.__version__,
        )

    def save(self, obj: Any, dest_dir: str, session: Optional[snowpark.Session] = None) -> ProtocolInfo:
        """Save the object to the destination directory."""
        if not isinstance(obj, self._np.ndarray):
            raise SerializationError(f"Expected {self._np.ndarray.__name__} object, got {type(obj).__name__}")
        result_path = posixpath.join(dest_dir, self.DEFAULT_PATH_PATTERN)
        with data_utils.open_stream(result_path, "wb", session=session) as stream:
            self._np.save(stream, obj)

        manifest: BinaryManifest = {"path": result_path}
        return self.protocol_info.with_manifest(manifest)

    def load(
        self,
        payload_info: ProtocolInfo,
        session: Optional[snowpark.Session] = None,
        path_transform: Optional[Callable[[str], str]] = None,
    ) -> Any:
        """Load the object from the source directory."""
        if payload_info.name != self.protocol_info.name:
            raise ProtocolMismatchError(
                f"Invalid payload protocol: expected '{self.protocol_info.name}', got '{payload_info.name}'"
            )

        payload_manifest = cast(BinaryManifest, payload_info.manifest)
        transformed_path = path_transform(payload_manifest["path"]) if path_transform else payload_manifest["path"]
        with data_utils.open_stream(transformed_path, "rb", session=session) as f:
            return self._np.load(f)


class AutoProtocol(SerializationProtocol):
    def __init__(self) -> None:
        self._protocols: list[SerializationProtocol] = []
        self._protocol_info = ProtocolInfo(
            name="auto",
            version=None,
            metadata=None,
        )

    @property
    def supported_types(self) -> Condition:
        return None  # All types are supported

    @property
    def protocol_info(self) -> ProtocolInfo:
        return self._protocol_info

    def try_register_protocol(
        self,
        klass: type[SerializationProtocol],
        *args: Any,
        index: int = 0,
        **kwargs: Any,
    ) -> None:
        """
        Try to construct and register a protocol. If the protocol cannot be constructed,
        log a warning and skip registration. By default (index=0), the most recently
        registered protocol takes precedence.

        Args:
            klass: The class of the protocol to register.
            args: The positional arguments to pass to the protocol constructor.
            index: The index to register the protocol at. If -1, the protocol is registered at the end of the list.
            kwargs: The keyword arguments to pass to the protocol constructor.
        """
        try:
            protocol = klass(*args, **kwargs)
            self.register_protocol(protocol, index=index)
        except Exception as e:
            logger.warning(f"Failed to register protocol {klass}: {e}")

    def register_protocol(
        self,
        protocol: SerializationProtocol,
        index: int = 0,
    ) -> None:
        """
        Register a protocol with a condition. By default (index=0), the most recently
        registered protocol takes precedence.

        Args:
            protocol: The protocol to register.
            index: The index to register the protocol at. If -1, the protocol is registered at the end of the list.

        Raises:
            ValueError: If the condition is invalid.
            ValueError: If the index is invalid.
        """
        # Validate condition
        # TODO: Build lookup table of supported types to protocols (in priority order)
        # for faster lookup at save/load time (instead of iterating over all protocols)
        if not isinstance(protocol, SerializationProtocol):
            raise ValueError(f"Invalid protocol type: {type(protocol)}. Expected SerializationProtocol.")
        if index == -1:
            self._protocols.append(protocol)
        elif index < 0:
            raise ValueError(f"Invalid index: {index}. Expected -1 or >= 0.")
        else:
            self._protocols.insert(index, protocol)

    def save(self, obj: Any, dest_dir: str, session: Optional[snowpark.Session] = None) -> ProtocolInfo:
        """Save the object to the destination directory."""
        last_protocol_error = None
        for protocol in self._protocols:
            try:
                if self._is_supported_type(obj, protocol):
                    logger.debug(f"Dumping object of type {type(obj)} with protocol {protocol}")
                    return protocol.save(obj, dest_dir, session)
            except Exception as e:
                logger.warning(f"Error dumping object {obj} with protocol {protocol}: {repr(e)}")
                last_protocol_error = (protocol.protocol_info, e)
        last_error_str = (
            f", most recent error ({last_protocol_error[0]}): {repr(last_protocol_error[1])}"
            if last_protocol_error
            else ""
        )
        raise ProtocolNotFoundError(
            f"No suitable protocol found for type {type(obj).__name__}"
            f" (available: {', '.join(str(p.protocol_info) for p in self._protocols)}){last_error_str}"
        )

    def load(
        self,
        payload_info: ProtocolInfo,
        session: Optional[snowpark.Session] = None,
        path_transform: Optional[Callable[[str], str]] = None,
    ) -> Any:
        """Load the object from the source directory."""
        last_error = None
        for protocol in self._protocols:
            if protocol.protocol_info.name == payload_info.name:
                try:
                    return protocol.load(payload_info, session, path_transform)
                except Exception as e:
                    logger.warning(f"Error loading object with protocol {protocol}: {repr(e)}")
                    last_error = e
        if last_error:
            raise last_error
        raise ProtocolNotFoundError(
            f"No protocol matching {payload_info} available"
            f" (available: {', '.join(str(p.protocol_info) for p in self._protocols)})"
            ", possibly due to snowflake-ml-python package version mismatch"
        )

    def _is_supported_type(self, obj: Any, protocol: SerializationProtocol) -> bool:
        if protocol.supported_types is None:
            return True  # None means all types are supported
        elif isinstance(protocol.supported_types, (type, tuple)):
            return isinstance(obj, protocol.supported_types)
        elif callable(protocol.supported_types):
            return protocol.supported_types(obj) is True
        raise ValueError(f"Invalid supported types: {protocol.supported_types} for protocol {protocol}")
