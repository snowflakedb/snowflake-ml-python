"""Common read-only access to registry model specifications."""

import abc
from collections.abc import Mapping
from typing import Any


class ModelSpec(abc.ABC):
    """Read-only interface shared by legacy and version 2.0 model specs."""

    @property
    @abc.abstractmethod
    def raw_spec(self) -> Mapping[str, Any]:
        """Return the model specification in its original schema."""

    @property
    @abc.abstractmethod
    def spec_version(self) -> str:
        """Return the specification version."""

    @property
    @abc.abstractmethod
    def signatures(self) -> Mapping[str, dict[str, Any]]:
        """Return owned copies of the function signatures, keyed by method name."""

    @property
    @abc.abstractmethod
    def model_type(self) -> str:
        """Return the lower case model framework or model artifact type."""

    @property
    @abc.abstractmethod
    def model_blobs(self) -> Mapping[str, Mapping[str, Any]]:
        """Return named model artifact definitions."""

    @property
    @abc.abstractmethod
    def model_options(self) -> Mapping[str, Mapping[str, Any]]:
        """Return model artifact options keyed by artifact name."""

    @property
    @abc.abstractmethod
    def model_tasks(self) -> Mapping[str, str | None]:
        """Return model tasks keyed by artifact name."""

    @property
    @abc.abstractmethod
    def supports_gpu(self) -> bool:
        """Return whether the specification declares GPU support."""

    @abc.abstractmethod
    def is_partitioned(self, function_name: str) -> bool:
        """Return whether a table function should use partitioned execution.

        Legacy packager specs default missing partition metadata to ``True``.
        Version 2.0 specs use ``properties.is_partition`` and treat an omitted
        key as ``False``.

        Args:
            function_name: Name of the model function.
        """

    @property
    @abc.abstractmethod
    def method_options(self) -> Mapping[str, Mapping[str, Any]]:
        """Return method options keyed by method name."""

    @property
    @abc.abstractmethod
    def case_sensitive(self) -> bool:
        """Return whether method and column names are case-sensitive."""
