"""Exception classes for the declarative feature store library."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snowflake.ml.feature_store.decl.types import ValidationResult


class SpecLoadError(Exception):
    """Raised when a spec file cannot be loaded or parsed."""


class ValidationError(Exception):
    """Raised when one or more invariant violations are detected.

    Attributes:
        results: The list of ``ValidationResult`` objects that triggered this error.
    """

    def __init__(self, results: list[ValidationResult]) -> None:
        self.results = results
        messages = "; ".join(f"[{r.severity}] {r.code}: {r.message}" for r in results)
        super().__init__(f"Validation failed: {messages}")


class DependencyError(Exception):
    """Raised when spec dependencies cannot be resolved or form a cycle."""


class StateDriftError(Exception):
    """Raised when a concurrent modification is detected on a Snowflake object.

    Attributes:
        object_name: The fully-qualified name of the drifted object.
        expected_version: The version the client expected to find.
        found_version: The version actually found in Snowflake.
    """

    def __init__(self, object_name: str, expected_version: str, found_version: str) -> None:
        self.object_name = object_name
        self.expected_version = expected_version
        self.found_version = found_version
        super().__init__(
            f"State drift detected on '{object_name}': "
            f"expected version '{expected_version}', found '{found_version}'. "
            "Re-run plan to refresh state."
        )


class FeatureStoreNotInitializedError(Exception):
    """Raised when a declarative operation targets a schema that has not
    been bootstrapped as a SnowML feature store.

    The declarative client mirrors the imperative client's invariant:
    every ``snow feature`` command except ``init`` requires the target
    schema to already carry the internal ``SNOWML_FEATURE_STORE_OBJECT``
    and ``SNOWML_FEATURE_VIEW_METADATA`` tags.  When those tags are
    missing, ``FeatureStore(creation_mode=FAIL_IF_NOT_EXIST)`` raises a
    ``NOT_FOUND`` ``SnowflakeMLException`` mentioning the missing
    feature-store schema or internal tag — we catch that, rewrap it as
    this error, and let the CLI surface an actionable message that
    points the operator at ``snow feature init``.

    Attributes:
        database: Target Snowflake database name.
        schema: Target Snowflake schema (the FeatureStore "name").
        wrapped: The underlying ``SnowflakeMLException`` (or other
            exception) raised by ``FeatureStore.__init__``; preserved
            so debugging tools can inspect the original error.
    """

    def __init__(
        self,
        database: str,
        schema: str,
        wrapped: BaseException,
    ) -> None:
        self.database = database
        self.schema = schema
        self.wrapped = wrapped
        super().__init__(
            f"Schema {database}.{schema} is not initialized as a feature store. "
            "Run `snow feature init` against this target to bootstrap. "
            f"Underlying error: {wrapped}"
        )
