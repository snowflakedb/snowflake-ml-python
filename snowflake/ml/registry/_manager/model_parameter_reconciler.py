import warnings
from dataclasses import dataclass
from typing import Optional

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import type_hints as model_types


@dataclass
class ReconciledParameters:
    """Holds the reconciled and validated parameters after processing."""

    conda_dependencies: Optional[list[str]] = None
    pip_requirements: Optional[list[str]] = None
    target_platforms: Optional[list[model_types.SupportedTargetPlatformType]] = None
    artifact_repository_map: Optional[dict[str, str]] = None
    options: Optional[model_types.ModelSaveOption] = None
    save_location: Optional[str] = None


class ModelParameterReconciler:
    """Centralizes all complex log_model parameter validation, transformation, and reconciliation logic."""

    def __init__(
        self,
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
        conda_dependencies: Optional[list[str]] = None,
        pip_requirements: Optional[list[str]] = None,
        target_platforms: Optional[list[model_types.SupportedTargetPlatformType]] = None,
        artifact_repository_map: Optional[dict[str, str]] = None,
        options: Optional[model_types.ModelSaveOption] = None,
    ) -> None:
        self._database_name = database_name
        self._schema_name = schema_name
        self._conda_dependencies = conda_dependencies
        self._pip_requirements = pip_requirements
        self._target_platforms = target_platforms
        self._artifact_repository_map = artifact_repository_map
        self._options = options

    def reconcile(self) -> ReconciledParameters:
        """Perform all parameter reconciliation and return clean parameters."""
        reconciled_artifact_repository_map = self._reconcile_artifact_repository_map()
        reconciled_save_location = self._extract_save_location()

        self._validate_pip_requirements_warehouse_compatibility(reconciled_artifact_repository_map)

        return ReconciledParameters(
            conda_dependencies=self._conda_dependencies,
            pip_requirements=self._pip_requirements,
            target_platforms=self._target_platforms,
            artifact_repository_map=reconciled_artifact_repository_map,
            options=self._options,
            save_location=reconciled_save_location,
        )

    def _reconcile_artifact_repository_map(self) -> Optional[dict[str, str]]:
        """Transform artifact_repository_map to use fully qualified names."""
        if not self._artifact_repository_map:
            return None

        transformed_map = {}

        for channel, artifact_repository_name in self._artifact_repository_map.items():
            db_id, schema_id, repo_id = sql_identifier.parse_fully_qualified_name(artifact_repository_name)

            transformed_map[channel] = sql_identifier.get_fully_qualified_name(
                db_id,
                schema_id,
                repo_id,
                self._database_name,
                self._schema_name,
            )

        return transformed_map

    def _extract_save_location(self) -> Optional[str]:
        """Extract save_location from options."""
        if self._options and "save_location" in self._options:
            return self._options.get("save_location")

        return None

    def _validate_pip_requirements_warehouse_compatibility(
        self, artifact_repository_map: Optional[dict[str, str]]
    ) -> None:
        """Validate pip_requirements compatibility with warehouse deployment."""
        if self._pip_requirements and not artifact_repository_map and self._targets_warehouse(self._target_platforms):
            warnings.warn(
                "Models logged specifying `pip_requirements` cannot be executed in a Snowflake Warehouse "
                "without specifying `artifact_repository_map`. This model can be run in Snowpark Container "
                "Services. See https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/container.",
                category=UserWarning,
                stacklevel=1,
            )

    @staticmethod
    def _targets_warehouse(target_platforms: Optional[list[model_types.SupportedTargetPlatformType]]) -> bool:
        """Returns True if warehouse is a target platform (None defaults to True)."""
        return (
            target_platforms is None
            or model_types.TargetPlatform.WAREHOUSE in target_platforms
            or "WAREHOUSE" in target_platforms
        )
