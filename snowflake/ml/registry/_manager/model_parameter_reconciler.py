import logging
import warnings
from dataclasses import dataclass
from typing import Any, Optional

from packaging import requirements

from snowflake.ml import version as snowml_version
from snowflake.ml._internal import env, env as snowml_env, env_utils
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import target_platform, type_hints as model_types
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.ml.model.volatility import DEFAULT_VOLATILITY_BY_MODEL_TYPE, Volatility
from snowflake.snowpark import Session
from snowflake.snowpark._internal import utils as snowpark_utils

logger = logging.getLogger(__name__)


@dataclass
class ReconciledParameters:
    """Holds the reconciled and validated parameters after processing."""

    conda_dependencies: Optional[list[str]] = None
    pip_requirements: Optional[list[str]] = None
    target_platforms: Optional[list[model_types.TargetPlatform]] = None
    artifact_repository_map: Optional[dict[str, str]] = None
    options: Optional[model_types.ModelSaveOption] = None
    save_location: Optional[str] = None


class ModelParameterReconciler:
    """Centralizes all complex log_model parameter validation, transformation, and reconciliation logic."""

    def __init__(
        self,
        model: model_types.SupportedModelType,
        session: Session,
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
        conda_dependencies: Optional[list[str]] = None,
        pip_requirements: Optional[list[str]] = None,
        target_platforms: Optional[list[model_types.SupportedTargetPlatformType]] = None,
        artifact_repository_map: Optional[dict[str, str]] = None,
        options: Optional[model_types.ModelSaveOption] = None,
        python_version: Optional[str] = None,
        statement_params: Optional[dict[str, str]] = None,
    ) -> None:
        self._model = model
        self._session = session
        self._database_name = database_name
        self._schema_name = schema_name
        self._conda_dependencies = conda_dependencies
        self._pip_requirements = pip_requirements
        self._target_platforms = target_platforms
        self._artifact_repository_map = artifact_repository_map
        self._options = options
        self._python_version = python_version
        self._statement_params = statement_params

    def reconcile(self) -> ReconciledParameters:
        """Perform all parameter reconciliation and return clean parameters."""

        reconciled_artifact_repository_map = self._reconcile_artifact_repository_map()
        reconciled_save_location = self._extract_save_location()

        self._validate_pip_requirements_warehouse_compatibility(reconciled_artifact_repository_map)

        reconciled_target_platforms = self._reconcile_target_platforms()
        reconciled_options = self._reconcile_explainability_options(reconciled_target_platforms)
        reconciled_options = self._reconcile_relax_version(reconciled_options, reconciled_target_platforms)
        reconciled_options = self._reconcile_volatility_defaults(reconciled_options)  # ADD THIS LINE

        return ReconciledParameters(
            conda_dependencies=self._conda_dependencies,
            pip_requirements=self._pip_requirements,
            target_platforms=reconciled_target_platforms,
            artifact_repository_map=reconciled_artifact_repository_map,
            options=reconciled_options,
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

    def _reconcile_target_platforms(self) -> Optional[list[model_types.TargetPlatform]]:
        """Reconcile target platforms with proper defaulting logic."""
        # User specified target platforms are defaulted to None and will not show up in the generated manifest.
        if self._target_platforms:
            # Convert any string target platforms to TargetPlatform objects
            return [model_types.TargetPlatform(platform) for platform in self._target_platforms]

        # Default the target platform to warehouse if not specified and any table function exists
        if self._has_table_function():
            logger.info(
                "Logging a partitioned model with a table function without specifying `target_platforms`. "
                'Default to `target_platforms=["WAREHOUSE"]`.'
            )
            return [target_platform.TargetPlatform.WAREHOUSE]

        # Default the target platform to SPCS if not specified when running in ML runtime
        if env.IN_ML_RUNTIME:
            logger.info(
                "Logging the model on Container Runtime for ML without specifying `target_platforms`. "
                'Default to `target_platforms=["SNOWPARK_CONTAINER_SERVICES"]`.'
            )
            return [target_platform.TargetPlatform.SNOWPARK_CONTAINER_SERVICES]

        return None

    def _has_table_function(self) -> bool:
        """Check if any table function exists in options."""
        if self._options is None:
            return False

        if self._options.get("function_type") == model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value:
            return True

        for opt in self._options.get("method_options", {}).values():
            if opt.get("function_type") == model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value:
                return True

        return False

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

    def _reconcile_explainability_options(
        self, target_platforms: Optional[list[model_types.TargetPlatform]]
    ) -> model_types.ModelSaveOption:
        """Reconcile explainability settings and embed_local_ml_library based on warehouse runnability."""
        options = self._options.copy() if self._options else model_types.BaseModelSaveOption()

        conda_dep_dict = env_utils.validate_conda_dependency_string_list(self._conda_dependencies or [])

        enable_explainability = options.get("enable_explainability", None)

        # Handle case where user explicitly disabled explainability
        if enable_explainability is False:
            return self._handle_embed_local_ml_library(options, target_platforms)

        target_platform_set = set(target_platforms) if target_platforms else set()

        is_warehouse_runnable = self._is_warehouse_runnable(conda_dep_dict)
        only_spcs = target_platform_set == set(target_platform.SNOWPARK_CONTAINER_SERVICES_ONLY)
        has_both_platforms = target_platform_set == set(target_platform.BOTH_WAREHOUSE_AND_SNOWPARK_CONTAINER_SERVICES)

        # Handle case where user explicitly requested explainability
        if enable_explainability:
            if only_spcs or not is_warehouse_runnable:
                raise ValueError(
                    "`enable_explainability` cannot be set to True when the model is not runnable in WH "
                    "or the target platforms include SPCS."
                )
            elif has_both_platforms:
                warnings.warn(
                    ("Explain function will only be available for model deployed to warehouse."),
                    category=UserWarning,
                    stacklevel=2,
                )

        # Handle case where explainability is not specified (None) - set default behavior
        if enable_explainability is None:
            if only_spcs or not is_warehouse_runnable:
                options["enable_explainability"] = False

        return self._handle_embed_local_ml_library(options, target_platforms)

    def _handle_embed_local_ml_library(
        self, options: model_types.ModelSaveOption, target_platforms: Optional[list[model_types.TargetPlatform]]
    ) -> model_types.ModelSaveOption:
        """Handle embed_local_ml_library logic."""
        if not snowpark_utils.is_in_stored_procedure() and target_platforms != [  # type: ignore[no-untyped-call]
            model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES  # no information schema check for SPCS-only models
        ]:
            snowml_matched_versions = env_utils.get_matched_package_versions_in_information_schema(
                self._session,
                reqs=[requirements.Requirement(f"{env_utils.SNOWPARK_ML_PKG_NAME}=={snowml_version.VERSION}")],
                python_version=self._python_version or snowml_env.PYTHON_VERSION,
                statement_params=self._statement_params,
            ).get(env_utils.SNOWPARK_ML_PKG_NAME, [])

            if len(snowml_matched_versions) < 1 and not options.get("embed_local_ml_library", False):
                logger.info(
                    f"Local snowflake-ml-python library has version {snowml_version.VERSION},"
                    " which is not available in the Snowflake server, embedding local ML library automatically."
                )
                options["embed_local_ml_library"] = True

        return options

    def _is_warehouse_runnable(self, conda_dep_dict: dict[str, list[Any]]) -> bool:
        """Check if model can run in warehouse based on conda channels and pip requirements."""
        # If pip requirements are present but no artifact repository map, model cannot run in warehouse
        if self._pip_requirements and not self._artifact_repository_map:
            return False

        # If no conda dependencies, model can run in warehouse
        if not conda_dep_dict:
            return True

        # Check if all conda channels are warehouse-compatible
        warehouse_compatible_channels = {env_utils.DEFAULT_CHANNEL_NAME, env_utils.SNOWFLAKE_CONDA_CHANNEL_URL}
        for channel in conda_dep_dict:
            if channel not in warehouse_compatible_channels:
                return False

        return True

    def _reconcile_relax_version(
        self,
        options: model_types.ModelSaveOption,
        target_platforms: Optional[list[model_types.TargetPlatform]],
    ) -> model_types.ModelSaveOption:
        """Reconcile relax_version setting based on pip requirements and target platforms."""
        target_platform_set = set(target_platforms) if target_platforms else set()
        has_pip_requirements = bool(self._pip_requirements)
        only_spcs = target_platform_set == set(target_platform.SNOWPARK_CONTAINER_SERVICES_ONLY)

        if "relax_version" not in options:
            if has_pip_requirements or only_spcs:
                logger.info(
                    "Setting `relax_version=False` as this model will run in Snowpark Container Services "
                    "or in Warehouse with a specified artifact_repository_map where exact version "
                    " specifications will be honored."
                )
                relax_version = False
            else:
                warnings.warn(
                    (
                        "`relax_version` is not set and therefore defaulted to True. Dependency version constraints"
                        " relaxed from ==x.y.z to >=x.y, <(x+1). To use specific dependency versions for compatibility,"
                        " reproducibility, etc., set `options={'relax_version': False}` when logging the model."
                    ),
                    category=UserWarning,
                    stacklevel=2,
                )
                relax_version = True
            options["relax_version"] = relax_version
            return options

        # Handle case where relax_version is already set
        relax_version = options["relax_version"]
        if relax_version and (has_pip_requirements or only_spcs):
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    "Setting `relax_version=True` is only allowed for models to be run in Warehouse with "
                    "Snowflake Conda Channel dependencies. It cannot be used with pip requirements or when "
                    "targeting only Snowpark Container Services."
                ),
            )

        return options

    def _get_default_volatility_for_model(self, model: model_types.SupportedModelType) -> Volatility:
        """Get default volatility for a model based on its type."""
        from snowflake.ml.model._packager import model_handler

        handler = model_handler.find_handler(model)
        # default to IMMUTABLE if no handler found or handler type not in defaults
        if not handler or handler.HANDLER_TYPE not in DEFAULT_VOLATILITY_BY_MODEL_TYPE:
            return Volatility.IMMUTABLE
        return DEFAULT_VOLATILITY_BY_MODEL_TYPE[handler.HANDLER_TYPE]

    def _reconcile_volatility_defaults(self, options: model_types.ModelSaveOption) -> model_types.ModelSaveOption:
        """Set global default volatility based on model type."""

        # Skip if default_volatility is already explicitly set
        if "volatility" in options:
            return options

        # Get default volatility for this model type
        default_volatility = self._get_default_volatility_for_model(self._model)
        options["volatility"] = default_volatility

        return options
