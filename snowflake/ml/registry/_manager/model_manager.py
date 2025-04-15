import os
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from absl.logging import logging

from snowflake.ml._internal import platform_capabilities, telemetry
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml._internal.human_readable_id import hrid_generator
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._client.model import model_impl, model_version_impl
from snowflake.ml.model._client.ops import metadata_ops, model_ops, service_ops
from snowflake.ml.model._model_composer import model_composer
from snowflake.ml.model._packager.model_meta import model_meta
from snowflake.ml.modeling._internal import constants
from snowflake.snowpark import exceptions as snowpark_exceptions, session

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(
        self,
        session: session.Session,
        *,
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
    ) -> None:
        self._database_name = database_name
        self._schema_name = schema_name
        self._model_ops = model_ops.ModelOperator(
            session, database_name=self._database_name, schema_name=self._schema_name
        )
        self._service_ops = service_ops.ServiceOperator(
            session, database_name=self._database_name, schema_name=self._schema_name
        )
        self._hrid_generator = hrid_generator.HRID16()

    def log_model(
        self,
        *,
        model: Union[model_types.SupportedModelType, model_version_impl.ModelVersion],
        model_name: str,
        version_name: Optional[str] = None,
        comment: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        conda_dependencies: Optional[List[str]] = None,
        pip_requirements: Optional[List[str]] = None,
        artifact_repository_map: Optional[Dict[str, str]] = None,
        target_platforms: Optional[List[model_types.SupportedTargetPlatformType]] = None,
        python_version: Optional[str] = None,
        signatures: Optional[Dict[str, model_signature.ModelSignature]] = None,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        user_files: Optional[Dict[str, List[str]]] = None,
        code_paths: Optional[List[str]] = None,
        ext_modules: Optional[List[ModuleType]] = None,
        task: model_types.Task = model_types.Task.UNKNOWN,
        options: Optional[model_types.ModelSaveOption] = None,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> model_version_impl.ModelVersion:

        database_name_id, schema_name_id, model_name_id = self._parse_fully_qualified_name(model_name)

        model_exists = self._model_ops.validate_existence(
            database_name=database_name_id,
            schema_name=schema_name_id,
            model_name=model_name_id,
            statement_params=statement_params,
        )

        if version_name is None:
            if model_exists:
                versions = self._model_ops.list_models_or_versions(
                    database_name=database_name_id,
                    schema_name=schema_name_id,
                    model_name=model_name_id,
                    statement_params=statement_params,
                )
                for _ in range(1000):
                    hrid = self._hrid_generator.generate()[1]
                    if sql_identifier.SqlIdentifier(hrid) not in versions:
                        version_name = hrid
                        break
                if version_name is None:
                    raise RuntimeError("Random version name generation failed.")
            else:
                version_name = self._hrid_generator.generate()[1]

        if isinstance(model, model_version_impl.ModelVersion):
            (
                source_database_name_id,
                source_schema_name_id,
                source_model_name_id,
            ) = sql_identifier.parse_fully_qualified_name(model.fully_qualified_model_name)

            self._model_ops.create_from_model_version(
                source_database_name=source_database_name_id,
                source_schema_name=source_schema_name_id,
                source_model_name=source_model_name_id,
                source_version_name=sql_identifier.SqlIdentifier(model.version_name),
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier(model_name),
                version_name=sql_identifier.SqlIdentifier(version_name),
                model_exists=model_exists,
                statement_params=statement_params,
            )
            return self.get_model(model_name=model_name, statement_params=statement_params).version(version_name)

        version_name_id = sql_identifier.SqlIdentifier(version_name)
        if model_exists and self._model_ops.validate_existence(
            database_name=database_name_id,
            schema_name=schema_name_id,
            model_name=model_name_id,
            version_name=version_name_id,
            statement_params=statement_params,
        ):
            raise ValueError(
                f"Model {model_name} version {version_name} already existed. "
                + "To auto-generate `version_name`, skip that argument."
            )

        return self._log_model(
            model=model,
            model_name=model_name,
            version_name=version_name,
            comment=comment,
            metrics=metrics,
            conda_dependencies=conda_dependencies,
            pip_requirements=pip_requirements,
            artifact_repository_map=artifact_repository_map,
            target_platforms=target_platforms,
            python_version=python_version,
            signatures=signatures,
            sample_input_data=sample_input_data,
            user_files=user_files,
            code_paths=code_paths,
            ext_modules=ext_modules,
            task=task,
            options=options,
            statement_params=statement_params,
        )

    def _log_model(
        self,
        model: model_types.SupportedModelType,
        *,
        model_name: str,
        version_name: str,
        comment: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        conda_dependencies: Optional[List[str]] = None,
        pip_requirements: Optional[List[str]] = None,
        artifact_repository_map: Optional[Dict[str, str]] = None,
        target_platforms: Optional[List[model_types.SupportedTargetPlatformType]] = None,
        python_version: Optional[str] = None,
        signatures: Optional[Dict[str, model_signature.ModelSignature]] = None,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        user_files: Optional[Dict[str, List[str]]] = None,
        code_paths: Optional[List[str]] = None,
        ext_modules: Optional[List[ModuleType]] = None,
        task: model_types.Task = model_types.Task.UNKNOWN,
        options: Optional[model_types.ModelSaveOption] = None,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> model_version_impl.ModelVersion:
        database_name_id, schema_name_id, model_name_id = sql_identifier.parse_fully_qualified_name(model_name)
        version_name_id = sql_identifier.SqlIdentifier(version_name)

        use_live_commit = platform_capabilities.PlatformCapabilities.get_instance().is_live_commit_enabled()
        if use_live_commit:
            logger.info("Using live commit model version")
        else:
            logger.info("Using non-live commit model version")

        if use_live_commit:
            # This step creates the live model version, and the files can be written directly to the stage
            # after this.
            try:
                self._model_ops.add_or_create_live_version(
                    database_name=database_name_id,
                    schema_name=schema_name_id,
                    model_name=model_name_id,
                    version_name=version_name_id,
                    statement_params=statement_params,
                )
            except (AssertionError, snowpark_exceptions.SnowparkSQLException) as e:
                logger.info(f"Failed to create live model version: {e}, falling back to regular model version creation")
                use_live_commit = False

        if use_live_commit:
            # using model version's stage path to write files directly to the stage
            stage_path = self._model_ops.get_model_version_stage_path(
                database_name=database_name_id,
                schema_name=schema_name_id,
                model_name=model_name_id,
                version_name=version_name_id,
            )
        else:
            # using a temp path to write files and then upload to the model version's stage
            stage_path = self._model_ops.prepare_model_temp_stage_path(
                database_name=database_name_id,
                schema_name=schema_name_id,
                statement_params=statement_params,
            )

        platforms = None
        # User specified target platforms are defaulted to None and will not show up in the generated manifest.
        if target_platforms:
            # Convert any string target platforms to TargetPlatform objects
            platforms = [model_types.TargetPlatform(platform) for platform in target_platforms]
        else:
            # Default the target platform to SPCS if not specified when running in ML runtime
            if os.getenv(constants.IN_ML_RUNTIME_ENV_VAR):
                logger.info(
                    "Logging the model on Container Runtime for ML without specifying `target_platforms`. "
                    'Default to `target_platforms=["SNOWPARK_CONTAINER_SERVICES"]`.'
                )
                platforms = [model_types.TargetPlatform.SNOWPARK_CONTAINER_SERVICES]

        if artifact_repository_map:
            for channel, artifact_repository_name in artifact_repository_map.items():
                db_id, schema_id, repo_id = sql_identifier.parse_fully_qualified_name(artifact_repository_name)

                artifact_repository_map[channel] = sql_identifier.get_fully_qualified_name(
                    db_id,
                    schema_id,
                    repo_id,
                    self._database_name,
                    self._schema_name,
                )

        logger.info("Start packaging and uploading your model. It might take some time based on the size of the model.")

        # Extract save_location from options if present
        save_location = None
        if options and "save_location" in options:
            save_location = options.get("save_location")
            logger.info(f"Model will be saved to local directory: {save_location}")

        mc = model_composer.ModelComposer(
            self._model_ops._session,
            stage_path=stage_path,
            statement_params=statement_params,
            save_location=save_location,
        )
        model_metadata: model_meta.ModelMetadata = mc.save(
            name=model_name_id.resolved(),
            model=model,
            signatures=signatures,
            sample_input_data=sample_input_data,
            conda_dependencies=conda_dependencies,
            pip_requirements=pip_requirements,
            artifact_repository_map=artifact_repository_map,
            target_platforms=platforms,
            python_version=python_version,
            user_files=user_files,
            code_paths=code_paths,
            ext_modules=ext_modules,
            options=options,
            task=task,
        )
        statement_params = telemetry.add_statement_params_custom_tags(
            statement_params, model_metadata.telemetry_metadata()
        )
        statement_params = telemetry.add_statement_params_custom_tags(
            statement_params, {"model_version_name": version_name_id}
        )

        logger.info("Start creating MODEL object for you in the Snowflake.")

        self._model_ops.create_from_stage(
            composed_model=mc,
            database_name=database_name_id,
            schema_name=schema_name_id,
            model_name=model_name_id,
            version_name=version_name_id,
            statement_params=statement_params,
            use_live_commit=use_live_commit,
        )

        mv = model_version_impl.ModelVersion._ref(
            model_ops=model_ops.ModelOperator(
                self._model_ops._session,
                database_name=database_name_id or self._database_name,
                schema_name=schema_name_id or self._schema_name,
            ),
            service_ops=service_ops.ServiceOperator(
                self._service_ops._session,
                database_name=database_name_id or self._database_name,
                schema_name=schema_name_id or self._schema_name,
            ),
            model_name=model_name_id,
            version_name=version_name_id,
        )

        if comment:
            mv.comment = comment

        if metrics:
            self._model_ops._metadata_ops.save(
                metadata_ops.ModelVersionMetadataSchema(metrics=metrics),
                database_name=database_name_id,
                schema_name=schema_name_id,
                model_name=model_name_id,
                version_name=version_name_id,
                statement_params=statement_params,
            )

        return mv

    def get_model(
        self,
        model_name: str,
        *,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> model_impl.Model:
        database_name_id, schema_name_id, model_name_id = self._parse_fully_qualified_name(model_name)
        if self._model_ops.validate_existence(
            database_name=database_name_id,
            schema_name=schema_name_id,
            model_name=model_name_id,
            statement_params=statement_params,
        ):
            return model_impl.Model._ref(
                model_ops.ModelOperator(
                    self._model_ops._session,
                    database_name=database_name_id or self._database_name,
                    schema_name=schema_name_id or self._schema_name,
                ),
                service_ops=service_ops.ServiceOperator(
                    self._service_ops._session,
                    database_name=database_name_id or self._database_name,
                    schema_name=schema_name_id or self._schema_name,
                ),
                model_name=model_name_id,
            )
        else:
            raise ValueError(f"Unable to find model {model_name}")

    def models(
        self,
        *,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> List[model_impl.Model]:
        model_names = self._model_ops.list_models_or_versions(
            database_name=None,
            schema_name=None,
            statement_params=statement_params,
        )
        return [
            model_impl.Model._ref(
                self._model_ops,
                service_ops=self._service_ops,
                model_name=model_name,
            )
            for model_name in model_names
        ]

    def show_models(
        self,
        *,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        rows = self._model_ops.show_models_or_versions(
            database_name=None,
            schema_name=None,
            statement_params=statement_params,
        )
        return pd.DataFrame([row.as_dict() for row in rows])

    def delete_model(
        self,
        model_name: str,
        *,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        database_name_id, schema_name_id, model_name_id = self._parse_fully_qualified_name(model_name)

        self._model_ops.delete_model_or_version(
            database_name=database_name_id,
            schema_name=schema_name_id,
            model_name=model_name_id,
            statement_params=statement_params,
        )

    def _parse_fully_qualified_name(
        self, model_name: str
    ) -> Tuple[
        Optional[sql_identifier.SqlIdentifier], Optional[sql_identifier.SqlIdentifier], sql_identifier.SqlIdentifier
    ]:
        try:
            return sql_identifier.parse_fully_qualified_name(model_name)
        except ValueError:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    f"The model_name `{model_name}` cannot be parsed as a SQL identifier. Alphanumeric characters and "
                    "underscores are permitted. See https://docs.snowflake.com/en/sql-reference/identifiers-syntax for "
                    "more information."
                ),
            )
