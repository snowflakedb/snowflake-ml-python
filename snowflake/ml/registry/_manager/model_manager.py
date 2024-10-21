from types import ModuleType
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from absl.logging import logging
from packaging import version

from snowflake.ml._internal import telemetry
from snowflake.ml._internal.human_readable_id import hrid_generator
from snowflake.ml._internal.utils import snowflake_env, sql_identifier
from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._client.model import model_impl, model_version_impl
from snowflake.ml.model._client.ops import metadata_ops, model_ops, service_ops
from snowflake.ml.model._model_composer import model_composer
from snowflake.ml.model._packager.model_meta import model_meta
from snowflake.snowpark import session

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
        target_platforms: Optional[List[model_types.SupportedTargetPlatformType]] = None,
        python_version: Optional[str] = None,
        signatures: Optional[Dict[str, model_signature.ModelSignature]] = None,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        code_paths: Optional[List[str]] = None,
        ext_modules: Optional[List[ModuleType]] = None,
        task: model_types.Task = model_types.Task.UNKNOWN,
        options: Optional[model_types.ModelSaveOption] = None,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> model_version_impl.ModelVersion:
        if not version_name:
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
                statement_params=statement_params,
            )
            return self.get_model(model_name=model_name, statement_params=statement_params).version(version_name)

        return self._log_model(
            model=model,
            model_name=model_name,
            version_name=version_name,
            comment=comment,
            metrics=metrics,
            conda_dependencies=conda_dependencies,
            pip_requirements=pip_requirements,
            target_platforms=target_platforms,
            python_version=python_version,
            signatures=signatures,
            sample_input_data=sample_input_data,
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
        version_name: Optional[str] = None,
        comment: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        conda_dependencies: Optional[List[str]] = None,
        pip_requirements: Optional[List[str]] = None,
        target_platforms: Optional[List[model_types.SupportedTargetPlatformType]] = None,
        python_version: Optional[str] = None,
        signatures: Optional[Dict[str, model_signature.ModelSignature]] = None,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        code_paths: Optional[List[str]] = None,
        ext_modules: Optional[List[ModuleType]] = None,
        task: model_types.Task = model_types.Task.UNKNOWN,
        options: Optional[model_types.ModelSaveOption] = None,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> model_version_impl.ModelVersion:
        database_name_id, schema_name_id, model_name_id = sql_identifier.parse_fully_qualified_name(model_name)

        if not version_name:
            version_name = self._hrid_generator.generate()[1]
        version_name_id = sql_identifier.SqlIdentifier(version_name)

        if self._model_ops.validate_existence(
            database_name=database_name_id,
            schema_name=schema_name_id,
            model_name=model_name_id,
            statement_params=statement_params,
        ) and self._model_ops.validate_existence(
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

        stage_path = self._model_ops.prepare_model_stage_path(
            database_name=database_name_id,
            schema_name=schema_name_id,
            statement_params=statement_params,
        )

        platforms = None
        # TODO(jbahk): Remove the version check after Snowflake 8.40.0 release
        # User specified target platforms are defaulted to None and will not show up in the generated manifest.
        # In the backend, we attempt to create a model for all platforms (WH, SPCS) regardless by default.
        if snowflake_env.get_current_snowflake_version(self._model_ops._session) >= version.parse("8.40.0"):
            # Convert any string target platforms to TargetPlatform objects
            if target_platforms:
                platforms = [model_types.TargetPlatform(platform) for platform in target_platforms]

        logger.info("Start packaging and uploading your model. It might take some time based on the size of the model.")

        mc = model_composer.ModelComposer(
            self._model_ops._session, stage_path=stage_path, statement_params=statement_params
        )
        model_metadata: model_meta.ModelMetadata = mc.save(
            name=model_name_id.resolved(),
            model=model,
            signatures=signatures,
            sample_input_data=sample_input_data,
            conda_dependencies=conda_dependencies,
            pip_requirements=pip_requirements,
            target_platforms=platforms,
            python_version=python_version,
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
        database_name_id, schema_name_id, model_name_id = sql_identifier.parse_fully_qualified_name(model_name)
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
        database_name_id, schema_name_id, model_name_id = sql_identifier.parse_fully_qualified_name(model_name)

        self._model_ops.delete_model_or_version(
            database_name=database_name_id,
            schema_name=schema_name_id,
            model_name=model_name_id,
            statement_params=statement_params,
        )
