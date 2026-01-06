import json
import logging
from types import ModuleType
from typing import TYPE_CHECKING, Any, Optional, Union

import pandas as pd
import yaml

from snowflake.ml._internal import platform_capabilities, telemetry
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml._internal.human_readable_id import hrid_generator
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import model_signature, task, type_hints
from snowflake.ml.model._client.model import model_impl, model_version_impl
from snowflake.ml.model._client.ops import metadata_ops, model_ops, service_ops
from snowflake.ml.model._client.service import (
    import_model_spec_schema,
    model_deployment_spec_schema,
)
from snowflake.ml.model._model_composer import model_composer
from snowflake.ml.model._packager.model_meta import model_meta
from snowflake.ml.model.models import huggingface
from snowflake.ml.registry._manager import model_parameter_reconciler
from snowflake.snowpark import exceptions as snowpark_exceptions, session
from snowflake.snowpark._internal import utils as snowpark_utils

if TYPE_CHECKING:
    from snowflake.ml.experiment._experiment_info import ExperimentInfo

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
        model: Union[type_hints.SupportedModelType, model_version_impl.ModelVersion],
        model_name: str,
        progress_status: type_hints.ProgressStatus,
        version_name: Optional[str] = None,
        comment: Optional[str] = None,
        metrics: Optional[dict[str, Any]] = None,
        conda_dependencies: Optional[list[str]] = None,
        pip_requirements: Optional[list[str]] = None,
        artifact_repository_map: Optional[dict[str, str]] = None,
        resource_constraint: Optional[dict[str, str]] = None,
        target_platforms: Optional[list[type_hints.SupportedTargetPlatformType]] = None,
        python_version: Optional[str] = None,
        signatures: Optional[dict[str, model_signature.ModelSignature]] = None,
        sample_input_data: Optional[type_hints.SupportedDataType] = None,
        user_files: Optional[dict[str, list[str]]] = None,
        code_paths: Optional[list[type_hints.CodePathLike]] = None,
        ext_modules: Optional[list[ModuleType]] = None,
        task: type_hints.Task = task.Task.UNKNOWN,
        experiment_info: Optional["ExperimentInfo"] = None,
        options: Optional[type_hints.ModelSaveOption] = None,
        statement_params: Optional[dict[str, Any]] = None,
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
            resource_constraint=resource_constraint,
            target_platforms=target_platforms,
            python_version=python_version,
            signatures=signatures,
            sample_input_data=sample_input_data,
            user_files=user_files,
            code_paths=code_paths,
            ext_modules=ext_modules,
            task=task,
            experiment_info=experiment_info,
            options=options,
            statement_params=statement_params,
            progress_status=progress_status,
        )

    def _log_model(
        self,
        model: type_hints.SupportedModelType,
        *,
        model_name: str,
        version_name: str,
        progress_status: type_hints.ProgressStatus,
        comment: Optional[str] = None,
        metrics: Optional[dict[str, Any]] = None,
        conda_dependencies: Optional[list[str]] = None,
        pip_requirements: Optional[list[str]] = None,
        artifact_repository_map: Optional[dict[str, str]] = None,
        resource_constraint: Optional[dict[str, str]] = None,
        target_platforms: Optional[list[type_hints.SupportedTargetPlatformType]] = None,
        python_version: Optional[str] = None,
        signatures: Optional[dict[str, model_signature.ModelSignature]] = None,
        sample_input_data: Optional[type_hints.SupportedDataType] = None,
        user_files: Optional[dict[str, list[str]]] = None,
        code_paths: Optional[list[type_hints.CodePathLike]] = None,
        ext_modules: Optional[list[ModuleType]] = None,
        task: type_hints.Task = task.Task.UNKNOWN,
        experiment_info: Optional["ExperimentInfo"] = None,
        options: Optional[type_hints.ModelSaveOption] = None,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> model_version_impl.ModelVersion:
        database_name_id, schema_name_id, model_name_id = sql_identifier.parse_fully_qualified_name(model_name)
        version_name_id = sql_identifier.SqlIdentifier(version_name)

        # Check if model is HuggingFace TransformersPipeline with no repo_snapshot_dir
        # If so, use remote logging via SYSTEM$IMPORT_MODEL
        if (
            isinstance(model, huggingface.TransformersPipeline)
            and model.compute_pool_for_log is not None
            and (not hasattr(model, "repo_snapshot_dir") or model.repo_snapshot_dir is None)
        ):
            logger.info("HuggingFace model has compute_pool_for_log, using remote logging")
            return self._remote_log_huggingface_model(
                model=model,
                model_name=model_name,
                version_name=version_name,
                database_name_id=database_name_id,
                schema_name_id=schema_name_id,
                model_name_id=model_name_id,
                version_name_id=version_name_id,
                comment=comment,
                conda_dependencies=conda_dependencies,
                pip_requirements=pip_requirements,
                target_platforms=target_platforms,
                options=options,
                statement_params=statement_params,
                progress_status=progress_status,
            )

        # TODO(SNOW-2091317): Remove this when the snowpark enables file PUT operation for snowurls
        use_live_commit = (
            not snowpark_utils.is_in_stored_procedure()  # type: ignore[no-untyped-call]
        ) and platform_capabilities.PlatformCapabilities.get_instance().is_live_commit_enabled()
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

        reconciler = model_parameter_reconciler.ModelParameterReconciler(
            model=model,
            session=self._model_ops._session,
            database_name=self._database_name,
            schema_name=self._schema_name,
            conda_dependencies=conda_dependencies,
            pip_requirements=pip_requirements,
            target_platforms=target_platforms,
            artifact_repository_map=artifact_repository_map,
            options=options,
            python_version=python_version,
            statement_params=statement_params,
        )

        model_params = reconciler.reconcile()

        # Use reconciled parameters
        artifact_repository_map = model_params.artifact_repository_map
        save_location = model_params.save_location

        logger.info("Start packaging and uploading your model. It might take some time based on the size of the model.")
        progress_status.update("packaging model...")
        progress_status.increment()

        if save_location:
            logger.info(f"Model will be saved to local directory: {save_location}")

        mc = model_composer.ModelComposer(
            self._model_ops._session,
            stage_path=stage_path,
            statement_params=statement_params,
            save_location=save_location,
        )

        progress_status.update("creating model manifest...")
        progress_status.increment()

        model_metadata: model_meta.ModelMetadata = mc.save(
            name=model_name_id.resolved(),
            model=model,
            signatures=signatures,
            sample_input_data=sample_input_data,
            conda_dependencies=conda_dependencies,
            pip_requirements=pip_requirements,
            artifact_repository_map=artifact_repository_map,
            resource_constraint=resource_constraint,
            target_platforms=model_params.target_platforms,
            python_version=python_version,
            user_files=user_files,
            code_paths=code_paths,
            ext_modules=ext_modules,
            options=model_params.options,
            task=task,
            experiment_info=experiment_info,
        )

        progress_status.update("uploading model files...")
        progress_status.increment()
        statement_params = telemetry.add_statement_params_custom_tags(
            statement_params, model_metadata.telemetry_metadata()
        )
        statement_params = telemetry.add_statement_params_custom_tags(
            statement_params, {"model_version_name": version_name_id}
        )

        progress_status.update("creating model object in Snowflake...")
        progress_status.increment()

        self._model_ops.create_from_stage(
            composed_model=mc,
            database_name=database_name_id,
            schema_name=schema_name_id,
            model_name=model_name_id,
            version_name=version_name_id,
            statement_params=statement_params,
            use_live_commit=use_live_commit,
        )

        mv = self._create_model_version_ref(
            database_name_id=database_name_id,
            schema_name_id=schema_name_id,
            model_name_id=model_name_id,
            version_name_id=version_name_id,
        )

        progress_status.update("setting model metadata...")
        progress_status.increment()

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

        progress_status.update("model logged successfully!")

        return mv

    def _remote_log_huggingface_model(
        self,
        model: huggingface.TransformersPipeline,
        model_name: str,
        version_name: str,
        database_name_id: Optional[sql_identifier.SqlIdentifier],
        schema_name_id: Optional[sql_identifier.SqlIdentifier],
        model_name_id: sql_identifier.SqlIdentifier,
        version_name_id: sql_identifier.SqlIdentifier,
        comment: Optional[str],
        conda_dependencies: Optional[list[str]],
        pip_requirements: Optional[list[str]],
        target_platforms: Optional[list[type_hints.SupportedTargetPlatformType]],
        options: Optional[type_hints.ModelSaveOption],
        statement_params: Optional[dict[str, Any]],
        progress_status: type_hints.ProgressStatus,
    ) -> model_version_impl.ModelVersion:
        """Log HuggingFace model remotely using SYSTEM$IMPORT_MODEL."""
        if not isinstance(model, huggingface.TransformersPipeline):
            raise ValueError(
                f"Model must be a TransformersPipeline object. The provided model is a {type(model)} object"
            )
        progress_status.update("preparing remote model logging...")
        progress_status.increment()

        # Get compute pool from options or use default
        compute_pool = model.compute_pool_for_log
        if compute_pool is None:
            raise ValueError("compute_pool_for_log is required for remote logging")

        # Construct fully qualified model name
        db_name = database_name_id.identifier() if database_name_id else self._database_name.identifier()
        schema_name = schema_name_id.identifier() if schema_name_id else self._schema_name.identifier()
        fq_model_name = f"{db_name}.{schema_name}.{model_name_id.identifier()}"

        # Build YAML spec for import model
        yaml_content = self._build_import_model_yaml_spec(
            model=model,
            fq_model_name=fq_model_name,
            version_name=version_name,
            compute_pool=compute_pool,
            comment=comment,
            conda_dependencies=conda_dependencies,
            pip_requirements=pip_requirements,
            target_platforms=target_platforms,
        )

        progress_status.update("Remotely logging the model...")
        progress_status.increment()

        self._model_ops.run_import_model_query(
            database_name=db_name,
            schema_name=schema_name,
            yaml_content=yaml_content,
            statement_params=statement_params,
        )
        progress_status.update("Remotely logged the model")
        progress_status.increment()

        # Return ModelVersion object
        return self._create_model_version_ref(
            database_name_id=database_name_id,
            schema_name_id=schema_name_id,
            model_name_id=model_name_id,
            version_name_id=version_name_id,
        )

    def get_model(
        self,
        model_name: str,
        *,
        statement_params: Optional[dict[str, Any]] = None,
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
        statement_params: Optional[dict[str, Any]] = None,
    ) -> list[model_impl.Model]:
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
        statement_params: Optional[dict[str, Any]] = None,
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
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        database_name_id, schema_name_id, model_name_id = self._parse_fully_qualified_name(model_name)

        self._model_ops.delete_model_or_version(
            database_name=database_name_id,
            schema_name=schema_name_id,
            model_name=model_name_id,
            statement_params=statement_params,
        )

    def _create_model_version_ref(
        self,
        database_name_id: Optional[sql_identifier.SqlIdentifier],
        schema_name_id: Optional[sql_identifier.SqlIdentifier],
        model_name_id: sql_identifier.SqlIdentifier,
        version_name_id: sql_identifier.SqlIdentifier,
    ) -> model_version_impl.ModelVersion:
        """Create a ModelVersion reference object.

        Args:
            database_name_id: Database name identifier, falls back to instance database if None.
            schema_name_id: Schema name identifier, falls back to instance schema if None.
            model_name_id: Model name identifier.
            version_name_id: Version name identifier.

        Returns:
            ModelVersion reference object.
        """
        return model_version_impl.ModelVersion._ref(
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

    def _build_import_model_yaml_spec(
        self,
        model: huggingface.TransformersPipeline,
        fq_model_name: str,
        version_name: str,
        compute_pool: str,
        comment: Optional[str],
        conda_dependencies: Optional[list[str]],
        pip_requirements: Optional[list[str]],
        target_platforms: Optional[list[type_hints.SupportedTargetPlatformType]],
    ) -> str:
        """Build YAML spec for SYSTEM$IMPORT_MODEL.

        Args:
            model: HuggingFace TransformersPipeline model.
            fq_model_name: Fully qualified model name.
            version_name: Model version name.
            compute_pool: Compute pool name.
            comment: Optional comment for the model.
            conda_dependencies: Optional conda dependencies.
            pip_requirements: Optional pip requirements.
            target_platforms: Optional target platforms.

        Returns:
            YAML string representing the import model spec.
        """
        # Convert target_platforms to list of strings
        target_platforms_list = self._convert_target_platforms_to_list(target_platforms)

        # Build HuggingFaceModel spec
        hf_model = model_deployment_spec_schema.HuggingFaceModel(
            hf_model_name=model.model,
            task=model.task,
            tokenizer=getattr(model, "tokenizer", None),
            token_secret_object=model.secret_identifier,
            trust_remote_code=model.trust_remote_code if model.trust_remote_code is not None else False,
            revision=model.revision,
            hf_model_kwargs=json.dumps(model.model_kwargs) if model.model_kwargs else "{}",
        )

        # Build LogModelArgs
        log_model_args = model_deployment_spec_schema.LogModelArgs(
            pip_requirements=pip_requirements,
            conda_dependencies=conda_dependencies,
            target_platforms=target_platforms_list,
            comment=comment,
        )

        # Build ModelSpec
        model_spec = import_model_spec_schema.ModelSpec(
            name=import_model_spec_schema.ModelName(
                model_name=fq_model_name,
                version_name=version_name,
            ),
            hf_model=hf_model,
            log_model_args=log_model_args,
        )

        # Build ImportModelSpec
        import_spec = import_model_spec_schema.ImportModelSpec(
            compute_pool=compute_pool,
            models=[model_spec],
        )

        # Convert to YAML
        return yaml.safe_dump(import_spec.model_dump(exclude_none=True))

    def _convert_target_platforms_to_list(
        self, target_platforms: Optional[list[type_hints.SupportedTargetPlatformType]]
    ) -> Optional[list[str]]:
        """Convert target_platforms to list of strings.

        Args:
            target_platforms: List of target platforms (enums or strings).

        Returns:
            List of platform strings, or None if input is None.
        """
        if not target_platforms:
            return None

        target_platforms_list = []
        for tp in target_platforms:
            if hasattr(tp, "value"):
                # It's an enum, get the value
                target_platforms_list.append(tp.value)
            else:
                # It's already a string
                target_platforms_list.append(str(tp))
        return target_platforms_list

    def _parse_fully_qualified_name(
        self, model_name: str
    ) -> tuple[
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
