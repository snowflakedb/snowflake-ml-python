from types import ModuleType
from typing import Any, Dict, List, Optional

import pandas as pd
from absl.logging import logging

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._client.model import model_impl, model_version_impl
from snowflake.ml.model._client.ops import metadata_ops, model_ops
from snowflake.ml.model._model_composer import model_composer
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

    def log_model(
        self,
        model: model_types.SupportedModelType,
        *,
        model_name: str,
        version_name: str,
        comment: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        conda_dependencies: Optional[List[str]] = None,
        pip_requirements: Optional[List[str]] = None,
        python_version: Optional[str] = None,
        signatures: Optional[Dict[str, model_signature.ModelSignature]] = None,
        sample_input_data: Optional[model_types.SupportedDataType] = None,
        code_paths: Optional[List[str]] = None,
        ext_modules: Optional[List[ModuleType]] = None,
        options: Optional[model_types.ModelSaveOption] = None,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> model_version_impl.ModelVersion:
        model_name_id = sql_identifier.SqlIdentifier(model_name)

        version_name_id = sql_identifier.SqlIdentifier(version_name)

        if self._model_ops.validate_existence(
            model_name=model_name_id, statement_params=statement_params
        ) and self._model_ops.validate_existence(
            model_name=model_name_id, version_name=version_name_id, statement_params=statement_params
        ):
            raise ValueError(f"Model {model_name} version {version_name} already existed.")

        stage_path = self._model_ops.prepare_model_stage_path(
            statement_params=statement_params,
        )

        logger.info("Start packaging and uploading your model. It might take some time based on the size of the model.")

        mc = model_composer.ModelComposer(self._model_ops._session, stage_path=stage_path)
        mc.save(
            name=model_name_id.resolved(),
            model=model,
            signatures=signatures,
            sample_input=sample_input_data,
            conda_dependencies=conda_dependencies,
            pip_requirements=pip_requirements,
            python_version=python_version,
            code_paths=code_paths,
            ext_modules=ext_modules,
            options=options,
        )

        logger.info("Start creating MODEL object for you in the Snowflake.")

        self._model_ops.create_from_stage(
            composed_model=mc,
            model_name=model_name_id,
            version_name=version_name_id,
            statement_params=statement_params,
        )

        mv = model_version_impl.ModelVersion._ref(
            self._model_ops,
            model_name=model_name_id,
            version_name=version_name_id,
        )

        if comment:
            mv.comment = comment

        if metrics:
            self._model_ops._metadata_ops.save(
                metadata_ops.ModelVersionMetadataSchema(metrics=metrics),
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
        model_name_id = sql_identifier.SqlIdentifier(model_name)
        if self._model_ops.validate_existence(
            model_name=model_name_id,
            statement_params=statement_params,
        ):
            return model_impl.Model._ref(
                self._model_ops,
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
            statement_params=statement_params,
        )
        return [
            model_impl.Model._ref(
                self._model_ops,
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
            statement_params=statement_params,
        )
        return pd.DataFrame([row.as_dict() for row in rows])

    def delete_model(
        self,
        model_name: str,
        *,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        model_name_id = sql_identifier.SqlIdentifier(model_name)

        self._model_ops.delete_model_or_version(
            model_name=model_name_id,
            statement_params=statement_params,
        )
