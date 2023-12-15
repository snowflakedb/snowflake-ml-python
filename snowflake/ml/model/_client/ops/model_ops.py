import pathlib
import tempfile
from typing import Any, Dict, List, Optional, Union, cast

import yaml

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import model_signature, type_hints
from snowflake.ml.model._client.ops import metadata_ops
from snowflake.ml.model._client.sql import (
    model as model_sql,
    model_version as model_version_sql,
    stage as stage_sql,
)
from snowflake.ml.model._model_composer import model_composer
from snowflake.ml.model._model_composer.model_manifest import (
    model_manifest,
    model_manifest_schema,
)
from snowflake.ml.model._packager.model_meta import model_meta, model_meta_schema
from snowflake.ml.model._signatures import snowpark_handler
from snowflake.snowpark import dataframe, session
from snowflake.snowpark._internal import utils as snowpark_utils


class ModelOperator:
    def __init__(
        self,
        session: session.Session,
        *,
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
    ) -> None:
        # Ideally, we should only keep session object inside the client, however, some components other than client
        # are requiring session object like ModelComposer and SnowparkDataFrameHandler. We currently cannot refractor
        # them all but we should try to avoid use the _session object here unless no other choice.
        self._session = session
        self._stage_client = stage_sql.StageSQLClient(
            session,
            database_name=database_name,
            schema_name=schema_name,
        )
        self._model_client = model_sql.ModelSQLClient(
            session,
            database_name=database_name,
            schema_name=schema_name,
        )
        self._model_version_client = model_version_sql.ModelVersionSQLClient(
            session,
            database_name=database_name,
            schema_name=schema_name,
        )
        self._metadata_ops = metadata_ops.MetadataOperator(
            session,
            database_name=database_name,
            schema_name=schema_name,
        )

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ModelOperator):
            return False
        return (
            self._stage_client == __value._stage_client
            and self._model_client == __value._model_client
            and self._model_version_client == __value._model_version_client
        )

    def prepare_model_stage_path(self, *, statement_params: Optional[Dict[str, Any]] = None) -> str:
        stage_name = sql_identifier.SqlIdentifier(
            snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.STAGE)
        )
        self._stage_client.create_tmp_stage(stage_name=stage_name, statement_params=statement_params)
        return f"@{self._stage_client.fully_qualified_stage_name(stage_name)}/model"

    def create_from_stage(
        self,
        composed_model: model_composer.ModelComposer,
        *,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        stage_path = str(composed_model.stage_path)
        if self.validate_existence(
            model_name=model_name,
            statement_params=statement_params,
        ):
            if self.validate_existence(
                model_name=model_name,
                version_name=version_name,
                statement_params=statement_params,
            ):
                raise ValueError(
                    f"Model {self._model_version_client.fully_qualified_model_name(model_name)} "
                    f"version {version_name} already existed."
                )
            else:
                self._model_version_client.add_version_from_stage(
                    stage_path=stage_path,
                    model_name=model_name,
                    version_name=version_name,
                    statement_params=statement_params,
                )
        else:
            self._model_version_client.create_from_stage(
                stage_path=stage_path,
                model_name=model_name,
                version_name=version_name,
                statement_params=statement_params,
            )

    def list_models_or_versions(
        self,
        *,
        model_name: Optional[sql_identifier.SqlIdentifier] = None,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> List[sql_identifier.SqlIdentifier]:
        if model_name:
            res = self._model_client.show_versions(
                model_name=model_name,
                statement_params=statement_params,
            )
        else:
            res = self._model_client.show_models(
                statement_params=statement_params,
            )
        return [sql_identifier.SqlIdentifier(row.name, case_sensitive=True) for row in res]

    def validate_existence(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        version_name: Optional[sql_identifier.SqlIdentifier] = None,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if version_name:
            res = self._model_client.show_versions(
                model_name=model_name,
                version_name=version_name,
                statement_params=statement_params,
            )
        else:
            res = self._model_client.show_models(
                model_name=model_name,
                statement_params=statement_params,
            )
        return len(res) == 1

    def get_comment(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        version_name: Optional[sql_identifier.SqlIdentifier] = None,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        if version_name:
            res = self._model_client.show_versions(
                model_name=model_name,
                version_name=version_name,
                statement_params=statement_params,
            )
        else:
            res = self._model_client.show_models(
                model_name=model_name,
                statement_params=statement_params,
            )
        assert len(res) == 1
        return cast(str, res[0].comment)

    def set_comment(
        self,
        *,
        comment: str,
        model_name: sql_identifier.SqlIdentifier,
        version_name: Optional[sql_identifier.SqlIdentifier] = None,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if version_name:
            self._model_version_client.set_comment(
                comment=comment,
                model_name=model_name,
                version_name=version_name,
                statement_params=statement_params,
            )
        else:
            self._model_client.set_comment(
                comment=comment,
                model_name=model_name,
                statement_params=statement_params,
            )

    def get_model_version_manifest(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> model_manifest_schema.ModelManifestDict:
        with tempfile.TemporaryDirectory() as tmpdir:
            self._model_version_client.get_file(
                model_name=model_name,
                version_name=version_name,
                file_path=pathlib.PurePosixPath(model_manifest.ModelManifest.MANIFEST_FILE_REL_PATH),
                target_path=pathlib.Path(tmpdir),
                statement_params=statement_params,
            )
            mm = model_manifest.ModelManifest(pathlib.Path(tmpdir))
            return mm.load()

    def get_model_version_native_packing_meta(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> model_meta_schema.ModelMetadataDict:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_meta_file_path = self._model_version_client.get_file(
                model_name=model_name,
                version_name=version_name,
                file_path=pathlib.PurePosixPath(
                    model_composer.ModelComposer.MODEL_DIR_REL_PATH, model_meta.MODEL_METADATA_FILE
                ),
                target_path=pathlib.Path(tmpdir),
                statement_params=statement_params,
            )
            with open(model_meta_file_path, encoding="utf-8") as f:
                raw_model_meta = yaml.safe_load(f)
            return model_meta.ModelMetadata._validate_model_metadata(raw_model_meta)

    def invoke_method(
        self,
        *,
        method_name: sql_identifier.SqlIdentifier,
        signature: model_signature.ModelSignature,
        X: Union[type_hints.SupportedDataType, dataframe.DataFrame],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, str]] = None,
    ) -> Union[type_hints.SupportedDataType, dataframe.DataFrame]:
        identifier_rule = model_signature.SnowparkIdentifierRule.INFERRED

        # Validate and prepare input
        if not isinstance(X, dataframe.DataFrame):
            keep_order = True
            output_with_input_features = False
            df = model_signature._convert_and_validate_local_data(X, signature.inputs)
            s_df = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(self._session, df, keep_order=keep_order)
        else:
            keep_order = False
            output_with_input_features = True
            identifier_rule = model_signature._validate_snowpark_data(X, signature.inputs)
            s_df = X

        original_cols = s_df.columns

        # Compose input and output names
        input_args = []
        for input_feature in signature.inputs:
            col_name = identifier_rule.get_sql_identifier_from_feature(input_feature.name)

            input_args.append(col_name)

        returns = []
        for output_feature in signature.outputs:
            output_name = identifier_rule.get_sql_identifier_from_feature(output_feature.name)
            returns.append((output_feature.name, output_feature.as_snowpark_type(), output_name))
            # Avoid removing output cols when output_with_input_features is False
            if output_name in original_cols:
                original_cols.remove(output_name)

        df_res = self._model_version_client.invoke_method(
            method_name=method_name,
            input_df=s_df,
            input_args=input_args,
            returns=returns,
            model_name=model_name,
            version_name=version_name,
            statement_params=statement_params,
        )

        if keep_order:
            df_res = df_res.sort(
                "_ID",
                ascending=True,
            )

        if not output_with_input_features:
            df_res = df_res.drop(*original_cols)

        # Get final result
        if not isinstance(X, dataframe.DataFrame):
            return snowpark_handler.SnowparkDataFrameHandler.convert_to_df(df_res, features=signature.outputs)
        else:
            return df_res

    def delete_model_or_version(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        version_name: Optional[sql_identifier.SqlIdentifier] = None,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        # TODO: Delete version is not supported yet.
        self._model_client.drop_model(
            model_name=model_name,
            statement_params=statement_params,
        )
