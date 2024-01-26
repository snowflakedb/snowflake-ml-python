import json
import pathlib
import tempfile
from typing import Any, Dict, List, Optional, Union, cast

import yaml

from snowflake.ml._internal.utils import identifier, sql_identifier
from snowflake.ml.model import model_signature, type_hints
from snowflake.ml.model._client.ops import metadata_ops
from snowflake.ml.model._client.sql import (
    model as model_sql,
    model_version as model_version_sql,
    stage as stage_sql,
    tag as tag_sql,
)
from snowflake.ml.model._model_composer import model_composer
from snowflake.ml.model._model_composer.model_manifest import (
    model_manifest,
    model_manifest_schema,
)
from snowflake.ml.model._packager.model_meta import model_meta, model_meta_schema
from snowflake.ml.model._signatures import snowpark_handler
from snowflake.snowpark import dataframe, row, session
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
        self._tag_client = tag_sql.ModuleTagSQLClient(
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

    def show_models_or_versions(
        self,
        *,
        model_name: Optional[sql_identifier.SqlIdentifier] = None,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> List[row.Row]:
        if model_name:
            return self._model_client.show_versions(
                model_name=model_name,
                validate_result=False,
                statement_params=statement_params,
            )
        else:
            return self._model_client.show_models(
                validate_result=False,
                statement_params=statement_params,
            )

    def list_models_or_versions(
        self,
        *,
        model_name: Optional[sql_identifier.SqlIdentifier] = None,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> List[sql_identifier.SqlIdentifier]:
        res = self.show_models_or_versions(
            model_name=model_name,
            statement_params=statement_params,
        )
        if model_name:
            col_name = self._model_client.MODEL_VERSION_NAME_COL_NAME
        else:
            col_name = self._model_client.MODEL_NAME_COL_NAME
        return [sql_identifier.SqlIdentifier(row[col_name], case_sensitive=True) for row in res]

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
                validate_result=False,
                statement_params=statement_params,
            )
        else:
            res = self._model_client.show_models(
                model_name=model_name,
                validate_result=False,
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
            col_name = self._model_client.MODEL_VERSION_COMMENT_COL_NAME
        else:
            res = self._model_client.show_models(
                model_name=model_name,
                statement_params=statement_params,
            )
            col_name = self._model_client.MODEL_COMMENT_COL_NAME
        return cast(str, res[0][col_name])

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

    def set_default_version(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.validate_existence(
            model_name=model_name, version_name=version_name, statement_params=statement_params
        ):
            raise ValueError(f"You cannot set version {version_name} as default version as it does not exist.")
        self._model_version_client.set_default_version(
            model_name=model_name, version_name=version_name, statement_params=statement_params
        )

    def get_default_version(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> sql_identifier.SqlIdentifier:
        res = self._model_client.show_models(model_name=model_name, statement_params=statement_params)[0]
        return sql_identifier.SqlIdentifier(
            res[self._model_client.MODEL_DEFAULT_VERSION_NAME_COL_NAME], case_sensitive=True
        )

    def get_tag_value(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        tag_database_name: sql_identifier.SqlIdentifier,
        tag_schema_name: sql_identifier.SqlIdentifier,
        tag_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        r = self._tag_client.get_tag_value(
            module_name=model_name,
            tag_database_name=tag_database_name,
            tag_schema_name=tag_schema_name,
            tag_name=tag_name,
            statement_params=statement_params,
        )
        value = r.TAG_VALUE
        if value is None:
            return value
        return str(value)

    def show_tags(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        tags_info = self._tag_client.get_tag_list(
            module_name=model_name,
            statement_params=statement_params,
        )
        res: Dict[str, str] = {
            identifier.get_schema_level_object_identifier(
                sql_identifier.SqlIdentifier(r.TAG_DATABASE, case_sensitive=True),
                sql_identifier.SqlIdentifier(r.TAG_SCHEMA, case_sensitive=True),
                sql_identifier.SqlIdentifier(r.TAG_NAME, case_sensitive=True),
            ): str(r.TAG_VALUE)
            for r in tags_info
        }
        return res

    def set_tag(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        tag_database_name: sql_identifier.SqlIdentifier,
        tag_schema_name: sql_identifier.SqlIdentifier,
        tag_name: sql_identifier.SqlIdentifier,
        tag_value: str,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._tag_client.set_tag_on_model(
            model_name=model_name,
            tag_database_name=tag_database_name,
            tag_schema_name=tag_schema_name,
            tag_name=tag_name,
            tag_value=tag_value,
            statement_params=statement_params,
        )

    def unset_tag(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        tag_database_name: sql_identifier.SqlIdentifier,
        tag_schema_name: sql_identifier.SqlIdentifier,
        tag_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._tag_client.unset_tag_on_model(
            model_name=model_name,
            tag_database_name=tag_database_name,
            tag_schema_name=tag_schema_name,
            tag_name=tag_name,
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

    def get_client_data_in_user_data(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> model_manifest_schema.SnowparkMLDataDict:
        raw_user_data_json_string = self._model_client.show_versions(
            model_name=model_name,
            version_name=version_name,
            statement_params=statement_params,
        )[0][self._model_client.MODEL_VERSION_USER_DATA_COL_NAME]
        raw_user_data = json.loads(raw_user_data_json_string)
        assert isinstance(raw_user_data, dict), "user data should be a dictionary"
        return model_manifest.ModelManifest.parse_client_data_from_user_data(raw_user_data)

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
