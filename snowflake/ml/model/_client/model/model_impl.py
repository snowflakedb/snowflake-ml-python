from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import identifier, sql_identifier
from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.model._client.ops import model_ops

_TELEMETRY_PROJECT = "MLOps"
_TELEMETRY_SUBPROJECT = "ModelManagement"


class Model:
    """Model Object containing multiple versions. Mapping to SQL's MODEL object."""

    _model_ops: model_ops.ModelOperator
    _model_name: sql_identifier.SqlIdentifier

    def __init__(self) -> None:
        raise RuntimeError("Model's initializer is not meant to be used. Use `get_model` from registry instead.")

    @classmethod
    def _ref(
        cls,
        model_ops: model_ops.ModelOperator,
        *,
        model_name: sql_identifier.SqlIdentifier,
    ) -> "Model":
        self: "Model" = object.__new__(cls)
        self._model_ops = model_ops
        self._model_name = model_name
        return self

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Model):
            return False
        return self._model_ops == __value._model_ops and self._model_name == __value._model_name

    @property
    def name(self) -> str:
        """Return the name of the model that can be used to refer to it in SQL."""
        return self._model_name.identifier()

    @property
    def fully_qualified_name(self) -> str:
        """Return the fully qualified name of the model that can be used to refer to it in SQL."""
        return self._model_ops._model_version_client.fully_qualified_model_name(self._model_name)

    @property
    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def description(self) -> str:
        """The description for the model. This is an alias of `comment`."""
        return self.comment

    @description.setter
    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def description(self, description: str) -> None:
        self.comment = description

    @property
    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def comment(self) -> str:
        """The comment to the model."""
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        return self._model_ops.get_comment(
            model_name=self._model_name,
            statement_params=statement_params,
        )

    @comment.setter
    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def comment(self, comment: str) -> None:
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        return self._model_ops.set_comment(
            comment=comment,
            model_name=self._model_name,
            statement_params=statement_params,
        )

    @property
    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def default(self) -> model_version_impl.ModelVersion:
        """The default version of the model."""
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
            class_name=self.__class__.__name__,
        )
        default_version_name = self._model_ops.get_default_version(
            model_name=self._model_name, statement_params=statement_params
        )
        return self.version(default_version_name)

    @default.setter
    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def default(self, version: Union[str, model_version_impl.ModelVersion]) -> None:
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
            class_name=self.__class__.__name__,
        )
        if isinstance(version, str):
            version_name = sql_identifier.SqlIdentifier(version)
        else:
            version_name = version._version_name
        self._model_ops.set_default_version(
            model_name=self._model_name, version_name=version_name, statement_params=statement_params
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def version(self, version_name: str) -> model_version_impl.ModelVersion:
        """
        Get a model version object given a version name in the model.

        Args:
            version_name: The name of the version.

        Raises:
            ValueError: When the requested version does not exist.

        Returns:
            The model version object.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        version_id = sql_identifier.SqlIdentifier(version_name)
        if self._model_ops.validate_existence(
            model_name=self._model_name,
            version_name=version_id,
            statement_params=statement_params,
        ):
            return model_version_impl.ModelVersion._ref(
                self._model_ops,
                model_name=self._model_name,
                version_name=version_id,
            )
        else:
            raise ValueError(
                f"Unable to find version with name {version_id.identifier()} in model {self.fully_qualified_name}"
            )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def versions(self) -> List[model_version_impl.ModelVersion]:
        """Get all versions in the model.

        Returns:
            A list of ModelVersion objects representing all versions in the model.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        version_names = self._model_ops.list_models_or_versions(
            model_name=self._model_name,
            statement_params=statement_params,
        )
        return [
            model_version_impl.ModelVersion._ref(
                self._model_ops,
                model_name=self._model_name,
                version_name=version_name,
            )
            for version_name in version_names
        ]

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def show_versions(self) -> pd.DataFrame:
        """Show information about all versions in the model.

        Returns:
            A Pandas DataFrame showing information about all versions in the model.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        rows = self._model_ops.show_models_or_versions(
            model_name=self._model_name,
            statement_params=statement_params,
        )
        return pd.DataFrame([row.as_dict() for row in rows])

    def delete_version(self, version_name: str) -> None:
        raise NotImplementedError("Deleting version has not been supported yet.")

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def show_tags(self) -> Dict[str, str]:
        """Get a dictionary showing the tag and its value attached to the model.

        Returns:
            The model version object.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        return self._model_ops.show_tags(model_name=self._model_name, statement_params=statement_params)

    def _parse_tag_name(
        self,
        tag_name: str,
    ) -> Tuple[sql_identifier.SqlIdentifier, sql_identifier.SqlIdentifier, sql_identifier.SqlIdentifier]:
        _tag_db, _tag_schema, _tag_name, _ = identifier.parse_schema_level_object_identifier(tag_name)
        if _tag_db is None:
            tag_db_id = self._model_ops._model_client._database_name
        else:
            tag_db_id = sql_identifier.SqlIdentifier(_tag_db)

        if _tag_schema is None:
            tag_schema_id = self._model_ops._model_client._schema_name
        else:
            tag_schema_id = sql_identifier.SqlIdentifier(_tag_schema)

        if _tag_name is None:
            raise ValueError(f"Unable parse the tag name `{tag_name}` you input.")

        tag_name_id = sql_identifier.SqlIdentifier(_tag_name)

        return tag_db_id, tag_schema_id, tag_name_id

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def get_tag(self, tag_name: str) -> Optional[str]:
        """Get the value of a tag attached to the model.

        Args:
            tag_name: The name of the tag, can be fully qualified. If not fully qualified, the database or schema of
                the model will be used.

        Returns:
            The tag value as a string if the tag is attached, otherwise None.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        tag_db_id, tag_schema_id, tag_name_id = self._parse_tag_name(tag_name)
        return self._model_ops.get_tag_value(
            model_name=self._model_name,
            tag_database_name=tag_db_id,
            tag_schema_name=tag_schema_id,
            tag_name=tag_name_id,
            statement_params=statement_params,
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def set_tag(self, tag_name: str, tag_value: str) -> None:
        """Set the value of a tag, attaching it to the model if not.

        Args:
            tag_name: The name of the tag, can be fully qualified. If not fully qualified, the database or schema of
                the model will be used.
            tag_value: The value of the tag
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        tag_db_id, tag_schema_id, tag_name_id = self._parse_tag_name(tag_name)
        self._model_ops.set_tag(
            model_name=self._model_name,
            tag_database_name=tag_db_id,
            tag_schema_name=tag_schema_id,
            tag_name=tag_name_id,
            tag_value=tag_value,
            statement_params=statement_params,
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def unset_tag(self, tag_name: str) -> None:
        """Unset a tag attached to a model.

        Args:
            tag_name: The name of the tag, can be fully qualified. If not fully qualified, the database or schema of
                the model will be used.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        tag_db_id, tag_schema_id, tag_name_id = self._parse_tag_name(tag_name)
        self._model_ops.unset_tag(
            model_name=self._model_name,
            tag_database_name=tag_db_id,
            tag_schema_name=tag_schema_id,
            tag_name=tag_name_id,
            statement_params=statement_params,
        )
