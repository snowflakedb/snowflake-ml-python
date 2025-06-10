from typing import Optional, Union

import pandas as pd

from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.model._client.ops import model_ops, service_ops

_TELEMETRY_PROJECT = "MLOps"
_TELEMETRY_SUBPROJECT = "ModelManagement"
SYSTEM_VERSION_ALIAS_DEFAULT = "DEFAULT"
SYSTEM_VERSION_ALIAS_FIRST = "FIRST"
SYSTEM_VERSION_ALIAS_LAST = "LAST"
SYSTEM_VERSION_ALIASES = (SYSTEM_VERSION_ALIAS_DEFAULT, SYSTEM_VERSION_ALIAS_FIRST, SYSTEM_VERSION_ALIAS_LAST)


class Model:
    """Model Object containing multiple versions. Mapping to SQL's MODEL object."""

    _model_ops: model_ops.ModelOperator
    _service_ops: service_ops.ServiceOperator
    _model_name: sql_identifier.SqlIdentifier

    def __init__(self) -> None:
        raise RuntimeError("Model's initializer is not meant to be used. Use `get_model` from registry instead.")

    @classmethod
    def _ref(
        cls,
        model_ops: model_ops.ModelOperator,
        *,
        service_ops: service_ops.ServiceOperator,
        model_name: sql_identifier.SqlIdentifier,
    ) -> "Model":
        self: "Model" = object.__new__(cls)
        self._model_ops = model_ops
        self._service_ops = service_ops
        self._model_name = model_name
        return self

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Model):
            return False
        return (
            self._model_ops == __value._model_ops
            and self._service_ops == __value._service_ops
            and self._model_name == __value._model_name
        )

    @property
    def name(self) -> str:
        """Return the name of the model that can be used to refer to it in SQL."""
        return self._model_name.identifier()

    @property
    def fully_qualified_name(self) -> str:
        """Return the fully qualified name of the model that can be used to refer to it in SQL."""
        return self._model_ops._model_version_client.fully_qualified_object_name(None, None, self._model_name)

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
            database_name=None,
            schema_name=None,
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
            database_name=None,
            schema_name=None,
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
            database_name=None, schema_name=None, model_name=self._model_name, statement_params=statement_params
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
            database_name=None,
            schema_name=None,
            model_name=self._model_name,
            version_name=version_name,
            statement_params=statement_params,
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def first(self) -> model_version_impl.ModelVersion:
        """The first version of the model."""
        return self.version(SYSTEM_VERSION_ALIAS_FIRST)

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def last(self) -> model_version_impl.ModelVersion:
        """The latest version of the model."""
        return self.version(SYSTEM_VERSION_ALIAS_LAST)

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def version(self, version_or_alias: str) -> model_version_impl.ModelVersion:
        """
        Get a model version object given a version name or version alias in the model.

        Args:
            version_or_alias: The name of the version or alias to a version.

        Raises:
            ValueError: When the requested version does not exist.

        Returns:
            The model version object.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )

        # check with system alias or with user defined alias
        version_id = self._model_ops.get_version_by_alias(
            database_name=None,
            schema_name=None,
            model_name=self._model_name,
            alias_name=sql_identifier.SqlIdentifier(version_or_alias),
            statement_params=statement_params,
        )

        # version_id is still None implies version_or_alias is not an alias. So it must be a version name.
        if version_id is None:
            version_id = sql_identifier.SqlIdentifier(version_or_alias)
            if not self._model_ops.validate_existence(
                database_name=None,
                schema_name=None,
                model_name=self._model_name,
                version_name=version_id,
                statement_params=statement_params,
            ):
                raise ValueError(
                    f"Unable to find version or alias with name {version_id.identifier()} "
                    f"in model {self.fully_qualified_name}"
                )

        return model_version_impl.ModelVersion._ref(
            self._model_ops,
            service_ops=self._service_ops,
            model_name=self._model_name,
            version_name=version_id,
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def versions(self) -> list[model_version_impl.ModelVersion]:
        """Get all versions in the model.

        Returns:
            A list of ModelVersion objects representing all versions in the model.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        version_names = self._model_ops.list_models_or_versions(
            database_name=None,
            schema_name=None,
            model_name=self._model_name,
            statement_params=statement_params,
        )
        return [
            model_version_impl.ModelVersion._ref(
                self._model_ops,
                service_ops=self._service_ops,
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
            database_name=None,
            schema_name=None,
            model_name=self._model_name,
            statement_params=statement_params,
        )
        return pd.DataFrame([row.as_dict() for row in rows])

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def delete_version(self, version_name: str) -> None:
        """Drop a version of the model.

        Args:
            version_name: The name of the version.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        self._model_ops.delete_model_or_version(
            database_name=None,
            schema_name=None,
            model_name=self._model_name,
            version_name=sql_identifier.SqlIdentifier(version_name),
            statement_params=statement_params,
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def show_tags(self) -> dict[str, str]:
        """Get a dictionary showing the tag and its value attached to the model.

        Returns:
            The model version object.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        return self._model_ops.show_tags(
            database_name=None, schema_name=None, model_name=self._model_name, statement_params=statement_params
        )

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
        tag_db_id, tag_schema_id, tag_name_id = sql_identifier.parse_fully_qualified_name(tag_name)
        return self._model_ops.get_tag_value(
            database_name=None,
            schema_name=None,
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
        tag_db_id, tag_schema_id, tag_name_id = sql_identifier.parse_fully_qualified_name(tag_name)
        self._model_ops.set_tag(
            database_name=None,
            schema_name=None,
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
        tag_db_id, tag_schema_id, tag_name_id = sql_identifier.parse_fully_qualified_name(tag_name)
        self._model_ops.unset_tag(
            database_name=None,
            schema_name=None,
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
    def rename(self, model_name: str) -> None:
        """Rename a model. Can be used to move a model when a fully qualified name is provided.

        Args:
            model_name: The new model name.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        new_db, new_schema, new_model = sql_identifier.parse_fully_qualified_name(model_name)

        self._model_ops.rename(
            database_name=None,
            schema_name=None,
            model_name=self._model_name,
            new_model_db=new_db,
            new_model_schema=new_schema,
            new_model_name=new_model,
            statement_params=statement_params,
        )
        self._model_ops = model_ops.ModelOperator(
            self._model_ops._session,
            database_name=new_db or self._model_ops._model_client._database_name,
            schema_name=new_schema or self._model_ops._model_client._schema_name,
        )
        self._model_name = new_model

    def _repr_html_(self) -> str:
        """Generate an HTML representation of the model.

        Returns:
            str: HTML string containing formatted model details.
        """
        from snowflake.ml.utils import html_utils

        # Get default version
        default_version = self.default.version_name

        # Get versions info
        try:
            versions_df = self.show_versions()
            versions_html = ""

            for _, row in versions_df.iterrows():
                versions_html += html_utils.create_version_item(
                    version_name=row["name"],
                    created_on=str(row["created_on"]),
                    comment=str(row.get("comment", "")),
                    is_default=bool(row["is_default_version"]),
                )
        except Exception:
            versions_html = html_utils.create_error_message("Error retrieving versions")

        # Get tags
        try:
            tags = self.show_tags()
            if not tags:
                tags_html = html_utils.create_error_message("No tags available")
            else:
                tags_html = ""
                for tag_name, tag_value in tags.items():
                    tags_html += html_utils.create_tag_item(tag_name, tag_value)
        except Exception:
            tags_html = html_utils.create_error_message("Error retrieving tags")

        # Create main content sections
        main_info = html_utils.create_grid_section(
            [
                ("Model Name", self.name),
                ("Full Name", self.fully_qualified_name),
                ("Description", self.description),
                ("Default Version", default_version),
            ]
        )

        versions_section = html_utils.create_section_header("Versions") + html_utils.create_content_section(
            versions_html
        )

        tags_section = html_utils.create_section_header("Tags") + html_utils.create_content_section(tags_html)

        content = main_info + versions_section + tags_section

        return html_utils.create_base_container("Model Details", content)
