from typing import Optional, cast

from snowflake import connector, snowpark
from snowflake.ml._internal.utils import formatting, table_manager
from snowflake.ml.registry import _initial_schema, artifact


class ArtifactManager:
    """It manages artifacts in model registry."""

    def __init__(
        self,
        session: snowpark.Session,
        database_name: str,
        schema_name: str,
    ) -> None:
        """Initializer of artifact manager.

        Args:
            session: Session object to communicate with Snowflake.
            database_name: Desired name of the model registry database.
            schema_name: Desired name of the schema used by this model registry inside the database.
        """
        self._session = session
        self._database_name = database_name
        self._schema_name = schema_name
        self._fully_qualified_table_name = table_manager.get_fully_qualified_table_name(
            self._database_name, self._schema_name, _initial_schema._ARTIFACT_TABLE_NAME
        )

    def exists(
        self,
        artifact_name: str,
        artifact_version: Optional[str] = None,
    ) -> bool:
        """Validate if an artifact exists.

        Args:
            artifact_name: Name of artifact.
            artifact_version: Version of artifact.

        Returns:
            bool: True if the artifact exists, False otherwise.
        """
        selected_artifact = self.get(artifact_name, artifact_version).collect()

        assert (
            len(selected_artifact) < 2
        ), f"""Multiple records found for artifact with name/version: {artifact_name}/{artifact_version}!"""

        return len(selected_artifact) == 1

    def add(
        self,
        artifact: artifact.Artifact,
        artifact_id: str,
        artifact_name: str,
        artifact_version: Optional[str] = None,
    ) -> artifact.Artifact:
        """
        Add a new artifact.

        Args:
            artifact: artifact object.
            artifact_id: id of artifact.
            artifact_name: name of artifact.
            artifact_version: version of artifact.

        Returns:
            A reference to artifact.
        """
        if artifact_version is None:
            artifact_version = ""
        assert artifact_id != "", "Artifact id can't be empty."

        new_artifact = {
            "ID": artifact_id,
            "TYPE": artifact.type.value,
            "NAME": artifact_name,
            "VERSION": artifact_version,
            "CREATION_ROLE": self._session.get_current_role(),
            "CREATION_TIME": formatting.SqlStr("CURRENT_TIMESTAMP()"),
            "ARTIFACT_SPEC": artifact._spec,
        }

        # TODO: Consider updating the METADATA table for artifact history tracking as well.
        table_manager.insert_table_entry(self._session, self._fully_qualified_table_name, new_artifact)
        artifact._log(name=artifact_name, version=artifact_version, id=artifact_id)
        return artifact

    def delete(
        self,
        artifact_name: str,
        artifact_version: Optional[str] = None,
        error_if_not_exist: bool = False,
    ) -> None:
        """
        Remove an artifact.

        Args:
            artifact_name: Name of artifact.
            artifact_version: Version of artifact.
            error_if_not_exist: Whether to raise errors if the target entry doesn't exist. Default to be false.

        Raises:
            DataError: If error_if_not_exist is true and the artifact doesn't exist in the database.
            RuntimeError: If the artifact deletion failed.
        """
        if not self.exists(artifact_name, artifact_version):
            if error_if_not_exist:
                raise connector.DataError(
                    f"Artifact {artifact_name}/{artifact_version} doesn't exist. Deletion failed."
                )
            else:
                return

        if artifact_version is None:
            artifact_version = ""
        delete_query = f"""DELETE FROM {self._fully_qualified_table_name}
                WHERE NAME='{artifact_name}' AND VERSION='{artifact_version}'
            """

        # TODO: Consider updating the METADATA table for artifact history tracking as well.
        try:
            self._session.sql(delete_query).collect()
        except Exception as e:
            raise RuntimeError(f"Delete artifact {artifact_name}/{artifact_version} failed due to {e}")

    def get(
        self,
        artifact_name: str,
        artifact_version: Optional[str] = None,
    ) -> snowpark.DataFrame:
        """Retrieve the Snowpark dataframe of the artifact matching the provided artifact id and type.

        Given that ID and TYPE act as a compound primary key for the artifact table,
        the resulting dataframe should have at most, one row.

        Args:
            artifact_name: Name of artifact.
            artifact_version: Version of artifact.

        Returns:
            A Snowpark dataframe representing the artifacts that match the given constraints.

        WARNING:
            The returned DataFrame is writable and shouldn't be made accessible to users.
        """
        if artifact_version is None:
            artifact_version = ""

        artifacts = self._session.sql(f"SELECT * FROM {self._fully_qualified_table_name}")
        target_artifact = artifacts.filter(snowpark.Column("NAME") == artifact_name).filter(
            snowpark.Column("VERSION") == artifact_version
        )
        return cast(snowpark.DataFrame, target_artifact)
