from typing import Any, Optional

from snowflake import snowpark
from snowflake.connector import connection
from snowflake.ml._internal.utils import identifier

from . import stage_fs


class SFEmbeddedStageFileSystem(stage_fs.SFStageFileSystem):
    def __init__(
        self,
        *,
        domain: str,
        name: str,
        snowpark_session: Optional[snowpark.Session] = None,
        sf_connection: Optional[connection.SnowflakeConnection] = None,
        **kwargs: Any,
    ) -> None:

        (db, schema, object_name, _) = identifier.parse_schema_level_object_identifier(name)
        self._name = name  # TODO: Require or resolve FQN
        self._domain = domain

        super().__init__(
            db=db,
            schema=schema,
            stage=object_name,
            snowpark_session=snowpark_session,
            sf_connection=sf_connection,
            **kwargs,
        )

    @property
    def stage_name(self) -> str:
        """Get the Snowflake path to this stage.

        Returns:
            A string in the format of snow://<domain>/<name>
                Example: snow://dataset/my_dataset

        # noqa: DAR203
        """
        return f"snow://{self._domain}/{self._name}"

    def _stage_path_to_relative_path(self, stage_path: str) -> str:
        """Convert a stage file path which comes from the LIST query to a relative file path in that stage.

        The file path returned by LIST query always has the format "versions/<version>/<relative_file_path>".
                The full "versions/<version>/<relative_file_path>" is returned

        Args:
            stage_path: A string started with the name of the stage.

        Returns:
            A string of the relative stage path.
        """
        return stage_path
