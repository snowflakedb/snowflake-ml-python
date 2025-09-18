import json
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Optional, Union

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import identifier, mixins

if TYPE_CHECKING:
    from snowflake.ml import dataset
    from snowflake.ml.feature_store import feature_view
    from snowflake.ml.model._client.model import model_version_impl

_PROJECT = "LINEAGE"
DOMAIN_LINEAGE_REGISTRY: dict[str, type["LineageNode"]] = {}


class LineageNode(mixins.SerializableSessionMixin):
    """
    Represents a node in a lineage graph and serves as the base class for all machine learning objects.
    """

    def __init__(
        self,
        session: snowpark.Session,
        name: str,
        domain: Union[Literal["feature_view", "dataset", "model", "table", "view"]],
        version: Optional[str] = None,
        status: Optional[Literal["ACTIVE", "DELETED", "MASKED"]] = None,
        created_on: Optional[datetime] = None,
    ) -> None:
        """
        Initializes a LineageNode instance.

        Args:
            session : The Snowflake session object.
            name : Fully qualified name of the lineage node, which is in the format '<db>.<schema>.<object_name>'.
            domain : The domain of the lineage node.
            version : The version of the lineage node, if applies.
            status : The status of the lineage node. Possible values are:
                      - 'MASKED': The user does not have the privilege to view the node.
                      - 'DELETED': The node has been deleted.
                      - 'ACTIVE': The node is currently active.
            created_on : The creation time of the lineage node.

        Raises:
            ValueError: If the name is not fully qualified.
        """
        if name and not identifier.is_fully_qualified_name(name):
            raise ValueError("name should be fully qualifed.")

        self._lineage_node_name = name
        self._lineage_node_domain = domain
        self._lineage_node_version = version
        self._lineage_node_status = status
        self._lineage_node_created_on = created_on
        self._session = session

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  name='{self._lineage_node_name}',\n"
            f"  version='{self._lineage_node_version}',\n"
            f"  domain='{self._lineage_node_domain}',\n"
            f"  status='{self._lineage_node_status}',\n"
            f"  created_on='{self._lineage_node_created_on}'\n"
            f")"
        )

    @staticmethod
    def _load_from_lineage_node(session: snowpark.Session, name: str, version: str) -> "LineageNode":
        """
        Loads the concrete object.

        Args:
            session : The Snowflake session object.
            name : Fully qualified name of the object.
            version : The version of object.

        Raises:
            NotImplementedError: If the derived class does not implement this method.
        """
        raise NotImplementedError()

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def lineage(
        self,
        direction: Literal["upstream", "downstream"] = "downstream",
        domain_filter: Optional[set[Literal["feature_view", "dataset", "model", "table", "view"]]] = None,
    ) -> list[Union["feature_view.FeatureView", "dataset.Dataset", "model_version_impl.ModelVersion", "LineageNode"]]:
        """
        Retrieves the lineage nodes connected to this node.

        Args:
            direction : The direction to trace lineage. Defaults to "downstream".
            domain_filter : Set of domains to filter nodes. Defaults to None.

        Returns:
            List[LineageNode]: A list of connected lineage nodes.
        """
        df = self._session.lineage.trace(
            self._lineage_node_name,
            self._lineage_node_domain.upper(),
            object_version=self._lineage_node_version,
            direction=direction,
            distance=1,
        )
        if domain_filter is not None:
            domain_filter = {d.lower() for d in domain_filter}  # type: ignore[misc]

        lineage_nodes: list["LineageNode"] = []
        for row in df.collect():
            lineage_object = (
                json.loads(row["TARGET_OBJECT"])
                if direction.lower() == "downstream"
                else json.loads(row["SOURCE_OBJECT"])
            )
            domain = lineage_object["domain"].lower()
            if domain_filter is None or domain in domain_filter:
                obj_name = ".".join(
                    identifier.rename_to_valid_snowflake_identifier(s)
                    for s in identifier.parse_schema_level_object_identifier(lineage_object["name"])
                    if s is not None
                )
                if domain in DOMAIN_LINEAGE_REGISTRY and lineage_object["status"] == "ACTIVE":
                    lineage_nodes.append(
                        DOMAIN_LINEAGE_REGISTRY[domain]._load_from_lineage_node(
                            self._session, obj_name, lineage_object.get("version")
                        )
                    )
                else:
                    lineage_nodes.append(
                        LineageNode(
                            name=obj_name,
                            version=lineage_object.get("version"),
                            domain=domain,
                            status=lineage_object["status"],
                            created_on=datetime.strptime(lineage_object["createdOn"], "%Y-%m-%dT%H:%M:%SZ"),
                            session=self._session,
                        )
                    )

        return lineage_nodes
