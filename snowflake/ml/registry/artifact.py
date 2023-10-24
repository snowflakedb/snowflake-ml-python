import enum
from typing import Optional


# Set of allowed artifact types.
class ArtifactType(enum.Enum):
    TESTTYPE = "TESTTYPE"  # A placeholder type just for unit test
    DATASET = "DATASET"


class Artifact:
    """
    A reference to artifact.

    Properties:
        id: A globally unique id represents this artifact.
        spec: Specification of artifact in json format.
        type: Type of artifact.
        name: Name of artifact.
        version: Version of artifact.
    """

    def __init__(self, type: ArtifactType, spec: str) -> None:
        """Create an artifact.

        Args:
            type: type of artifact.
            spec: specification in json format.
        """
        self.type: ArtifactType = type
        self.name: Optional[str] = None
        self.version: Optional[str] = None
        self._spec: str = spec
        self._id: Optional[str] = None

    def _log(self, name: str, version: str, id: str) -> None:
        """Additional information when this artifact is logged.

        Args:
            name: name of artifact.
            version: version of artifact.
            id: A global unique id represents this artifact.
        """
        self.name = name
        self.version = version
        self._id = id
