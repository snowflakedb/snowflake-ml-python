import enum
from dataclasses import dataclass
from typing import Optional


# Set of allowed artifact types.
class ArtifactType(enum.Enum):
    TESTTYPE = "TESTTYPE"  # A placeholder type just for unit test
    DATASET = "DATASET"


@dataclass(frozen=True)
class ArtifactReference:
    """
    A reference to artifact.

    Properties:
        id: A globally unique id represents this artifact.
        spec: Specification of artifact in json format.
        type: Type of artifact.
        name: Name of artifact.
        version: Version of artifact.
    """

    _id: str
    _spec: str

    type: ArtifactType
    name: str
    version: Optional[str] = None
