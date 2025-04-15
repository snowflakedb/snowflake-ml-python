import dataclasses
from typing import Optional, Union


@dataclasses.dataclass(frozen=True)
class DataFrameInfo:
    """Serializable information from Snowpark DataFrames"""

    sql: str
    query_id: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class DatasetInfo:
    """Serializable information from SnowML Datasets"""

    fully_qualified_name: str
    version: str
    url: Optional[str] = None
    exclude_cols: Optional[list[str]] = None


DataSource = Union[DataFrameInfo, DatasetInfo, str]
