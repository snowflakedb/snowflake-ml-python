import dataclasses
from typing import Union


@dataclasses.dataclass(frozen=True)
class DataFrameInfo:
    """Serializable information from Snowpark DataFrames"""

    sql: str
    query_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DatasetInfo:
    """Serializable information from SnowML Datasets"""

    fully_qualified_name: str
    version: str
    url: str | None = None
    exclude_cols: list[str] | None = None


DataSource = Union[DataFrameInfo, DatasetInfo, str]
