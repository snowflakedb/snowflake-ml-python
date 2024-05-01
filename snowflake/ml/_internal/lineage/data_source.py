import dataclasses
from typing import List, Optional


@dataclasses.dataclass(frozen=True)
class DataSource:
    fully_qualified_name: str
    version: str
    url: str
    exclude_cols: Optional[List[str]] = None
