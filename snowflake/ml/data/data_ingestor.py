from typing import TYPE_CHECKING, Any, Iterator, Optional, Protocol, Sequence, TypeVar

from numpy import typing as npt

from snowflake import snowpark
from snowflake.ml.data import data_source

if TYPE_CHECKING:
    import pandas as pd
    import ray


DataIngestorType = TypeVar("DataIngestorType", bound="DataIngestor")


class DataIngestor(Protocol):
    @classmethod
    def from_sources(
        cls: type[DataIngestorType], session: snowpark.Session, sources: Sequence[data_source.DataSource]
    ) -> DataIngestorType:
        raise NotImplementedError

    @classmethod
    def from_ray_dataset(
        cls: type[DataIngestorType],
        ray_ds: "ray.data.Dataset",
    ) -> DataIngestorType:
        raise NotImplementedError

    @property
    def data_sources(self) -> list[data_source.DataSource]:
        raise NotImplementedError

    def to_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last_batch: bool = True,
    ) -> Iterator[dict[str, npt.NDArray[Any]]]:
        raise NotImplementedError

    def to_pandas(self, limit: Optional[int] = None) -> "pd.DataFrame":
        raise NotImplementedError
