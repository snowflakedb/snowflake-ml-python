import math
import random
import sched
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, cast

from absl.logging import logging

from snowflake.snowpark import Session
from snowflake.snowpark.types import (
    DateType,
    StructType,
    TimestampType,
    TimeType,
    _FractionalType,
    _IntegralType,
    _NumericType,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stats:
    mean: float
    min_val: float
    max_val: float
    std: float


def replace_nan(val: Any, default_val: Any = 0) -> Any:
    return default_val if math.isnan(val) else val


class SyntheticDataGenerator:
    def __init__(self, session: Session, database: str, schema: str, source_table: str) -> None:
        self._session = session
        self._database = database
        self._schema = schema
        self._source_table = source_table
        self._table_schema, self._stats = self._collect_metadata()
        self._trigger_thread: Optional[threading.Thread] = None

    def trigger(self, batch_size: int, num_batches: int, freq: int = 10) -> None:
        def _run() -> None:
            s = sched.scheduler(time.monotonic, time.sleep)
            for i in range(num_batches):
                s.enter(freq * i, 1, self._generate_new_data, argument=(batch_size,))
            s.run()

        self._trigger_thread = threading.Thread(target=_run)
        self._trigger_thread.start()

    def __del__(self) -> None:
        if self._trigger_thread is not None:
            self._trigger_thread.join()

    def _collect_metadata(self) -> Tuple[StructType, Dict[str, Stats]]:
        df = self._session.table([self._database, self._schema, self._source_table])
        df_stats = df.to_pandas().describe()
        stats = {}
        for s in df.schema.fields:
            if not isinstance(s.datatype, (DateType, TimeType, TimestampType, _NumericType)):
                raise RuntimeError(f"Unsupported source column type {s.datatype} for {s.name}")

            cur_stats = df_stats[s.name]
            stats[s.name] = Stats(
                mean=replace_nan(cur_stats["mean"]),
                min_val=replace_nan(cur_stats["min"]),
                max_val=replace_nan(cur_stats["max"]),
                std=replace_nan(cur_stats["std"]),
            )

        return df.schema, stats

    def _generate_new_data(self, num_rows: int) -> None:
        batch = []
        for _ in range(num_rows):
            row = []
            for field in self._table_schema.fields:
                stats = self._stats[field.name]
                if isinstance(field.datatype, TimestampType):
                    row.append(time.time())
                elif isinstance(field.datatype, _IntegralType):
                    row.append(random.randint(cast(int, stats.min_val), cast(int, stats.max_val)))
                elif isinstance(field.datatype, _FractionalType):
                    row.append(random.uniform(stats.min_val, stats.max_val))
                else:
                    raise RuntimeError(f"Unsupported type: {field.datatype}")
            batch.append(row)

        df = self._session.create_dataframe(batch, self._table_schema)
        df.write.mode("append").save_as_table([self._database, self._schema, self._source_table], block=True)
        logger.info(f"Dumped {num_rows} rows to table {self._source_table}.")
