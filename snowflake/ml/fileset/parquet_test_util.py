import tempfile
from typing import Any, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_DATA0 = {"col1": [0], "col2": [10], "col3": ["a"]}
_DATA1 = {"col1": [1, 2], "col2": [11, 12], "col3": ["ab", "abc"]}
_DATA2 = {"col1": [3, 4, 5, 6], "col2": [13, 14, np.NaN, 16], "col3": ["m", "mn", "mnm", "mnmn"]}


def write_parquet_file() -> Tuple[Any, ...]:  # Use "Any" as type hints to satisfy mypy check.
    """Creates 3 temporary parquet files for testing."""
    files = []
    for data in [_DATA0, _DATA1, _DATA2]:
        f = tempfile.NamedTemporaryFile()
        t = pa.table(
            data,
            schema=pa.schema(
                [pa.field("col1", pa.int64()), pa.field("col2", pa.float64()), pa.field("col3", pa.string())]
            ),
        )
        pq.write_table(t, f.name)
        files.append(f)
    return tuple(files)
