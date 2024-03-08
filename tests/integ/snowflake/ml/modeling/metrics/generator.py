from typing import Any, List, Tuple, Union

from snowflake.ml.modeling.metrics import metrics_utils
from tests.integ.snowflake.ml.modeling.framework import utils
from tests.integ.snowflake.ml.modeling.framework.utils import MAX_INT, MIN_INT, DataType

_NUM_ROWS = [
    metrics_utils.BATCH_SIZE // 2,  # row # < batch size
    metrics_utils.BATCH_SIZE + 7,  # row # > batch size
    metrics_utils.BATCH_SIZE * 4,  # row # is a multiple of batch size
]
_NUM_ROW_LARGE = metrics_utils.BATCH_SIZE * 100 + 7  # large row #
_NUM_ROWS.append(_NUM_ROW_LARGE)


def gen_test_cases(
    types: List[DataType], low: Union[int, List[int]] = MIN_INT, high: Union[int, List[int]] = MAX_INT
) -> Tuple[List[List[Any]], List[str]]:
    """
    Generate metrics test cases. The last test case has a large data size.

    Args:
        types: type per column
        low: lower bound(s) of the output interval (inclusive)
        high: upper bound(s) of the output interval (exclusive)

    Returns:
        A tuple of test data of multiple sizes and column names.
        The 1st column of test data is "ID".
    """
    data_list = []
    snowflake_identifiers: List[str] = []
    for num_row in _NUM_ROWS:
        data, identifiers = utils.gen_fuzz_data(
            rows=num_row,
            types=types,
            low=low,
            high=high,
        )
        data_list.append(data)
        if len(snowflake_identifiers) == 0:
            snowflake_identifiers = identifiers
    return data_list, snowflake_identifiers
