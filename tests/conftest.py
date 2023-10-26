import inspect
import os
from unittest import mock

import cloudpickle as cp
import pytest

from snowflake.snowpark._internal.utils import TempObjectType

TEMP_OBJECT_NAME_PREFIX = "SNOWPARK_TEMP_"


def _random_name_for_temp_object(object_type: TempObjectType) -> str:
    return f"{TEMP_OBJECT_NAME_PREFIX}{object_type.value}_{os.urandom(9).hex().upper()}"


@pytest.fixture(scope="session", autouse=True)
def random_name_for_temp_object_mock():
    with mock.patch(
        "snowflake.ml.modeling._internal.snowpark_handlers.random_name_for_temp_object", _random_name_for_temp_object
    ) as _fixture:
        cp.register_pickle_by_value(inspect.getmodule(_random_name_for_temp_object))
        yield _fixture
