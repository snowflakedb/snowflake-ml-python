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
    try:
        # Modeling requires scikit-learn, which is an optional dependency. Importing the module here both
        # skips the patch when modeling is unavailable and binds the submodule on its parent namespace
        # package, which mock.patch needs to resolve the dotted path below.
        from snowflake.ml.modeling._internal.snowpark_implementations import (  # noqa: F401
            snowpark_handlers,
        )
    except ImportError:
        yield None
        return

    with mock.patch(
        "snowflake.ml.modeling._internal.snowpark_implementations.snowpark_handlers.random_name_for_temp_object",
        _random_name_for_temp_object,
    ) as _fixture:
        cp.register_pickle_by_value(inspect.getmodule(_random_name_for_temp_object))
        yield _fixture
