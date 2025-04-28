import os

import cloudpickle

from snowflake.ml._internal import init_utils
from snowflake.ml._internal.utils import result

pkg_dir = os.path.dirname(__file__)
pkg_name = __name__
exportable_functions = init_utils.fetch_functions_from_modules_in_pkg_dir(pkg_dir=pkg_dir, pkg_name=pkg_name)
for k, v in exportable_functions.items():
    globals()[k] = v

registered_modules = cloudpickle.list_registry_pickle_by_value()
if result not in registered_modules:
    cloudpickle.register_pickle_by_value(result)
