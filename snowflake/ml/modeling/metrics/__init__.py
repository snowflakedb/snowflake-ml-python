import os

import cloudpickle

from snowflake.ml._internal import init_utils
from snowflake.ml._internal.utils import result
from snowflake.ml.modeling._internal import optional_dependency

pkg_name = __name__

# The distributed metrics functions are built on scikit-learn (optional as of 2.0); fail fast with
# an actionable error when it is missing. These are functions, not estimators, so no deprecation
# warning is emitted.
optional_dependency.check_modeling_dependencies(pkg_name, emit_deprecation_warning=False)

pkg_dir = os.path.dirname(__file__)
exportable_functions = init_utils.fetch_functions_from_modules_in_pkg_dir(pkg_dir=pkg_dir, pkg_name=pkg_name)
for k, v in exportable_functions.items():
    globals()[k] = v

registered_modules = cloudpickle.list_registry_pickle_by_value()
if result not in registered_modules:
    cloudpickle.register_pickle_by_value(result)
