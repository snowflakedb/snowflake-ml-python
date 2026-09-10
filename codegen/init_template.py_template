import os

from snowflake.ml._internal import init_utils
from snowflake.ml.modeling._internal import optional_dependency

pkg_name = __name__

# Fail fast with an actionable error when a required optional dependency (scikit-learn, and
# xgboost/lightgbm for those subpackages) is missing, and warn that snowflake.ml.modeling
# estimators are deprecated when the import succeeds.
optional_dependency.check_modeling_dependencies(pkg_name)

pkg_dir = os.path.dirname(os.path.abspath(__file__))
exportable_classes = init_utils.fetch_classes_from_modules_in_pkg_dir(pkg_dir=pkg_dir, pkg_name=pkg_name)
for k, v in exportable_classes.items():
    globals()[k] = v
