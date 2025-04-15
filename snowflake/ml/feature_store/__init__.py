import os

from snowflake.ml._internal import init_utils

from .access_manager import setup_feature_store

pkg_dir = os.path.dirname(__file__)
pkg_name = __name__
exportable_classes = init_utils.fetch_classes_from_modules_in_pkg_dir(pkg_dir=pkg_dir, pkg_name=pkg_name)
for k, v in exportable_classes.items():
    globals()[k] = v

__all__ = list(exportable_classes.keys()) + [
    "setup_feature_store",
]
