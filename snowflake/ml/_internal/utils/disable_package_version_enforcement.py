"""
Enables relax version:

The API and results of this might lead to more issues caused by
the different versioning from packages, such as sklearn, pandas, ...

Importing this file dynamically sets _relax_version = True

    >>> # explicitly import this package
    >>> from snowflake.ml._internal.utils import disable_package_version_enforcement  # noqa: F401
    >>> # now you can import other package normally without any version errors
    >>> from snowflake.ml.modeling.linear_model import LogisticRegression
"""

from snowflake.ml._internal.utils import pkg_version_utils

pkg_version_utils._relax_version = True
