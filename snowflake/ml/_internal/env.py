import platform

from snowflake.ml import version

SOURCE = "SnowML"
VERSION = version.get_version()
PYTHON_VERSION = platform.python_version()
OS = platform.system()
