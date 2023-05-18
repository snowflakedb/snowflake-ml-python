import platform

from snowflake.ml import version

SOURCE = "SnowML"
VERSION = version.VERSION
PYTHON_VERSION = platform.python_version()
OS = platform.system()
