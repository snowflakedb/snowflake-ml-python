import os
import platform
import sys
import sysconfig

SOURCE = "SnowML"
PYTHON_VERSION = platform.python_version()
OS = platform.system()
IN_ML_RUNTIME_ENV_VAR = "IN_SPCS_ML_RUNTIME"
IN_ML_RUNTIME = os.getenv(IN_ML_RUNTIME_ENV_VAR)
USE_OPTIMIZED_DATA_INGESTOR = "USE_OPTIMIZED_DATA_INGESTOR"


def interpreter_is_free_threaded() -> bool:
    """Return whether this interpreter was built with the ``t`` ABI flag.

    ``platform.python_version()`` omits that flag. The check is the build, not
    whether the GIL is currently enabled.

    Returns:
        True when the ``t`` ABI flag is present.
    """
    abiflags = getattr(sys, "abiflags", "") or ""
    if "t" in abiflags:
        return True
    gil_disabled = sysconfig.get_config_var("Py_GIL_DISABLED")
    if gil_disabled is None:
        return False
    return str(gil_disabled) == "1"
