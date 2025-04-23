import os
import platform

SOURCE = "SnowML"
PYTHON_VERSION = platform.python_version()
OS = platform.system()
IN_ML_RUNTIME_ENV_VAR = "IN_SPCS_ML_RUNTIME"
IN_ML_RUNTIME = os.getenv(IN_ML_RUNTIME_ENV_VAR)
USE_OPTIMIZED_DATA_INGESTOR = "USE_OPTIMIZED_DATA_INGESTOR"
