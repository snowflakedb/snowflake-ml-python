import sys
import warnings

from snowflake.ml.model._client.model.batch_inference_specs import (
    JobSpec,
    OutputSpec,
    SaveMode,
)
from snowflake.ml.model._client.model.model_impl import Model
from snowflake.ml.model._client.model.model_version_impl import ExportMode, ModelVersion
from snowflake.ml.model.models.huggingface_pipeline import HuggingFacePipelineModel
from snowflake.ml.model.volatility import Volatility

__all__ = [
    "Model",
    "ModelVersion",
    "ExportMode",
    "HuggingFacePipelineModel",
    "JobSpec",
    "OutputSpec",
    "SaveMode",
    "Volatility",
]

_deprecation_warning_msg_for_3_9 = (
    "Python 3.9 is deprecated in snowflake-ml-python. " "Please upgrade to Python 3.10 or greater."
)

warnings.filterwarnings(
    "once",
    message=_deprecation_warning_msg_for_3_9,
)

if sys.version_info.major == 3 and sys.version_info.minor == 9:
    warnings.warn(
        _deprecation_warning_msg_for_3_9,
        category=DeprecationWarning,
        stacklevel=2,
    )
