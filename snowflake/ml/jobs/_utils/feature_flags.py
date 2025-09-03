import os
from enum import Enum


class FeatureFlags(Enum):
    USE_SUBMIT_JOB_V2 = "MLRS_USE_SUBMIT_JOB_V2"
    ENABLE_IMAGE_VERSION_ENV_VAR = "MLRS_ENABLE_RUNTIME_VERSIONS"

    def is_enabled(self) -> bool:
        return os.getenv(self.value, "false").lower() == "true"

    def is_disabled(self) -> bool:
        return not self.is_enabled()

    def __str__(self) -> str:
        return self.value
