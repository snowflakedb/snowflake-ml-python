import copy
from abc import abstractmethod
from typing import Any, Protocol, final

from snowflake.ml._internal import migrator_utils


class _BaseModelMetaMigratorProtocol(Protocol):
    source_version: str
    target_version: str

    @staticmethod
    @abstractmethod
    def upgrade(original_meta_dict: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class BaseModelMetaMigrator(_BaseModelMetaMigratorProtocol):
    @final
    def try_upgrade(self, original_meta_dict: dict[str, Any]) -> dict[str, Any]:
        loaded_meta_version = original_meta_dict.get("version", None)
        if not loaded_meta_version or str(loaded_meta_version) != self.source_version:
            raise NotImplementedError(
                f"Unknown or unsupported model metadata file with version {loaded_meta_version} found."
            )
        try:
            return self.upgrade(copy.deepcopy(original_meta_dict))
        except migrator_utils.UnableToUpgradeError as e:
            raise RuntimeError(
                f"Can not upgrade your model metadata from version {self.__class__.source_version} to"
                f" {self.__class__.target_version}."
                f"The latest version support the original version of Snowpark ML library is {e.last_supported_version}."
            )
