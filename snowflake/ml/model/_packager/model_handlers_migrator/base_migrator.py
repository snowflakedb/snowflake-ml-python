from abc import abstractmethod
from typing import Protocol, final

from snowflake.ml._internal import migrator_utils
from snowflake.ml.model._packager.model_meta import model_meta


class _BaseModelHandlerMigratorProtocol(Protocol):
    source_version: str
    target_version: str

    @staticmethod
    @abstractmethod
    def upgrade(
        name: str,
        model_meta: model_meta.ModelMetadata,
        model_blobs_dir_path: str,
    ) -> None:
        raise NotImplementedError


class BaseModelHandlerMigrator(_BaseModelHandlerMigratorProtocol):
    @final
    def try_upgrade(self, name: str, model_meta: model_meta.ModelMetadata, model_blobs_dir_path: str) -> None:
        assert (
            model_meta.models[name].handler_version == self.__class__.source_version
        ), "Incorrect source handler version found."
        try:
            self.upgrade(name=name, model_meta=model_meta, model_blobs_dir_path=model_blobs_dir_path)
            model_meta.models[name].handler_version = self.__class__.target_version
        except migrator_utils.UnableToUpgradeError as e:
            raise RuntimeError(
                f"Can not upgrade your model {name} from version {self.__class__.source_version} to"
                f" {self.__class__.target_version}."
                f"The latest version support the original version of Snowpark ML library is {e.last_supported_version}."
            )
