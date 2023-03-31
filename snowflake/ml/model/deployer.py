from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import pandas as pd

from snowflake import snowpark
from snowflake.ml.model import udf_util
from snowflake.snowpark._internal import type_utils


class TargetPlatform:
    SNOWPARK = "snowpark"
    WAREHOUSE = "warehouse"


@dataclass
class Deployment:
    name: str
    model_dir_path: str
    platform: str
    options: Dict[str, str]


class DeploymentManager(ABC):
    """WIP: Intended to provide model deployment management."""

    @abstractmethod
    def create(self, name: str, model_dir_path: str, platform: str, options: Dict[str, str]) -> None:
        """Create a deployment.

        Args:
            name: Name of the deployment for the model.
            model_dir_path: Directory of the model.
            platform: Target platform to deploy the model.
            options: Each target platform will have their own specifications of options.
        """
        pass

    @abstractmethod
    def list(self) -> List[Deployment]:
        pass

    @abstractmethod
    def get(self, name: str) -> Optional[Deployment]:
        pass

    @abstractmethod
    def delete(self, name: str) -> None:
        pass


class LocalDeploymentManager(DeploymentManager):
    def __init__(self) -> None:
        self._storage: Dict[str, Deployment] = dict()

    def create(self, name: str, model_dir_path: str, platform: str, options: Dict[str, str]) -> None:
        self._storage[name] = Deployment(name, model_dir_path, platform, options)

    def list(self) -> List[Deployment]:
        return list(self._storage.values())

    def get(self, name: str) -> Optional[Deployment]:
        if name in self._storage:
            return self._storage[name]
        else:
            return None

    def delete(self, name: str) -> None:
        self._storage.pop(name)


class Deployer:
    """TODO: Better data modeling for deployment interface."""

    def __init__(self, session: snowpark.Session, manager: DeploymentManager) -> None:
        self._manager = manager
        self._session = session

    def _deploy(self, name: str, model_dir_path: str, platform: str, options: Dict[str, str]) -> None:
        """Actual deployment to Snowflake based on platform."""
        print(f"Model from {model_dir_path} is deployed as {name} to {platform} with options: {options}")
        # TODO: Support `options` with schema and subtype.
        if platform == TargetPlatform.WAREHOUSE:
            udf_util.deploy_to_warehouse(self._session, udf_name=name, model_dir_path=model_dir_path)
        else:
            raise NotImplementedError()

    def create_deployment(self, name: str, model_dir_path: str, platform: str, options: Dict[str, str]) -> None:
        is_success = False
        try:
            self._manager.create(name, model_dir_path, platform, options)
            self._deploy(name, model_dir_path, platform, options)
            is_success = True
        except Exception as e:
            print(e)
        finally:
            if not is_success:
                self._manager.delete(name)

    def list_deployments(self) -> List[Deployment]:
        return self._manager.list()

    def get_deployment(self, name: str) -> Optional[Deployment]:
        return self._manager.get(name)

    def delete_deployment(self, name: str) -> None:
        """Must be idopotement."""
        self._manager.delete(name)

    def predict(self, name: str, df: Union[pd.DataFrame, snowpark.DataFrame]) -> pd.DataFrame:
        if isinstance(df, pd.DataFrame):
            df = self._session.create_dataframe(df)
        d = self.get_deployment(name)
        if not d:
            raise ValueError(f"Deployment {name} does not exist.")
        return df.select(
            snowpark.functions.call_udf(
                name, *[type_utils.ColumnOrLiteral(snowpark.functions.col(x)) for x in df.columns]
            )
        ).to_pandas()
